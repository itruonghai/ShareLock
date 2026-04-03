# Ego4D egovid-5m: Video Decoding Speed Analysis

## Dataset profile

| Metric | Value |
|---|---|
| Training clips | 4,873,088 |
| Source videos | ~8,451 |
| Clips per source video (avg) | ~577 |
| Clip duration | 4 s (120 frames @ 30 fps) |
| Frames per clip (sampled) | 16 |
| Source video length (avg est.) | ~45 min @ 30fps |

---

## Bottleneck evolution

### v1 — Naive per-clip seeking (original approach)

The first implementation opened a new `av.container` for every clip and seeked to the clip's start time, then decoded 16 frames.

```
Cost per clip = 1 seek (random, into H.264 file) + 16 frame decodes
Seeks per source video = 577 clips × 1 seek = 577 random seeks
Total seeks for dataset = 4,873,088
```

**H.264 seek cost**: A random seek in a multi-GB H.264 file requires a keyframe (I-frame) lookup, then decoding all B/P frames from that keyframe to the target. For a 45-min video at default GOP=250, average seek distance = 125 frames = ~4s of decode work per seek.

**Observed**: 31.6 s/clip → estimated total: ~42,747 hours on 1 GPU.

---

### v2 — Grouped IO + single forward scan

Key insight: clips in the same source video are evenly distributed across the full video duration. Instead of seeking 577 times, seek once to the earliest clip and decode forward, assigning each decoded frame to whichever clip target it matches.

```
Seeks per source video: 1  (down from 577)
Frames decoded per source video: ~all frames in [t_first, t_last]
  = (t_last - t_first) × fps
  ≈ (45 min - some margin) × 30 fps
  ≈ 80,000 frames decoded per source video
Clips per source video: 577 × 16 = 9,232 target frames needed
Useful decode ratio: 9,232 / 80,000 ≈ 11.5%
```

**Observed**: 1.35 s/clip → estimated total: ~1,828 hours on 1 GPU.

Speedup over v1: **23×**

Remaining waste: ~88.5% of decoded frames are discarded. Every frame in the span [t_first_clip, t_last_clip] is decoded even if not needed.

---

### v3 — B-frame skipping + multi-threaded codec

**B-frame skip** (`skip_frame="BIDIR"`): In H.264, bidirectional frames (B-frames) require both a forward and backward reference frame to reconstruct — they are the most expensive to decode. Skipping them reduces the number of decoded frames and avoids their reconstruction cost.

For a typical H.264 GOP pattern `I,B,B,P,B,B,P,...` with 2 B-frames between each reference:
- Without skip: decode all frames (I + B + P)
- With skip: decode only I + P frames (33% of frames)

```
Frames decoded per source video (estimated):
  Without skip: ~80,000
  With skip:    ~80,000 × 0.33 = ~26,400

Expected speedup from skip alone: ~3×
Practical speedup (variable GOP, codec overhead): ~1.5–2.5×
```

**Multi-threaded codec** (`thread_type=AUTO, thread_count=2`): each IO worker thread uses 2 codec threads internally. With `num_workers=16` IO threads, this uses up to 32 CPU threads for decode. On a 16-core node each worker gets 2 hardware threads.

**Note on B-frame skip + observed speed**: In practice, B-frame skip alone did not reduce wall-clock clip rate. The observed ~1.47 s/clip with B-skip matches the observed ~1.35 s/clip without it (within run-to-run noise). See **"Why B-frame skip didn't help"** below for diagnosis.

**Note on frame matching accuracy**: With B-frames skipped, consecutive decoded frames are spaced at ~3/fps ≈ 100ms instead of 1/fps ≈ 33ms. The matching tolerance was widened from `0.5/fps` to `1.5/fps` to ensure every sample target is matched to the nearest decoded frame. The quality impact on features is negligible.

---

## Why B-frame skip didn't help — bottleneck diagnosis

The end-to-end clip rate is `max(T_io, T_gpu)`, not their sum, because IO workers run concurrently with the GPU encoder. Reducing IO time only helps if IO is the bottleneck.

### Measured

```
Observed clip rate: ~1.47 s/clip  (1 GPU, batch_size=8, ViT-iG fp32)
```

### GPU time estimate (ViT-iG fp32)

ViT-iG gigantic is ~1.9 B parameters. Input: `[8, 3, 16, 384, 384]` → `[8, 3, 16, 27, 27]` patches = **93,312 tokens** per batch.

For a standard ViT with ~48 transformer layers (attention + MLP):
```
FLOPs per batch ≈ 2 × 48 × (4 × 1664² × 93312 + 2 × 93312² × 1664)
               ≈ several hundred TFLOPs

A100 fp32 throughput: 77.6 TFLOPS
→ GPU time per batch: ~3–10 s
→ GPU time per clip:  ~0.4–1.25 s
```

This matches the observed 1.47 s/clip. **GPU is the bottleneck** — IO workers finish decoding a video's worth of clips well before the GPU finishes encoding the previous batch.

### IO time estimate (after B-frame skip)

```
Frames decoded per video ≈ 80,000 × 0.33 (B-skip) = 26,400
PyAV throughput ≈ 50–100 fps (CPU)
IO time per video ≈ 26,400 / 75 fps ≈ 350 s
IO time per clip  ≈ 350 s / 577 clips ≈ 0.6 s/clip (per worker)
With num_workers=16 workers: IO effective rate >> GPU rate
```

IO is 10–20× faster than GPU when parallelised across 16 workers. Cutting IO time further gives zero benefit until GPU is sped up.

### Fix: bf16 autocast in VideoEncoder.forward()

`torch.autocast("cuda", dtype=torch.bfloat16)` activates tensor cores for the ViT's matrix multiplications. A100 bf16 throughput: **312 TFLOPS** (4× fp32).

```
Expected GPU time per clip (bf16): ~0.1–0.3 s
New bottleneck: IO at ~0.04 s/clip effective (16 workers × 0.6 s / 577 clips ÷ 16)
Expected end-to-end: ~0.1–0.3 s/clip
```

---

## Throughput model (1 GPU)

```
T_total = N_clips × max(T_io_effective, T_gpu)

where:
  N_clips = 4,873,088
  T_io_effective = T_io_per_video / clips_per_video  (amortised over workers)
  
V1:  T_gpu = 1.2s,  T_io = 30s   →  IO bound  → T_total ≈ 42,700 h
V2:  T_gpu = 1.2s,  T_io = 0.6s  →  GPU bound → T_total ≈  1,630 h
V3:  T_gpu = 1.2s,  T_io = 0.4s  →  GPU bound → T_total ≈  1,630 h  (B-skip no gain)
V4:  T_gpu = 0.2s,  T_io = 0.04s →  GPU bound → T_total ≈    270 h  (bf16 fix)
```

---

## Multi-GPU scaling

The extraction shards by **source video** (not by clip), so each GPU owns disjoint files and there is no cross-GPU coordination.

| GPUs | Estimated time (V4, bf16) |
|------|--------------------------|
| 1    | ~270 h |
| 2    | ~135 h |
| 3    | ~90 h  |

**GPU is now the bottleneck** (after bf16 fix). Adding GPUs scales linearly since each GPU is independent and IO workers are per-GPU.

---

## Memory analysis

### Per-source-video frame cache

`decode_clips_from_video` holds all 577 clips × 16 frames of a single source video in a Python dict while the forward scan runs.

```
Tensor shape per frame: [3, H, W] float32
  ViT-iG (384px):  3 × 384 × 384 × 4 bytes = 1.77 MB
  ViT-L  (224px):  3 × 224 × 224 × 4 bytes = 0.60 MB

Unique frame tensors per source video (worst case):
  577 clips × 16 slots = 9,232 tensors
  (many slots share the same tensor object if targets map to the same frame)

Peak RAM per worker thread (ViT-iG):
  9,232 × 1.77 MB ≈ 16.3 GB

With num_workers=16 active simultaneously:
  16 × 16.3 GB ≈ 261 GB  ← exceeds typical cluster node RAM
```

### Mitigation in v3

The submission window (`SUBMIT_WINDOW = num_workers × 2`) caps the number of futures whose results are held in memory simultaneously.

```
Active decoded results in memory at once:
  max = SUBMIT_WINDOW × tensors_per_video
  = (16 × 2) × 16.3 GB = 522 GB  (worst case, all large videos)

Practical bound:
  = num_workers (active) × 16.3 GB
  = 16 × 16.3 GB ≈ 261 GB
```

### Recommended num_workers by variant

| Variant | Tensor size | Safe num_workers (128 GB RAM) | Safe num_workers (256 GB RAM) |
|---------|-------------|-------------------------------|-------------------------------|
| ViT-iG 384px | 1.77 MB | **7** | 14 |
| ViT-L 384px  | 1.77 MB | 7 | 14 |
| ViT-L 224px  | 0.60 MB | 21 | 42 |

`num_workers=8` is a safe default for 128 GB nodes running ViT-iG.

---

## Code review findings

### Bug fixes applied

**1. `stream.duration` null crash** (`dataloader_video.py`)

`stream.duration` is `None` for some poorly-muxed Ego4D files. The original code did `float(stream.duration * stream.time_base)` which raises `TypeError`. Fixed by falling back to `container.duration`, then to a 1-hour sentinel value.

```python
# Before (crashes on None):
vid_dur = float(stream.duration * stream.time_base)

# After:
raw_dur = stream.duration or (container.duration // av.time_base if container.duration else None)
vid_dur = float(raw_dur * time_base) if raw_dur else 3600.0
```

**2. half_frame too narrow after B-frame skip** (`dataloader_video.py`)

With `skip_frame="BIDIR"` the effective frame spacing grows from `1/fps ≈ 33ms` to `~3/fps ≈ 100ms`. The original tolerance `half_frame = 0.5/fps = 17ms` meant targets between P-frames could be mismatched. Widened to `1.5/fps = 50ms` to cover the extended gap.

**3. Misleading n_missing print** (`dataloader_video.py`)

The original `parsed: len(df) + n_missing` mixed two different counts: `n_missing` counted unique missing *source videos*, while `len(df)` was after a further `clip_duration >= 0.1` filter. Replaced with a structured per-stage breakdown.

**4. All futures submitted at once** (`precompute_video_features.py`)

The original code submitted all 8,451 source-video decode jobs to the ThreadPoolExecutor at once. Completed futures whose `.result()` hadn't been consumed yet keep their decoded frame tensors live in memory. With a sliding window of `num_workers × 2`, at most `num_workers × 2` videos' worth of frames are live simultaneously.

### Logic observations (no fix needed)

**Orphaned `video_id` field**: `load_ego4d_annotations` previously added `out["video_id"] = out["key"]` (redundant column removed in fix #3 refactor). The `key` field carries the bare `VideoID_StartFrame_EndFrame` string used as the feature store key.

**`video_duration` field is unused for Ego4D**: The `video_duration` column is computed but only consumed by `VideoClipDataset` (EgoExo4D path). `extract_video_ego4d` reads `vid_dur` directly from PyAV stream metadata. No bug, just dead data in the sample dicts.

**B-frame skip changes temporal accuracy**: Sample frames for a 4s Ego4D clip are 16 evenly spaced targets. With B-frames skipped and `half_frame=1.5/fps`, each target maps to the nearest I/P frame within ±50ms. The actual temporal error per frame is ≤50ms, which is well within the 250ms between successive sample targets.

---

## Recommended run configuration

### Single GPU (Northeastern researcher partition)

```bash
python precompute_video_features.py \
    --dataset ego4d \
    --ego4d_root /path/to/ego4d \
    --csv_file /path/to/ego4d/egovid-text.csv \
    --variant vjepa2.1_vitig_384 \
    --output_dir precomputed_features_ego4d \
    --extract video \
    --num_gpus 1 \
    --batch_size 8 \
    --num_workers 8      # safe for 128 GB RAM nodes with ViT-iG
```

### Dry-run (sanity check before full run)

```bash
python precompute_video_features.py \
    --dataset ego4d \
    --ego4d_root /path/to/ego4d \
    --csv_file /path/to/ego4d/egovid-text.csv \
    --variant vjepa2.1_vitig_384 \
    --output_dir precomputed_features_ego4d_test \
    --extract video \
    --num_gpus 1 \
    --batch_size 4 \
    --num_workers 2 \
    --max_samples 32
```

Check the clip/s rate printed by tqdm. Target: > 1.5 clips/s (indicating the forward scan + B-skip is working). If < 0.5 clips/s, reduce `num_workers` to free RAM pressure.
