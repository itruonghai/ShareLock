"""
Precompute V-JEPA-2 / V-JEPA-2.1 video features and LLaMA-3-8B language features.

Supports two datasets via --dataset:
  egoexo4d  (default) — EgoExo4D Atomic Action Descriptions
  ego4d               — Ego4D egovid-5m (CSV with video_id + llava_cap)

Keys:
  EgoExo4D: "{take_uid}__{frame_idx}"  (frame-accurate, FPS-based)
  Ego4D:    "{video_id}"               (one clip per video_id)

──────────────────────────────────────────────────────────────────────────────
EgoExo4D usage:

  # Video features (multi-GPU)
  python precompute_video_features.py \\
      --dataset egoexo4d \\
      --video_root EgoExo/train_videos/takes \\
      --annotation_json EgoExo/annotations/atomic_descriptions_train.json \\
      --takes_json EgoExo/takes.json \\
      --split_file EgoExo/train_takes.txt \\
      --variant vjepa2.1_vitl_384 \\
      --output_dir precomputed_features_video \\
      --extract video --num_gpus 4 --batch_size 32 --num_workers 8

  # Language features (multi-GPU)
  python precompute_video_features.py \\
      --dataset egoexo4d \\
      --video_root EgoExo/train_videos/takes \\
      --annotation_json EgoExo/annotations/atomic_descriptions_train.json \\
      --takes_json EgoExo/takes.json \\
      --split_file EgoExo/train_takes.txt \\
      --language_model meta-llama/Meta-Llama-3-8B \\
      --caption_name atomic_train \\
      --output_dir precomputed_features_video \\
      --extract language --num_gpus 4

──────────────────────────────────────────────────────────────────────────────
Ego4D usage:

  # Video features
  python precompute_video_features.py \\
      --dataset ego4d \\
      --ego4d_root /path/to/ego4d \\
      --csv_file /path/to/ego4d/egovid-text.csv \\
      --variant vjepa2.1_vitl_384 \\
      --output_dir precomputed_features_ego4d \\
      --extract video --num_gpus 4 --batch_size 32 --num_workers 8

  # Language features
  python precompute_video_features.py \\
      --dataset ego4d \\
      --ego4d_root /path/to/ego4d \\
      --csv_file /path/to/ego4d/egovid-text.csv \\
      --language_model meta-llama/Meta-Llama-3-8B \\
      --caption_name egovid_train \\
      --output_dir precomputed_features_ego4d \\
      --extract language --num_gpus 4

  # Dry-run (single GPU, small batch)
  python precompute_video_features.py \\
      --dataset ego4d \\
      --ego4d_root /path/to/ego4d \\
      --csv_file /path/to/ego4d/egovid-text.csv \\
      --variant vjepa2.1_vitl_384 \\
      --output_dir precomputed_features_ego4d \\
      --extract video --num_gpus 1 --batch_size 4 --num_workers 4 --max_samples 8
"""

import os
import gc
import math
import argparse

import av
import torch
import torch.multiprocessing as mp
import tqdm

from featureutils.core import FeatureUtils
from sharelock.models.video_encoder import VideoEncoder
from sharelock.models.language_encoder import LanguageEncoder, EgoVLPv2TextEncoder
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict

from dataloader_video import (
    load_egoexo4d_annotations,
    load_ego4d_annotations,
    decode_clips_from_video,
    sample_frames_centered,
    sample_frames_adaptive,
    SamplingConfig,
    FilterConfig,
)


# ---------------------------------------------------------------------------
# Key helpers
# ---------------------------------------------------------------------------

# Per-video FPS cache: 65k samples → ~2,560 unique paths, avoids repeated av.open
_FPS_CACHE: dict = {}


def _get_fps(video_path: str) -> float:
    """Read FPS from a video file. Returns 30.0 on failure. Results are cached."""
    if video_path in _FPS_CACHE:
        return _FPS_CACHE[video_path]
    try:
        container = av.open(video_path)
        fps = float(container.streams.video[0].average_rate)
        container.close()
        _FPS_CACHE[video_path] = fps if fps > 0 else 30.0
    except Exception:
        _FPS_CACHE[video_path] = 30.0
    return _FPS_CACHE[video_path]


def make_key(take_uid: str, anchor_t: float, video_path: str) -> str:
    """Frame-accurate feature key: {take_uid}__{frame_idx}."""
    fps = _get_fps(video_path)
    frame_idx = int(round(anchor_t * fps))
    return f"{take_uid}__{frame_idx}"


# ---------------------------------------------------------------------------
# Video feature extraction
# ---------------------------------------------------------------------------

class VideoClipDataset(torch.utils.data.Dataset):
    """Wraps a list of annotation samples for async frame decoding via DataLoader.

    Works for both EgoExo4D (frame-accurate make_key) and Ego4D (video_id key).
    Samples that already carry a "key" field (Ego4D) skip make_key computation.
    Samples that carry a "clip_duration" field (Ego4D full-video sampling) use
    that value instead of the global cfg.clip_duration.
    """

    def __init__(self, samples: list, sampling_cfg: SamplingConfig, existing_keys: set):
        self.cfg = sampling_cfg
        # Pre-filter already-extracted keys so workers never touch them
        self.items = [
            s for s in samples
            if self._sample_key(s) not in existing_keys
        ]

    def _sample_key(self, s: dict) -> str:
        """Return the feature-store key for a sample."""
        if "key" in s:
            return s["key"]
        return make_key(s["take_uid"], s["timestamp"], s["video_path"])

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        s   = self.items[idx]
        key = self._sample_key(s)

        # Per-sample clip duration override (Ego4D: full-video sampling)
        if "clip_duration" in s:
            from dataclasses import replace
            cfg = replace(self.cfg, clip_duration=s["clip_duration"])
        else:
            cfg = self.cfg

        if cfg.mode == "adaptive":
            frames = sample_frames_adaptive(
                video_path=s["video_path"],
                timestamp=s["timestamp"],
                cfg=cfg,
                video_duration=s["video_duration"],
                prev_timestamp=s.get("prev_timestamp"),
                next_timestamp=s.get("next_timestamp"),
            )
        else:
            frames = sample_frames_centered(
                video_path=s["video_path"],
                timestamp=s["timestamp"],
                cfg=cfg,
                video_duration=s["video_duration"],
            )
        if frames is None:
            return None, key          # sentinel for failed decode
        return frames, key            # [T, C, H, W], str


def _collate_clips(batch):
    """Drop None samples (failed decodes) and stack the rest."""
    valid = [(frames, key) for frames, key in batch if frames is not None]
    if not valid:
        return None, []
    frames_list, keys = zip(*valid)
    return torch.stack(frames_list), list(keys)   # [B, T, C, H, W], [B]


def extract_video(rank: int, num_gpus: int, samples: list, args) -> None:
    """Encode video clips with VideoEncoder on the assigned GPU."""
    device = torch.device(f"cuda:{rank}")
    output_dir = f"{args.output_dir}/{args.variant}"

    feature_utils = FeatureUtils(
        base_dir=output_dir, staging_dir=args.cache_dir, feature_num=1
    )
    existing_keys = set(feature_utils.list_keys())
    print(f"[Video GPU {rank}] Existing features: {len(existing_keys)}")

    sampling_cfg = SamplingConfig(
        num_frames=args.num_frames,
        clip_duration=args.clip_duration,
        frame_size=VideoEncoder.VARIANTS[args.variant][2],
        mode=args.sampling_mode,
    )

    encoder = VideoEncoder(variant=args.variant).to(device)

    # Assign samples to this GPU by index, then apply max_samples cap
    assigned = [s for i, s in enumerate(samples) if i % num_gpus == rank]
    if args.max_samples is not None:
        assigned = assigned[: args.max_samples]
    print(f"[Video GPU {rank}] {len(assigned)} clips to encode")

    dataset = VideoClipDataset(assigned, sampling_cfg, existing_keys)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=_collate_clips,
        pin_memory=True,
        prefetch_factor=2 if args.num_workers > 0 else None,
    )
    print(f"[Video GPU {rank}] {len(dataset)} clips after filtering existing keys")

    for frames_batch, keys in tqdm.tqdm(loader, desc=f"[Video GPU {rank}]"):
        if frames_batch is None:
            continue
        frames_batch = frames_batch.to(device)      # [B, T, C, H, W]
        features = encoder(frames_batch)             # [B, embed_dim]
        for key, feat in zip(keys, features):
            feature_utils.save_feature(key, vision_features=feat.detach().cpu())

    feature_utils.save()
    del encoder
    torch.cuda.empty_cache()
    print(f"[Video GPU {rank}] Done.")


# ---------------------------------------------------------------------------
# Ego4D-optimized video extraction: group by source video, one open per file
# ---------------------------------------------------------------------------

def extract_video_ego4d(rank: int, num_gpus: int, samples: list, args) -> None:
    """
    Ego4D video feature extraction with overlapped IO and GPU encode.

    Pipeline:
      IO workers (ThreadPoolExecutor) → bounded clip_queue → GPU encoder thread

    The GPU encoder thread runs independently of the IO futures loop, so the
    GPU encodes one batch while IO workers decode the next videos concurrently.
    Previously the main thread alternated IO-wait → GPU-encode → IO-wait,
    leaving the GPU idle during each IO-wait (~25s per video).
    """
    import queue
    import threading
    import time as _time

    device     = torch.device(f"cuda:{rank}")
    output_dir = f"{args.output_dir}/{args.variant}"

    feature_utils = FeatureUtils(
        base_dir=output_dir, staging_dir=args.cache_dir, feature_num=1
    )
    existing_keys = set(feature_utils.list_keys())
    print(f"[Video GPU {rank}] Existing features: {len(existing_keys)}")

    # ── Group clips by source video; skip already-extracted ──────────────────
    groups: dict = defaultdict(list)
    for s in samples:
        if s["key"] not in existing_keys:
            groups[s["video_path"]].append(s)

    all_paths      = sorted(groups.keys())
    assigned_paths = [p for i, p in enumerate(all_paths) if i % num_gpus == rank]
    n_clips        = sum(len(groups[p]) for p in assigned_paths)
    print(f"[Video GPU {rank}] {len(assigned_paths)} source videos, "
          f"{n_clips} clips to encode")

    if args.max_samples is not None:
        capped, count = [], 0
        for p in assigned_paths:
            capped.append(p)
            count += len(groups[p])
            if count >= args.max_samples:
                break
        assigned_paths = capped

    sampling_cfg = SamplingConfig(
        num_frames=args.num_frames,
        clip_duration=args.clip_duration,
        frame_size=VideoEncoder.VARIANTS[args.variant][2],
        mode="centered",
    )
    encoder = VideoEncoder(variant=args.variant).to(device)

    clip_bar = tqdm.tqdm(
        total=n_clips, desc=f"[GPU {rank}] clips", unit="clip",
        dynamic_ncols=True,
    )
    vid_bar = tqdm.tqdm(
        total=len(assigned_paths), desc=f"[GPU {rank}] videos", unit="vid",
        dynamic_ncols=True, position=1, leave=True,
    )

    # ── Shared profiling state (written by GPU thread, read by main thread) ───
    _stats = {"t_gpu": 0.0, "n_flushes": 0, "t_io": 0.0, "n_vids": 0}
    _PROFILE_EVERY   = 5
    avg_clips_per_vid = n_clips / max(1, len(assigned_paths))

    # ── GPU encoder thread ────────────────────────────────────────────────────
    # Bounded queue: blocks main thread (backpressure) if GPU falls behind,
    # preventing unbounded CPU RAM growth from decoded-but-unenqueued clips.
    # Size = 4× batch_size gives the GPU ~4 batches of prefetch headroom.
    QUEUE_MAXSIZE = args.batch_size * 4
    clip_queue: queue.Queue = queue.Queue(maxsize=QUEUE_MAXSIZE)

    def _gpu_worker():
        frames_buf: list = []
        keys_buf:   list = []

        def _flush():
            if not frames_buf:
                return
            t0 = _time.perf_counter()
            with torch.no_grad():
                batch    = torch.stack(frames_buf).to(device)
                features = encoder(batch)
                torch.cuda.synchronize(device)
            for key, feat in zip(keys_buf, features):
                feature_utils.save_feature(key, vision_features=feat.detach().cpu())
            _stats["t_gpu"]     += _time.perf_counter() - t0
            _stats["n_flushes"] += 1
            frames_buf.clear()
            keys_buf.clear()

        while True:
            item = clip_queue.get()
            if item is None:          # sentinel: no more clips
                _flush()
                clip_queue.task_done()
                break
            frames, key = item
            clip_bar.update(1)
            if frames is not None:
                frames_buf.append(frames)
                keys_buf.append(key)
                if len(frames_buf) >= args.batch_size:
                    _flush()
            clip_queue.task_done()

    gpu_thread = threading.Thread(target=_gpu_worker, daemon=True)
    gpu_thread.start()

    # ── IO loop: main thread only feeds the queue ─────────────────────────────
    SUBMIT_WINDOW = args.num_workers * 2

    with ThreadPoolExecutor(max_workers=args.num_workers) as pool:
        pending:   dict = {}
        path_iter = iter(assigned_paths)

        def _fill_window():
            while len(pending) < SUBMIT_WINDOW:
                try:
                    vpath = next(path_iter)
                except StopIteration:
                    break
                f = pool.submit(decode_clips_from_video, vpath, groups[vpath], sampling_cfg)
                pending[f] = (vpath, len(groups[vpath]))

        _fill_window()
        while pending:
            t_wait = _time.perf_counter()
            for future in as_completed(pending):
                _stats["t_io"]  += _time.perf_counter() - t_wait
                _stats["n_vids"] += 1

                vpath, _ = pending.pop(future)
                for item in future.result():
                    clip_queue.put(item)   # blocks if GPU thread is behind

                vid_bar.update(1)
                vid_bar.set_postfix({"last": os.path.basename(vpath),
                                     "q": clip_queue.qsize()})

                # ── Bottleneck report ─────────────────────────────────────
                nv = _stats["n_vids"]
                nf = _stats["n_flushes"]
                if nv % _PROFILE_EVERY == 0 and nf > 0:
                    avg_gpu_ms      = _stats["t_gpu"] / nf * 1000
                    avg_io_wait_ms  = _stats["t_io"]  / nv * 1000
                    ms_clip_gpu     = avg_gpu_ms / args.batch_size
                    ms_clip_io_eff  = (avg_io_wait_ms / avg_clips_per_vid) / args.num_workers
                    gpu_ms_per_vid  = (avg_clips_per_vid / args.batch_size) * avg_gpu_ms
                    tqdm.tqdm.write(
                        f"[Bottleneck @ {nv} vids | {nf} flushes]\n"
                        f"  GPU : {avg_gpu_ms:.0f}ms/flush = {ms_clip_gpu:.1f}ms/clip\n"
                        f"  IO  : {avg_io_wait_ms:.0f}ms/vid-wait"
                        f" = {ms_clip_io_eff:.2f}ms/clip eff ({args.num_workers} workers)\n"
                        f"  Per-video: GPU={gpu_ms_per_vid/1000:.0f}s  "
                        f"IO-wait={avg_io_wait_ms/1000:.0f}s  "
                        f"(GPU {gpu_ms_per_vid/avg_io_wait_ms:.1f}x IO-wait)  "
                        f"queue={clip_queue.qsize()}/{QUEUE_MAXSIZE}\n"
                        f"  → {'** GPU-BOUND **' if ms_clip_gpu > ms_clip_io_eff else '** IO-BOUND **'}"
                    )

                _fill_window()
                break

    # Signal GPU thread to finish remaining clips and stop
    clip_queue.put(None)
    clip_queue.join()
    gpu_thread.join()

    clip_bar.close()
    vid_bar.close()
    feature_utils.save()
    del encoder
    torch.cuda.empty_cache()
    print(f"[Video GPU {rank}] Done.")


# ---------------------------------------------------------------------------
# Language feature extraction (mirrors precompute_features.py)
# ---------------------------------------------------------------------------

def extract_language(rank: int, num_gpus: int, captions: dict, args) -> None:
    """Encode captions with LanguageEncoder on the assigned GPU."""
    device = torch.device(f"cuda:{rank}")
    output_dir = (
        f"{args.output_dir}"
        f"/{args.language_model.split('/')[-1]}/{args.caption_name}"
    )

    feature_utils = FeatureUtils(
        base_dir=output_dir, staging_dir=args.cache_dir, feature_num=1
    )
    print(f"[Lang GPU {rank}] Existing features: {len(feature_utils.list_keys())}")

    if args.language_model == "egovlpv2":
        model = EgoVLPv2TextEncoder(
            args.egovlpv2_checkpoint, cache_dir=args.model_cache_dir
        ).to(device)
    else:
        model = LanguageEncoder(
            args.language_model, cache_dir=args.model_cache_dir
        ).to(device)

    items = [
        (key, caption)
        for idx, (key, caption) in enumerate(captions.items())
        if idx % num_gpus == rank and not feature_utils.exists(key)
    ]
    print(f"[Lang GPU {rank}] {len(items)} captions to encode")

    for batch_start in tqdm.tqdm(
        range(0, len(items), args.language_batch_size),
        desc=f"[Lang GPU {rank}]",
    ):
        batch = items[batch_start : batch_start + args.language_batch_size]
        batch_keys = [b[0] for b in batch]
        batch_texts = [b[1] for b in batch]

        features = model(batch_texts)
        if features.dim() == 1:
            features = features.unsqueeze(0)

        for key, feat in zip(batch_keys, features):
            feature_utils.save_feature(key, language_features=feat.unsqueeze(0))

    feature_utils.save()
    del model
    torch.cuda.empty_cache()
    print(f"[Lang GPU {rank}] Done.")


# ---------------------------------------------------------------------------
# Worker (one per GPU)
# ---------------------------------------------------------------------------

def _load_samples(args) -> list:
    """Load annotation samples for the selected dataset."""
    if args.dataset == "ego4d":
        import pathlib
        video_root = str(pathlib.Path(args.ego4d_root) / "v2" / "video_540ss")
        return load_ego4d_annotations(
            csv_file=args.csv_file,
            video_root=video_root,
            split_file=args.split_file,
        )
    else:  # egoexo4d
        filter_cfg = FilterConfig(
            keep_subject_C_only=True,
            drop_unsure=True,
            require_ego_visible=True,
            min_timestamp=2.0,
        )
        sampling_cfg = SamplingConfig(
            num_frames=args.num_frames,
            clip_duration=args.clip_duration,
            mode=args.sampling_mode,
        )
        return load_egoexo4d_annotations(
            annotation_json=args.annotation_json,
            takes_json=args.takes_json,
            video_root=args.video_root if args.video_root else "",
            filter_cfg=filter_cfg,
            sampling_cfg=sampling_cfg,
            split_file=args.split_file,
        )


def _sample_key(sample: dict) -> str:
    """Return the feature-store key for a sample (dataset-agnostic)."""
    if "key" in sample:
        return sample["key"]
    return make_key(sample["take_uid"], sample["timestamp"], sample["video_path"])


def worker(rank: int, args) -> None:
    """Load annotations once, then extract video and/or language features."""
    samples = _load_samples(args)

    if args.extract in ("video", "both"):
        if args.dataset == "ego4d":
            extract_video_ego4d(rank, args.num_gpus, samples, args)
        else:
            extract_video(rank, args.num_gpus, samples, args)
        gc.collect()
        torch.cuda.empty_cache()

    if args.extract in ("language", "both"):
        # Build {key: text} dict — keys must match video feature keys exactly
        captions: dict = {}
        for sample in samples:
            key = _sample_key(sample)
            if key not in captions:
                captions[key] = sample["text"]
        print(f"[Lang GPU {rank}] {len(captions)} unique captions")
        extract_language(rank, args.num_gpus, captions, args)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Precompute V-JEPA-2 video + language features (EgoExo4D or Ego4D)"
    )

    # ── Dataset selector ─────────────────────────────────────────────────────
    parser.add_argument("--dataset", type=str, default="egoexo4d",
                        choices=["egoexo4d", "ego4d"],
                        help="Dataset to process: egoexo4d (default) or ego4d")

    # ── EgoExo4D paths ────────────────────────────────────────────────────────
    parser.add_argument("--annotation_json", type=str, default=None,
                        help="[EgoExo4D] atomic_descriptions_{train|val}.json")
    parser.add_argument("--takes_json", type=str, default=None,
                        help="[EgoExo4D] takes.json")
    parser.add_argument("--video_root", type=str, default=None,
                        help="[EgoExo4D] Root of video files, e.g. EgoExo/train_videos/takes")

    # ── Ego4D paths ───────────────────────────────────────────────────────────
    parser.add_argument("--ego4d_root", type=str, default=None,
                        help="[Ego4D] Path to ego4d root directory "
                             "(videos are at ego4d_root/v2/video_540ss/)")
    parser.add_argument("--csv_file", type=str, default=None,
                        help="[Ego4D] Path to annotation CSV "
                             "(e.g. ego4d/egovid-text.csv or egovid-val.csv)")

    # ── Shared split filter ───────────────────────────────────────────────────
    parser.add_argument("--split_file", type=str, default=None,
                        help="Optional file of allowed take/video IDs (one per line)")

    # Model / variant
    parser.add_argument("--variant", type=str, default="vjepa2.1_vitl_384",
                        choices=list(VideoEncoder.VARIANTS.keys()),
                        help="V-JEPA-2/2.1 variant to use")
    parser.add_argument("--language_model", type=str,
                        default="meta-llama/Meta-Llama-3-8B")
    parser.add_argument("--egovlpv2_checkpoint", type=str, default=None,
                        help="Path to EgoVLPv2.pth (required when --language_model egovlpv2)")
    parser.add_argument("--caption_name", type=str, default="atomic_train",
                        help="Subdirectory name for language features")

    # Sampling
    parser.add_argument("--num_frames", type=int, default=16,
                        help="Frames per clip (V-JEPA-2 default: 16)")
    parser.add_argument("--clip_duration", type=float, default=4.0,
                        help="Clip window in seconds centered on annotation")

    # Output
    parser.add_argument("--output_dir", type=str, default="precomputed_features_video")
    parser.add_argument("--cache_dir", type=str,
                        default=os.environ.get("TMPDIR", None),
                        help="FeatureUtils staging directory")
    parser.add_argument("--model_cache_dir", type=str,
                        default=os.environ.get("HF_HOME", None),
                        help="HuggingFace model cache directory")

    # Extraction control
    parser.add_argument("--extract", type=str, default="both",
                        choices=["video", "language", "both"])
    parser.add_argument("--num_gpus", type=int, default=1)
    parser.add_argument("--gpu_id", type=int, default=None,
                        help="GPU ID for single-GPU mode (default: 0)")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Clips per GPU batch for video encoding")
    parser.add_argument("--num_workers", type=int, default=8,
                        help="DataLoader worker processes for async frame decoding")
    parser.add_argument("--language_batch_size", type=int, default=512,
                        help="Captions per GPU batch for language encoding")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Cap samples per GPU (useful for dry-runs)")
    parser.add_argument("--sampling_mode", type=str, default="centered",
                        choices=["centered", "adaptive"],
                        help="[EgoExo4D] Frame sampling: centered (fixed window) or "
                             "adaptive (Voronoi). Ego4D always samples the full clip.")

    args = parser.parse_args()

    # ── Validate required args per dataset ───────────────────────────────────
    if args.dataset == "egoexo4d":
        if args.annotation_json is None:
            parser.error("--annotation_json is required for --dataset egoexo4d")
        if args.takes_json is None:
            parser.error("--takes_json is required for --dataset egoexo4d")
    elif args.dataset == "ego4d":
        if args.ego4d_root is None:
            parser.error("--ego4d_root is required for --dataset ego4d")
        if args.csv_file is None:
            parser.error("--csv_file is required for --dataset ego4d")

    return args


if __name__ == "__main__":
    args = parse_args()

    print(f"Dataset: {args.dataset} | Extract: {args.extract} | GPUs: {args.num_gpus}")
    if args.extract in ("video", "both"):
        print(f"  Video variant: {args.variant} (batch_size={args.batch_size})")
    if args.extract in ("language", "both"):
        print(f"  Language model: {args.language_model} "
              f"(batch_size={args.language_batch_size})")

    if args.num_gpus > 1:
        mp.spawn(worker, nprocs=args.num_gpus, args=(args,))
    else:
        rank = args.gpu_id if args.gpu_id is not None else 0
        worker(rank, args)
