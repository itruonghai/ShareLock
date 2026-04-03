"""
EgoExo4D Atomic Action Descriptions — Dataloader
==================================================
Annotation format (atomic_descriptions_train.json):
{
  "annotations": {
    "<take_uid>": [                          # list of annotators for this take
      {
        "annotation_uid": "...",
        "annotator_id": "...",
        "rejected": false,
        "descriptions": [
          {
            "text": "C touches the screw with his right hand.",
            "timestamp": 18.48,             # single point, within 1-2s of action
            "subject": "C",                  # "C" = camera wearer, "O" = other
            "ego_visible": true,             # is the action visible in ego view?
            "best_exo": {"cam_id": "cam02"}, # best exo camera (ignored here)
            "unsure": false                  # annotator uncertainty flag
          },
          ...
        ]
      },
      ...  # second annotator for same take
    ]
  }
}

Video structure on disk (EgoExo4D):
  {video_root}/{take_name}/frame_aligned_videos/downscaled/448/aria01_214-1.mp4
  (filename taken from takes.json → frame_aligned_videos.aria01.rgb.relative_path)

Key design decisions:
  1. Filter ego_visible=False  → avoids blank/uninformative egocentric frames
  2. Filter unsure=True        → removes noisy annotations
  3. Filter subject != "C"     → keep camera-wearer actions only (egocentric focus)
  4. Multi-annotator positives → group descriptions by (take_uid, timestamp_bucket)
                                  for EgoNCE multi-positive loss
  5. Window strategy           → center [t - half_dur, t + half_dur] on timestamp,
                                  clamp to video boundaries (no drop, no padding)
  6. Sparse uniform sampling   → V-JEPA-2 style: linspace over window → 16 frames
"""

import os
import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import av
import numpy as np
import torch
import tqdm
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class SamplingConfig:
    num_frames: int = 16           # total frames per clip (V-JEPA-2 default)
    clip_duration: float = 4.0    # seconds centered on annotation timestamp
    frame_size: int = 224         # spatial resolution
    mode: str = "centered"        # "centered" | "adaptive" (Voronoi neighbor-aware)
    min_adaptive_half: float = 0.0  # min half-window for adaptive mode (0 = disabled)

@dataclass
class FilterConfig:
    keep_subject_C_only: bool = True   # only camera-wearer actions
    drop_unsure: bool = True           # drop annotator-uncertain descriptions
    require_ego_visible: bool = True   # CRITICAL: drop if ego camera can't see it
    min_timestamp: float = 2.0        # skip annotations too close to video start
    # (need at least half clip_duration of headroom)


# ---------------------------------------------------------------------------
# Video frame sampler (V-JEPA-2 style sparse uniform)
# ---------------------------------------------------------------------------

def get_video_duration(video_path: str) -> Optional[float]:
    try:
        container = av.open(video_path)
        stream = container.streams.video[0]
        duration = float(stream.duration * stream.time_base)
        container.close()
        return duration
    except Exception as e:
        print(f"[WARN] Cannot read duration: {video_path} — {e}")
        return None


def sample_frames_centered(
    video_path: str,
    timestamp: float,
    cfg: SamplingConfig,
    video_duration: Optional[float] = None,
) -> Optional[torch.Tensor]:
    """
    Sample cfg.num_frames frames centered on `timestamp`.
    Uses V-JEPA-2 sparse uniform sampling: linspace over [t_start, t_end].
    Returns [T, C, H, W] or None on failure.
    """
    if video_duration is None:
        video_duration = get_video_duration(video_path)
    if video_duration is None:
        return None

    half = cfg.clip_duration / 2.0

    # Compute window, then clamp to [0, video_duration]
    # Clamp = shift window inward while preserving clip_duration
    t_start = timestamp - half
    t_end   = timestamp + half

    if t_start < 0:
        t_start = 0.0
        t_end   = min(cfg.clip_duration, video_duration)
    if t_end > video_duration:
        t_end   = video_duration
        t_start = max(0.0, video_duration - cfg.clip_duration)

    # V-JEPA-2 sparse uniform: sample num_frames evenly in [t_start, t_end]
    sample_times = np.linspace(t_start, t_end, cfg.num_frames)

    frames = _decode_at_timestamps(video_path, sample_times, cfg.frame_size)
    if frames is None or len(frames) == 0:
        return None

    # Pad with last frame if decoding missed some timestamps
    while len(frames) < cfg.num_frames:
        frames.append(frames[-1])
    frames = frames[:cfg.num_frames]

    return torch.stack(frames, dim=0)  # [T, C, H, W]


def sample_frames_adaptive(
    video_path: str,
    timestamp: float,
    cfg: SamplingConfig,
    video_duration: Optional[float] = None,
    prev_timestamp: Optional[float] = None,
    next_timestamp: Optional[float] = None,
) -> Optional[torch.Tensor]:
    """
    Voronoi-based adaptive sampling. Shrinks the clip window to the midpoints
    between neighboring annotation timestamps, eliminating inter-clip overlap.

    Falls back to the full centered window when no neighbor exists (first/last
    annotation in a take, or single-annotation takes).

    Returns [T, C, H, W] or None on failure.
    """
    if video_duration is None:
        video_duration = get_video_duration(video_path)
    if video_duration is None:
        return None

    max_half = cfg.clip_duration / 2.0

    # Voronoi boundaries: midpoint to left/right neighbors, capped at max_half
    left = (
        max(timestamp - max_half, (prev_timestamp + timestamp) / 2.0)
        if prev_timestamp is not None
        else timestamp - max_half
    )
    right = (
        min(timestamp + max_half, (timestamp + next_timestamp) / 2.0)
        if next_timestamp is not None
        else timestamp + max_half
    )

    # Optional minimum-window floor — expands to min_adaptive_half on each side,
    # clamped to video bounds. May create slight overlap for very dense takes (
    # neighbor gap < 2 * min_adaptive_half), which is an explicit trade-off.
    if cfg.min_adaptive_half > 0.0 and (right - left) < 2.0 * cfg.min_adaptive_half:
        left  = max(0.0, timestamp - cfg.min_adaptive_half)
        right = min(video_duration, timestamp + cfg.min_adaptive_half)

    left  = max(0.0, left)
    right = min(video_duration, right)

    # Degenerate fallback (shouldn't happen given upstream filters)
    if left >= right:
        half  = max(cfg.min_adaptive_half, 0.1)
        left  = max(0.0, timestamp - half)
        right = min(video_duration, timestamp + half)

    sample_times = np.linspace(left, right, cfg.num_frames)
    frames = _decode_at_timestamps(video_path, sample_times, cfg.frame_size)
    if frames is None or len(frames) == 0:
        return None

    while len(frames) < cfg.num_frames:
        frames.append(frames[-1])
    return torch.stack(frames[:cfg.num_frames], dim=0)  # [T, C, H, W]


def _decode_at_timestamps(
    video_path: str,
    timestamps: np.ndarray,
    frame_size: int,
) -> Optional[list]:
    """Seek-based frame decoding using PyAV."""

    tfm = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((frame_size, frame_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    frames = []
    prev_frame = None

    try:
        container = av.open(video_path)
        stream = container.streams.video[0]

        for t in timestamps:
            t = float(np.clip(t, 0.0, float(stream.duration * stream.time_base) - 1e-4))
            pts = int(t / float(stream.time_base))

            try:
                container.seek(pts, stream=stream, any_frame=False)
                for frame in container.decode(video=0):
                    # Skip frames that are before our target pts
                    if frame.pts is not None and frame.pts < pts:
                        continue
                    img = frame.to_ndarray(format="rgb24")
                    tensor = tfm(img)
                    frames.append(tensor)
                    prev_frame = tensor
                    break
            except Exception:
                if prev_frame is not None:
                    frames.append(prev_frame)

        container.close()
    except Exception as e:
        print(f"[WARN] Decode error {video_path}: {e}")
        return None

    return frames


# ---------------------------------------------------------------------------
# Annotation loader
# ---------------------------------------------------------------------------

def load_egoexo4d_annotations(
    annotation_json: str,
    takes_json: str,           # takes.json — maps take_uid → take_name, duration
    video_root: str,
    filter_cfg: FilterConfig,
    sampling_cfg: SamplingConfig,
    split_file: Optional[str] = None,  # train_takes.txt / val_takes.txt
) -> list:
    """
    Parse atomic_descriptions_{train|val}.json and return a flat list of samples.

    Each sample:
    {
      "take_uid":        str,
      "video_path":      str,
      "timestamp":       float,
      "text":            str,
      "multi_positives": [str, ...],   # other annotators' text at same timestamp
      "video_duration":  float,
      "prev_timestamp":  float | None, # neighbor timestamps for adaptive sampling
      "next_timestamp":  float | None,
    }

    Multi-positives: descriptions from different annotators within ±1s of the
    same timestamp are grouped together. These are used as positive pairs in
    EgoNCE loss — they describe the same action in different words.
    """

    # --- Load split file (optional): restrict to official train/val takes ----
    valid_take_names: Optional[set] = None
    if split_file is not None:
        with open(split_file, "r") as f:
            valid_take_names = {line.strip() for line in f if line.strip()}
        print(f"[EgoExo4D] Split filter: {len(valid_take_names)} takes from {split_file}")

    # --- Load takes metadata: take_uid → {take_name, duration_sec} ----------
    with open(takes_json, "r") as f:
        takes_raw = json.load(f)

    # takes.json is a list of take objects
    takes_meta = {}
    for take in takes_raw:
        rgb_rel = (
            take.get("frame_aligned_videos", {})
                .get("aria01", {})
                .get("rgb", {})
                .get("relative_path", None)
        )
        takes_meta[take["take_uid"]] = {
            "take_name": take["take_name"],
            "duration_sec": take.get("duration_sec", None),
            "rgb_relative_path": rgb_rel,
        }

    # --- Load atomic descriptions ------------------------------------------
    with open(annotation_json, "r") as f:
        data = json.load(f)
    annotations = data["annotations"]  # dict: take_uid → [annotator_obj, ...]

    half = sampling_cfg.clip_duration / 2.0
    all_samples = []

    for take_uid, annotator_list in annotations.items():
        if take_uid not in takes_meta:
            continue

        take_name = takes_meta[take_uid]["take_name"]

        # Skip takes not in the official split (faster than video_path.exists())
        if valid_take_names is not None and take_name not in valid_take_names:
            continue
        duration  = takes_meta[take_uid]["duration_sec"]

        # Use aria01 RGB filename under the standard downscaled/448 directory
        rgb_rel = takes_meta[take_uid]["rgb_relative_path"]
        if rgb_rel is None:
            continue
        video_path = Path(video_root) / take_name / \
                     "frame_aligned_videos" / "downscaled" / "448" / Path(rgb_rel).name

        if not video_path.exists():
            continue

        # Get video duration from file if not in metadata
        if duration is None:
            duration = get_video_duration(str(video_path))
        if duration is None:
            continue

        # --- Collect all valid descriptions per annotator ------------------
        # Structure: {timestamp_bucket → [text, ...]}
        # Bucket = round to nearest 0.5s → groups descriptions within ±0.25s
        bucket_texts: dict = {}

        for annotator in annotator_list:
            if annotator.get("rejected", False):
                continue

            for desc in annotator.get("descriptions", []):
                # Apply filters
                if filter_cfg.drop_unsure and desc.get("unsure", False):
                    continue
                if filter_cfg.keep_subject_C_only and \
                   desc.get("subject", "C") != "C":
                    continue
                if filter_cfg.require_ego_visible and \
                   not desc.get("ego_visible", True):
                    continue

                t = float(desc["timestamp"])
                text = desc["text"].strip()

                if not text:
                    continue

                # Skip timestamps too close to video boundaries
                if t < filter_cfg.min_timestamp:
                    continue
                if t > duration - half:
                    continue

                # Bucket by 1s floor to group multi-annotator positives
                # math.floor gives deterministic buckets (avoids banker's rounding edge cases)
                bucket = float(math.floor(t))  # e.g. t=10.49 and t=10.82 both → 10.0
                if bucket not in bucket_texts:
                    bucket_texts[bucket] = []
                bucket_texts[bucket].append((t, text))

        # --- Build samples from buckets ------------------------------------
        take_samples = []
        for bucket, entries in bucket_texts.items():
            timestamps = [e[0] for e in entries]
            texts      = [e[1] for e in entries]
            anchor_t   = float(np.median(timestamps))
            take_samples.append({
                "take_uid":        take_uid,
                "video_path":      str(video_path),
                "timestamp":       anchor_t,
                "text":            texts[0],
                "multi_positives": texts[1:] if len(texts) > 1 else [],
                "video_duration":  duration,
                "prev_timestamp":  None,   # populated below after sorting
                "next_timestamp":  None,
            })

        # Sort by timestamp within take, then inject Voronoi neighbor info
        take_samples.sort(key=lambda s: s["timestamp"])
        for i, s in enumerate(take_samples):
            s["prev_timestamp"] = take_samples[i - 1]["timestamp"] if i > 0 else None
            s["next_timestamp"] = take_samples[i + 1]["timestamp"] if i < len(take_samples) - 1 else None

        all_samples.extend(take_samples)

    print(f"[EgoExo4D] Loaded {len(all_samples)} samples from "
          f"{len(annotations)} takes")
    print(f"[EgoExo4D] Samples with multi-positives: "
          f"{sum(1 for s in all_samples if s['multi_positives'])}")

    return all_samples


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class EgoExo4DAtomicDataset(Dataset):
    """
    EgoExo4D Atomic Action Descriptions dataset for ShareLock-style
    text-video alignment training.

    Returns per-sample:
      frames           [T, C, H, W]   — V-JEPA-2 input
      input_ids        [L]            — tokenized primary text
      attention_mask   [L]            — attention mask
      text             str            — primary description
      multi_positives  list[str]      — other annotators' descriptions (for EgoNCE)
      take_uid         str            — for cross-take negative sampling
    """

    def __init__(
        self,
        annotation_json: str,       # atomic_descriptions_train.json
        takes_json: str,            # takes.json
        video_root: str,
        tokenizer,
        split_file: Optional[str] = None,  # train_takes.txt / val_takes.txt
        sampling_cfg: SamplingConfig = SamplingConfig(),
        filter_cfg: FilterConfig = FilterConfig(),
        max_text_len: int = 77,
        augment_text: bool = True,  # randomly pick from multi_positives
    ):
        self.tokenizer    = tokenizer
        self.sampling_cfg = sampling_cfg
        self.max_text_len = max_text_len
        self.augment_text = augment_text

        self.samples = load_egoexo4d_annotations(
            annotation_json, takes_json, video_root,
            filter_cfg, sampling_cfg, split_file=split_file,
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]

        # --- Video frames --------------------------------------------------
        if self.sampling_cfg.mode == "adaptive":
            frames = sample_frames_adaptive(
                video_path=sample["video_path"],
                timestamp=sample["timestamp"],
                cfg=self.sampling_cfg,
                video_duration=sample["video_duration"],
                prev_timestamp=sample.get("prev_timestamp"),
                next_timestamp=sample.get("next_timestamp"),
            )
        else:
            frames = sample_frames_centered(
                video_path=sample["video_path"],
                timestamp=sample["timestamp"],
                cfg=self.sampling_cfg,
                video_duration=sample["video_duration"],
            )

        if frames is None:
            # Fallback to next sample — avoids returning None to DataLoader
            return self.__getitem__((idx + 1) % len(self))

        # --- Text: optionally augment by sampling from multi-positives -----
        if self.augment_text and sample["multi_positives"]:
            import random
            all_texts = [sample["text"]] + sample["multi_positives"]
            text = random.choice(all_texts)
        else:
            text = sample["text"]

        # --- Tokenize ------------------------------------------------------
        tokens = self.tokenizer(
            text,
            max_length=self.max_text_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "frames":          frames,                               # [T, C, H, W]
            "input_ids":       tokens["input_ids"].squeeze(0),       # [L]
            "attention_mask":  tokens["attention_mask"].squeeze(0),  # [L]
            "text":            text,
            "multi_positives": sample["multi_positives"],
            "take_uid":        sample["take_uid"],
            "timestamp":       sample["timestamp"],
        }


# ---------------------------------------------------------------------------
# Collate: handles variable-length multi_positives
# ---------------------------------------------------------------------------

def egoexo_collate_fn(batch: list) -> dict:
    batch = [b for b in batch if b is not None and "frames" in b]
    if not batch:
        return {}

    return {
        "frames":          torch.stack([b["frames"] for b in batch]),         # [B,T,C,H,W]
        "input_ids":       torch.stack([b["input_ids"] for b in batch]),      # [B,L]
        "attention_mask":  torch.stack([b["attention_mask"] for b in batch]), # [B,L]
        "texts":           [b["text"] for b in batch],
        "multi_positives": [b["multi_positives"] for b in batch],  # list of lists
        "take_uids":       [b["take_uid"] for b in batch],
        "timestamps":      [b["timestamp"] for b in batch],
    }


# ---------------------------------------------------------------------------
# Diagnostic: validate dataset before training
# ---------------------------------------------------------------------------

def run_dataset_diagnostics(dataset: EgoExo4DAtomicDataset, n_samples: int = 5):
    """
    Sanity check: print sample info and verify frame shapes.
    Run this before training to catch path/duration issues early.
    """
    print("\n=== EgoExo4D Dataset Diagnostics ===")
    print(f"Total samples:  {len(dataset)}")

    multi_pos_count = sum(1 for s in dataset.samples if s["multi_positives"])
    print(f"With multi-pos: {multi_pos_count} ({100*multi_pos_count/len(dataset):.1f}%)")

    durations = [s["video_duration"] for s in dataset.samples]
    print(f"Video duration: min={min(durations):.1f}s  "
          f"max={max(durations):.1f}s  mean={np.mean(durations):.1f}s")

    timestamps = [s["timestamp"] for s in dataset.samples]
    print(f"Timestamps:     min={min(timestamps):.1f}s  "
          f"max={max(timestamps):.1f}s")

    print(f"\nSample {n_samples} items:")
    for i in range(min(n_samples, len(dataset))):
        item = dataset[i]
        print(f"  [{i}] frames={tuple(item['frames'].shape)}  "
              f"text='{item['text'][:60]}'  "
              f"multi_pos={len(item['multi_positives'])}")

    print("=================================\n")


# ---------------------------------------------------------------------------
# Usage
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from transformers import AutoTokenizer

    sampling_cfg = SamplingConfig(
        num_frames=16,
        clip_duration=4.0,   # 2s before + 2s after annotation point
        frame_size=224,
    )

    filter_cfg = FilterConfig(
        keep_subject_C_only=True,   # egocentric focus
        drop_unsure=True,
        require_ego_visible=True,   # critical: only visible ego actions
        min_timestamp=2.0,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        "sentence-transformers/all-mpnet-base-v2"
    )

    dataset = EgoExo4DAtomicDataset(
        annotation_json="EgoExo/annotations/atomic_descriptions_train.json",
        takes_json="EgoExo/takes.json",
        video_root="EgoExo/train_videos/takes",
        split_file="EgoExo/train_takes.txt",
        tokenizer=tokenizer,
        sampling_cfg=sampling_cfg,
        filter_cfg=filter_cfg,
        augment_text=True,   # randomly sample from multi-positive texts
    )

    # Always run diagnostics before training
    run_dataset_diagnostics(dataset, n_samples=5)

    loader = DataLoader(
        dataset,
        batch_size=64,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        collate_fn=egoexo_collate_fn,
        prefetch_factor=2,
    )

    for batch in loader:
        print("frames:     ", batch["frames"].shape)    # [64, 16, 3, 224, 224]
        print("input_ids:  ", batch["input_ids"].shape) # [64, 77]
        print("sample:     ", batch["texts"][0])
        print("multi_pos:  ", batch["multi_positives"][0])
        break


# ---------------------------------------------------------------------------
# Multi-clip decoder: one av.open per source video
# ---------------------------------------------------------------------------

def decode_clips_from_video(
    video_path: str,
    clips: list,
    sampling_cfg: SamplingConfig,
) -> list:
    """
    Open *video_path* once and decode all clips in a single pass.

    Clips are sorted by timestamp before decoding so seeks are always forward
    (backward seeks are much slower for most codecs).

    Args:
        video_path:   Path to the source video file.
        clips:        List of sample dicts, each with:
                        "key"            – feature-store key (returned as-is)
                        "timestamp"      – midpoint of clip in seconds
                        "clip_duration"  – exact clip length in seconds
                        "video_duration" – safe upper bound for clamping
        sampling_cfg: Shared SamplingConfig (num_frames, frame_size).
                      clip_duration is overridden per-clip from the sample dict.

    Returns:
        List of (frames_tensor_or_None, key) in timestamp-sorted order.
        frames_tensor: [T, C, H, W] float32 on CPU, or None on decode failure.
    """
    clips = sorted(clips, key=lambda s: s["timestamp"])

    tfm = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((sampling_cfg.frame_size, sampling_cfg.frame_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    results = []
    container = None
    try:
        container = av.open(video_path)
        stream    = container.streams.video[0]
        vid_dur   = float(stream.duration * stream.time_base)
        time_base = float(stream.time_base)

        for clip in tqdm.tqdm(clips, desc=os.path.basename(video_path),
                              unit="clip", leave=False, dynamic_ncols=True):
            key      = clip["key"]
            clip_dur = clip.get("clip_duration", sampling_cfg.clip_duration)
            ts       = clip["timestamp"]

            half    = clip_dur / 2.0
            t_start = max(0.0,    ts - half)
            t_end   = min(vid_dur, ts + half)

            if t_end <= t_start:
                results.append((None, key))
                continue

            sample_times = np.linspace(t_start, t_end, sampling_cfg.num_frames)
            frames: list = []
            prev_frame   = None

            for t in sample_times:
                t   = float(np.clip(t, 0.0, vid_dur - 1e-4))
                pts = int(t / time_base)
                try:
                    container.seek(pts, stream=stream, any_frame=False)
                    for frame in container.decode(video=0):
                        if frame.pts is not None and frame.pts < pts:
                            continue
                        tensor = tfm(frame.to_ndarray(format="rgb24"))
                        frames.append(tensor)
                        prev_frame = tensor
                        break
                except Exception:
                    if prev_frame is not None:
                        frames.append(prev_frame)

            # Pad with last frame if some timestamps failed to decode
            while len(frames) < sampling_cfg.num_frames:
                if frames:
                    frames.append(frames[-1])
                else:
                    break

            if not frames:
                results.append((None, key))
            else:
                results.append((torch.stack(frames[:sampling_cfg.num_frames]), key))

    except Exception as e:
        print(f"[WARN] decode_clips_from_video failed for {video_path}: {e}")
        decoded = {r[1] for r in results}
        for clip in clips:
            if clip["key"] not in decoded:
                results.append((None, clip["key"]))
    finally:
        if container is not None:
            container.close()

    return results


# ---------------------------------------------------------------------------
# Ego4D egovid-5m annotation loader
# ---------------------------------------------------------------------------

def load_ego4d_annotations(
    csv_file: str,
    video_root: str,
    split_file: Optional[str] = None,
) -> list:
    """
    Parse an Ego4D egovid CSV and return a flat list of samples, one per clip.

    video_id format: {VideoID}_{StartFrame}_{EndFrame}[.mp4]
    Required CSV columns: video_id, llava_cap, frame_num, fps

    Optimised for 5M-row CSVs:
      - reads only needed columns (usecols)
      - vectorised string parsing (no per-row Python callbacks)
      - file-existence check only for unique source videos (~8k instead of 5M)
    """
    import time
    import pandas as pd

    needed = ["video_id", "llava_cap", "frame_num", "fps"]
    print(f"[Ego4D] Reading CSV: {csv_file}", flush=True)
    t0 = time.time()
    df = pd.read_csv(
        csv_file,
        usecols=needed,
        dtype={"video_id": str, "frame_num": "Int32"},
    )
    print(f"[Ego4D] CSV read: {len(df):,} rows in {time.time()-t0:.1f}s", flush=True)

    n_total = len(df)
    df = df.dropna(subset=needed).copy()
    df["frame_num"] = df["frame_num"].astype(int)
    df["fps"]       = df["fps"].astype(float)

    # ── Vectorised parse: strip .mp4, split into source_id / start / end ─────
    t0 = time.time()
    bare   = df["video_id"].str.replace(r"\.mp4$", "", regex=True)
    parts  = bare.str.rsplit("_", n=2, expand=True)   # cols: 0=VideoID, 1=start, 2=end
    df["bare"]        = bare
    df["source_id"]   = parts[0]
    df["start_frame"] = pd.to_numeric(parts[1], errors="coerce")
    df["end_frame"]   = pd.to_numeric(parts[2], errors="coerce")
    print(f"[Ego4D] Parsed video_ids in {time.time()-t0:.1f}s", flush=True)

    n_skipped = df["start_frame"].isna().sum()
    df = df.dropna(subset=["source_id", "start_frame", "end_frame"])
    df["start_frame"] = df["start_frame"].astype(int)
    df["end_frame"]   = df["end_frame"].astype(int)

    # ── Optional allow-list filter ────────────────────────────────────────────
    if split_file is not None:
        with open(split_file) as f:
            allowed = {line.strip() for line in f if line.strip()}
        df = df[df["bare"].isin(allowed) | df["video_id"].isin(allowed)]
        print(f"[Ego4D] Split filter: {len(allowed)} IDs → {len(df)} rows kept")

    # ── File-existence check — only for unique source videos (~8k) ────────────
    video_dir  = Path(video_root)
    unique_ids = df["source_id"].unique()
    exists_map = {
        sid: (video_dir / (sid + ".mp4")).exists()
        for sid in tqdm.tqdm(unique_ids, desc="[Ego4D] Checking source videos",
                             unit="video", leave=False)
    }
    n_missing = sum(1 for v in exists_map.values() if not v)
    df        = df[df["source_id"].map(exists_map)]

    # ── Vectorised time computation ───────────────────────────────────────────
    fps              = df["fps"]
    start_time       = df["start_frame"] / fps
    end_time         = df["end_frame"]   / fps
    df["clip_duration"] = df["frame_num"] / fps
    df["timestamp"]     = (start_time + end_time) / 2.0
    df["video_duration"] = end_time + 10.0          # safe upper bound; no av.open needed
    df["video_path"]    = str(video_dir) + "/" + df["source_id"] + ".mp4"

    df = df[df["clip_duration"] >= 0.1]

    # ── Build output records ──────────────────────────────────────────────────
    out = df[["bare", "video_path", "llava_cap", "video_duration",
              "timestamp", "clip_duration"]].copy()
    out.rename(columns={"bare": "key", "llava_cap": "text"}, inplace=True)
    out["video_id"] = out["key"]
    samples = out.to_dict("records")

    print(
        f"[Ego4D] CSV rows: {n_total}  parsed: {len(df) + n_missing}  "
        f"loaded: {len(samples)}  "
        f"missing on disk: {n_missing}  skipped (bad format): {n_skipped}"
    )
    return samples