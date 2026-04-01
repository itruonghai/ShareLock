"""
Precompute V-JEPA-2 / V-JEPA-2.1 video features and LLaMA-3-8B language features
for EgoExo4D Atomic Action Descriptions.

Features are stored via featureutils (same format as precompute_features.py),
keyed by  "{take_uid}__{frame_idx}"  where frame_idx = round(anchor_t * fps).
This gives frame-accurate, collision-free keys that are stable across runs.

Usage:
  # Step 1 — video features (multi-GPU)
  python precompute_video_features.py \\
      --video_root EgoExo/train_videos/takes \\
      --annotation_json EgoExo/annotations/atomic_descriptions_train.json \\
      --takes_json EgoExo/takes.json \\
      --split_file EgoExo/train_takes.txt \\
      --variant vjepa2.1_vitl_384 \\
      --output_dir precomputed_features_video \\
      --extract video --num_gpus 4 --batch_size 32 --num_workers 8

  # Step 2 — language features (multi-GPU)
  # NOTE: --video_root is required — keys are frame-accurate (FPS read from video)
  #       and load_egoexo4d_annotations filters by video_path.exists()
  python precompute_video_features.py \\
      --video_root EgoExo/train_videos/takes \\
      --annotation_json EgoExo/annotations/atomic_descriptions_train.json \\
      --takes_json EgoExo/takes.json \\
      --split_file EgoExo/train_takes.txt \\
      --language_model meta-llama/Meta-Llama-3-8B \\
      --caption_name atomic_train \\
      --output_dir precomputed_features_video \\
      --extract language --num_gpus 4

  # Dry-run (single GPU, small batch) to verify paths
  python precompute_video_features.py \\
      --video_root EgoExo/train_videos/takes \\
      --annotation_json EgoExo/annotations/atomic_descriptions_train.json \\
      --takes_json EgoExo/takes.json \\
      --split_file EgoExo/train_takes.txt \\
      --variant vjepa2.1_vitl_384 \\
      --output_dir precomputed_features_video \\
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
from dataloader_video import (
    load_egoexo4d_annotations,
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
    """Wraps a list of annotation samples for async frame decoding via DataLoader."""

    def __init__(self, samples: list, sampling_cfg: SamplingConfig, existing_keys: set):
        # Pre-filter already-extracted keys so workers never touch them
        self.items = [
            s for s in samples
            if make_key(s["take_uid"], s["timestamp"], s["video_path"]) not in existing_keys
        ]
        self.cfg = sampling_cfg

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        s = self.items[idx]
        key = make_key(s["take_uid"], s["timestamp"], s["video_path"])
        if self.cfg.mode == "adaptive":
            frames = sample_frames_adaptive(
                video_path=s["video_path"],
                timestamp=s["timestamp"],
                cfg=self.cfg,
                video_duration=s["video_duration"],
                prev_timestamp=s.get("prev_timestamp"),
                next_timestamp=s.get("next_timestamp"),
            )
        else:
            frames = sample_frames_centered(
                video_path=s["video_path"],
                timestamp=s["timestamp"],
                cfg=self.cfg,
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

def worker(rank: int, args) -> None:
    """Load annotations once, then extract video and/or language features."""
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

    samples = load_egoexo4d_annotations(
        annotation_json=args.annotation_json,
        takes_json=args.takes_json,
        video_root=args.video_root if args.video_root else "",
        filter_cfg=filter_cfg,
        sampling_cfg=sampling_cfg,
        split_file=args.split_file,
    )

    if args.extract in ("video", "both"):
        extract_video(rank, args.num_gpus, samples, args)
        gc.collect()
        torch.cuda.empty_cache()

    if args.extract in ("language", "both"):
        # Build {key: text} dict — keys must match video feature keys exactly
        captions: dict = {}
        for sample in samples:
            key = make_key(
                sample["take_uid"], sample["timestamp"], sample["video_path"]
            )
            if key not in captions:
                captions[key] = sample["text"]
        print(f"[Lang GPU {rank}] {len(captions)} unique captions")
        extract_language(rank, args.num_gpus, captions, args)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Precompute V-JEPA-2 video + LLaMA language features for EgoExo4D"
    )
    # Data paths
    parser.add_argument("--annotation_json", type=str, required=True,
                        help="atomic_descriptions_{train|val}.json")
    parser.add_argument("--takes_json", type=str, required=True,
                        help="takes.json")
    parser.add_argument("--video_root", type=str, default=None,
                        help="Root of video files, e.g. EgoExo/train_videos/takes")
    parser.add_argument("--split_file", type=str, default=None,
                        help="train_takes.txt or val_takes.txt")

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
                        help="Frame sampling: centered (fixed 4s window) or adaptive (Voronoi, no inter-clip overlap)")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    print(f"Extract: {args.extract} | GPUs: {args.num_gpus}")
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
