"""
Zero-shot action recognition on UCF101 / HMDB51 / Kinetics-400.

Analogous to eval_zero_shot_imagenet.py but for video:
  1. Encode action class names with templates → language_projector → class prototypes
  2. Encode video clips with V-JEPA-2 → vision_projector
  3. Classify via cosine similarity, report top-1 and top-5 accuracy

Dataset format (standard for UCF101 / HMDB51 / Kinetics-400):
    {dataset_dir}/
      class_name_1/
        video1.avi
        video2.avi
      class_name_2/
        ...

Download datasets:
    # UCF101 (~7 GB)
    wget https://www.crcv.ucf.edu/data/UCF101/UCF101.rar
    unrar x UCF101.rar

    # HMDB51 (~2 GB)
    wget https://serre-lab.clps.brown.edu/wp-content/uploads/2013/10/hmdb51_org.rar
    unrar x hmdb51_org.rar && for f in *.rar; do unrar x "$f"; done

Usage:
    python eval_zero_shot_action.py \\
        --checkpoint logs/.../best_model.ckpt \\
        --config configs/egoexo4d_vjepa2_config.yaml \\
        --dataset_dir UCF101 --dataset ucf101

    python eval_zero_shot_action.py \\
        --checkpoint logs/.../best_model.ckpt \\
        --config configs/egoexo4d_vjepa2_config.yaml \\
        --dataset_dir hmdb51_org --dataset hmdb51

    # Quick sanity check (10 videos per class)
    python eval_zero_shot_action.py \\
        --checkpoint logs/.../best_model.ckpt \\
        --config configs/egoexo4d_vjepa2_config.yaml \\
        --dataset_dir UCF101 --dataset ucf101 \\
        --max_videos_per_class 10
"""

import os
import re
import glob
import argparse
import tempfile

import av
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from omegaconf import OmegaConf
from torchvision import transforms

# PL checkpoints contain OmegaConf objects — force weights_only=False
_orig_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    kwargs["weights_only"] = False
    return _orig_torch_load(*args, **kwargs)
torch.load = _patched_torch_load

from sharelock.models.model import ShareLock, ShareLockWithTextEncoder
from sharelock.models.video_encoder import VideoEncoder


# Egocentric + standard action description templates
ACTION_TEMPLATES = [
    "a video of a person {}.",
    "a person is {}.",
    "someone {}.",
    "the person is {}.",
    "a clip of someone {}.",
    "a video of someone {}.",
    "an egocentric video of a person {}.",
    "a first-person view of someone {}.",
]


def clean_class_name(name: str, dataset: str) -> str:
    """Normalize raw directory name to natural language."""
    name = name.replace("_", " ").replace("-", " ")
    if dataset == "ucf101":
        # UCF101 uses CamelCase: "BasketballDunk" → "basketball dunk"
        name = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", name)
    return name.lower().strip()


def sample_video_uniform(video_path: str, num_frames: int, frame_size: int) -> torch.Tensor | None:
    """Sample num_frames uniformly over the full video duration.

    Returns [T, C, H, W] float32 tensor (ImageNet normalized), or None on failure.
    """
    preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((frame_size, frame_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    try:
        container = av.open(video_path)
        stream = container.streams.video[0]
        duration = float(stream.duration * stream.time_base) if stream.duration else None

        if duration is None or duration <= 0:
            return None

        # Avoid very start/end where content may be missing
        sample_times = np.linspace(duration * 0.05, duration * 0.95, num_frames)
        sample_pts = [int(t / float(stream.time_base)) for t in sample_times]
        container.close()

        container = av.open(video_path)
        stream = container.streams.video[0]
        frames = []
        for pts in sample_pts:
            container.seek(pts, stream=stream, any_frame=False)
            frame = next(container.decode(video=0), None)
            if frame is None:
                continue
            img = frame.to_ndarray(format="rgb24")   # [H, W, 3]
            frames.append(preprocess(img))

        container.close()

        if len(frames) < num_frames:
            return None

        return torch.stack(frames[:num_frames])      # [T, C, H, W]

    except Exception:
        return None


def build_action_prototypes(model, class_names, templates, device,
                            text_batch_size=256, cache_path=None):
    """Encode class names with templates and mean-pool → class prototype matrix.

    Returns [dim, N_classes] prototype tensor (same convention as eval_zero_shot_imagenet.py).
    """
    if cache_path and os.path.exists(cache_path):
        print(f"Loading cached prototypes from {cache_path}")
        return torch.load(cache_path, map_location=device, weights_only=True)

    n_classes, n_tmpl = len(class_names), len(templates)
    total = n_classes * n_tmpl
    print(f"Encoding {n_classes} classes × {n_tmpl} templates = {total} texts "
          f"(batch={text_batch_size})...")

    all_texts = [
        tmpl.format(cls)
        for cls in class_names
        for tmpl in templates
    ]

    all_embs = []
    with torch.no_grad():
        for i in tqdm(range(0, total, text_batch_size), desc="Encoding class names"):
            batch = all_texts[i : i + text_batch_size]
            emb = model.encode_text(batch)          # [B, dim], already L2-normed
            if emb.dim() == 1:
                emb = emb.unsqueeze(0)
            all_embs.append(emb.cpu().float())

    all_embs = torch.cat(all_embs, dim=0)                      # [N*T, dim]
    all_embs = all_embs.view(n_classes, n_tmpl, -1)            # [N, T, dim]
    all_embs = F.normalize(all_embs, dim=-1)
    class_embs = F.normalize(all_embs.mean(dim=1), dim=-1)     # [N, dim]
    prototypes = class_embs.to(device).T                        # [dim, N]

    if cache_path:
        os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)
        torch.save(prototypes.cpu(), cache_path)
        print(f"Cached prototypes → {cache_path}")

    return prototypes


def main():
    parser = argparse.ArgumentParser(
        description="Zero-shot action recognition for EgoExo4D-trained ShareLock"
    )
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config", type=str, default="configs/egoexo4d_vjepa2_config.yaml")
    parser.add_argument("--dataset_dir", type=str, required=True,
                        help="Root dir with class_name/video subdirectory structure")
    parser.add_argument("--dataset", type=str, default="ucf101",
                        choices=["ucf101", "hmdb51", "kinetics400"],
                        help="Dataset name (affects class name normalization)")
    parser.add_argument("--num_frames", type=int, default=16,
                        help="Frames per clip (should match training, default 16)")
    parser.add_argument("--video_extensions", type=str, default="avi,mp4,mkv",
                        help="Comma-separated video file extensions to include")
    parser.add_argument("--max_videos_per_class", type=int, default=None,
                        help="Cap per class for quick sanity checks")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Videos per V-JEPA-2 forward pass")
    parser.add_argument("--text_batch_size", type=int, default=256,
                        help="Texts per LLaMA forward pass")
    parser.add_argument("--prototype_cache", type=str, default=None,
                        help="Cache path for class prototype tensors")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Device: {device}")

    # ── Load config + checkpoint ─────────────────────────────────────────────
    config = OmegaConf.load(args.config)
    raw_ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = OmegaConf.merge(config, raw_ckpt["hyper_parameters"])

    print(f"Vision encoder  : {config.model.vision_encoder}")
    print(f"Language encoder: {config.model.language_encoder}")

    train_lang = config.model.get("train_language_encoder", False)
    model_cls = ShareLockWithTextEncoder if train_lang else ShareLock
    model = model_cls.load_from_checkpoint(
        args.checkpoint, config=config, map_location=device, strict=False
    )
    model = model.to(device).eval()

    # ── Discover classes ─────────────────────────────────────────────────────
    class_dirs = sorted([
        d for d in os.listdir(args.dataset_dir)
        if os.path.isdir(os.path.join(args.dataset_dir, d))
    ])
    if not class_dirs:
        raise ValueError(f"No class subdirectories found in {args.dataset_dir}")

    class_names_clean = [clean_class_name(c, args.dataset) for c in class_dirs]
    print(f"Classes: {len(class_dirs)}  (e.g. '{class_dirs[0]}' → '{class_names_clean[0]}')")

    # ── Build class prototypes (LLaMA) ───────────────────────────────────────
    cache_path = args.prototype_cache or (
        f"{os.path.splitext(args.checkpoint)[0]}_protos_{args.dataset}.pt"
    )
    class_prototypes = build_action_prototypes(
        model, class_names_clean, ACTION_TEMPLATES, device,
        text_batch_size=args.text_batch_size, cache_path=cache_path,
    )  # [dim, N_classes]

    # Unload language model to free GPU memory
    if model.language_encoder is not None:
        if hasattr(model.language_encoder, "unload_model"):
            model.language_encoder.unload_model()
    torch.cuda.empty_cache()

    # ── Load video encoder ───────────────────────────────────────────────────
    variant = config.model.vision_encoder
    frame_size = VideoEncoder.VARIANTS[variant][2]
    video_encoder = VideoEncoder(variant=variant).to(device)
    print(f"Loaded VideoEncoder ({variant}, frame_size={frame_size})")

    # ── Evaluate ─────────────────────────────────────────────────────────────
    extensions = set(args.video_extensions.split(","))
    correct_top1 = correct_top5 = total = skipped = 0

    for cls_idx, (cls_dir, cls_name) in enumerate(tqdm(
        zip(class_dirs, class_names_clean),
        total=len(class_dirs),
        desc="Evaluating",
    )):
        video_paths = []
        for ext in extensions:
            video_paths.extend(
                glob.glob(os.path.join(args.dataset_dir, cls_dir, f"*.{ext}"))
            )
        video_paths = sorted(video_paths)
        if args.max_videos_per_class:
            video_paths = video_paths[: args.max_videos_per_class]

        buffer = []
        label = torch.tensor([cls_idx], device=device)

        def _eval_buffer(buf):
            nonlocal correct_top1, correct_top5, total
            batch = torch.stack(buf).to(device)             # [B, T, C, H, W]
            with torch.no_grad():
                raw  = video_encoder(batch)                  # [B, embed_dim]
                proj = F.normalize(model.vision_projector(raw), dim=-1)
                logits = proj @ class_prototypes             # [B, N_classes]
            labels = torch.full((len(buf),), cls_idx, device=device)
            correct_top1 += (logits.argmax(dim=-1) == labels).sum().item()
            correct_top5 += (
                (logits.topk(min(5, logits.size(1)), dim=-1).indices == labels.unsqueeze(1))
                .any(dim=1).sum().item()
            )
            total += len(buf)

        for vpath in video_paths:
            frames = sample_video_uniform(vpath, args.num_frames, frame_size)
            if frames is None:
                skipped += 1
                continue
            buffer.append(frames)
            if len(buffer) >= args.batch_size:
                _eval_buffer(buffer)
                buffer = []

        if buffer:
            _eval_buffer(buffer)

    top1 = correct_top1 / total if total > 0 else 0.0
    top5 = correct_top5 / total if total > 0 else 0.0

    print("\n" + "=" * 60)
    print(f"Zero-shot {args.dataset.upper()} Action Recognition")
    print(f"  Checkpoint  : {args.checkpoint}")
    print(f"  Classes     : {len(class_dirs)}")
    print(f"  Videos      : {total} evaluated, {skipped} skipped")
    print(f"  Templates   : {len(ACTION_TEMPLATES)}")
    print("-" * 60)
    print(f"  Top-1 Accuracy: {top1 * 100:.2f}%")
    print(f"  Top-5 Accuracy: {top5 * 100:.2f}%")
    print("=" * 60)


if __name__ == "__main__":
    main()
