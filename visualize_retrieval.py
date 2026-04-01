"""
Visualize EgoExo4D video-text retrieval results as an MP4 video.

For each query video clip, a slide shows:
  - Strip of 8 video frames
  - Ground truth caption
  - Top-K retrieved captions with similarity scores
    (GT caption highlighted in green if retrieved at rank 1)

Usage:
    python visualize_retrieval.py \\
        --checkpoint logs/.../best_model.ckpt \\
        --config configs/egoexo4d_vjepa2_config.yaml \\
        --precomputed_features_dir precomputed_features_video_val \\
        --video_root EgoExo/val_videos/takes \\
        --annotation_json EgoExo/annotations/atomic_descriptions_val.json \\
        --takes_json EgoExo/takes.json \\
        --split_file EgoExo/val_takes.txt \\
        --output retrieval_viz.mp4 \\
        --num_queries 20 --top_k 5

    # Only show correct retrievals (R@1 hits)
    python visualize_retrieval.py ... --mode correct

    # Mix of correct + incorrect
    python visualize_retrieval.py ... --mode mixed

    # Text-to-video retrieval only
    python visualize_retrieval.py ... --direction t2v

    # Both directions (V→T slide then T→V slide per query)
    python visualize_retrieval.py ... --direction both
"""

import os
import json
import argparse
import random
import tempfile
import textwrap

import av
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import imageio.v3 as iio
from tqdm import tqdm
from omegaconf import OmegaConf
from torchvision import transforms

_orig_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    kwargs["weights_only"] = False
    return _orig_torch_load(*args, **kwargs)
torch.load = _patched_torch_load

from featureutils.core import FeatureUtils
from sharelock.models.model import ShareLock, ShareLockWithTextEncoder
from sharelock.data.datasets import _load_features_parallel
from dataloader_video import (
    load_egoexo4d_annotations, SamplingConfig, FilterConfig, sample_frames_centered,
)
from precompute_video_features import make_key
from eval_egoexo4d_retrieval import build_caption_map, build_mcq_instances


# ── Constants ────────────────────────────────────────────────────────────────

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406])
IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225])

SLIDE_W, SLIDE_H = 1280, 720   # output resolution
SLIDE_DPI = 100
SLIDE_DURATION_SEC = 4         # seconds per query slide
FPS = 24


# ── Helpers ──────────────────────────────────────────────────────────────────

def denormalize(tensor: torch.Tensor) -> np.ndarray:
    """[C, H, W] ImageNet-normalized tensor → [H, W, 3] uint8."""
    t = tensor.cpu().float() * IMAGENET_STD[:, None, None] + IMAGENET_MEAN[:, None, None]
    t = t.clamp(0, 1).permute(1, 2, 0).numpy()
    return (t * 255).astype(np.uint8)


def load_clip_frames(video_path: str, timestamp: float, video_duration: float,
                     num_frames: int = 8, frame_size: int = 224) -> list[np.ndarray] | None:
    """Sample num_frames centered on timestamp, return list of [H, W, 3] uint8 arrays."""
    cfg = SamplingConfig(num_frames=num_frames, clip_duration=4.0, frame_size=frame_size)
    frames_tensor = sample_frames_centered(video_path, timestamp, cfg, video_duration)
    if frames_tensor is None:
        return None
    return [denormalize(f) for f in frames_tensor]   # list of [H, W, 3] uint8


def build_annotation_maps(annotation_json, takes_json, video_root, split_file):
    """Return two dicts keyed by feature key ({take_uid}__{frame_idx}):
        key_to_caption:   key → primary text caption
        key_to_video_info: key → {"video_path": str, "timestamp": float, "duration": float}
    """
    filter_cfg = FilterConfig(
        keep_subject_C_only=True, drop_unsure=True,
        require_ego_visible=True, min_timestamp=2.0,
    )
    sampling_cfg = SamplingConfig(num_frames=16, clip_duration=4.0)
    samples = load_egoexo4d_annotations(
        annotation_json, takes_json, video_root,
        filter_cfg, sampling_cfg, split_file=split_file,
    )
    key_to_caption    = {}
    key_to_video_info = {}
    for s in samples:
        key = make_key(s["take_uid"], s["timestamp"], s["video_path"])
        if key not in key_to_caption:
            key_to_caption[key] = s["text"]
            key_to_video_info[key] = {
                "video_path": s["video_path"],
                "timestamp":  s["timestamp"],
                "duration":   s["video_duration"],
            }
    return key_to_caption, key_to_video_info


# ── Slide rendering ──────────────────────────────────────────────────────────

def wrap(text: str, width: int = 70) -> str:
    return "\n".join(textwrap.wrap(text, width))


def render_category_header(category: str, n: int, r1: float) -> np.ndarray:
    """Render a full-screen category title card → [SLIDE_H, SLIDE_W, 3] uint8."""
    fig = plt.figure(figsize=(SLIDE_W / SLIDE_DPI, SLIDE_H / SLIDE_DPI), dpi=SLIDE_DPI)
    fig.patch.set_facecolor("#0d0d1a")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_facecolor("#0d0d1a")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Category name
    ax.text(0.5, 0.60, category, ha="center", va="center",
            color="#e2c96e", fontsize=38, fontweight="bold")
    # Stats line
    ax.text(0.5, 0.44, f"{n} samples  ·  V→T R@1 = {r1:.1f}%  (full pool)",
            ha="center", va="center", color="#aaaaaa", fontsize=16)
    # Decorative rule
    ax.axhline(y=0.52, xmin=0.25, xmax=0.75, color="#e2c96e", linewidth=1.2, alpha=0.5)

    fig.canvas.draw()
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    img = buf.reshape(SLIDE_H, SLIDE_W, 4)[..., :3].copy()
    plt.close(fig)
    return img


def render_slide(
    frames: list[np.ndarray],          # 8× [H, W, 3] uint8
    gt_caption: str,
    retrieved: list[tuple[str, float]],  # [(caption, score), ...]
    query_idx: int,
    gt_rank: int,                        # 1-based rank of GT in retrieved list, or None
) -> np.ndarray:
    """Render one query slide → [SLIDE_H, SLIDE_W, 3] uint8."""
    fig = plt.figure(figsize=(SLIDE_W / SLIDE_DPI, SLIDE_H / SLIDE_DPI), dpi=SLIDE_DPI)
    fig.patch.set_facecolor("#1a1a2e")

    n_frames = len(frames)
    gs = gridspec.GridSpec(
        2, 1,
        figure=fig,
        height_ratios=[0.42, 0.58],
        hspace=0.03,
        left=0.01, right=0.99, top=0.97, bottom=0.02,
    )

    # ── Top: video frames ────────────────────────────────────────────────────
    gs_frames = gridspec.GridSpecFromSubplotSpec(
        1, n_frames, subplot_spec=gs[0], wspace=0.02,
    )
    for i, frame in enumerate(frames):
        ax = fig.add_subplot(gs_frames[i])
        ax.imshow(frame)
        ax.axis("off")
        ax.set_facecolor("#1a1a2e")
        # Frame timestamp marker
        ax.text(0.5, -0.04, f"f{i+1}", transform=ax.transAxes,
                ha="center", va="top", color="#888888", fontsize=6)

    # ── Bottom: text information ─────────────────────────────────────────────
    ax_text = fig.add_subplot(gs[1])
    ax_text.set_facecolor("#0f0f23")
    ax_text.set_xlim(0, 1)
    ax_text.set_ylim(0, 1)
    ax_text.axis("off")

    y = 0.97
    line_h = 0.13

    # Query header
    ax_text.text(0.01, y, f"Query #{query_idx + 1}",
                 transform=ax_text.transAxes,
                 color="#aaaaaa", fontsize=9, fontweight="bold", va="top")
    y -= line_h * 0.8

    # Ground truth
    gt_color = "#06d6a0"  # green
    ax_text.text(0.01, y, "Ground Truth:",
                 transform=ax_text.transAxes,
                 color=gt_color, fontsize=9, fontweight="bold", va="top")
    ax_text.text(0.16, y, wrap(gt_caption, 100),
                 transform=ax_text.transAxes,
                 color="#e0e0e0", fontsize=9, va="top")
    y -= line_h

    # Separator
    ax_text.axhline(y=y + line_h * 0.4, xmin=0.01, xmax=0.99,
                    color="#333355", linewidth=0.8)

    # Retrieved captions
    ax_text.text(0.01, y, "Retrieved:",
                 transform=ax_text.transAxes,
                 color="#aaaaaa", fontsize=9, fontweight="bold", va="top")
    y -= line_h * 0.85

    for rank, (caption, score) in enumerate(retrieved, start=1):
        is_gt_match = (gt_rank is not None and rank == gt_rank)
        rank_color  = "#06d6a0" if is_gt_match else "#ef4444" if rank == 1 else "#cccccc"
        bg_color    = "#0d3320" if is_gt_match else None

        label = f"[{rank}] {score:.3f}"
        ax_text.text(0.01, y, label,
                     transform=ax_text.transAxes,
                     color=rank_color, fontsize=8.5, fontweight="bold", va="top",
                     fontfamily="monospace")

        suffix = "  ✓ GT" if is_gt_match else ""
        ax_text.text(0.12, y, wrap(caption, 95) + suffix,
                     transform=ax_text.transAxes,
                     color=rank_color if is_gt_match else "#dddddd",
                     fontsize=8.5, va="top")
        y -= line_h
        if y < 0.02:
            break

    fig.canvas.draw()
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    img = buf.reshape(SLIDE_H, SLIDE_W, 4)[..., :3].copy()  # RGBA → RGB
    plt.close(fig)
    return img


def render_t2v_slide(
    query_caption: str,
    retrieved: list[tuple[list[np.ndarray], str, float]],  # (frames, caption, score)
    query_idx: int,
    gt_rank: int | None,
) -> np.ndarray:
    """Render T→V slide: text query → retrieved video strips.
    Layout: query text header (top 18%), then one row per retrieved video.
    Each row: rank+score label | 4 frame thumbnails | caption snippet.
    """
    n_ret = len(retrieved)
    n_frames_show = 4

    fig = plt.figure(figsize=(SLIDE_W / SLIDE_DPI, SLIDE_H / SLIDE_DPI), dpi=SLIDE_DPI)
    fig.patch.set_facecolor("#1a1a2e")

    gs_outer = gridspec.GridSpec(
        n_ret + 1, 1,
        figure=fig,
        height_ratios=[0.18] + [0.82 / n_ret] * n_ret,
        hspace=0.04,
        left=0.01, right=0.99, top=0.97, bottom=0.02,
    )

    # ── Query text header ─────────────────────────────────────────────────────
    ax_q = fig.add_subplot(gs_outer[0])
    ax_q.set_facecolor("#0f0f23")
    ax_q.set_xlim(0, 1)
    ax_q.set_ylim(0, 1)
    ax_q.axis("off")
    ax_q.text(0.01, 0.90, f"T→V Query #{query_idx + 1}",
              transform=ax_q.transAxes,
              color="#aaaaaa", fontsize=9, fontweight="bold", va="top")
    ax_q.text(0.01, 0.55, wrap(query_caption, 130),
              transform=ax_q.transAxes,
              color="#e0e0e0", fontsize=9, va="top")

    # ── Retrieved video rows ──────────────────────────────────────────────────
    for rank, (frames, caption, score) in enumerate(retrieved, start=1):
        is_gt  = (gt_rank is not None and rank == gt_rank)
        r_color = "#06d6a0" if is_gt else "#ef4444" if rank == 1 else "#cccccc"
        bg     = "#0d3320"  if is_gt else "#1a0d0d" if rank == 1 else "#111122"

        gs_row = gridspec.GridSpecFromSubplotSpec(
            1, n_frames_show + 1,
            subplot_spec=gs_outer[rank],
            wspace=0.015,
            width_ratios=[1] * n_frames_show + [1.4],
        )

        # Frame thumbnails
        for fi in range(n_frames_show):
            ax_f = fig.add_subplot(gs_row[fi])
            ax_f.set_facecolor(bg)
            ax_f.axis("off")
            if fi < len(frames):
                ax_f.imshow(frames[fi])
            # Rank + score badge on the first frame
            if fi == 0:
                ax_f.text(0.04, 0.96, f"[{rank}] {score:.3f}",
                          transform=ax_f.transAxes,
                          color=r_color, fontsize=7, fontweight="bold",
                          va="top", fontfamily="monospace",
                          bbox=dict(facecolor="#00000088", edgecolor="none", pad=1))

        # Caption snippet
        ax_t = fig.add_subplot(gs_row[n_frames_show])
        ax_t.set_facecolor(bg)
        ax_t.set_xlim(0, 1)
        ax_t.set_ylim(0, 1)
        ax_t.axis("off")
        suffix = "  ✓ GT" if is_gt else ""
        ax_t.text(0.05, 0.5, wrap(caption, 38) + suffix,
                  transform=ax_t.transAxes,
                  color=r_color if is_gt else "#dddddd",
                  fontsize=7.5, va="center")

    fig.canvas.draw()
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    img = buf.reshape(SLIDE_H, SLIDE_W, 4)[..., :3].copy()
    plt.close(fig)
    return img


def render_mcq_slide(
    frames: list[np.ndarray],         # 8 video frames
    options: list[str],               # 5 captions in display order (shuffled)
    gt_pos: int,                      # which option index is ground truth (0-4)
    pred_pos: int,                    # which option index model chose (0-4)
    query_idx: int,
    scores: list[float] | None = None,  # cosine similarity per option (same order)
) -> np.ndarray:
    """Render a 5-way MCQ slide: video frames + 5 labelled text options A–E."""
    LABELS = ["A", "B", "C", "D", "E"]
    is_correct = (pred_pos == gt_pos)

    fig = plt.figure(figsize=(SLIDE_W / SLIDE_DPI, SLIDE_H / SLIDE_DPI), dpi=SLIDE_DPI)
    fig.patch.set_facecolor("#1a1a2e")

    gs = gridspec.GridSpec(
        2, 1, figure=fig,
        height_ratios=[0.38, 0.62],
        hspace=0.03,
        left=0.01, right=0.99, top=0.97, bottom=0.02,
    )

    # ── Top: video frames ─────────────────────────────────────────────────────
    gs_frames = gridspec.GridSpecFromSubplotSpec(1, len(frames), subplot_spec=gs[0], wspace=0.02)
    for i, frame in enumerate(frames):
        ax = fig.add_subplot(gs_frames[i])
        ax.imshow(frame)
        ax.axis("off")
        ax.set_facecolor("#1a1a2e")
        ax.text(0.5, -0.04, f"f{i+1}", transform=ax.transAxes,
                ha="center", va="top", color="#888888", fontsize=6)

    # Header badge over the frames
    ax_hdr = fig.add_subplot(gs[0])
    ax_hdr.set_xlim(0, 1); ax_hdr.set_ylim(0, 1); ax_hdr.axis("off")
    result_txt = "✓ CORRECT" if is_correct else "✗ WRONG"
    result_col = "#06d6a0"   if is_correct else "#ef4444"
    ax_hdr.text(0.99, 0.98, f"MCQ #{query_idx + 1}  {result_txt}",
                transform=ax_hdr.transAxes, ha="right", va="top",
                color=result_col, fontsize=9, fontweight="bold",
                bbox=dict(facecolor="#00000099", edgecolor="none", pad=3))

    # ── Bottom: 5 MCQ options stacked ────────────────────────────────────────
    gs_opts = gridspec.GridSpecFromSubplotSpec(5, 1, subplot_spec=gs[1], hspace=0.06)
    for pos, caption in enumerate(options):
        is_gt   = (pos == gt_pos)
        is_pred = (pos == pred_pos)

        if is_gt and is_pred:          # correct prediction
            bg, border, txt_col = "#0d3320", "#06d6a0", "#06d6a0"
            badge = "✓ GT  ◀ Model"
        elif is_gt and not is_pred:    # GT not picked
            bg, border, txt_col = "#0d3320", "#06d6a0", "#06d6a0"
            badge = "✓ GT"
        elif is_pred and not is_gt:    # wrong prediction
            bg, border, txt_col = "#2a0a0a", "#ef4444", "#ef4444"
            badge = "✗ Model"
        else:
            bg, border, txt_col = "#111128", "#334", "#cccccc"
            badge = ""

        ax = fig.add_subplot(gs_opts[pos])
        ax.set_facecolor(bg)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
        for spine in ax.spines.values():
            spine.set_visible(True); spine.set_color(border); spine.set_linewidth(1.4)

        label_txt = LABELS[pos]
        ax.text(0.012, 0.5, label_txt,
                transform=ax.transAxes, va="center",
                color=txt_col, fontsize=12, fontweight="bold")
        ax.text(0.05, 0.5, wrap(caption, 88),
                transform=ax.transAxes, va="center",
                color=txt_col, fontsize=9)

        # Score badge + mini bar on the right
        if scores is not None:
            score_val = scores[pos]
            # Normalise bar width: map [min_score, max_score] → [0, 1]
            s_min, s_max = min(scores), max(scores)
            bar_w = (score_val - s_min) / (s_max - s_min + 1e-8)
            # Thin horizontal bar just above the bottom border
            ax.barh(0.08, bar_w * 0.18, height=0.18, left=0.795,
                    color=txt_col, alpha=0.55, transform=ax.transAxes)
            score_txt = f"{score_val:.4f}"
            ax.text(0.985, 0.5, score_txt,
                    transform=ax.transAxes, va="center", ha="right",
                    color=txt_col, fontsize=8.5, fontweight="bold",
                    fontfamily="monospace")
            if badge:
                ax.text(0.985, 0.12, badge,
                        transform=ax.transAxes, va="bottom", ha="right",
                        color=txt_col, fontsize=7, fontweight="bold",
                        fontfamily="monospace")
        elif badge:
            ax.text(0.988, 0.5, badge,
                    transform=ax.transAxes, va="center", ha="right",
                    color=txt_col, fontsize=8, fontweight="bold",
                    fontfamily="monospace")

    fig.canvas.draw()
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    img = buf.reshape(SLIDE_H, SLIDE_W, 4)[..., :3].copy()
    plt.close(fig)
    return img


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Visualize EgoExo4D retrieval results as MP4"
    )
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config", type=str, default="configs/egoexo4d_vjepa2_config.yaml")
    parser.add_argument("--precomputed_features_dir", type=str, required=True)
    parser.add_argument("--video_root", type=str, required=True,
                        help="Root dir of video files (e.g. EgoExo/val_videos/takes)")
    parser.add_argument("--annotation_json", type=str, required=True,
                        help="atomic_descriptions_val.json")
    parser.add_argument("--takes_json", type=str, default="EgoExo/takes.json")
    parser.add_argument("--split_file", type=str, default=None)
    parser.add_argument("--caption_name", type=str, default=None)
    parser.add_argument("--output", type=str, default="retrieval_viz.mp4")
    parser.add_argument("--num_queries", type=int, default=20,
                        help="Number of query slides to render")
    parser.add_argument("--top_k", type=int, default=5,
                        help="Number of retrieved captions to show per query")
    parser.add_argument("--num_frames", type=int, default=8,
                        help="Number of video frames to display per query")
    parser.add_argument("--mode", type=str, default="random",
                        choices=["random", "correct", "incorrect", "mixed", "per_category"],
                        help="Which queries to visualize: random / correct (R@1 hits) "
                             "/ incorrect (R@1 misses) / mixed (50/50) "
                             "/ per_category (category header + examples, full pool retrieval)")
    parser.add_argument("--queries_per_category", type=int, default=2,
                        help="Queries to show per category in per_category mode")
    parser.add_argument("--direction", type=str, default="v2t",
                        choices=["v2t", "t2v", "both", "mcq"],
                        help="Retrieval direction: v2t (video→text), t2v (text→video), "
                             "both, or mcq (5-way multiple choice)")
    parser.add_argument("--mcq_seed", type=int, default=42,
                        help="RNG seed for MCQ distractor sampling")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    random.seed(args.seed)
    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Device: {device}")

    # ── Load model ───────────────────────────────────────────────────────────
    config = OmegaConf.load(args.config)
    raw_ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = OmegaConf.merge(config, raw_ckpt["hyper_parameters"])

    train_lang = config.model.get("train_language_encoder", False)
    model_cls  = ShareLockWithTextEncoder if train_lang else ShareLock
    model = model_cls.load_from_checkpoint(
        args.checkpoint, config=config, map_location=device, strict=False
    )
    model = model.to(device).eval()
    print(f"Loaded model ({config.model.vision_encoder})")

    # ── Load precomputed features ────────────────────────────────────────────
    vision_dir = os.path.join(
        args.precomputed_features_dir,
        config.model.vision_encoder.split("/")[-1],
    )
    caption_files = config.data.caption_files
    if not isinstance(caption_files, str):
        caption_files = list(caption_files)[0]
    caption_name = args.caption_name or caption_files
    lang_dir = os.path.join(
        args.precomputed_features_dir,
        config.model.language_encoder.split("/")[-1],
        caption_name.replace(".json", ""),
    )

    staging_dir = os.environ.get("TMPDIR", tempfile.mkdtemp())
    vision_fu = FeatureUtils(base_dir=vision_dir, staging_dir=staging_dir, require_features_exist=True)
    lang_fu   = FeatureUtils(base_dir=lang_dir,   staging_dir=staging_dir, require_features_exist=True)
    vision_fu.stage_data(features=["vision_features"])
    lang_fu.stage_data(features=["language_features"])

    feature_ids = sorted(set(vision_fu.list_keys()) & set(lang_fu.list_keys()))
    print(f"Aligned features: {len(feature_ids)}")

    print("Loading vision features...")
    vis_loaded     = _load_features_parallel(vision_fu, feature_ids, ["vision_features"])
    vision_tensor  = vis_loaded["vision_features"]
    print("Loading language features...")
    lang_loaded    = _load_features_parallel(lang_fu, feature_ids, ["language_features"])
    language_tensor = lang_loaded["language_features"].squeeze(1)

    # ── Project all features ─────────────────────────────────────────────────
    print("Projecting...")
    all_vis, all_lang = [], []
    with torch.no_grad():
        for i in tqdm(range(0, len(feature_ids), args.batch_size), desc="Projecting"):
            vb = vision_tensor[i : i + args.batch_size].to(device)
            lb = language_tensor[i : i + args.batch_size].to(device)
            all_vis.append(model.vision_projector(vb).cpu())
            all_lang.append(model.language_projector(lb).cpu())
    vis_emb  = F.normalize(torch.cat(all_vis,  dim=0), dim=-1)
    lang_emb = F.normalize(torch.cat(all_lang, dim=0), dim=-1)

    # ── Build annotation maps (key → caption, video info) ───────────────────
    print("Building annotation maps from raw video files...")
    key_to_caption, key_to_video_info = build_annotation_maps(
        args.annotation_json, args.takes_json, args.video_root, args.split_file
    )

    # Filter feature_ids to those with known captions AND existing video files
    valid_ids = [
        fid for fid in feature_ids
        if fid in key_to_caption and fid in key_to_video_info
        and os.path.exists(key_to_video_info[fid]["video_path"])
    ]
    print(f"Features with valid video files: {len(valid_ids)}")

    # ── Select queries ───────────────────────────────────────────────────────
    # Pre-compute R@1 hits for mode filtering
    valid_idx_in_all = [feature_ids.index(fid) for fid in valid_ids]
    vis_valid  = vis_emb[valid_idx_in_all]    # [M, dim]
    lang_valid = lang_emb[valid_idx_in_all]   # [M, dim]
    sim_valid  = vis_valid.to(device) @ lang_valid.to(device).T  # [M, M]
    top1_preds = sim_valid.argmax(dim=1).cpu()
    gt_indices = torch.arange(len(valid_ids))
    r1_hits    = (top1_preds == gt_indices).tolist()

    correct_pool   = [i for i, h in enumerate(r1_hits) if h]
    incorrect_pool = [i for i, h in enumerate(r1_hits) if not h]

    # ── Build ordered list of (local_idx, header_img_or_None) ────────────────
    # Each entry is either (local_idx, None) = query slide
    #                   or (None, header_img)  = category header card
    render_queue: list[tuple] = []

    if args.mode == "per_category":
        # Load category map: take_uid → category name
        with open(args.takes_json) as f:
            takes_raw = json.load(f)
        takes_list = takes_raw if isinstance(takes_raw, list) else takes_raw.get("takes", [])
        uid_to_category = {t["take_uid"]: t.get("parent_task_name", "Unknown")
                           for t in takes_list}

        # Group valid_ids by category
        from collections import defaultdict
        cat_to_local = defaultdict(list)
        for local_i, fid in enumerate(valid_ids):
            take_uid = fid.rsplit("__", 1)[0]
            cat = uid_to_category.get(take_uid, "Unknown")
            cat_to_local[cat].append(local_i)

        # Per-category V→T R@1 (full pool)
        for cat in sorted(cat_to_local.keys()):
            idxs = cat_to_local[cat]
            cat_r1 = sum(r1_hits[i] for i in idxs) / len(idxs) * 100
            n_cat  = len(idxs)
            header_img = render_category_header(cat, n_cat, cat_r1)
            render_queue.append((None, header_img))

            picks = random.sample(idxs, min(args.queries_per_category, len(idxs)))
            for local_i in picks:
                render_queue.append((local_i, None))
    else:
        if args.mode == "correct":
            pool = correct_pool
        elif args.mode == "incorrect":
            pool = incorrect_pool
        elif args.mode == "mixed":
            half = args.num_queries // 2
            pool = (random.sample(correct_pool,   min(half, len(correct_pool))) +
                    random.sample(incorrect_pool, min(args.num_queries - half, len(incorrect_pool))))
        else:
            pool = list(range(len(valid_ids)))

        if len(pool) < args.num_queries:
            print(f"[WARN] Only {len(pool)} queries available for mode='{args.mode}', "
                  f"requested {args.num_queries}")
        for local_i in random.sample(pool, min(args.num_queries, len(pool))):
            render_queue.append((local_i, None))

    # ── Pre-build MCQ instances (before render loop, if needed) ─────────────
    mcq_instances: dict = {}   # local_idx → (option_local_idxs, pred_pos, gt_pos)
    if args.direction == "mcq":
        fid_to_text = build_caption_map(
            args.annotation_json, args.takes_json, args.video_root, args.split_file
        )
        mcq_q_idxs, mcq_op_idxs, skip_count = build_mcq_instances(
            valid_ids, fid_to_text, seed=args.mcq_seed
        )
        print(f"[MCQ] {len(mcq_q_idxs)} valid instances, {skip_count} skipped")
        # Compute model predictions for all MCQ instances at once
        rng_mcq = random.Random(args.mcq_seed)
        if mcq_q_idxs:
            q_t   = torch.tensor(mcq_q_idxs, dtype=torch.long)
            op_t  = torch.tensor(mcq_op_idxs, dtype=torch.long)
            scores = (vis_valid[q_t].to(device).unsqueeze(1) *
                      lang_valid[op_t].to(device)).sum(-1).cpu()   # [M, 5]
            preds  = scores.argmax(1).tolist()
            for qi, ops, pred in zip(mcq_q_idxs, mcq_op_idxs, preds):
                # Shuffle display order (GT is at ops[0])
                order = list(range(5))
                rng_mcq.shuffle(order)
                shuffled_ops   = [ops[o] for o in order]
                gt_display_pos = order.index(0)   # where GT ended up after shuffle
                mcq_instances[qi] = (shuffled_ops, pred, gt_display_pos)
        # Rebuild render_queue keeping only MCQ-eligible queries
        mcq_pool = set(mcq_instances.keys())
        render_queue = [(li, hi) for li, hi in render_queue
                        if hi is not None or li in mcq_pool]

    n_query_slides = sum(1 for li, _ in render_queue if li is not None)
    dir_mult = 2 if args.direction == "both" else 1
    print(f"Rendering {n_query_slides * dir_mult} query slides + "
          f"{len(render_queue) - n_query_slides} header cards "
          f"(mode={args.mode}, direction={args.direction})")

    # ── Render slides ────────────────────────────────────────────────────────
    frame_size = 224  # display size per frame
    n_frames_t2v = 4  # frames shown per retrieved row in T→V slides
    slides_frames: list[np.ndarray] = []
    frames_per_slide = SLIDE_DURATION_SEC * FPS

    for local_idx, header_img in tqdm(render_queue, desc="Rendering slides"):
        # Category header card
        if header_img is not None:
            for _ in range(frames_per_slide):
                slides_frames.append(header_img)
            continue

        fid      = valid_ids[local_idx]
        vid_info = key_to_video_info[fid]
        gt_cap   = key_to_caption[fid]

        # ── V→T: video query → retrieve captions ────────────────────────────
        if args.direction in ("v2t", "both"):
            row_sim = sim_valid[local_idx].cpu()
            topk    = min(args.top_k, row_sim.size(0))
            top_vals, top_idx = row_sim.topk(topk)

            retrieved_v2t = [
                (key_to_caption.get(valid_ids[idx.item()], "[unknown]"), val.item())
                for idx, val in zip(top_idx, top_vals)
            ]
            gt_rank = next(
                (r for r, (cap, _) in enumerate(retrieved_v2t, 1) if cap == gt_cap), None
            )

            clip_frames = load_clip_frames(
                vid_info["video_path"], vid_info["timestamp"], vid_info["duration"],
                num_frames=args.num_frames, frame_size=frame_size,
            )
            if clip_frames is None:
                print(f"[WARN] Could not decode frames for {fid}, skipping")
                continue

            slide_img = render_slide(clip_frames, gt_cap, retrieved_v2t, local_idx, gt_rank)
            for _ in range(frames_per_slide):
                slides_frames.append(slide_img)

        # ── MCQ: 5-way multiple choice ───────────────────────────────────────
        if args.direction == "mcq":
            if local_idx not in mcq_instances:
                continue
            shuffled_ops, _, gt_display_pos = mcq_instances[local_idx]
            # Score in shuffled display order
            q_v  = vis_valid[local_idx].to(device)                       # [dim]
            op_t = torch.tensor(shuffled_ops, dtype=torch.long)
            sc   = (q_v.unsqueeze(0) * lang_valid[op_t].to(device)).sum(-1).cpu()
            pred_display_pos = sc.argmax().item()
            option_scores    = sc.tolist()

            options_text = [key_to_caption.get(valid_ids[idx], "[unknown]")
                            for idx in shuffled_ops]
            clip_frames = load_clip_frames(
                vid_info["video_path"], vid_info["timestamp"], vid_info["duration"],
                num_frames=args.num_frames, frame_size=frame_size,
            )
            if clip_frames is None:
                print(f"[WARN] Could not decode frames for {fid}, skipping")
                continue
            slide_img = render_mcq_slide(
                clip_frames, options_text,
                gt_pos=gt_display_pos, pred_pos=pred_display_pos,
                query_idx=local_idx, scores=option_scores,
            )
            for _ in range(frames_per_slide):
                slides_frames.append(slide_img)
            continue

        # ── T→V: text query → retrieve video clips ───────────────────────────
        if args.direction in ("t2v", "both"):
            col_sim = sim_valid.T[local_idx].cpu()
            topk    = min(args.top_k, col_sim.size(0))
            top_vals, top_idx = col_sim.topk(topk)

            retrieved_t2v = []
            for idx, val in zip(top_idx, top_vals):
                vid_idx = idx.item()
                vfid    = valid_ids[vid_idx]
                vinfo   = key_to_video_info[vfid]
                cap     = key_to_caption.get(vfid, "[unknown]")
                frames  = load_clip_frames(
                    vinfo["video_path"], vinfo["timestamp"], vinfo["duration"],
                    num_frames=n_frames_t2v, frame_size=frame_size,
                )
                retrieved_t2v.append((frames or [], cap, val.item()))

            gt_rank_t2v = next(
                (r for r, idx in enumerate(top_idx, 1) if idx.item() == local_idx), None
            )
            slide_img = render_t2v_slide(gt_cap, retrieved_t2v, local_idx, gt_rank_t2v)
            for _ in range(frames_per_slide):
                slides_frames.append(slide_img)

    # ── Save MP4 ─────────────────────────────────────────────────────────────
    if not slides_frames:
        print("No slides rendered — check video paths and annotation maps.")
        return

    print(f"Saving {len(slides_frames)} frames → {args.output} ...")
    iio.imwrite(args.output, slides_frames, fps=FPS, codec="libx264")

    n_slides = len(slides_frames) // frames_per_slide
    print(f"Saved: {args.output}  ({n_slides} queries, {len(slides_frames)/FPS:.0f}s)")


if __name__ == "__main__":
    main()
