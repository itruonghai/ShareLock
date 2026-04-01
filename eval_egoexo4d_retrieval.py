"""
Video-text retrieval evaluation on EgoExo4D.

Metrics: R@1, R@5, R@10 for both Text→Video and Video→Text directions.
Supports overall eval and per-category breakdown (Basketball, Bike Repair, etc.).

Prerequisites — precompute val features:
    python precompute_video_features.py \\
        --video_root EgoExo/val_videos/takes \\
        --annotation_json EgoExo/annotations/atomic_descriptions_val.json \\
        --takes_json EgoExo/takes.json \\
        --split_file EgoExo/val_takes.txt \\
        --variant vjepa2.1_vitig_384 \\
        --output_dir precomputed_features_video_val \\
        --extract both --num_gpus 3 --batch_size 32

Usage:
    # Overall eval
    python eval_egoexo4d_retrieval.py \\
        --checkpoint logs/.../best_model.ckpt \\
        --config configs/egoexo4d_vjepa2_config.yaml \\
        --precomputed_features_dir precomputed_features_video_val

    # Per-category breakdown
    python eval_egoexo4d_retrieval.py \\
        --checkpoint logs/.../best_model.ckpt \\
        --config configs/egoexo4d_vjepa2_config.yaml \\
        --precomputed_features_dir precomputed_features_video_val \\
        --takes_json EgoExo/takes.json \\
        --per_category

    # Internal val split (no re-precomputation needed)
    python eval_egoexo4d_retrieval.py \\
        --checkpoint logs/.../best_model.ckpt \\
        --config configs/egoexo4d_vjepa2_config.yaml \\
        --precomputed_features_dir precomputed_features_video \\
        --use_internal_val_split
"""

import os
import json
import argparse
import random
import tempfile
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from omegaconf import OmegaConf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# PL checkpoints contain OmegaConf objects — force weights_only=False
_orig_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    kwargs["weights_only"] = False
    return _orig_torch_load(*args, **kwargs)
torch.load = _patched_torch_load

from featureutils.core import FeatureUtils
from sharelock.models.model import ShareLock, ShareLockWithTextEncoder
from sharelock.data.datasets import _load_features_parallel


def recall_at_k(sim_matrix: torch.Tensor, ks=(1, 5, 10)) -> dict:
    """
    sim_matrix: [N, N]  sim[i, j] = similarity(video_i, text_j)
    Ground truth: diagonal (video i pairs with text i).
    Returns R@K for both directions as percentages.
    """
    results = {}
    for tag, mat in [("v2t", sim_matrix), ("t2v", sim_matrix.T)]:
        ranks = mat.argsort(dim=1, descending=True)
        gt = torch.arange(mat.size(0), device=mat.device)
        for k in ks:
            hit = (ranks[:, :k] == gt.unsqueeze(1)).any(dim=1).float()
            results[f"{tag}_R@{k}"] = hit.mean().item() * 100
    return results


def compute_extended_metrics(sim_matrix: torch.Tensor) -> dict:
    """
    Compute MedR, MeanR, score gap, and GT score percentile.
    sim_matrix: [N, N] — sim[i, j] = similarity(video_i, text_j), GT on diagonal.

    Returns dict with keys: {v2t,t2v}_{MedR, MeanR, score_gap, gt_pct}
    """
    N = sim_matrix.size(0)
    results = {}
    for tag, mat in [("v2t", sim_matrix), ("t2v", sim_matrix.T)]:
        gt_scores = mat.diagonal()  # [N]
        # 1-based rank: number of elements per row strictly greater than GT score, + 1
        gt_ranks = (mat > gt_scores.unsqueeze(1)).sum(dim=1).float() + 1   # [N]
        results[f"{tag}_MedR"]  = gt_ranks.median().item()
        results[f"{tag}_MeanR"] = gt_ranks.mean().item()
        # Score gap: GT pair mean vs mean of all off-diagonal elements
        off_diag_mean = (mat.sum() - gt_scores.sum()) / (N * N - N)
        results[f"{tag}_score_gap"] = (gt_scores.mean() - off_diag_mean).item()
        # GT score percentile: fraction of per-row elements the GT score beats (higher = better)
        pct = (mat < gt_scores.unsqueeze(1)).float().mean(dim=1)   # [N], 0–1
        results[f"{tag}_gt_pct"] = pct.mean().item() * 100         # as %
    return results


def print_extended_metrics(metrics: dict, n: int, label: str = "Overall"):
    """Print MedR / MeanR / ScoreGap / GT-pct row (companion to print_retrieval_table)."""
    rnd = n / 2
    print(f"  {label:<18}  N={n:>5}  "
          f"V→T  MedR={metrics['v2t_MedR']:6.0f}(rnd≈{rnd:.0f})  "
          f"MeanR={metrics['v2t_MeanR']:6.0f}  "
          f"ScoreGap={metrics['v2t_score_gap']:+.4f}  "
          f"GT-pct={metrics['v2t_gt_pct']:5.1f}%  |  "
          f"T→V  MedR={metrics['t2v_MedR']:6.0f}  "
          f"MeanR={metrics['t2v_MeanR']:6.0f}  "
          f"ScoreGap={metrics['t2v_score_gap']:+.4f}  "
          f"GT-pct={metrics['t2v_gt_pct']:5.1f}%")


def save_analysis_plot(sim: torch.Tensor, output_path: str,
                       mcq_per_query: torch.Tensor | None = None,
                       mcq_cat_correct: dict | None = None) -> None:
    """
    Save analysis PNG:
      Panel 1 — V→T GT rank histogram vs. uniform random baseline
      Panel 2 — GT pair scores vs. background score distribution
      Panel 3 — MCQ per-category accuracy bar chart (only if mcq_per_query given)
    """
    N = sim.size(0)
    sim_cpu = sim.cpu()

    gt_scores_v2t = sim_cpu.diagonal()
    gt_ranks_v2t  = (sim_cpu > gt_scores_v2t.unsqueeze(1)).sum(dim=1).numpy() + 1
    gt_scores_np  = gt_scores_v2t.numpy()

    n_bg      = min(50_000, N * (N - 1))
    all_scores = sim_cpu.numpy().flatten()
    mask = np.ones(N * N, dtype=bool)
    mask[np.arange(N) * (N + 1)] = False
    bg_scores = all_scores[mask]
    rng_np    = np.random.default_rng(0)
    bg_sample = rng_np.choice(bg_scores, size=min(n_bg, len(bg_scores)), replace=False)

    has_mcq  = mcq_per_query is not None and len(mcq_per_query) > 0
    n_panels = 3 if (has_mcq and mcq_cat_correct) else 2
    fig, axes = plt.subplots(1, n_panels, figsize=(6.5 * n_panels, 5))
    if n_panels == 2:
        axes = list(axes)
    fig.patch.set_facecolor("#1a1a2e")
    for ax in axes:
        ax.set_facecolor("#111133")
        ax.tick_params(colors="#cccccc")
        ax.spines[:].set_color("#444466")
        ax.title.set_color("#e0e0e0")
        ax.xaxis.label.set_color("#cccccc")
        ax.yaxis.label.set_color("#cccccc")

    # ── Panel 1: GT rank histogram ───────────────────────────────────────────
    ax = axes[0]
    bins = min(50, N)
    ax.hist(gt_ranks_v2t, bins=bins, color="#5588ff", alpha=0.85,
            edgecolor="#3366cc", label="Model GT ranks")
    expected_per_bin = N / bins
    ax.axhline(expected_per_bin, color="#ff9933", linewidth=1.5,
               linestyle="--", label=f"Random ({expected_per_bin:.0f}/bin)")
    med_r = float(np.median(gt_ranks_v2t))
    ax.axvline(med_r, color="#06d6a0", linewidth=1.5,
               linestyle=":", label=f"MedR = {med_r:.0f}")
    ax.set_title(f"V→T GT Rank Distribution  (N={N})", fontsize=12)
    ax.set_xlabel("GT Rank (lower is better)")
    ax.set_ylabel("# Queries")
    ax.legend(fontsize=9, facecolor="#222244", labelcolor="#cccccc")

    # ── Panel 2: score distributions ─────────────────────────────────────────
    ax = axes[1]
    ax.hist(bg_sample, bins=60, color="#888888", alpha=0.6, density=True,
            label=f"Background ({len(bg_sample):,} sampled)")
    ax.hist(gt_scores_np, bins=60, color="#06d6a0", alpha=0.85, density=True,
            label=f"GT pairs (N={N})")
    gap = float(gt_scores_np.mean() - bg_sample.mean())
    ax.set_title(f"Similarity Score: GT vs Background  (gap={gap:+.4f})", fontsize=12)
    ax.set_xlabel("Cosine similarity")
    ax.set_ylabel("Density")
    ax.legend(fontsize=9, facecolor="#222244", labelcolor="#cccccc")

    # ── Panel 3: MCQ per-category accuracy bar chart ─────────────────────────
    if n_panels == 3:
        ax = axes[2]
        cats    = sorted(mcq_cat_correct.keys())
        accs    = [100.0 * sum(mcq_cat_correct[c]) / len(mcq_cat_correct[c]) for c in cats]
        ns      = [len(mcq_cat_correct[c]) for c in cats]
        colors  = ["#06d6a0" if a >= 50 else "#ef4444" for a in accs]
        y_pos   = np.arange(len(cats))
        bars = ax.barh(y_pos, accs, color=colors, alpha=0.85, edgecolor="#222244")
        ax.axvline(20, color="#ff9933", linewidth=1.5, linestyle="--", label="Random (20%)")
        ax.axvline(100 * sum(mcq_per_query.tolist()) / len(mcq_per_query),
                   color="#e2c96e", linewidth=1.5, linestyle=":", label="Overall MCQ")
        for bar, acc, n in zip(bars, accs, ns):
            ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                    f"{acc:.0f}% (N={n})", va="center", color="#cccccc", fontsize=7)
        short_cats = [c[:22] + "…" if len(c) > 22 else c for c in cats]
        ax.set_yticks(y_pos)
        ax.set_yticklabels(short_cats, fontsize=8, color="#cccccc")
        ax.set_xlim(0, 110)
        ax.set_xlabel("MCQ Accuracy (%)")
        ax.set_title("5-Way MCQ Accuracy per Category", fontsize=12)
        ax.legend(fontsize=9, facecolor="#222244", labelcolor="#cccccc")

    overall_mcq = (f"  |  MCQ={100*sum(mcq_per_query.tolist())/len(mcq_per_query):.1f}%"
                   if has_mcq else "")
    fig.suptitle(f"EgoExo4D Retrieval Analysis  (N={N},  MedR={med_r:.0f}{overall_mcq})",
                 color="#e2c96e", fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig(output_path, dpi=120, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Analysis plot saved → {output_path}")


# ── 5-way MCQ helpers ────────────────────────────────────────────────────────

_PUNCT     = str.maketrans("", "", ".,;:?!()")
_VERB_SKIP = frozenset({
    "and", "the", "a", "an", "or", "but", "with",
    "c", "man", "woman", "person", "x",
    "his", "her", "their", "its",
})


def extract_primary_verb(caption: str) -> str | None:
    """Extract action verb from EgoExo4D caption "C [verb] ..."."""
    words = caption.split()
    if not words or words[0].lower() != "c":
        return None
    for w in words[1:]:
        clean = w.translate(_PUNCT).lower()
        if not clean or clean in _VERB_SKIP:
            continue
        if w[0].isupper():   # proper noun / name
            continue
        return clean
    return None


def build_caption_map(annotation_json: str, takes_json: str,
                      video_root: str, split_file: str | None = None) -> dict:
    """Return {feature_id → caption_text} using same key formula as precompute."""
    import av as _av
    from dataloader_video import load_egoexo4d_annotations, FilterConfig, SamplingConfig
    _fps_cache: dict = {}

    def _fps(path):
        if path not in _fps_cache:
            try:
                c = _av.open(path)
                _fps_cache[path] = float(c.streams.video[0].average_rate) or 30.0
                c.close()
            except Exception:
                _fps_cache[path] = 30.0
        return _fps_cache[path]

    print("[MCQ] Loading annotations for caption map...")
    samples = load_egoexo4d_annotations(
        annotation_json, takes_json, video_root,
        FilterConfig(keep_subject_C_only=True, drop_unsure=True,
                     require_ego_visible=True, min_timestamp=2.0),
        SamplingConfig(num_frames=16, clip_duration=4.0),
        split_file=split_file,
    )
    out: dict = {}
    for s in samples:
        key = f"{s['take_uid']}__{int(round(s['timestamp'] * _fps(s['video_path'])))}"
        out.setdefault(key, s["text"])
    print(f"[MCQ] Caption map: {len(out)} entries from {len(_fps_cache)} unique videos")
    return out


def build_mcq_instances(feature_ids: list, fid_to_text: dict,
                        seed: int = 42) -> tuple:
    """
    Build 5-way MCQ instances: for each feature_id find 4 same-take distractors
    with a different primary action verb.

    Returns (query_idxs, option_idxs, skip_count).
    option_idxs[i] = [gt_idx, dist1, dist2, dist3, dist4]  (GT at position 0)
    """
    rng = random.Random(seed)
    take_to_idxs: dict = defaultdict(list)
    for i, fid in enumerate(feature_ids):
        take_to_idxs[fid.rsplit("__", 1)[0]].append(i)

    idx_to_verb = {
        i: extract_primary_verb(fid_to_text[fid]) if fid in fid_to_text else None
        for i, fid in enumerate(feature_ids)
    }

    query_idxs, option_idxs, skip_count = [], [], 0
    for i, fid in enumerate(feature_ids):
        gv = idx_to_verb[i]
        if gv is None:
            skip_count += 1
            continue
        same_take = take_to_idxs[fid.rsplit("__", 1)[0]]
        candidates = [j for j in same_take
                      if j != i and idx_to_verb[j] and idx_to_verb[j] != gv]
        if len(candidates) < 4:
            skip_count += 1
            continue
        query_idxs.append(i)
        option_idxs.append([i] + rng.sample(candidates, 4))  # GT at index 0
    return query_idxs, option_idxs, skip_count


def evaluate_mcq_5way(vis_emb: torch.Tensor, lang_emb: torch.Tensor,
                      query_idxs: list, option_idxs: list,
                      device: torch.device) -> tuple:
    """
    Vectorized 5-way MCQ. GT is at option index 0.
    Returns (accuracy_pct, per_query_correct BoolTensor [M]).
    """
    if not query_idxs:
        return 0.0, torch.zeros(0, dtype=torch.bool)
    q_idx  = torch.tensor(query_idxs,  dtype=torch.long)   # [M]  keep on cpu for indexing
    op_idx = torch.tensor(option_idxs, dtype=torch.long)   # [M, 5]
    # [M, 1, dim] · [M, 5, dim] → [M, 5] cosine similarities
    scores  = (vis_emb[q_idx].to(device).unsqueeze(1) * lang_emb[op_idx].to(device)).sum(-1)
    correct = (scores.argmax(1) == 0).cpu()
    return correct.float().mean().item() * 100.0, correct


def load_category_map(takes_json: str) -> dict:
    """Return {take_uid: parent_task_name} from takes.json."""
    with open(takes_json) as f:
        takes = json.load(f)
    # takes.json is a list of take dicts
    if isinstance(takes, list):
        return {t["take_uid"]: t.get("parent_task_name", "Unknown") for t in takes}
    # some versions wrap in a dict
    if "takes" in takes:
        return {t["take_uid"]: t.get("parent_task_name", "Unknown") for t in takes["takes"]}
    return {}


def print_retrieval_table(metrics: dict, n: int, label: str = "Overall"):
    """Print a formatted retrieval results row."""
    print(f"  {label:<18}  N={n:>5}  "
          f"V→T  R@1={metrics['v2t_R@1']:5.1f}%  R@5={metrics['v2t_R@5']:5.1f}%  R@10={metrics['v2t_R@10']:5.1f}%  |  "
          f"T→V  R@1={metrics['t2v_R@1']:5.1f}%  R@5={metrics['t2v_R@5']:5.1f}%  R@10={metrics['t2v_R@10']:5.1f}%")


def main():
    parser = argparse.ArgumentParser(
        description="EgoExo4D video-text retrieval evaluation (R@1/5/10)"
    )
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to ShareLock .ckpt checkpoint")
    parser.add_argument("--config", type=str, default="configs/egoexo4d_vjepa2_config.yaml")
    parser.add_argument("--precomputed_features_dir", type=str, required=True,
                        help="Base dir of precomputed features (same layout as training)")
    parser.add_argument("--caption_name", type=str, default=None,
                        help="Caption subdirectory (default: config caption_files[0])")
    parser.add_argument("--takes_json", type=str, default="EgoExo/takes.json",
                        help="Path to takes.json for category lookup")
    parser.add_argument("--per_category", action="store_true",
                        help="Report per-category breakdown (requires --takes_json)")
    parser.add_argument("--mcq_5way", action="store_true",
                        help="Evaluate 5-way MCQ accuracy (1 GT + 4 same-take hard negatives)")
    parser.add_argument("--annotation_json", type=str, default=None,
                        help="atomic_descriptions_val.json (required for --mcq_5way)")
    parser.add_argument("--video_root", type=str, default=None,
                        help="Root of val video takes (required for --mcq_5way)")
    parser.add_argument("--split_file", type=str, default=None,
                        help="val_takes.txt split filter (optional)")
    parser.add_argument("--mcq_seed", type=int, default=42,
                        help="RNG seed for distractor sampling")
    parser.add_argument("--use_internal_val_split", action="store_true",
                        help="Use the same seeded val holdout as training "
                             "(no separate val precomputation needed)")
    parser.add_argument("--max_eval_pairs", type=int, default=None,
                        help="Randomly subsample N pairs (e.g. 1000) for faster eval")
    parser.add_argument("--batch_size", type=int, default=4096,
                        help="Batch size for projector inference")
    parser.add_argument("--save_plot", type=str, default=None,
                        help="If set, save rank-distribution + score-gap PNG to this path")
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

    # ── Resolve feature directories ──────────────────────────────────────────
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
    print(f"Vision features : {vision_dir}")
    print(f"Language features: {lang_dir}")

    # ── Load features ────────────────────────────────────────────────────────
    staging_dir = os.environ.get("TMPDIR", tempfile.mkdtemp())
    vision_fu = FeatureUtils(base_dir=vision_dir, staging_dir=staging_dir, require_features_exist=True)
    lang_fu   = FeatureUtils(base_dir=lang_dir,   staging_dir=staging_dir, require_features_exist=True)

    vision_fu.stage_data(features=["vision_features"])
    lang_fu.stage_data(features=["language_features"])

    vision_keys = set(vision_fu.list_keys())
    lang_keys   = set(lang_fu.list_keys())
    feature_ids = sorted(vision_keys & lang_keys)
    print(f"Aligned features: {len(feature_ids)}")

    if args.use_internal_val_split:
        rng = random.Random(config.seed)
        rng.shuffle(feature_ids)
        feature_ids = feature_ids[: config.data.val_split_num]
        print(f"Using internal val split: {len(feature_ids)} samples")

    if args.max_eval_pairs is not None and args.max_eval_pairs < len(feature_ids):
        random.Random(config.seed).shuffle(feature_ids)
        feature_ids = feature_ids[: args.max_eval_pairs]
        print(f"Subsampled to {len(feature_ids)} pairs for eval")

    print(f"Loading vision features ({len(feature_ids)} samples)...")
    vis_loaded = _load_features_parallel(vision_fu, feature_ids, ["vision_features"])
    vision_tensor = vis_loaded["vision_features"]           # [N, vision_dim]

    print(f"Loading language features ({len(feature_ids)} samples)...")
    lang_loaded = _load_features_parallel(lang_fu, feature_ids, ["language_features"])
    language_tensor = lang_loaded["language_features"].squeeze(1)  # [N, lang_dim]

    # ── Project all features ─────────────────────────────────────────────────
    print("Projecting through trained projectors...")
    all_vis, all_lang = [], []
    with torch.no_grad():
        for i in tqdm(range(0, len(feature_ids), args.batch_size), desc="Projecting"):
            vis_batch  = vision_tensor[i : i + args.batch_size].to(device)
            lang_batch = language_tensor[i : i + args.batch_size].to(device)
            all_vis.append(model.vision_projector(vis_batch).cpu())
            all_lang.append(model.language_projector(lang_batch).cpu())

    vis_emb  = F.normalize(torch.cat(all_vis,  dim=0), dim=-1)   # [N, dim]
    lang_emb = F.normalize(torch.cat(all_lang, dim=0), dim=-1)   # [N, dim]

    # ── 5-way MCQ evaluation ─────────────────────────────────────────────────
    mcq_accuracy      = None
    mcq_per_query     = None
    mcq_q_idxs        = []
    mcq_cat_correct: dict = defaultdict(list)

    if args.mcq_5way:
        if not args.annotation_json or not args.video_root:
            print("[MCQ] ERROR: --mcq_5way requires --annotation_json and --video_root")
        else:
            fid_to_text = build_caption_map(
                args.annotation_json, args.takes_json, args.video_root, args.split_file
            )
            print("[MCQ] Building MCQ instances (same-take, different-verb)...")
            mcq_q_idxs, mcq_op_idxs, skip_count = build_mcq_instances(
                feature_ids, fid_to_text, seed=args.mcq_seed
            )
            print(f"[MCQ] {len(mcq_q_idxs)} valid queries  |  {skip_count} skipped "
                  f"(no verb / < 4 same-take different-verb distractors)")
            if mcq_q_idxs:
                mcq_accuracy, mcq_per_query = evaluate_mcq_5way(
                    vis_emb, lang_emb, mcq_q_idxs, mcq_op_idxs, device
                )
                # Build per-category MCQ accuracy map
                if os.path.exists(args.takes_json):
                    cat_map_mcq = load_category_map(args.takes_json)
                    for qi, ok in zip(mcq_q_idxs, mcq_per_query.tolist()):
                        cat = cat_map_mcq.get(feature_ids[qi].rsplit("__", 1)[0], "Unknown")
                        mcq_cat_correct[cat].append(ok)

    # ── Overall retrieval metrics ────────────────────────────────────────────
    N = vis_emb.size(0)
    print(f"Computing {N}×{N} similarity matrix...")
    sim = vis_emb.to(device) @ lang_emb.to(device).T              # [N, N]
    overall     = recall_at_k(sim)
    overall_ext = compute_extended_metrics(sim)

    print("\n" + "=" * 130)
    print(f"EgoExo4D Video-Text Retrieval")
    print(f"  Checkpoint  : {args.checkpoint}")
    print(f"  Features    : {args.precomputed_features_dir}")
    print(f"  Pool size   : {N}  (random R@1 = {100/N:.3f}%,  random MedR ≈ {N//2})")
    print("-" * 130)
    print_retrieval_table(overall, N, "Overall")
    print_extended_metrics(overall_ext, N, "Overall")
    if mcq_accuracy is not None:
        print(f"  {'MCQ 5-way':<18}  N={len(mcq_q_idxs):>5}  "
              f"Accuracy={mcq_accuracy:5.1f}%  (random=20.0%,  gain={mcq_accuracy-20:.1f}pp)")

    # ── Per-category breakdown ───────────────────────────────────────────────
    if args.per_category:
        if not os.path.exists(args.takes_json):
            print(f"\n[WARN] takes_json not found: {args.takes_json} — skipping per-category eval")
        else:
            category_map = load_category_map(args.takes_json)  # {take_uid: category_name}

            # Map each feature_id → category via take_uid (key format: {take_uid}__{frame_idx})
            idx_by_category = defaultdict(list)
            for idx, fid in enumerate(feature_ids):
                take_uid = fid.rsplit("__", 1)[0]
                category = category_map.get(take_uid, "Unknown")
                idx_by_category[category].append(idx)

            print("-" * 130)
            for category in sorted(idx_by_category.keys()):
                idxs = torch.tensor(idx_by_category[category])
                if len(idxs) < 2:
                    continue
                # Sub-matrix: only pairs from this category
                cat_sim = sim[idxs][:, idxs]
                cat_metrics = recall_at_k(cat_sim)
                cat_ext     = compute_extended_metrics(cat_sim)
                print_retrieval_table(cat_metrics, len(idxs), category)
                print_extended_metrics(cat_ext,    len(idxs), "")
                if mcq_cat_correct and category in mcq_cat_correct:
                    vals = mcq_cat_correct[category]
                    cat_mcq_acc = 100.0 * sum(vals) / len(vals)
                    print(f"  {'':18}           MCQ={cat_mcq_acc:5.1f}%  (N={len(vals)},  gain={cat_mcq_acc-20:.1f}pp)")

    print("=" * 130)

    if args.save_plot:
        save_analysis_plot(sim, args.save_plot,
                           mcq_per_query=mcq_per_query,
                           mcq_cat_correct=mcq_cat_correct if mcq_cat_correct else None)


if __name__ == "__main__":
    main()
