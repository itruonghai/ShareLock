"""
Check Ego4D egovid-5m dataset integrity.

video_id format: {VideoID}_{StartFrame}_{EndFrame}[.mp4]
  - VideoID     → source video filename (VideoID.mp4 in v2/video_540ss/)
  - StartFrame  → first frame index of the clip
  - EndFrame    → last frame index of the clip

CSV columns used: video_id, frame_num, fps, llava_cap

Usage:
    python check_ego4d_dataset.py --ego4d_root /path/to/ego4d
    python check_ego4d_dataset.py --ego4d_root /path/to/ego4d --split train
    python check_ego4d_dataset.py --ego4d_root /path/to/ego4d --save_missing
"""

import argparse
from pathlib import Path

import pandas as pd


def check_split(ego4d_root: Path, csv_file: Path, split_name: str, save_missing: bool = False):
    print(f"\n{'='*60}")
    print(f"Split : {split_name}")
    print(f"CSV   : {csv_file}")
    print(f"{'='*60}")

    if not csv_file.exists():
        print(f"[ERROR] CSV not found: {csv_file}")
        return []

    # ── Read only the columns we need ────────────────────────────────────────
    needed = ["video_id", "llava_cap", "frame_num", "fps"]
    df = pd.read_csv(
        csv_file,
        usecols=needed,
        dtype={"video_id": str, "frame_num": "Int32"},
    )
    print(f"Total rows in CSV : {len(df)}")

    for col in needed:
        n = df[col].isna().sum()
        if n:
            print(f"[WARN] Rows missing '{col}': {n}")

    df = df.dropna(subset=needed).copy()
    df["frame_num"] = df["frame_num"].astype(int)
    df["fps"]       = df["fps"].astype(float)
    print(f"Valid rows        : {len(df)}")

    # ── Vectorised parse: VideoID_StartFrame_EndFrame[.mp4] ──────────────────
    bare = df["video_id"].str.replace(r"\.mp4$", "", regex=True)
    parts = bare.str.rsplit("_", n=2, expand=True)   # columns 0, 1, 2
    df["source_id"]   = parts[0]
    df["start_frame"] = pd.to_numeric(parts[1], errors="coerce")
    df["end_frame"]   = pd.to_numeric(parts[2], errors="coerce")

    n_unparsed = df["start_frame"].isna().sum()
    if n_unparsed:
        print(f"[WARN] {n_unparsed} video_ids could not be parsed as "
              f"VideoID_StartFrame_EndFrame")

    parseable = df["start_frame"].notna()
    df_p = df[parseable].copy()
    df_p["start_frame"] = df_p["start_frame"].astype(int)
    df_p["end_frame"]   = df_p["end_frame"].astype(int)

    # ── Clip consistency ──────────────────────────────────────────────────────
    bad_order = df_p["start_frame"] >= df_p["end_frame"]
    if bad_order.any():
        print(f"[WARN] {bad_order.sum()} clips where start_frame >= end_frame")

    dur = df_p["frame_num"] / df_p["fps"]
    print(f"\nClip duration (s) — "
          f"min={dur.min():.2f}  max={dur.max():.2f}  "
          f"mean={dur.mean():.2f}  median={dur.median():.2f}")
    if (dur < 1.0).any():
        print(f"[WARN] {(dur < 1.0).sum()} clips shorter than 1 s")
    if (dur > 300.0).any():
        print(f"[WARN] {(dur > 300.0).sum()} clips longer than 300 s")

    # ── Source video presence ─────────────────────────────────────────────────
    video_dir = ego4d_root / "v2" / "video_540ss"
    if not video_dir.exists():
        print(f"\n[ERROR] Video directory not found: {video_dir}")
        return []

    disk_files       = {p.stem for p in video_dir.iterdir() if p.suffix == ".mp4"}
    source_ids_in_csv = set(df_p["source_id"].unique())
    print(f"\nSource video files on disk : {len(disk_files)}")
    print(f"Unique source videos in CSV: {len(source_ids_in_csv)}")

    missing_sources = sorted(source_ids_in_csv - disk_files)
    print(f"Source videos missing      : {len(missing_sources)}")

    if missing_sources:
        # Value-counts gives per-source clip count without a per-row loop
        counts = df_p[df_p["source_id"].isin(missing_sources)]["source_id"].value_counts()
        print(f"\nFirst 30 missing source video IDs:")
        for sid in missing_sources[:30]:
            print(f"  {sid}  ({counts.get(sid, 0)} clips)")
        if len(missing_sources) > 30:
            print(f"  … and {len(missing_sources) - 30} more")

    clips_missing = df_p["source_id"].isin(missing_sources).sum()
    print(f"Clips affected (source missing): {clips_missing} / {len(df_p)}")

    # ── Orphaned source videos ────────────────────────────────────────────────
    orphaned = sorted(disk_files - source_ids_in_csv)
    print(f"\nOrphaned source videos on disk (no clip in CSV): {len(orphaned)}")
    if orphaned:
        print(f"  First 10: {orphaned[:10]}")

    # ── Duplicates ────────────────────────────────────────────────────────────
    n_dups = df["video_id"].duplicated(keep=False).sum()
    if n_dups:
        print(f"\n[WARN] Duplicate video_id entries: {n_dups} rows")

    # ── Caption stats ─────────────────────────────────────────────────────────
    cap_lens = df["llava_cap"].str.split().str.len()
    print(f"\nCaption length (words) — "
          f"min={cap_lens.min()}  max={cap_lens.max()}  "
          f"mean={cap_lens.mean():.1f}  median={cap_lens.median():.1f}")

    if save_missing and missing_sources:
        out = Path(f"missing_ego4d_{split_name}.txt")
        out.write_text("\n".join(missing_sources) + "\n")
        print(f"\nSaved missing source IDs → {out}")

    return missing_sources


def main():
    parser = argparse.ArgumentParser(
        description="Check Ego4D egovid-5m dataset integrity"
    )
    parser.add_argument("--ego4d_root", type=str, required=True)
    parser.add_argument("--train_csv",  type=str, default=None,
                        help="Training CSV (default: ego4d_root/egovid-text.csv)")
    parser.add_argument("--val_csv",    type=str, default=None,
                        help="Validation CSV (default: ego4d_root/egovid-val.csv)")
    parser.add_argument("--split",      type=str, default="both",
                        choices=["train", "val", "both"])
    parser.add_argument("--save_missing", action="store_true")
    args = parser.parse_args()

    ego4d_root = Path(args.ego4d_root)
    train_csv  = Path(args.train_csv) if args.train_csv else ego4d_root / "egovid-text.csv"
    val_csv    = Path(args.val_csv)   if args.val_csv   else ego4d_root / "egovid-val.csv"

    if args.split in ("train", "both"):
        check_split(ego4d_root, train_csv, "train", args.save_missing)
    if args.split in ("val", "both"):
        check_split(ego4d_root, val_csv, "val", args.save_missing)

    print("\nDone.")


if __name__ == "__main__":
    main()
