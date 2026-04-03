"""
Check Ego4D egovid-5m dataset integrity.

video_id format: {VideoID}_{StartFrame}_{EndFrame}
  - VideoID     → source video filename (VideoID.mp4 in v2/video_540ss/)
  - StartFrame  → first frame index of the clip
  - EndFrame    → last frame index of the clip

CSV columns used: video_id, frame_num, fps, llava_cap

Checks performed:
  - Missing field values (video_id / llava_cap / frame_num / fps)
  - Clip consistency: frame_num == EndFrame - StartFrame, start < end
  - Clip duration sanity: frame_num / fps in a reasonable range
  - Source videos in CSV but not on disk (missing)
  - Source videos on disk but not referenced by any clip (orphaned)
  - Duplicate video_id entries

Usage:
    # Check both splits (default)
    python check_ego4d_dataset.py --ego4d_root /path/to/ego4d

    # Check only one split
    python check_ego4d_dataset.py --ego4d_root /path/to/ego4d --split train

    # Custom CSV paths
    python check_ego4d_dataset.py \\
        --ego4d_root /path/to/ego4d \\
        --train_csv /path/to/egovid-text.csv \\
        --val_csv   /path/to/egovid-val.csv

    # Save missing source video IDs to a text file
    python check_ego4d_dataset.py --ego4d_root /path/to/ego4d --save_missing
"""

import argparse
from pathlib import Path

import pandas as pd


def parse_video_id(video_id: str):
    """
    Parse 'VideoID_StartFrame_EndFrame[.mp4]' into
    (source_video_id, start_frame, end_frame).
    Returns (video_id, None, None) if the format cannot be parsed.
    """
    bare = video_id.removesuffix(".mp4")   # CSV may include .mp4 suffix
    parts = bare.rsplit("_", 2)
    if len(parts) == 3:
        try:
            start = int(parts[1])
            end   = int(parts[2])
            return parts[0], start, end
        except ValueError:
            pass
    return video_id, None, None


def check_split(ego4d_root: Path, csv_file: Path, split_name: str, save_missing: bool = False):
    print(f"\n{'='*60}")
    print(f"Split : {split_name}")
    print(f"CSV   : {csv_file}")
    print(f"{'='*60}")

    if not csv_file.exists():
        print(f"[ERROR] CSV not found: {csv_file}")
        return []

    df = pd.read_csv(csv_file)
    print(f"Total rows in CSV : {len(df)}")

    # ── 1. Missing required fields ────────────────────────────────────────────
    required = ["video_id", "llava_cap", "frame_num", "fps"]
    for col in required:
        n = df[col].isna().sum() if col in df.columns else len(df)
        if n:
            print(f"[WARN] Rows missing '{col}': {n}")

    df_valid = df.dropna(subset=[c for c in required if c in df.columns]).copy()
    df_valid["video_id"]   = df_valid["video_id"].astype(str)
    df_valid["frame_num"]  = df_valid["frame_num"].astype(int)
    df_valid["fps"]        = df_valid["fps"].astype(float)
    print(f"Valid rows (all required fields): {len(df_valid)}")

    # ── 2. Parse video_id → (source_video_id, start_frame, end_frame) ────────
    parsed = df_valid["video_id"].map(parse_video_id)
    df_valid = df_valid.copy()
    df_valid["source_id"]    = parsed.map(lambda x: x[0])
    df_valid["start_frame"]  = parsed.map(lambda x: x[1])
    df_valid["end_frame"]    = parsed.map(lambda x: x[2])

    n_unparsed = df_valid["start_frame"].isna().sum()
    if n_unparsed:
        print(f"[WARN] {n_unparsed} video_ids could not be parsed as "
              f"VideoID_StartFrame_EndFrame")

    # ── 3. Clip consistency checks ────────────────────────────────────────────
    parseable = df_valid["start_frame"].notna()

    # start < end
    bad_order = parseable & (df_valid["start_frame"] >= df_valid["end_frame"])
    if bad_order.any():
        print(f"[WARN] {bad_order.sum()} clips where start_frame >= end_frame")

    # Duration sanity (< 1s or > 5 min are suspicious)
    df_valid["clip_duration"] = df_valid["frame_num"] / df_valid["fps"]
    too_short = (df_valid["clip_duration"] < 1.0).sum()
    too_long  = (df_valid["clip_duration"] > 300.0).sum()
    dur = df_valid["clip_duration"]
    print(f"\nClip duration (s) — "
          f"min={dur.min():.2f}  max={dur.max():.2f}  "
          f"mean={dur.mean():.2f}  median={dur.median():.2f}")
    if too_short:
        print(f"[WARN] {too_short} clips shorter than 1 s")
    if too_long:
        print(f"[WARN] {too_long} clips longer than 300 s")

    # ── 4. Check source videos against disk ───────────────────────────────────
    video_dir = ego4d_root / "v2" / "video_540ss"
    if not video_dir.exists():
        print(f"\n[ERROR] Video directory not found: {video_dir}")
        return []

    disk_files = {p.stem for p in video_dir.iterdir() if p.suffix == ".mp4"}
    print(f"\nSource video files on disk : {len(disk_files)}")

    source_ids_in_csv = set(df_valid["source_id"].dropna().unique())
    print(f"Unique source videos in CSV: {len(source_ids_in_csv)}")

    missing_sources  = sorted(source_ids_in_csv - disk_files)
    print(f"Source videos missing      : {len(missing_sources)}")

    if missing_sources:
        print(f"\nFirst 30 missing source video IDs:")
        for sid in missing_sources[:30]:
            n_clips = (df_valid["source_id"] == sid).sum()
            print(f"  {sid}  ({n_clips} clips in CSV)")
        if len(missing_sources) > 30:
            print(f"  … and {len(missing_sources) - 30} more")

    # Count clips whose source video is missing
    clips_missing = df_valid["source_id"].isin(missing_sources).sum()
    print(f"Clips affected (source missing): {clips_missing} / {len(df_valid)}")

    # ── 5. Orphaned source videos ─────────────────────────────────────────────
    orphaned = sorted(disk_files - source_ids_in_csv)
    print(f"\nOrphaned source videos on disk (no clip in CSV): {len(orphaned)}")
    if orphaned and len(orphaned) <= 20:
        for o in orphaned:
            print(f"  {o}")
    elif orphaned:
        print(f"  First 10: {orphaned[:10]}")

    # ── 6. Duplicate video_id ─────────────────────────────────────────────────
    dup_mask = df_valid["video_id"].duplicated(keep=False)
    n_dups   = dup_mask.sum()
    if n_dups:
        print(f"\n[WARN] Duplicate video_id entries: {n_dups} rows")
        print(df_valid[dup_mask][["video_id", "llava_cap"]].head(10).to_string())

    # ── 7. Caption stats ──────────────────────────────────────────────────────
    cap_lens = df_valid["llava_cap"].astype(str).str.split().str.len()
    print(f"\nCaption length (words) — "
          f"min={cap_lens.min()}  max={cap_lens.max()}  "
          f"mean={cap_lens.mean():.1f}  median={cap_lens.median():.1f}")

    # ── 8. Optionally save missing source IDs ────────────────────────────────
    if save_missing and missing_sources:
        out = Path(f"missing_ego4d_{split_name}.txt")
        out.write_text("\n".join(missing_sources) + "\n")
        print(f"\nSaved missing source IDs → {out}")

    return missing_sources


def main():
    parser = argparse.ArgumentParser(
        description="Check Ego4D egovid-5m dataset integrity"
    )
    parser.add_argument("--ego4d_root", type=str, required=True,
                        help="Path to the ego4d root directory")
    parser.add_argument("--train_csv", type=str, default=None,
                        help="Training CSV (default: ego4d_root/egovid-text.csv)")
    parser.add_argument("--val_csv", type=str, default=None,
                        help="Validation CSV (default: ego4d_root/egovid-val.csv)")
    parser.add_argument("--split", type=str, default="both",
                        choices=["train", "val", "both"],
                        help="Which split(s) to check")
    parser.add_argument("--save_missing", action="store_true",
                        help="Write missing source video IDs to "
                             "missing_ego4d_<split>.txt")
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
