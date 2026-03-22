"""
Download captions (text only) from the CC3M-LLaVA 595k HuggingFace dataset and save
them in the format expected by precompute_language_features.py.

Uses streaming so only text metadata is downloaded — no images.

Outputs:
  {output_dir}/llava-filtered-cc3m-595k/captions.json       {int_idx: caption}
  {output_dir}/llava-filtered-cc3m-595k/id_mapping.json     {int_idx: original_hf_id}
"""

import os
import json
import argparse
import tqdm

from datasets import load_dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare CC3M-LLaVA captions")
    parser.add_argument("--output_dir", type=str, default="datasets", help="Root output directory")
    parser.add_argument("--hf_dataset", type=str, default="pingzhili/llava-filtered-cc3m-595k",
                        help="HuggingFace dataset identifier")
    args = parser.parse_args()

    save_dir = os.path.join(args.output_dir, args.hf_dataset.split("/")[-1])
    os.makedirs(save_dir, exist_ok=True)

    captions_path = os.path.join(save_dir, "captions.json")
    mapping_path = os.path.join(save_dir, "id_mapping.json")

    print(f"Streaming captions from {args.hf_dataset} ...")
    # streaming=True avoids downloading the images — only parquet text columns are fetched
    dataset = load_dataset(args.hf_dataset, split="train", streaming=True)

    captions = {}
    id_mapping = {}

    for idx, sample in enumerate(tqdm.tqdm(dataset, desc="Reading captions")):
        captions[str(idx)] = sample["caption"]
        id_mapping[str(idx)] = sample["id"]

    print(f"Total captions: {len(captions)}")

    with open(captions_path, "w", encoding="utf-8") as f:
        json.dump(captions, f)
    print(f"Saved captions -> {captions_path}")

    with open(mapping_path, "w", encoding="utf-8") as f:
        json.dump(id_mapping, f)
    print(f"Saved ID mapping -> {mapping_path}")
