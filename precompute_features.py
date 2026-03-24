"""
Unified feature extractor for ShareLock.

Precomputes both vision (DINOv2) and language (Llama-3-8B) features from a
HuggingFace dataset in a single script. The dataset is loaded once per worker
and captions are collected in memory, eliminating the separate caption
preparation step.

Multi-GPU is handled automatically via torch.multiprocessing.spawn:

    # Single GPU
    python precompute_features.py \
        --vision_model dinov2_vitl14 \
        --language_model meta-llama/Meta-Llama-3-8B

    # 3 GPUs (automatic sharding)
    python precompute_features.py \
        --vision_model dinov2_vitl14 \
        --language_model meta-llama/Meta-Llama-3-8B \
        --num_gpus 3

    # Language only (streams text, no image download)
    python precompute_features.py --extract language \
        --language_model meta-llama/Meta-Llama-3-8B --num_gpus 3
"""

import os
import gc
import torch
import torch.multiprocessing as mp
import tqdm
import argparse

from datasets import load_dataset
from torch.utils.data import DataLoader

from featureutils.core import FeatureUtils
from sharelock.models.vision_encoder import VisionEncoder
from sharelock.models.language_encoder import LanguageEncoder
from sharelock.utils.misc import get_transforms


_IMG_TRANSFORMS = get_transforms()


def _vision_transform_fn(batch):
    """Applied lazily per-batch by HF Dataset.with_transform (must be picklable)."""
    batch["image"] = [_IMG_TRANSFORMS(img.convert("RGB")) for img in batch["image"]]
    return batch


def _vision_collate_fn(samples):
    """Stack image tensors into a batch (must be picklable)."""
    return torch.stack([s["image"] for s in samples])


def extract_vision(rank, num_gpus, dataset, args):
    """Encode images with VisionEncoder on the assigned GPU."""
    device = torch.device(f"cuda:{rank}")
    output_dir = f"{args.output_dir}/{args.vision_model.split('/')[-1]}"

    feature_utils = FeatureUtils(base_dir=output_dir, staging_dir=args.cache_dir, feature_num=1)
    existing_keys = set(feature_utils.list_keys())
    print(f"[Vision GPU {rank}] Existing features: {len(existing_keys)}")

    transformed = dataset.with_transform(_vision_transform_fn)

    dataloader = DataLoader(
        transformed, batch_size=args.vision_batch_size, num_workers=args.num_workers,
        collate_fn=_vision_collate_fn, pin_memory=True,
        multiprocessing_context="fork" if args.num_gpus > 1 else None,
    )

    model = VisionEncoder(args.vision_model).to(device)

    image_id = 0
    for batch_idx, images in enumerate(tqdm.tqdm(dataloader, desc=f"[Vision GPU {rank}]")):
        batch_len = len(images)

        if batch_idx % num_gpus != rank:
            image_id += batch_len
            continue

        batch_ids = [str(i) for i in range(image_id, image_id + batch_len)]
        new_indices = [i for i, key in enumerate(batch_ids) if key not in existing_keys]
        if not new_indices:
            image_id += batch_len
            continue

        features = model(images[new_indices].to(device))

        for local_idx, global_idx in enumerate(new_indices):
            feature_utils.save_feature(
                batch_ids[global_idx],
                vision_features=features[local_idx].detach().clone(),
            )

        image_id += batch_len

    feature_utils.save()
    del model
    torch.cuda.empty_cache()
    print(f"[Vision GPU {rank}] Done.")


def extract_language(rank, num_gpus, captions, args):
    """Encode captions with LanguageEncoder on the assigned GPU."""
    device = torch.device(f"cuda:{rank}")
    output_dir = (
        f"{args.output_dir}"
        f"/{args.language_model.split('/')[-1]}/{args.caption_name}"
    )

    feature_utils = FeatureUtils(base_dir=output_dir, staging_dir=args.cache_dir, feature_num=1)

    model = LanguageEncoder(args.language_model, cache_dir=args.model_cache_dir).to(device)

    print(f"[Lang GPU {rank}] Existing features: {len(feature_utils.list_keys())}")

    items = [
        (image_id, caption)
        for image_idx, (image_id, caption) in enumerate(captions.items())
        if image_idx % num_gpus == rank and not feature_utils.exists(image_id)
    ]
    print(f"[Lang GPU {rank}] {len(items)} captions to encode")

    for batch_start in tqdm.tqdm(
        range(0, len(items), args.language_batch_size), desc=f"[Lang GPU {rank}]"
    ):
        batch = items[batch_start : batch_start + args.language_batch_size]
        batch_ids = [item[0] for item in batch]
        batch_captions = [item[1] for item in batch]

        features = model(batch_captions)
        if features.dim() == 1:
            features = features.unsqueeze(0)

        for image_id, feat in zip(batch_ids, features):
            feature_utils.save_feature(image_id, language_features=feat.unsqueeze(0))

    feature_utils.save()
    del model
    torch.cuda.empty_cache()
    print(f"[Lang GPU {rank}] Done.")


def worker(rank, args):
    """Per-GPU worker: load dataset once, extract vision then language features."""
    num_gpus = args.num_gpus

    if args.extract in ("vision", "both"):
        print(f"[GPU {rank}] Loading dataset {args.dataset_dir}...")
        dataset = load_dataset(args.dataset_dir, split="train")

        captions = {}
        if args.extract == "both":
            print(f"[GPU {rank}] Collecting captions...")
            all_captions = dataset["caption"]
            captions = {str(i): cap for i, cap in enumerate(all_captions)}

        extract_vision(rank, num_gpus, dataset, args)
        del dataset
        gc.collect()
        torch.cuda.empty_cache()

        if args.extract == "both":
            extract_language(rank, num_gpus, captions, args)

    elif args.extract == "language":
        print(f"[GPU {rank}] Streaming captions from {args.dataset_dir}...")
        ds = load_dataset(args.dataset_dir, split="train", streaming=True)
        captions = {}
        for idx, sample in enumerate(tqdm.tqdm(ds, desc=f"[GPU {rank}] Reading captions")):
            captions[str(idx)] = sample["caption"]

        extract_language(rank, num_gpus, captions, args)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Precompute vision and language features for ShareLock",
    )
    parser.add_argument("--dataset_dir", type=str, default="datasets",
                        help="Local directory for HuggingFace dataset.")
    parser.add_argument("--vision_model", type=str, default="dinov2_vitl14",
                        help="Vision model name")
    parser.add_argument("--language_model", type=str, default="meta-llama/Meta-Llama-3-8B",
                        help="Language model name")
    parser.add_argument("--output_dir", type=str, default="precomputed_features",
                        help="Root directory for saved features")
    parser.add_argument("--cache_dir", type=str, default=os.environ.get("TMPDIR", None),
                        help="FeatureUtils staging/cache directory")
    parser.add_argument("--model_cache_dir", type=str, default=os.environ.get("HF_HOME", None),
                        help="HuggingFace model cache directory")
    parser.add_argument("--extract", type=str, default="both",
                        choices=["vision", "language", "both"],
                        help="Which features to extract")
    parser.add_argument("--num_gpus", type=int, default=1,
                        help="Number of GPUs for parallel extraction")
    parser.add_argument("--gpu_id", type=int, default=None,
                        help="Specific GPU ID for single-GPU mode (default: 0)")
    parser.add_argument("--vision_batch_size", type=int, default=2048,
                        help="Batch size for vision feature extraction")
    parser.add_argument("--language_batch_size", type=int, default=2048,
                        help="Batch size for language feature extraction")
    parser.add_argument("--num_workers", type=int, default=16,
                        help="DataLoader workers for vision extraction")
    parser.add_argument("--caption_name", type=str, default="captions",
                        help="Subdirectory name for language features (matches caption_files config)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    print(f"Extract: {args.extract} | GPUs: {args.num_gpus} | Dataset: {args.dataset_dir}")
    if args.extract in ("vision", "both"):
        print(f"  Vision model: {args.vision_model} (batch_size={args.vision_batch_size})")
    if args.extract in ("language", "both"):
        print(f"  Language model: {args.language_model} (batch_size={args.language_batch_size})")

    if args.num_gpus > 1:
        mp.spawn(worker, nprocs=args.num_gpus, args=(args,))
    else:
        rank = args.gpu_id if args.gpu_id is not None else 0
        worker(rank, args)
