"""
Extract vision features from the CC3M-LLaVA 595k HuggingFace dataset.

Adapted from precompute_image_features_hf.py — no label/class_names logic.
Sequential integer IDs are used as FeatureUtils keys, matching prepare_cc3m_captions.py.

Multi-GPU: run two processes with --gpu_num 2 --gpu_id 0 and --gpu_num 2 --gpu_id 1.
"""

import os
import torch
import tqdm
import argparse

from datasets import load_dataset
from torch.utils.data import DataLoader

from featureutils.core import FeatureUtils
from sharelock.models.vision_encoder import VisionEncoder
from sharelock.utils.misc import get_transforms


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Precompute CC3M-LLaVA vision features")
    parser.add_argument("--hf_dataset", type=str, default="pingzhili/llava-filtered-cc3m-595k",
                        help="HuggingFace dataset identifier")
    parser.add_argument("--vision_model", type=str, required=True, help="Vision model to use")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save features")
    parser.add_argument("--cache_dir", type=str, default=os.environ.get("TMPDIR", None),
                        help="FeatureUtils staging/cache directory")
    parser.add_argument("--gpu_num", type=int, default=1, help="Total number of GPUs being used")
    parser.add_argument("--gpu_id", type=int, default=0, help="ID of this GPU")
    parser.add_argument("--batch_size", type=int, default=1024, help="Inference batch size")
    parser.add_argument("--num_workers", type=int, default=16, help="DataLoader workers")
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")

    print(f"Extracting vision features for {args.hf_dataset} with {args.vision_model}")
    print(f"GPU {args.gpu_id}/{args.gpu_num}, device: {device}, batch_size: {args.batch_size}")

    # Feature storage
    dataset_name = args.hf_dataset.split("/")[-1]
    output_dir = f"{args.output_dir}/{dataset_name}/{args.vision_model.split('/')[-1]}"
    feature_utils = FeatureUtils(base_dir=output_dir, staging_dir=args.cache_dir, feature_num=1)
    existing_keys = set(feature_utils.list_keys())
    print(f"Existing features: {len(existing_keys)}")

    # Load dataset and apply transforms
    transforms = get_transforms()

    def transform_fn(batch):
        batch["image"] = [transforms(img.convert("RGB")) for img in batch["image"]]
        return batch

    print("Loading dataset...")
    dataset = load_dataset(args.hf_dataset, split="train")
    dataset = dataset.with_transform(transform_fn)

    def collate_fn(samples):
        images = torch.stack([s["image"] for s in samples])
        return images

    dataloader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                            collate_fn=collate_fn, pin_memory=True)

    # Load vision model
    model = VisionEncoder(args.vision_model).to(device)

    image_id = 0
    for batch_idx, images in enumerate(tqdm.tqdm(dataloader, desc=f"GPU {args.gpu_id}")):
        batch_len = len(images)

        if batch_idx % args.gpu_num != args.gpu_id:
            image_id += batch_len
            continue

        # Skip if all keys in this batch already exist
        batch_ids = [str(i) for i in range(image_id, image_id + batch_len)]
        new_indices = [i for i, key in enumerate(batch_ids) if key not in existing_keys]
        if not new_indices:
            image_id += batch_len
            continue

        new_images = images[new_indices].to(device)
        features = model(new_images)

        for local_idx, global_idx in enumerate(new_indices):
            feature_utils.save_feature(
                batch_ids[global_idx],
                vision_features=features[local_idx].detach().clone()
            )

        image_id += batch_len

    feature_utils.save()
    print(f"GPU {args.gpu_id} done.")
