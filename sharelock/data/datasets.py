import os
import json
import torch
import random
import tqdm
from concurrent.futures import ThreadPoolExecutor
from torch.utils.data import Dataset, IterableDataset

from featureutils.core import FeatureUtils


def _load_features_parallel(feature_utils, ids, feature_names, num_workers=16):
    """Load all features for given ids into a stacked tensor using parallel file reads."""
    def load_one(image_id):
        return feature_utils.load_feature(image_id, feature_names)

    with ThreadPoolExecutor(max_workers=num_workers) as ex:
        results = list(tqdm.tqdm(ex.map(load_one, ids), total=len(ids), leave=False))

    return {name: torch.stack([r[name].squeeze() for r in results]) for name in feature_names}


def _tensor_cache_path(config, split):
    """Return a stable path for the pre-built tensor cache for a given split."""
    caption_files = config.data.caption_files
    if isinstance(caption_files, str):
        caption_files = [caption_files]
    caption_tag = "+".join(f.replace(".json", "") for f in sorted(caption_files))
    vis = config.model.vision_encoder.split("/")[-1]
    lang = config.model.language_encoder.split("/")[-1]
    filename = (
        f"{vis}__{lang}__{caption_tag}"
        f"__{split}_seed{config.seed}_val{config.data.val_split_num}.pt"
    ).replace("/", "-")
    cache_dir = os.path.join(config.data.precomputed_features_dir, "_tensor_cache")
    os.makedirs(cache_dir, exist_ok=True)
    return os.path.join(cache_dir, filename)


class VisionLanguageFeatureDataset(Dataset):
    def __init__(self, config, split):
        self.config = config.copy()
        self.split = split

        # Normalise caption_files to a list early so cache key is consistent
        self.config.data.caption_files = (
            [self.config.data.caption_files]
            if isinstance(self.config.data.caption_files, str)
            else list(self.config.data.caption_files)
        )

        cache_path = _tensor_cache_path(self.config, split)
        if os.path.exists(cache_path):
            print(f"Loading {split} tensors from cache ({cache_path})...", flush=True)
            cached = torch.load(cache_path, weights_only=True)
            self.vision_tensor = cached["vision"]
            self.language_tensors = cached["language"]
            return

        data_staging_dir = os.environ.get("TMPDIR", None)
        rng = random.Random(self.config.seed)

        # Loading precomputed image features
        feature_dir = f"{self.config.data.precomputed_features_dir}/{self.config.model.vision_encoder.split('/')[-1]}"
        image_features = FeatureUtils(base_dir=feature_dir, staging_dir=data_staging_dir, require_features_exist=True)
        if len(image_features.list_keys()) == 0:
            raise ValueError(f"No vision features found for vision encoder {self.config.model.vision_encoder} in {feature_dir}")
        image_features.stage_data(features=["vision_features"])

        # Setup image ids and create splits
        feature_ids = image_features.list_keys()
        rng.shuffle(feature_ids)
        if split == "train":
            feature_ids = feature_ids[self.config.data.val_split_num:]
        elif split == "val":
            feature_ids = feature_ids[:self.config.data.val_split_num]

        # Pre-load all vision features into RAM
        print(f"Pre-loading {split} vision features ({len(feature_ids)} samples) into RAM...", flush=True)
        loaded = _load_features_parallel(image_features, feature_ids, ["vision_features"])
        self.vision_tensor = loaded["vision_features"]  # [N, vision_dim]
        del image_features

        # Loading precomputed language features for each caption file (randomly select caption at each iteration)
        self.language_tensors = []
        for caption_file in self.config.data.caption_files:
            feature_dir = f"{self.config.data.precomputed_features_dir}/{self.config.model.language_encoder.split('/')[-1]}/{caption_file.replace('.json', '')}"
            language_features = FeatureUtils(base_dir=feature_dir, staging_dir=data_staging_dir, require_features_exist=True)
            if len(language_features.list_keys()) == 0:
                raise ValueError(f"No language features found for {self.config.model.language_encoder} / {caption_file} in {feature_dir}")
            language_features.stage_data()
            print(f"Pre-loading {split} language features [{caption_file}] into RAM...", flush=True)
            loaded = _load_features_parallel(language_features, feature_ids, ["language_features"])
            self.language_tensors.append(loaded["language_features"])  # [N, language_dim]
            del language_features

        print(f"Saving {split} tensor cache to {cache_path}...", flush=True)
        torch.save({"vision": self.vision_tensor, "language": self.language_tensors}, cache_path)

    def __len__(self):
        return len(self.vision_tensor)

    def __getitem__(self, idx):
        return {
            "vision_features": self.vision_tensor[idx],
            "language_features": random.choice(self.language_tensors)[idx],
        }

    def to_batch_dataset(self, batch_size, shuffle):
        """Wrap tensors in an InMemoryBatchDataset for zero-IPC, single-gather batching."""
        return InMemoryBatchDataset(self.vision_tensor, self.language_tensors, batch_size, shuffle)


class InMemoryBatchDataset(IterableDataset):
    """Yields pre-built batches via a single tensor gather — no workers, no IPC.

    With data already in RAM, DataLoader workers only add inter-process
    communication overhead (~640 MB/batch via shared memory queues for
    batch_size=32768). This class bypasses that entirely: each batch is
    built with one tensor index operation and yielded directly.
    """
    def __init__(self, vision_tensor, language_tensors, batch_size, shuffle):
        self.vision_tensor = vision_tensor
        self.language_tensors = language_tensors
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.n = len(vision_tensor)

    def __iter__(self):
        perm = torch.randperm(self.n) if self.shuffle else torch.arange(self.n)
        lang = random.choice(self.language_tensors)
        for i in range(0, self.n - self.batch_size + 1, self.batch_size):
            idx = perm[i:i + self.batch_size]
            yield {
                "vision_features": self.vision_tensor[idx],
                "language_features": lang[idx],
            }

    def __len__(self):
        return self.n // self.batch_size
    
class VisionCaptionDataset(Dataset):
    """Loads precomputed vision features + raw caption strings for online text-encoder training.

    Returns {"vision_features": Tensor[vision_dim], "caption": str}.
    Uses a standard Dataset so PL's DistributedSampler shards data correctly in DDP.
    """

    def __init__(self, config, split):
        self.config = config.copy()
        self.split = split

        cache_path = self._cache_path(split)
        if os.path.exists(cache_path):
            print(f"Loading {split} tensors from cache ({cache_path})...", flush=True)
            cached = torch.load(cache_path, weights_only=False)
            self.vision_tensor = cached["vision"]
            self.captions = cached["captions"]
            return

        data_staging_dir = os.environ.get("TMPDIR", None)
        rng = random.Random(self.config.seed)

        # Load precomputed vision features
        feature_dir = f"{self.config.data.precomputed_features_dir}/{self.config.model.vision_encoder.split('/')[-1]}"
        image_features = FeatureUtils(base_dir=feature_dir, staging_dir=data_staging_dir, require_features_exist=True)
        if len(image_features.list_keys()) == 0:
            raise ValueError(f"No vision features found for {self.config.model.vision_encoder} in {feature_dir}")
        image_features.stage_data(features=["vision_features"])

        # Create train/val split
        feature_ids = image_features.list_keys()
        rng.shuffle(feature_ids)
        if split == "train":
            feature_ids = feature_ids[self.config.data.val_split_num:]
        elif split == "val":
            feature_ids = feature_ids[:self.config.data.val_split_num]

        # Pre-load vision features into RAM
        print(f"Pre-loading {split} vision features ({len(feature_ids)} samples)...", flush=True)
        loaded = _load_features_parallel(image_features, feature_ids, ["vision_features"])
        self.vision_tensor = loaded["vision_features"]  # [N, vision_dim]
        del image_features

        # Load raw captions from JSON {image_id: caption}
        captions_file = self.config.data.captions_file
        with open(captions_file) as f:
            all_captions = json.load(f)

        self.captions = [all_captions[fid] for fid in feature_ids]

        print(f"Saving {split} tensor cache to {cache_path}...", flush=True)
        torch.save({"vision": self.vision_tensor, "captions": self.captions}, cache_path)

    def _cache_path(self, split):
        vis = self.config.model.vision_encoder.split("/")[-1]
        filename = (
            f"{vis}__captions"
            f"__{split}_seed{self.config.seed}_val{self.config.data.val_split_num}.pt"
        ).replace("/", "-")
        cache_dir = os.path.join(self.config.data.precomputed_features_dir, "_tensor_cache")
        os.makedirs(cache_dir, exist_ok=True)
        return os.path.join(cache_dir, filename)

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        return {
            "vision_features": self.vision_tensor[idx],
            "caption": self.captions[idx],
        }


class ClassificationFeatureDataset(Dataset):
    def __init__(self, config):
        self.config = config.copy()
        
        data_staging_dir = os.environ.get("TMPDIR", None)
        
        feature_dir = f"{self.config.data.precomputed_features_dir}/{self.config.model.vision_encoder.split('/')[-1]}"
        self.image_features = FeatureUtils(base_dir=feature_dir, staging_dir=data_staging_dir, require_features_exist=True)
        if len(self.image_features.list_keys()) == 0:
            raise ValueError(f"No vision features found for {self.config.model.vision_encoder} in {feature_dir}")
        self.image_features.stage_data()
        self.feature_idxs = self.image_features.list_keys()
        
        caption_files = config.data.caption_files
        if isinstance(caption_files, list):
            caption_files = caption_files[0]
        class_names_feature_dir = f"{self.config.data.precomputed_features_dir}/{self.config.model.language_encoder.split('/')[-1]}/{caption_files.replace('.json', '')}"
        self.class_names_features = FeatureUtils(base_dir=class_names_feature_dir, staging_dir=data_staging_dir, require_features_exist=True)
        if len(self.class_names_features.list_keys()) == 0:
            raise ValueError(f"No language features found for {self.config.model.language_encoder} in {class_names_feature_dir}")
        self.class_names_features.stage_data()
        
    def __len__(self):
        return len(self.image_features.list_keys())
    
    def __getitem__(self, idx):
        image_id = self.feature_idxs[idx]
        features = self.image_features.load_feature(image_id, ["vision_features", "label"])
        return features
    
    def get_class_features(self):
        class_features = []
        for class_id in self.class_names_features.list_keys():
            features = self.class_names_features.load_feature(class_id, ["language_features"])
            class_features.append(features["language_features"].squeeze())
        return torch.stack(class_features)