import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from sharelock.data.datasets import VisionLanguageFeatureDataset, InMemoryBatchDataset, ClassificationFeatureDataset

class DataModule(pl.LightningDataModule):
    def __init__(self, config):
        super(DataModule, self).__init__()
        
        self.config = config.copy()
        
        self.num_workers = self.config.data.num_workers
        self.batch_size = self.config.training.batch_size
                
    def setup(self, stage=None):        
        if stage == "fit" or stage is None:
            if isinstance(self.config.data.dataset, str):
                # Perform training and validation on single dataset
                train_raw = VisionLanguageFeatureDataset(self.config, split="train")
                val_raw = VisionLanguageFeatureDataset(self.config, split="val")
            else:
                # Perform training and validation on multiple datasets — concat tensors directly
                train_vision = []
                train_language = []
                val_vision = []
                val_language = []
                for dataset in self.config.data.dataset:
                    config = self.config.copy()
                    config.data.dataset = dataset
                    t = VisionLanguageFeatureDataset(config, split="train")
                    v = VisionLanguageFeatureDataset(config, split="val")
                    train_vision.append(t.vision_tensor)
                    train_language.append(t.language_tensors)
                    val_vision.append(v.vision_tensor)
                    val_language.append(v.language_tensors)
                # Merge into single VisionLanguageFeatureDataset-like objects
                train_raw = VisionLanguageFeatureDataset.__new__(VisionLanguageFeatureDataset)
                train_raw.vision_tensor = torch.cat(train_vision)
                train_raw.language_tensors = [torch.cat([lt[i] for lt in train_language]) for i in range(len(train_language[0]))]
                val_raw = VisionLanguageFeatureDataset.__new__(VisionLanguageFeatureDataset)
                val_raw.vision_tensor = torch.cat(val_vision)
                val_raw.language_tensors = [torch.cat([lt[i] for lt in val_language]) for i in range(len(val_language[0]))]

            self.train_dataset = InMemoryBatchDataset(train_raw.vision_tensor, train_raw.language_tensors, self.batch_size, shuffle=True)
            self.val_dataset = InMemoryBatchDataset(val_raw.vision_tensor, val_raw.language_tensors, self.batch_size, shuffle=False)
                    
            
        if stage == "test" or stage is None:
            # Perform test on ImageNet1k
            config = self.config.copy()
            config.data.dataset = "imagenet-1k"
            config.data.caption_files = "class_names.json"
            self.test_dataset = ClassificationFeatureDataset(config)
        
    def train_dataloader(self):
        # batch_size=None: dataset yields pre-built batches; num_workers=0: no IPC overhead
        return DataLoader(self.train_dataset, batch_size=None, num_workers=0, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=None, num_workers=0, pin_memory=True)
    
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True, prefetch_factor=12)
