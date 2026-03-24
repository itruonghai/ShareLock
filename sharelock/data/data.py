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
            train_raw = VisionLanguageFeatureDataset(self.config, split="train")
            val_raw = VisionLanguageFeatureDataset(self.config, split="val")

            self.train_dataset = InMemoryBatchDataset(train_raw.vision_tensor, train_raw.language_tensors, self.batch_size, shuffle=True)
            self.val_dataset = InMemoryBatchDataset(val_raw.vision_tensor, val_raw.language_tensors, self.batch_size, shuffle=False)
            
        if stage == "test" or stage is None:
            config = self.config.copy()
            config.data.caption_files = "class_names.json"
            self.test_dataset = ClassificationFeatureDataset(config)
        
    def train_dataloader(self):
        # batch_size=None: dataset yields pre-built batches; num_workers=0: no IPC overhead
        return DataLoader(self.train_dataset, batch_size=None, num_workers=0, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=None, num_workers=0, pin_memory=True)
    
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True, prefetch_factor=12)
