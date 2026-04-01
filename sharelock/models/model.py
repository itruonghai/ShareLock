import torch
import torch.nn as nn
import pytorch_lightning as pl
from PIL import Image

from transformers import get_cosine_schedule_with_warmup

from sharelock.models.language_encoder import LanguageEncoder, EgoVLPv2TextEncoder, TrainableCLIPTextEncoder
from sharelock.models.vision_encoder import VisionEncoder
from sharelock.models.projection import build_projector
from sharelock.utils.misc import loss, feature_dimensions_vision, feature_dimensions_language, get_transforms

class ShareLock(pl.LightningModule):
    def __init__(self, config):
        super(ShareLock, self).__init__()
        self.config = config.copy()
        
        self.save_hyperparameters(config)
        
        
        self.loss = loss
        self.logit_scale = nn.Parameter(torch.tensor(2.6592))
        
        # Setup of projection layers
        try:
            self.image_feature_size = feature_dimensions_vision[config.model.vision_encoder]
        except KeyError:
            raise ValueError(f"Feature dimensions for vision encoder {config.model.vision_encoder} not found. Please add it to the ‘feature_dimensions_vision‘ dictionary (sharelock/utils/misc.py).")
        try:
            self.language_feature_size = feature_dimensions_language[config.model.language_encoder]
        except KeyError:
            raise ValueError(f"Feature dimensions for language encoder {config.model.language_encoder} not found. Please add it to the ‘feature_dimensions_language‘ dictionary (sharelock/utils/misc.py).")
        if self.config.model.vision_projector.num_layers < 1:
            self.embedding_space_dim = self.image_feature_size
        elif self.config.model.language_projector.num_layers < 1:
            self.embedding_space_dim = self.language_feature_size
        else:
            self.embedding_space_dim = self.config.model.embedding_space_dim
        
        self.vision_projector = build_projector(config.model.vision_projector, self.embedding_space_dim, input_size=self.image_feature_size)
        self.vision_encoder = None

        self.language_projector = build_projector(config.model.language_projector, self.embedding_space_dim, input_size=self.language_feature_size)
        self.language_encoder = None
        
        self.class_prototypes = None
        
        
    def forward(self, batch: dict):
        image_features_projected = self.vision_projector(batch["vision_features"])
        langauge_features_projected = self.language_projector(batch["language_features"].squeeze())
        return image_features_projected, langauge_features_projected
    
    def training_step(self, batch: dict, batch_idx: int):
        self.logit_scale.data.clamp_(0, 4.6052)
        image_features_projected, langauge_features_projected = self(batch)
        loss = self.loss(image_features_projected, langauge_features_projected, self.logit_scale.exp())

        self.log('train_loss', loss, prog_bar=True)
        self.log('logit_scale', self.logit_scale.exp(), prog_bar=False)
        return loss

    def validation_step(self, batch: dict, batch_idx: int):
        image_features_projected, langauge_features_projected = self(batch)
        loss = self.loss(image_features_projected, langauge_features_projected, self.logit_scale.exp())

        self.log(f"validation_loss", loss, prog_bar=True)
        return loss
    
        
    def test_step(self, batch: dict, batch_idx: int):
        if batch_idx == 0:
            self.setup_class_prototypes()
        
        image_features_projected = self.vision_projector(batch["vision_features"])
        predictions = self.predict(image_features_projected)
        accuracy = (predictions == batch["label"]).float().mean()
        
        self.log(f"test_accuracy", accuracy)
        return accuracy
        
    def predict(self, image_features_projected: torch.Tensor):
        logits = image_features_projected @ self.class_prototypes
        return torch.argmax(logits, dim=1)
    
    def setup_class_prototypes(self):
        dataset = self.trainer.datamodule.test_dataloader().dataset
        class_features = dataset.get_class_features().to(self.device)
        self.class_prototypes = self.language_projector(class_features).T
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config.training.learning_rate, weight_decay=self.config.training.weight_decay)
        
        if self.config.training.lr_schedule == "constant":
            return optimizer
        lr_scheduler = {
                            "scheduler": get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=self.config.training.lr_warmup_steps, num_training_steps=self.config.training.max_steps),
                            "interval": "step",
                            "frequency": 1,
                        }
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}
    
    
    def encode_image(self, image: [torch.Tensor, Image.Image]):
        """Inference function to encode and project an image (preprocessed tensor or Image)"""
        
        if isinstance(image, Image.Image):
            image = get_transforms(self.config.model.vision_encoder)(image).unsqueeze(0)
            
        if self.vision_encoder is None:
            self.vision_encoder = VisionEncoder(self.config.model.vision_encoder).to(self.device)
            
        return self.vision_projector(self.vision_encoder(image.to(self.device)))
    
    def encode_text(self, text: str):
        """Inference function to encode and project a text (unprocessed raw-text)"""

        if self.language_encoder is None:
            if self.config.model.language_encoder == "egovlpv2":
                checkpoint = self.config.model.get("egovlpv2_checkpoint")
                self.language_encoder = EgoVLPv2TextEncoder(checkpoint).to(self.device)
            else:
                self.language_encoder = LanguageEncoder(self.config.model.language_encoder).to(self.device)

        return self.language_projector(self.language_encoder(text))


class ShareLockWithTextEncoder(pl.LightningModule):
    """ShareLock variant with a trainable CLIP text encoder.

    Vision features are still precomputed (frozen DINOv2). The CLIP text encoder
    is trained end-to-end, optionally initialized from scratch (random weights).
    """

    def __init__(self, config):
        super().__init__()
        self.config = config.copy()
        self.save_hyperparameters(config)

        self.loss = loss
        self.logit_scale = nn.Parameter(torch.tensor(2.6592))

        # Vision projector (identity + L2-norm over precomputed DINOv2 features)
        self.image_feature_size = feature_dimensions_vision[config.model.vision_encoder]
        if config.model.vision_projector.num_layers < 1:
            self.embedding_space_dim = self.image_feature_size
        else:
            self.embedding_space_dim = config.model.embedding_space_dim

        self.vision_projector = build_projector(
            config.model.vision_projector, self.embedding_space_dim, input_size=self.image_feature_size
        )

        # Trainable CLIP text encoder
        from_scratch = config.model.get("language_from_scratch", False)
        cache_dir = None  # relies on HF_HOME env var
        self.language_encoder = TrainableCLIPTextEncoder(
            config.model.language_encoder,
            from_scratch=from_scratch,
            cache_dir=cache_dir,
        )

        # Language projector (CLIP 768-dim → embedding space)
        self.language_feature_size = feature_dimensions_language[config.model.language_encoder]
        self.language_projector = build_projector(
            config.model.language_projector, self.embedding_space_dim, input_size=self.language_feature_size
        )

    def forward(self, batch: dict):
        vision_proj = self.vision_projector(batch["vision_features"])
        lang_features = self.language_encoder(batch["caption"])
        lang_proj = self.language_projector(lang_features)
        return vision_proj, lang_proj

    def training_step(self, batch: dict, batch_idx: int):
        self.logit_scale.data.clamp_(0, 4.6052)
        vision_proj, lang_proj = self(batch)
        l = self.loss(vision_proj, lang_proj, self.logit_scale.exp())
        self.log("train_loss", l, prog_bar=True, sync_dist=True)
        self.log("logit_scale", self.logit_scale.exp(), prog_bar=False)
        return l

    def validation_step(self, batch: dict, batch_idx: int):
        vision_proj, lang_proj = self(batch)
        l = self.loss(vision_proj, lang_proj, self.logit_scale.exp())
        self.log("validation_loss", l, prog_bar=True, sync_dist=True)
        return l

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.config.training.learning_rate,
            weight_decay=self.config.training.weight_decay,
        )
        if self.config.training.lr_schedule == "constant":
            return optimizer
        lr_scheduler = {
            "scheduler": get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=self.config.training.lr_warmup_steps,
                num_training_steps=self.config.training.max_steps,
            ),
            "interval": "step",
            "frequency": 1,
        }
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}

    def encode_image(self, image):
        """Inference: encode and project an image (PIL Image or preprocessed tensor)."""
        if isinstance(image, __import__("PIL").Image.Image):
            image = get_transforms(self.config.model.vision_encoder)(image).unsqueeze(0)
        if not hasattr(self, "_vision_encoder") or self._vision_encoder is None:
            self._vision_encoder = VisionEncoder(self.config.model.vision_encoder).to(self.device)
        return self.vision_projector(self._vision_encoder(image.to(self.device)))

    def encode_text(self, text: str):
        """Inference: encode and project text."""
        return self.language_projector(self.language_encoder([text] if isinstance(text, str) else text))
