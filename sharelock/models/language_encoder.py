import os
import torch
import torch.nn as nn
import pytorch_lightning as pl
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
from typing import List

class LanguageEncoder(pl.LightningModule):
    def __init__(self, model_name, cache_dir=os.environ.get("HF_HOME", None)):
        super(LanguageEncoder, self).__init__()
        
        self.model_name = model_name
        self.cache_dir = cache_dir
        self._is_clip = "clip" in model_name.lower()
        
        self.tokenizer = None
        self.language_model = None
        self._device = None
        
    def load_model(self):
        hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")

        if self._is_clip:
            from transformers import CLIPTextModel
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, cache_dir=self.cache_dir, token=hf_token)
            self.language_model = CLIPTextModel.from_pretrained(
                self.model_name, cache_dir=self.cache_dir,
                torch_dtype=torch.bfloat16, token=hf_token)
        else:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_name, cache_dir=self.cache_dir, padding_side="left", token=hf_token)
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
                self.tokenizer.pad_token = self.tokenizer.eos_token
            except Exception as e:
                raise ValueError(f"Language model {self.model_name} not found on HuggingFace model hub. Error: {e}")

            try:
                self.language_model = AutoModel.from_pretrained(
                    self.model_name, cache_dir=self.cache_dir, trust_remote_code=True,
                    torch_dtype=torch.bfloat16, token=hf_token)
            except ValueError as e:
                print(f"Error loading model with AutoModel: {e}", "Attempting to load with AutoModelForCausalLM...", sep="\n")
                self.language_model = AutoModelForCausalLM.from_pretrained(
                    self.model_name, cache_dir=self.cache_dir, trust_remote_code=True,
                    torch_dtype=torch.bfloat16, token=hf_token)
            
        for param in self.language_model.parameters():
            param.requires_grad = False
        self.language_model.eval()
        self.language_model.to(self._device)
        
    def to(self, device):
        self._device = device
        if self.language_model is not None:
            self.language_model.to(device)
        return self
            
    def unload_model(self):
        self.tokenizer = None
        self.language_model = None
        torch.cuda.empty_cache()
    
    def forward(self, texts: List[str]):
        texts = [texts] if isinstance(texts, str) else texts

        if self.language_model is None:
            self.load_model()

        with torch.no_grad():
            tokens = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(self._device)
            output = self.language_model(**tokens, return_dict=True)
            if self._is_clip:
                return output.pooler_output.float().squeeze()
            return output.last_hidden_state[:, -1].float().squeeze()


class EgoVLPv2TextEncoder(pl.LightningModule):
    """Frozen RoBERTa-base text encoder with EgoVLPv2 pretrained weights (EgoCLIP/Ego4D)."""

    FEATURE_DIM = 768

    def __init__(self, checkpoint_path: str, cache_dir=os.environ.get("HF_HOME", None)):
        super().__init__()
        self.checkpoint_path = checkpoint_path
        self.cache_dir = cache_dir
        self.model = None
        self.tokenizer = None
        self._device = None

    def load_model(self):
        from transformers import RobertaModel, RobertaTokenizer
        self.tokenizer = RobertaTokenizer.from_pretrained("roberta-base", cache_dir=self.cache_dir)
        self.model = RobertaModel.from_pretrained("roberta-base", cache_dir=self.cache_dir, add_pooling_layer=False)
        # Load EgoVLPv2 checkpoint and transplant text_model weights
        # EgoVLPv2 checkpoints pickle a parse_config.ConfigParser object.
        # Inject a dummy stub so pickle can reconstruct it without the EgoVLPv2 source tree.
        import sys, types
        if "parse_config" not in sys.modules:
            _m = types.ModuleType("parse_config")
            class _ConfigParser:
                def __init__(self, *a, **kw): pass
                def __setstate__(self, s): self.__dict__.update(s)
            _m.ConfigParser = _ConfigParser
            sys.modules["parse_config"] = _m
            _injected = True
        else:
            _injected = False
        ckpt = torch.load(self.checkpoint_path, map_location="cpu", weights_only=False)
        if _injected:
            del sys.modules["parse_config"]
        state_dict = ckpt.get("state_dict", ckpt)
        # Strip DataParallel 'module.' prefix if present
        if any(k.startswith("module.") for k in state_dict):
            state_dict = {k[len("module."):]: v for k, v in state_dict.items()}
        text_state_dict = {
            k[len("text_model."):]: v
            for k, v in state_dict.items()
            if k.startswith("text_model.")
        }
        if not text_state_dict:
            top_keys = list(state_dict.keys())[:10]
            raise ValueError(
                f"No 'text_model.*' keys found in checkpoint: {self.checkpoint_path}\n"
                f"Top-level keys (first 10): {top_keys}"
            )
        missing, unexpected = self.model.load_state_dict(text_state_dict, strict=False)
        # Buffers like position_ids may differ across transformers versions — that's OK.
        # Error only if core weight matrices are missing.
        weight_missing = [k for k in missing if not k.endswith("position_ids")]
        if weight_missing:
            raise RuntimeError(f"Missing keys when loading EgoVLPv2 text_model: {weight_missing}")
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.eval()
        self.model.to(self._device)

    def to(self, device):
        self._device = device
        if self.model is not None:
            self.model.to(device)
        return self

    def unload_model(self):
        self.tokenizer = None
        self.model = None
        torch.cuda.empty_cache()

    def forward(self, texts: List[str]):
        texts = [texts] if isinstance(texts, str) else texts
        if self.model is None:
            self.load_model()
        with torch.no_grad():
            tokens = self.tokenizer(
                texts, return_tensors="pt", padding=True, truncation=True, max_length=77
            ).to(self._device)
            output = self.model(**tokens, return_dict=True)
            # CLS token (position 0) — shape [B, 768]
            return output.last_hidden_state[:, 0].float().squeeze()


class TrainableCLIPTextEncoder(nn.Module):
    """Trainable CLIP text encoder — gradients enabled, optionally initialized from scratch."""

    def __init__(self, model_name, from_scratch=False, cache_dir=None):
        super().__init__()
        from transformers import CLIPTokenizer, CLIPTextModel, CLIPTextConfig

        hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
        self.tokenizer = CLIPTokenizer.from_pretrained(
            model_name, cache_dir=cache_dir, token=hf_token
        )
        if from_scratch:
            config = CLIPTextConfig.from_pretrained(
                model_name, cache_dir=cache_dir, token=hf_token
            )
            self.model = CLIPTextModel(config)
        else:
            self.model = CLIPTextModel.from_pretrained(
                model_name, cache_dir=cache_dir, token=hf_token
            )

    def forward(self, texts: List[str]):
        tokens = self.tokenizer(
            texts, return_tensors="pt", padding=True, truncation=True
        ).to(self.model.device)
        output = self.model(**tokens, return_dict=True)
        return output.pooler_output.float()  # [B, 768]