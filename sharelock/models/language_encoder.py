import os
import torch
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