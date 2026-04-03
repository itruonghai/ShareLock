"""
Frozen V-JEPA-2 / V-JEPA-2.1 video encoder for ShareLock.

Loads via torch.hub from facebookresearch/vjepa2. All parameters are frozen
(requires_grad=False) and the model stays in eval() permanently.

Input:  [B, T, C, H, W]  float32 — T=16 frames, H=W=384 (2.1) or 224 (2.0)
Output: [B, embed_dim]   float32 — mean-pooled over all spatial+temporal tokens

Variants:
  V-JEPA-2.1 (384px, preferred):
    vjepa2.1_vitb_384   → 768-dim
    vjepa2.1_vitl_384   → 1024-dim  (default)
    vjepa2.1_vitg_384   → 1408-dim
    vjepa2.1_vitig_384  → 1536-dim

  V-JEPA-2 (224px, fallback):
    vjepa2_vitl         → 1024-dim
    vjepa2_vith         → 1280-dim
    vjepa2_vitg         → 1408-dim
"""

import os
from typing import Optional

import torch
import torch.nn as nn


class VideoEncoder(nn.Module):
    """Frozen V-JEPA-2 / V-JEPA-2.1 encoder with spatial-temporal mean pooling."""

    # Maps ShareLock variant name → (torch.hub model name, embed_dim, frame_size)
    VARIANTS: dict = {
        # V-JEPA-2.1 @ 384px
        "vjepa2.1_vitb_384":  ("vjepa2_1_vit_base_384",     768, 384),
        "vjepa2.1_vitl_384":  ("vjepa2_1_vit_large_384",   1024, 384),
        "vjepa2.1_vitg_384":  ("vjepa2_1_vit_giant_384",   1408, 384),
        "vjepa2.1_vitig_384": ("vjepa2_1_vit_gigantic_384",1664, 384),
        # V-JEPA-2 @ 224px
        "vjepa2_vitl":        ("vjepa2_vit_large",          1024, 224),
        "vjepa2_vith":        ("vjepa2_vit_huge",           1280, 224),
        "vjepa2_vitg":        ("vjepa2_vit_giant",          1408, 224),
    }

    def __init__(
        self,
        variant: str = "vjepa2.1_vitl_384",
        cache_dir: Optional[str] = None,
    ):
        super().__init__()
        if variant not in self.VARIANTS:
            raise ValueError(
                f"Unknown variant '{variant}'. "
                f"Choose from: {list(self.VARIANTS.keys())}"
            )
        hub_name, self.embed_dim, self.frame_size = self.VARIANTS[variant]
        self.variant = variant

        if cache_dir is not None:
            os.environ.setdefault("TORCH_HOME", cache_dir)

        print(f"[VideoEncoder] Loading {variant} ({hub_name}) from torch.hub…")

        # Fix localhost:8300 URL bug and SSL cert issues common in HPC environments
        self._patch_vjepa_base_url()
        self._patch_ssl()

        # torch.hub returns (encoder, predictor) — we only need the encoder
        encoder, _predictor = torch.hub.load(
            "facebookresearch/vjepa2",
            hub_name,
            pretrained=True,
            trust_repo=True,
        )
        self.model = encoder

        # Freeze all parameters
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.eval()

        print(f"[VideoEncoder] Loaded. embed_dim={self.embed_dim}, frame_size={self.frame_size}")

    @staticmethod
    def _patch_vjepa_base_url() -> None:
        """Fix localhost:8300 bug in the cached torch.hub backbones.py.

        torch.hub adds the repo root to sys.path and imports hubconf, which in
        turn imports src.hub.backbones. Pre-importing that module and patching
        VJEPA_BASE_URL before torch.hub.load runs means Python's module cache
        serves the patched version.
        """
        import sys, glob, importlib.util
        correct_url = "https://dl.fbaipublicfiles.com/vjepa2"

        # 1. Patch any already-imported backbones module
        # Use vars() to avoid triggering __getattr__ on transformers modules
        # (which would produce hundreds of deprecation warnings for VJEPA_BASE_URL aliases)
        for mod in list(sys.modules.values()):
            if "VJEPA_BASE_URL" in vars(mod):
                mod.VJEPA_BASE_URL = correct_url

        # 2. Pre-import from the cached hub repo so it's in sys.modules before
        #    torch.hub.load tries to import it
        hub_dir = torch.hub.get_dir()
        for repo_dir in glob.glob(f"{hub_dir}/facebookresearch_vjepa2_*"):
            if repo_dir not in sys.path:
                sys.path.insert(0, repo_dir)
            try:
                import src.hub.backbones as _bb  # noqa: F401
                _bb.VJEPA_BASE_URL = correct_url
            except Exception:
                pass

    @staticmethod
    def _patch_ssl() -> None:
        """Disable SSL cert verification for HPC environments with missing CA bundles."""
        import ssl
        ssl._create_default_https_context = ssl._create_unverified_context

    def train(self, mode: bool = True):
        # Always stay in eval — this encoder must never be un-frozen
        return super().train(False)

    @torch.no_grad()
    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Args:
            frames: [B, T, C, H, W] — normalized video frames (ImageNet stats)
        Returns:
            [B, embed_dim] — mean-pooled spatial+temporal patch tokens, float32
        """
        # V-JEPA-2 encoder expects [B, C, T, H, W]
        x = frames.permute(0, 2, 1, 3, 4)   # [B, T, C, H, W] → [B, C, T, H, W]
        # Use bf16 autocast for ~2–4× speedup on NVIDIA tensor cores (Volta+).
        # bf16 has the same exponent range as fp32 so ViT activations stay stable.
        # Output is cast back to float32 so downstream projectors are unaffected.
        with torch.autocast(x.device.type, dtype=torch.bfloat16):
            # Model outputs all patch tokens: [B, num_tokens, dim]
            tokens = self.model(x)
        # Mean-pool all tokens → [B, dim]; cast to float32 for downstream projector
        return tokens.mean(dim=1).float()
