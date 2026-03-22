import torch
import torch.nn as nn


class Projection(nn.Module):
    """Original MLP projector: Linear → BatchNorm1d → ReLU → Dropout (N layers)."""
    def __init__(self, config, embedding_size, input_size):
        super().__init__()
        self.config = config.copy()
        self.normalize = self.config.normalize
        hidden_size = self.config.hidden_size
        num_layers = self.config.num_layers

        if num_layers > 0:
            layers = []
            for i in range(num_layers - 1):
                layers.append(nn.Linear(input_size, hidden_size))
                layers.append(nn.BatchNorm1d(hidden_size))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(getattr(config, 'dropout', 0.2)))
                input_size = hidden_size
            layers.append(nn.Linear(input_size, embedding_size))
            self.projection = nn.Sequential(*layers)
        else:
            self.projection = nn.Identity()

    def forward(self, x):
        projection = self.projection(x)
        if self.normalize:
            projection = projection / projection.norm(dim=1, keepdim=True)
        return projection


class MLPv2Projector(nn.Module):
    """Enhanced MLP: LayerNorm + GELU + residual connections.

    Improvements over the baseline:
    - LayerNorm: more stable than BatchNorm for embedding inputs (no batch-stat dependency)
    - GELU: smoother activation standard in transformer-era models
    - Residual connections: added when input_size == hidden_size
    """
    def __init__(self, config, embedding_size, input_size):
        super().__init__()
        self.normalize = config.normalize
        hidden_size = config.hidden_size
        num_layers = config.num_layers
        dropout = getattr(config, 'dropout', 0.1)

        layers = nn.ModuleList()
        curr = input_size
        for i in range(num_layers - 1):
            layers.append(_MLPv2Block(curr, hidden_size, dropout))
            curr = hidden_size
        self.layers = layers
        self.out = nn.Linear(curr, embedding_size)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = self.out(x)
        if self.normalize:
            x = x / x.norm(dim=1, keepdim=True)
        return x


class _MLPv2Block(nn.Module):
    def __init__(self, input_size, hidden_size, dropout):
        super().__init__()
        self.linear = nn.Linear(input_size, hidden_size)
        self.norm = nn.LayerNorm(hidden_size)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.residual = (input_size == hidden_size)

    def forward(self, x):
        out = self.drop(self.act(self.norm(self.linear(x))))
        if self.residual:
            out = out + x
        return out


class QFormerProjector(nn.Module):
    """Q-Former adapted for precomputed single-vector features.

    Inspired by BLIP-2. Learned query tokens cross-attend to the (single)
    input feature vector, allowing richer multi-head extraction. Works with
    the precomputed last-hidden-state representation (no full token sequence needed).

    Pipeline:
        Input [B, d_in] → Linear → [B, 1, hidden]   (single "context token")
        Queries: [B, num_queries, hidden]             (learned, broadcast over batch)
        For each transformer layer:
            - Self-attention among queries
            - Cross-attention: queries attend to the single context token
            - FFN
        Mean-pool queries → [B, hidden] → Linear → [B, embedding_size]
    """
    def __init__(self, config, embedding_size, input_size):
        super().__init__()
        self.normalize = config.normalize
        hidden_size = config.hidden_size
        num_queries = getattr(config, 'num_queries', 32)
        num_heads = getattr(config, 'num_heads', 8)
        num_layers = getattr(config, 'num_transformer_layers', 2)
        dropout = getattr(config, 'dropout', 0.1)

        self.input_norm = nn.BatchNorm1d(input_size)
        self.input_proj = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.GELU(),
        )
        self.queries = nn.Parameter(torch.randn(1, num_queries, hidden_size) * 0.02)
        self.layers = nn.ModuleList([
            _QFormerLayer(hidden_size, num_heads, dropout) for _ in range(num_layers)
        ])
        self.out = nn.Linear(hidden_size, embedding_size)

    def forward(self, x):
        B = x.shape[0]
        # context: single token from input features
        ctx = self.input_proj(self.input_norm(x)).unsqueeze(1)  # [B, 1, hidden]
        q = self.queries.expand(B, -1, -1)                     # [B, num_queries, hidden]
        for layer in self.layers:
            q = layer(q, ctx)
        pooled = q.mean(dim=1)                          # [B, hidden]
        out = self.out(pooled)
        if self.normalize:
            out = out / out.norm(dim=1, keepdim=True)
        return out


class _QFormerLayer(nn.Module):
    def __init__(self, hidden_size, num_heads, dropout):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(hidden_size, num_heads, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(hidden_size, num_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 4, hidden_size),
        )
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.norm3 = nn.LayerNorm(hidden_size)
        self.drop = nn.Dropout(dropout)

    def forward(self, q, ctx):
        # self-attention among queries
        q2, _ = self.self_attn(q, q, q)
        q = self.norm1(q + self.drop(q2))
        # cross-attention: queries attend to input context token
        q2, _ = self.cross_attn(q, ctx, ctx)
        q = self.norm2(q + self.drop(q2))
        # FFN
        q = self.norm3(q + self.drop(self.ffn(q)))
        return q


class NVFormerProjector(nn.Module):
    """NV-Former: single-layer attention pooling inspired by NV-Embed.

    Simpler than Q-Former — one cross-attention layer pools the input into
    a fixed set of learned query representations, then projects to embedding space.

    Pipeline:
        Input [B, d_in] → Linear → K, V: [B, 1, hidden]
        Queries Q: [B, num_queries, hidden]   (learned, broadcast over batch)
        out = softmax(Q @ K.T / sqrt(d)) @ V  (cross-attend to single input token)
        Mean-pool → [B, hidden] → Linear → [B, embedding_size]
    """
    def __init__(self, config, embedding_size, input_size):
        super().__init__()
        self.normalize = config.normalize
        hidden_size = config.hidden_size
        num_queries = getattr(config, 'num_queries', 32)
        num_heads = getattr(config, 'num_heads', 8)

        self.input_proj = nn.Linear(input_size, hidden_size)
        self.queries = nn.Parameter(torch.randn(1, num_queries, hidden_size) * 0.02)
        self.attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(hidden_size)
        self.out = nn.Linear(hidden_size, embedding_size)

    def forward(self, x):
        B = x.shape[0]
        kv = self.input_proj(x).unsqueeze(1)            # [B, 1, hidden]
        q = self.queries.expand(B, -1, -1)              # [B, num_queries, hidden]
        attn_out, _ = self.attn(q, kv, kv)
        pooled = self.norm(attn_out).mean(dim=1)        # [B, hidden]
        out = self.out(pooled)
        if self.normalize:
            out = out / out.norm(dim=1, keepdim=True)
        return out


def build_projector(config, embedding_size, input_size) -> nn.Module:
    """Factory: instantiate the right projector based on config.type.

    Supported types:
        'mlp'       — original Linear→BatchNorm→ReLU→Dropout (default)
        'mlp_v2'    — LayerNorm + GELU + residual connections
        'qformer'   — cross-attention transformer with learned queries (BLIP-2 style)
        'nv_former' — single cross-attention pooling (NV-Embed style)
    """
    ptype = getattr(config, 'type', 'mlp')
    if ptype == 'mlp_v2':
        return MLPv2Projector(config, embedding_size, input_size)
    if ptype == 'qformer':
        return QFormerProjector(config, embedding_size, input_size)
    if ptype == 'nv_former':
        return NVFormerProjector(config, embedding_size, input_size)
    return Projection(config, embedding_size, input_size)
