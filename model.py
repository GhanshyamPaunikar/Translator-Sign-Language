"""Transformer model for word-level ASL recognition from MediaPipe landmarks."""
from __future__ import annotations

import torch
import torch.nn as nn


class AttentionPool(nn.Module):
    """Aggregate a sequence via a single learnable query token.

    Better than mean-pool because the query learns to weight informative frames
    (typically movement peaks and handshape transitions in signs).
    """

    def __init__(self, dim: int, heads: int = 4):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.size(0)
        q = self.query.expand(B, -1, -1)
        out, _ = self.attn(q, x, x)
        return self.norm(out.squeeze(1))


class SignTransformer(nn.Module):
    """Transformer encoder for classifying sign-language landmark sequences.

    Args:
        input_dim: Landmark vector dim per frame (1662 for MediaPipe Holistic:
                   pose 33×4 + face 468×3 + LH 21×3 + RH 21×3).
        d_model: Transformer hidden dim.
        nhead: Number of attention heads.
        num_layers: Stacked transformer encoder blocks.
        dim_ff: FFN hidden dim inside each block.
        seq_len: Max frames per clip (learned positional embeddings sized for this).
        num_classes: Vocabulary size.
        dropout: Applied in input projection, inside encoder, and in classifier head.
    """

    def __init__(
        self,
        input_dim: int = 1662,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 4,
        dim_ff: int = 512,
        seq_len: int = 30,
        num_classes: int = 100,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.pos_embed = nn.Parameter(torch.randn(1, seq_len, d_model) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # pre-norm is more stable
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pool = AttentionPool(d_model, heads=4)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, input_dim) → logits (B, num_classes)."""
        x = self.input_proj(x) + self.pos_embed[:, : x.size(1)]
        x = self.encoder(x)
        x = self.pool(x)
        return self.classifier(x)

    @classmethod
    def from_checkpoint(cls, path: str, map_location: str | torch.device = "cpu"):
        """Load model + config from a .pt file saved by the training notebook."""
        ckpt = torch.load(path, map_location=map_location, weights_only=False)
        model = cls(**ckpt["config"])
        model.load_state_dict(ckpt["model_state_dict"])
        return model, ckpt.get("idx_to_gloss", {})
