import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict


# ──────────────────────────────────────────────────────────────────────────
# Helper: one attention-based MIL stream
# ──────────────────────────────────────────────────────────────────────────
class _MILStream(nn.Module):
    def __init__(
        self,
        in_dim:   int,          # input feature dimension for this stream
        hid_dim:  int,          # internal hidden dimension shared by all streams
        dropout:  float,
        num_heads: int,
        num_queries: int
    ):
        super().__init__()
        self.num_queries = num_queries
        self.project = nn.Sequential(
            nn.Linear(in_dim, hid_dim),
            nn.LayerNorm(hid_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Learnable query (1 x hid_dim) that will attend to the bag
        self.query = nn.Parameter(torch.empty(num_queries, hid_dim))
        nn.init.xavier_uniform_(self.query)


        self.attn = nn.MultiheadAttention(
            embed_dim   = hid_dim,
            num_heads   = num_heads,
            dropout     = dropout,
            batch_first = True          # (B, N, D)
        )
        # self.ln_post = nn.LayerNorm(hid_dim)

    def forward(
        self,
        x:    torch.Tensor,             # (B, N, in_dim)
        mask: Optional[torch.Tensor] = None   # (B, N) with True for padding
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        emb  : (B, hid_dim) bag embedding
        alpha: (B, 1, N)    attention weights the query pays to each instance
        """
        B, N, _ = x.shape
        h = self.project(x)                 # (B, N, hid_dim)
        q = self.query.unsqueeze(0).expand(B, -1, -1)   # (B, 1, hid_dim)
        out, alpha = self.attn(q, h, h, key_padding_mask=mask)
        # out = self.ln_post(out)
        if self.num_queries == 1:
            emb = out.squeeze(1)                # (B, hid_dim)
        else:
            emb = out.mean(dim=1)
        return emb, alpha                   # alpha kept for visualisation

class LiteMIL(nn.Module):
    def __init__(
        self,
        input_dim:  int,
        hidden_dim: int,
        num_classes: int,
        dropout:     float,
        num_heads:   int,
        num_queries: int
    ):
        super().__init__()
        self.stream = _MILStream(input_dim, hidden_dim, dropout, num_heads, num_queries)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            # nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )

    def forward(
        self,
        x:    torch.Tensor,                 # (B, N, F)
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict]:
        emb, alpha = self.stream(x, mask)
        logits = self.classifier(emb)
        return logits, {'attn': alpha}
