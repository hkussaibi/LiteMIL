import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------
# Attention blocks
# ----------------------------
class Attn_Net(nn.Module):
    """
    Ungated attention: Linear(L->D) + Tanh + [Dropout] + Linear(D->num_classes)
    Returns (A, x) where A is attention logits and x is the input passthrough.
    """
    def __init__(self, L=512, D=256, dropout=False, num_classes=1):
        super().__init__()
        layers = [nn.Linear(L, D), nn.Tanh()]
        if dropout:
            layers.append(nn.Dropout(0.25))
        layers.append(nn.Linear(D, num_classes))
        self.module = nn.Sequential(*layers)

    def forward(self, x):
        # x: (B, N, L) or (N, L)
        return self.module(x), x


class Attn_Net_Gated(nn.Module):
    """
    Gated attention (Ilse et al.): tanh(Wx) ⊙ sigmoid(Vx) -> Linear(D->num_classes)
    Returns (A, x) where x is the input passthrough.
    """
    def __init__(self, L=1024, D=256, dropout=False, num_classes=1):
        super().__init__()
        att_a = [nn.Linear(L, D), nn.Tanh()]
        att_b = [nn.Linear(L, D), nn.Sigmoid()]
        if dropout:
            att_a.append(nn.Dropout(0.25))
            att_b.append(nn.Dropout(0.25))
        self.attention_a = nn.Sequential(*att_a)
        self.attention_b = nn.Sequential(*att_b)
        self.attention_c = nn.Linear(D, num_classes)

    def forward(self, x):
        # x: (B, N, L) or (N, L)
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = self.attention_c(a.mul(b))
        return A, x


def initialize_weights(module):
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            m.bias.data.zero_()

        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
# ----------------------------
# Single-head ABMIL (batched)
# ----------------------------
class ABMIL(nn.Module):
    """
    Batched ABMIL.
    - Projects input to proj_dim
    - Computes attention over instances
    - Aggregates to bag embedding
    - Classifies

    Input:  x (B, N, F) or (N, F)
    Output: logits (B, C), results_dict
    """
    def __init__(self, gate=True, size_arg="small", dropout=False, num_classes=2):
        super().__init__()
        # (in_dim, proj_dim, attn_hidden)
        self.size_dict = {
            "small": (1024, 512, 256),
            "big":   (1024, 512, 384)
        }
        in_dim, proj_dim, attn_hidden = self.size_dict[size_arg]

        # Shared projection
        proj_layers = [nn.Linear(in_dim, proj_dim), nn.ReLU()]
        if dropout:
            proj_layers.append(nn.Dropout(0.25))
        self.feature_proj = nn.Sequential(*proj_layers)

        # Attention
        if gate:
            self.attention = Attn_Net_Gated(L=proj_dim, D=attn_hidden, dropout=dropout, num_classes=1)
        else:
            self.attention = Attn_Net(L=proj_dim, D=attn_hidden, dropout=dropout, num_classes=1)

        # Classifier on aggregated bag embedding
        self.classifier = nn.Linear(proj_dim, num_classes)

        initialize_weights(self)

    def forward(self, x, attention_only=False):
        # x: (B, N, F) or (N, F)
        if x.dim() == 2:
            x = x.unsqueeze(0)  # (1, N, F)
        x = x.float()

        # Project features
        h = self.feature_proj(x)                      # (B, N, proj_dim)

        # Attention logits and passthrough (same as h)
        A, h_proj = self.attention(h)                 # A: (B, N, 1), h_proj: (B, N, proj_dim)

        # Mask padded rows (all zeros after projection)
        pad_mask = (h_proj.abs().sum(dim=-1) == 0)    # (B, N) True where padded

        # Softmax over instances
        A = A.transpose(1, 2)                         # (B, 1, N)
        if pad_mask.any():
            A = A.masked_fill(pad_mask.unsqueeze(1), float("-inf"))
        A = F.softmax(A, dim=2)                       # (B, 1, N)

        if attention_only:
            return A                                   # (B, 1, N)

        # Aggregate to bag embedding
        M = torch.bmm(A, h_proj).squeeze(1)           # (B, proj_dim)

        # Classify
        logits = self.classifier(M)                   # (B, C)
        Y_hat = torch.argmax(logits, dim=1)
        Y_prob = F.softmax(logits, dim=1)
        return logits, {'logits': logits, 'Y_prob': Y_prob, 'Y_hat': Y_hat}


# ----------------------------
# Multi-head ABMIL (batched)
# ----------------------------
class ABMIL_Multihead(nn.Module):
    """
    Multihead ABMIL:
    - Shared projection to proj_total_dim
    - Split features across n heads (equal step per head)
    - Per-head attention and aggregation
    - Concatenate head embeddings and classify

    Input:  x (B, N, F) or (N, F)
    Output: logits (B, C), results_dict
    """
    def __init__(self, gate=True, size_arg="small", dropout=False,
                 num_classes=2, n=2, head_size="small"):
        super().__init__()
        assert n >= 1, "Number of heads must be >= 1"
        self.n_heads = n

        # (in_dim, proj_total_dim, attn_hidden_default)
        self.size_dict = {
            "tiny":  (1024, 256,  64),
            "small": (1024, 512, 256),
            "big":   (1024, 768, 384)
        }
        in_dim, proj_total_dim, _ = self.size_dict[size_arg]

        # Make proj_total_dim divisible by n_heads
        step = math.ceil(proj_total_dim / self.n_heads)
        self.proj_total_dim = step * self.n_heads
        self.step = step  # per-head feature dim

        # Per-head hidden (attention MLP) width
        if head_size == "tiny":
            per_head_hidden = max(8, self.step // 4)
        elif head_size == "small":
            per_head_hidden = max(16, self.step // 2)
        elif head_size == "big":
            per_head_hidden = self.step
        else:
            per_head_hidden = self.step

        # Shared projection: F -> proj_total_dim
        proj_layers = [nn.Linear(in_dim, self.proj_total_dim), nn.ReLU()]
        if dropout:
            proj_layers.append(nn.Dropout(0.25))
        self.net_general = nn.Sequential(*proj_layers)

        # Per-head attention modules
        att_modules = []
        for _ in range(self.n_heads):
            if gate:
                att_modules.append(Attn_Net_Gated(L=self.step, D=per_head_hidden, dropout=dropout, num_classes=1))
            else:
                att_modules.append(Attn_Net(L=self.step, D=per_head_hidden, dropout=dropout, num_classes=1))
        self.attention_net = nn.ModuleList(att_modules)

        # Classifier over concatenated head embeddings
        self.classifier = nn.Linear(self.proj_total_dim, num_classes)

        initialize_weights(self)

    def forward(self, x, attention_only=False):
        # x: (B, N, F) or (N, F)
        if x.dim() == 2:
            x = x.unsqueeze(0)
        x = x.float()
        B, N, _ = x.shape

        # Shared projection
        h_proj = self.net_general(x)                         # (B, N, proj_total_dim)

        # Split into heads: (B, N, n_heads, step)
        h_heads = h_proj.view(B, N, self.n_heads, self.step)

        # Pad mask on tokens (True where padded)
        pad_mask = (h_proj.abs().sum(dim=-1) == 0)           # (B, N)

        # Per-head attention logits (B, n_heads, N)
        A_heads = []
        for i in range(self.n_heads):
            A_i, _ = self.attention_net[i](h_heads[:, :, i, :])  # (B, N, 1)
            A_i = A_i.transpose(1, 2)                             # (B, 1, N)
            A_heads.append(A_i)
        A_all = torch.cat(A_heads, dim=1)                         # (B, n_heads, N)

        # Mask pads before softmax
        if pad_mask.any():
            A_all = A_all.masked_fill(pad_mask.unsqueeze(1), float("-inf"))

        # Normalize attention per head over instances
        A_all = F.softmax(A_all, dim=2)                           # (B, n_heads, N)

        if attention_only:
            return A_all                                          # (B, n_heads, N)

        # Aggregate per head:
        # (B, n_heads, 1, N) @ (B, n_heads, N, step) -> (B, n_heads, 1, step)
        A_all_exp = A_all.unsqueeze(2)                             # (B, n_heads, 1, N)
        h_heads_tr = h_heads.transpose(1, 2)                       # (B, n_heads, N, step)
        M_heads = torch.matmul(A_all_exp, h_heads_tr).squeeze(2)   # (B, n_heads, step)

        # Concatenate heads -> (B, proj_total_dim)
        M = M_heads.reshape(B, self.n_heads * self.step)

        # Classify
        logits = self.classifier(M)                                # (B, C)
        Y_hat = torch.argmax(logits, dim=1)
        Y_prob = F.softmax(logits, dim=1)
        return logits, {'logits': logits, 'Y_prob': Y_prob, 'Y_hat': Y_hat}