
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from nystrom_attention import NystromAttention

class TransLayer(nn.Module):
    def __init__(self, norm_layer=nn.LayerNorm, dim=512):
        super().__init__()
        self.norm = norm_layer(dim)
        self.attn = NystromAttention(
            dim=dim,
            dim_head=dim // 8,
            heads=8,
            num_landmarks=dim // 2,  # number of landmarks
            pinv_iterations=6,
            # number of moore-penrose iterations for approximating pinverse. 6 was recommended by the paper
            residual=True,
            # whether to do an extra residual with the value or not. supposedly faster convergence if turned on
            dropout=0.1
        )

    def forward(self, x):
        x = x + self.attn(self.norm(x))
        return x

class PPEG(nn.Module):
    def __init__(self, dim=512):
        super(PPEG, self).__init__()
        self.proj = nn.Conv2d(dim, dim, 7, 1, 7 // 2, groups=dim)
        self.proj1 = nn.Conv2d(dim, dim, 5, 1, 5 // 2, groups=dim)
        self.proj2 = nn.Conv2d(dim, dim, 3, 1, 3 // 2, groups=dim)

    def forward(self, x, H, W):
        B, _, C = x.shape
        cls_token, feat_token = x[:, 0], x[:, 1:]
        cnn_feat = feat_token.transpose(1, 2).view(B, C, H, W)
        x = self.proj(cnn_feat) + cnn_feat + self.proj1(cnn_feat) + self.proj2(cnn_feat)
        x = x.flatten(2).transpose(1, 2)
        x = torch.cat((cls_token.unsqueeze(1), x), dim=1)
        return x


class TransMIL(nn.Module):
    def __init__(self, input_size, num_classes):
        super(TransMIL, self).__init__()
        self.pos_layer = PPEG(dim=512)
        self._fc1 = nn.Sequential(nn.Linear(input_size, 512), nn.ReLU())
        self.cls_token = nn.Parameter(torch.randn(1, 1, 512))
        self.n_classes = num_classes
        self.layer1 = TransLayer(dim=512)
        self.layer2 = TransLayer(dim=512)
        self.norm = nn.LayerNorm(512)
        self._fc2 = nn.Linear(512, self.n_classes)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None):
        # x: (B, N, input_size) or (N, input_size)
        if x.dim() == 2:
            x = x.unsqueeze(0)

        h = x.float()  # (B, N, input_size)
        h = self._fc1(h)  # (B, N, 512)

        # Pad tokens to square for PPEG
        N = h.size(1)
        side = int(np.ceil(np.sqrt(N)))
        add_len = side * side - N
        if add_len > 0:
            h = torch.cat([h, h[:, :add_len, :]], dim=1)  # (B, side*side, 512)

        # Prepend CLS token on correct device/dtype
        B = h.size(0)
        cls_tokens = self.cls_token.expand(B, -1, -1).to(h.device, dtype=h.dtype)
        h = torch.cat((cls_tokens, h), dim=1)  # (B, 1 + side*side, 512)

        # Transformer + positional encoding
        h = self.layer1(h)
        h = self.pos_layer(h, side, side)
        h = self.layer2(h)

        # CLS pooling, normalize, classify
        h = self.norm(h)[:, 0]
        logits = self._fc2(h)  # (B, n_classes)
        Y_hat = torch.argmax(logits, dim=1)
        Y_prob = F.softmax(logits, dim=1)

        results_dict = {'logits': logits, 'Y_prob': Y_prob, 'Y_hat': Y_hat}
        return logits, results_dict
