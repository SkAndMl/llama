from ._base import ModelConfig, MHABase, BlockBase, GPTBase, FFN, RMSNorm
from torch import nn, Tensor
from torch.nn import functional as F

import torch
import math


def get_alibi_slopes(cfg: ModelConfig) -> Tensor:
    start = 2 ** (-8 / cfg.n_heads)
    ratio = start
    return torch.tensor([start * (ratio ** i) for i in range(cfg.n_heads)])


def get_linear_bias(cfg: ModelConfig) -> Tensor:
    slopes = get_alibi_slopes(cfg).view(cfg.n_heads, 1, 1)
    pos = torch.arange(cfg.ctx_size)
    distances = pos[None, :] - pos[:, None]
    distances = torch.where(distances > 0, 0, distances)  # ctx_size, ctx_size
    distances.unsqueeze_(0)  # 1, ctx_size, ctx_size
    linear_bias = distances * slopes  # n_heads, ctx_size, ctx_size
    return linear_bias.unsqueeze(0)  # 1, n_heads, ctx_size, ctx_size


class MHA(MHABase):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__(cfg)
        linear_bias = get_linear_bias(cfg)
        self.register_buffer("linear_bias", linear_bias)

    def forward(self, x: Tensor) -> Tensor:
        # x -> bsz, seq_len, n_embd
        bsz, seq_len, n_embd = x.shape
        linear_bias = self.linear_bias[:, :, :seq_len, :seq_len]
        qkv: Tensor = self.QKV(x)
        q, k, v = qkv.split(n_embd, 2)
        q = q.view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        attn = q @ k.transpose(-1, -2) / math.sqrt(self.head_dim)
        attn = attn + linear_bias + self.mask[:, :, :seq_len, :seq_len]
        attn = F.softmax(attn, dim=-1)  # bsz, n_heads, seq_len, seq_len
        y = attn @ v  # bsz, n_heads, seq_len, head_dim
        y = y.transpose(1, 2).contiguous().view(bsz, seq_len, -1)
        return self.O(y)


class Block(BlockBase):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__(cfg)
        self.mha = MHA(cfg)  # Override the base mha


class GPT(GPTBase):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__(cfg)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.apply(self._init_weights)

    def forward(self, x: Tensor) -> Tensor:
        x = self.tok_emb(x)  # bsz, seq_len, n_embd
        for block in self.blocks:
            x = block(x)
        
        return self.lm_head(self.norm(x))