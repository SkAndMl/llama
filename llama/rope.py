from ._base import ModelConfig, MHABase, BlockBase, GPTBase, FFN, RMSNorm
from torch import nn, Tensor
from torch.nn import functional as F

import torch
import math


def get_freqs_cis(cfg: ModelConfig) -> Tensor:
    head_dim = cfg.n_embd // cfg.n_heads
    i = torch.arange(head_dim // 2)
    thetas = cfg.theta ** (-2 * i / head_dim)  # head_dim // 2
    pos = torch.arange(cfg.ctx_size)  # pos
    freqs = torch.outer(pos, thetas)  # pos, head_dim // 2
    real = torch.cos(freqs)
    imag = torch.sin(freqs)
    return torch.complex(real, imag)


def apply_rot_emb(x: Tensor, freqs: Tensor) -> Tensor:
    # x -> bsz, n_heads, seq_len, head_dim; freqs -> pos, head_dim // 2
    bsz, n_heads, seq_len, head_dim = x.shape
    half = head_dim // 2
    f = freqs[:seq_len]

    x = x.reshape(bsz, n_heads, seq_len, half, 2)
    x_rot = torch.view_as_complex(x) * f.view(1, 1, seq_len, half)  # bsz, n_heads, seq_len, head_dim // 2
    x_real = torch.view_as_real(x_rot)  # bsz, n_heads, seq_len, head_dim // 2, 2
    return x_real.reshape(bsz, n_heads, seq_len, head_dim)


class MHA(MHABase):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__(cfg)
        
        freqs = get_freqs_cis(cfg)
        self.register_buffer("freqs", freqs)

    def forward(self, x: Tensor) -> Tensor:
        # x -> bsz, seq_len, n_embd
        bsz, seq_len, n_embd = x.shape
        qkv: Tensor = self.QKV(x)
        q, k, v = qkv.split(n_embd, 2)
        q = q.view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        q = apply_rot_emb(q, self.freqs)
        k = apply_rot_emb(k, self.freqs)

        attn = q @ k.transpose(-1, -2) / math.sqrt(self.head_dim)
        attn = attn + self.mask[:, :, :seq_len, :seq_len]
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