from ._base import ModelConfig, MHABase, BlockBase, GPTBase, FFN, RMSNorm
from torch import nn, Tensor
from torch.nn import functional as F

import torch
import math


def get_pos_emb(cfg: ModelConfig) -> Tensor:
    thetas = cfg.theta ** ((-2 * torch.arange(0, cfg.n_embd // 2)) / cfg.n_embd)  # n_embd // 2
    pos = torch.arange(cfg.ctx_size)  # ctx_size
    freqs = torch.outer(pos, thetas)  # ctx_size, n_embd // 2
    even_pos = torch.sin(freqs)
    odd_pos = torch.cos(freqs)
    pos_emb = torch.stack((even_pos, odd_pos), dim=2)
    pos_emb = pos_emb.reshape(cfg.ctx_size, -1)  # ctx_size, n_embd
    return pos_emb.unsqueeze(0)


class MHA(MHABase):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__(cfg)

    def forward(self, x: Tensor) -> Tensor:
        # x -> bsz, seq_len, n_embd
        bsz, seq_len, n_embd = x.shape
        qkv: Tensor = self.QKV(x)
        q, k, v = qkv.split(n_embd, 2)
        q = q.view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

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
        
        self.register_buffer("pos_emb", get_pos_emb(cfg))
        self.apply(self._init_weights)

    def forward(self, x: Tensor) -> Tensor:
        tok_emb = self.tok_emb(x)  # bsz, seq_len, n_embd
        pos_emb = self.pos_emb[:, :x.shape[1], :]
        x = tok_emb + pos_emb
        for block in self.blocks:
            x = block(x)
        
        return self.lm_head(self.norm(x))