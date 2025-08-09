from pydantic import BaseModel, PositiveInt, PositiveFloat
from typing import Literal
from torch import nn, Tensor
from torch.nn import functional as F

import torch
import math


class ModelConfig(BaseModel):
    vocab_size: PositiveInt
    ctx_size: PositiveInt
    n_embd: PositiveInt
    n_heads: PositiveInt
    n_layers: PositiveInt
    bias: bool
    attn_bias: bool
    device: Literal["cpu", "mps", "cuda"]
    theta: int
    eps: PositiveFloat
    ffn_dim: PositiveInt

def get_freqs_cis(cfg: ModelConfig) -> Tensor:
    head_dim = cfg.n_embd // cfg.n_heads
    i = torch.arange(head_dim // 2)
    thetas = cfg.theta ** (-2 * i / head_dim) # head_dim // 2
    pos = torch.arange(cfg.ctx_size) # pos
    freqs = torch.outer(pos, thetas) # pos, head_dim // 2
    real = torch.cos(freqs)
    imag = torch.sin(freqs)
    return torch.complex(real, imag)


def apply_rot_emb(x: Tensor, freqs: Tensor) -> Tensor:
    # x -> bsz, n_heads, seq_len, head_dim; freqs -> pos, head_dim // 2
    bsz, n_heads, seq_len, head_dim = x.shape
    half = head_dim // 2
    f = freqs[:seq_len]

    x = x.reshape(bsz, n_heads, seq_len, half, 2)
    x_rot = torch.view_as_complex(x) * f.view(1, 1, seq_len, half) # bsz, n_heads, seq_len, head_dim // 2
    x_real =  torch.view_as_real(x_rot) # bsz, n_heads, seq_len, head_dim // 2, 2
    return x_real.reshape(bsz, n_heads, seq_len, head_dim)


class RMSNorm(nn.Module):

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.w = nn.Parameter(torch.ones(cfg.n_embd))
        self.eps = cfg.eps
    
    def forward(self, x: Tensor) -> Tensor:
        # x -> bsz, ctx_size, n_embd
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return self.w * x * rms

class MHA(nn.Module):

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        assert cfg.n_embd % cfg.n_heads == 0
        self.head_dim = cfg.n_embd // cfg.n_heads
        self.n_heads = cfg.n_heads
        self.QKV = nn.Linear(cfg.n_embd, cfg.n_embd * 3, bias=cfg.attn_bias)
        self.O = nn.Linear(cfg.n_embd, cfg.n_embd, bias=cfg.attn_bias)

        mask = torch.triu(
            torch.ones(1, 1, cfg.ctx_size, cfg.ctx_size), diagonal=1
        ) * float("-inf")
        self.register_buffer("mask", mask)

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
        attn = F.softmax(attn, dim=-1) # bsz, n_heads, seq_len, seq_len
        y = attn @ v # bsz, n_heads, seq_len, head_dim
        y = y.transpose(1, 2).contiguous().view(bsz, seq_len, -1)
        return self.O(y)


class FFN(nn.Module):

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(cfg.n_embd, cfg.ffn_dim, bias=cfg.bias)
        self.up_proj = nn.Linear(cfg.n_embd, cfg.ffn_dim, bias=cfg.bias)
        self.down_proj = nn.Linear(cfg.ffn_dim, cfg.n_embd, bias=cfg.bias)
    
    def forward(self, x: Tensor) -> Tensor:
        x1, x2 = self.gate_proj(x), self.up_proj(x)
        x = F.silu(x1) * x2
        x = self.down_proj(x)
        return x


class Block(nn.Module):

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.mha = MHA(cfg)
        self.ffn = FFN(cfg)
        self.norm1 = RMSNorm(cfg)
        self.norm2 = RMSNorm(cfg)
    
    def forward(self, x: Tensor) -> Tensor:
        x = x + self.mha(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class GPT(nn.Module):

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.tok_emb = nn.Embedding(num_embeddings=cfg.vocab_size, embedding_dim=cfg.n_embd)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.norm = RMSNorm(cfg)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size)
    
    def forward(self, x: Tensor) -> Tensor:
        
        x = self.tok_emb(x) # bsz, seq_len, n_embd
        for block in self.blocks:
            x = block(x)
        
        return self.lm_head(self.norm(x))