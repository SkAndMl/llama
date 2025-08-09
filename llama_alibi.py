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



def get_linear_bias(cfg: ModelConfig) -> Tensor:
    pos = torch.zeros(cfg.ctx_size, cfg.ctx_size)
    slopes = (2 ** (-8 / torch.arange(1, cfg.n_heads+1))).view(cfg.n_heads, 1, 1) # n_heads, 1, 1
    for i in range(cfg.ctx_size):
        for j in range(i, -1, -1):
            pos[i, j] = j - i
    
    pos.unsqueeze_(0) # 1, ctx_size, ctx_size
    linear_bias = pos * slopes # n_heads, ctx_size, ctx_size
    return linear_bias.unsqueeze(0) # 1, n_heads, ctx_size, ctx_size 


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

        attn = q @ k.transpose(-1, -2)/ math.sqrt(self.head_dim)
        attn = attn + linear_bias + self.mask[:, :, :seq_len, :seq_len]
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