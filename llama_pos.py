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

def get_pos_emb(cfg: ModelConfig) -> Tensor:

    thetas = cfg.theta ** ((-2 * torch.arange(0, cfg.n_embd // 2)) / cfg.n_embd) # n_embd // 2
    pos = torch.arange(cfg.ctx_size) # ctx_size
    freqs = torch.outer(pos, thetas) # ctx_size, n_embd // 2
    even_pos = torch.sin(freqs)
    odd_pos = torch.cos(freqs)
    pos_emb = torch.stack((even_pos, odd_pos), dim=2)
    pos_emb = pos_emb.reshape(cfg.ctx_size, -1) # ctx_size, n_embd
    return pos_emb.unsqueeze(0)


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
            torch.ones(1, 1, cfg.ctx_size, cfg.ctx_size) * float("-inf"), diagonal=1
        )
        self.register_buffer("mask", mask)

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

        self.register_buffer(
            "pos_emb", get_pos_emb(cfg)
        )

        self.apply(self._init_weights)
    
    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.xavier_uniform_(m.weight)
        elif isinstance(m, RMSNorm):
            torch.nn.init.ones_(m.w)


    def forward(self, x: Tensor) -> Tensor:
        
        tok_emb = self.tok_emb(x) # bsz, seq_len, n_embd
        pos_emb = self.pos_emb[:, :x.shape[1], :]
        x = tok_emb + pos_emb
        for block in self.blocks:
            x = block(x)
        
        return self.lm_head(self.norm(x))

    def generate(self, x, max_new_tokens):
        for _ in range(max_new_tokens):
            logits = self(x)[:, -1]
            token = torch.argmax(logits, -1).unsqueeze(1)
            x = torch.cat([x, token], dim=1)
        return x