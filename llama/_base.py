from pydantic import BaseModel, PositiveInt, PositiveFloat
from typing import Literal, Optional
from torch import nn, Tensor
from torch.nn import functional as F
from abc import ABC, abstractmethod

import torch


class ModelConfig(BaseModel):
    vocab_size: PositiveInt
    ctx_size: PositiveInt
    n_embd: PositiveInt
    n_heads: PositiveInt
    n_layers: PositiveInt
    bias: bool
    attn_bias: bool
    device: Literal["cpu", "mps", "cuda"]
    theta: Optional[int] = None
    eps: PositiveFloat
    ffn_dim: PositiveInt


class RMSNorm(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.w = nn.Parameter(torch.ones(cfg.n_embd))
        self.eps = cfg.eps
    
    def forward(self, x: Tensor) -> Tensor:
        # x -> bsz, ctx_size, n_embd
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return self.w * x * rms


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


class MHABase(nn.Module, ABC):
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

    @abstractmethod
    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError


class BlockBase(nn.Module, ABC):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        # this will be overridden in concrete implementations
        self.mha = None  
        self.ffn = FFN(cfg)
        self.norm1 = RMSNorm(cfg)
        self.norm2 = RMSNorm(cfg)
    
    def forward(self, x: Tensor) -> Tensor:
        x = x + self.mha(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class GPTBase(nn.Module, ABC):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.tok_emb = nn.Embedding(num_embeddings=cfg.vocab_size, embedding_dim=cfg.n_embd)
        # this will be set by concrete implementations
        self.blocks = None
        self.norm = RMSNorm(cfg)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size)

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

    @abstractmethod
    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError

    def generate(self, x, max_new_tokens):
        for _ in range(max_new_tokens):
            logits = self(x)[:, -1]
            token = torch.argmax(logits, -1).unsqueeze(1)
            x = torch.cat([x, token], dim=1)
        return x