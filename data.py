from typing import Tuple

import torch
import random

class DataLoader:
    def __init__(self, filepath: str, tokenizer, batch_size: int, ctx_size: int):
        with open(filepath, "r") as f:
            self.data = f.read().split("<|endoftext|>")
        
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.ctx_size = ctx_size
        self.i = 0

    def _convert_data_to_tensors(self, data) -> torch.Tensor:
        tokens = []
        for text in data:
            _tokens = self.tokenizer.encode(text)
            # Pad or truncate to ctx_size + 1
            if len(_tokens) < self.ctx_size + 1:
                _tokens += self.tokenizer.encode("<|endoftext|>", allowed_special="all") * (self.ctx_size + 1 - len(_tokens))
            tokens.append(torch.tensor(_tokens[:self.ctx_size + 1]))
        return torch.stack(tokens, dim=0)

    def __len__(self) -> int:
        return len(self.data) // self.batch_size

    def __iter__(self):
        self.i = 0
        random.shuffle(self.data)
        return self
            
    def __next__(self) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.i + self.batch_size > len(self.data):
            raise StopIteration
        
        tokens = self._convert_data_to_tensors(self.data[self.i:self.i + self.batch_size])
        x, y = tokens[:, :-1], tokens[:, 1:]
        self.i += self.batch_size
        return x, y
