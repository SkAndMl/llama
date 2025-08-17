from pathlib import Path
from datetime import datetime
from typing import Dict, Any

import torch

def save_checkpoint(model, optimizer, step: int, loss: float, save_dir: Path, embedding_type: str):
    """Save model checkpoint"""
    save_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_path = save_dir / f"checkpoint_{embedding_type}_{timestamp}_step_{step}.pt"
    
    checkpoint = {
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'timestamp': datetime.now().isoformat()
    }
    
    torch.save(checkpoint, checkpoint_path)
    
    # Also save as latest
    latest_path = save_dir / f"latest_{embedding_type}.pt"
    torch.save(checkpoint, latest_path)
    
    return checkpoint_path


def load_checkpoint(checkpoint_path: Path, model, optimizer=None):
    """Load model checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return checkpoint['step'], checkpoint['loss']


def get_model_info(model) -> Dict[str, Any]:
    """Get model information for logging"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'model_size_mb': total_params * 4 / (1024 * 1024)  # Assuming float32
    }

def get_model_and_config(embedding_type: str, tokenizer, device: str):
    """Factory function to get model and config based on embedding type"""
    base_config = {
        "vocab_size": tokenizer.n_vocab,
        "n_embd": 256,
        "n_heads": 4,
        "n_layers": 4,
        "ctx_size": 256,
        "device": device,
        "bias": False,
        "attn_bias": False,
        "eps": 1e-9,
        "ffn_dim": 384
    }
    
    if embedding_type in ["pos", "rope"]:
        base_config["theta"] = 10000
    
    if embedding_type == "alibi":
        from llama.alibi import ModelConfig, GPT
    elif embedding_type == "pos":
        from llama.pos import ModelConfig, GPT
    elif embedding_type == "rope":
        from llama.rope import ModelConfig, GPT
    else:
        raise ValueError(f"Unknown embedding type: {embedding_type}")
    
    cfg = ModelConfig(**base_config)
    model = GPT(cfg)
    
    return model, cfg