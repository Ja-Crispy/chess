"""
Base classes for chess models.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


class ChessOutputHeads(nn.Module):
    """
    Standard output heads for chess models.

    Two heads:
    1. Policy head: probability distribution over moves (4096-dim)
    2. Value head: position evaluation in [-1, 1]
    """

    def __init__(self, hidden_dim: int, dropout: float = 0.1):
        super().__init__()

        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 4096)
        )

        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Tanh()
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: (batch, hidden_dim) feature tensor

        Returns:
            Dictionary with 'policy_logits' and 'value'
        """
        return {
            'policy_logits': self.policy_head(x),
            'value': self.value_head(x).squeeze(-1)
        }


class ChessModelBase(nn.Module):
    """Base class for all chess models."""

    def __init__(self):
        super().__init__()

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_model_size_mb(self) -> float:
        """Get model size in megabytes."""
        param_size = sum(p.numel() * p.element_size() for p in self.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in self.buffers())
        return (param_size + buffer_size) / (1024 ** 2)

    def print_model_stats(self):
        """Print model statistics."""
        print(f"Model: {self.__class__.__name__}")
        print(f"  Parameters: {self.count_parameters():,}")
        print(f"  Size: {self.get_model_size_mb():.2f} MB")


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for transformers."""

    def __init__(self, d_model: int, max_len: int = 64):
        super().__init__()

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-torch.log(torch.tensor(10000.0)) / d_model))

        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)

        Returns:
            x with positional encoding added
        """
        return x + self.pe[:x.size(1)]
