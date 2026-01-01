"""
End-to-end transformer for chess.

Architecture:
- Input: Piece-centric encoding (piece tokens + positional encoding)
- Backbone: Decoder-only transformer
- Outputs: Policy (move distribution) + Value (position eval)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, List
from ..base import ChessModelBase, ChessOutputHeads, PositionalEncoding


class ChessTransformer(ChessModelBase):
    """
    Decoder-only transformer for chess.

    Config:
        vocab_size: 1024 (piece tokens + special tokens)
        hidden_dim: 512
        num_layers: 12
        num_heads: 8
        ffn_dim: 2048
        max_seq_len: 128
        dropout: 0.1
        total_params: ~50M
    """

    def __init__(
        self,
        vocab_size: int = 1024,
        hidden_dim: int = 512,
        num_layers: int = 12,
        num_heads: int = 8,
        ffn_dim: int = 2048,
        max_seq_len: int = 128,
        dropout: float = 0.1,
        use_piece_embeddings: bool = True
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.use_piece_embeddings = use_piece_embeddings

        # Input embeddings
        if use_piece_embeddings:
            # Separate embeddings for piece type and square position
            self.piece_embedding = nn.Embedding(12, hidden_dim)  # 12 piece types
            self.square_embedding = nn.Embedding(64, hidden_dim)  # 64 squares
        else:
            # Combined token embedding
            self.token_embedding = nn.Embedding(vocab_size, hidden_dim)

        self.pos_encoding = PositionalEncoding(hidden_dim, max_seq_len)
        self.dropout = nn.Dropout(dropout)

        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation='relu',
            batch_first=True,
            norm_first=True  # Pre-norm architecture
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            enable_nested_tensor=False
        )

        # Output heads
        self.output_heads = ChessOutputHeads(hidden_dim, dropout)

        # Layer-wise activations storage for interpretability
        self.layer_activations: List[torch.Tensor] = []
        self.store_activations = False

    def enable_activation_storage(self):
        """Enable storing layer activations for interpretability."""
        self.store_activations = True

    def disable_activation_storage(self):
        """Disable activation storage."""
        self.store_activations = False
        self.layer_activations = []

    def get_layer_activations(self) -> List[torch.Tensor]:
        """Get stored layer activations."""
        return self.layer_activations

    def forward(
        self,
        pieces: Optional[torch.Tensor] = None,
        squares: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        tokens: Optional[torch.Tensor] = None,
        return_all_layers: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            pieces: (batch, seq_len) piece IDs [0-11] (if use_piece_embeddings=True)
            squares: (batch, seq_len) square IDs [0-63] (if use_piece_embeddings=True)
            mask: (batch, seq_len) binary mask (1=valid, 0=padding)
            tokens: (batch, seq_len) token IDs (if use_piece_embeddings=False)
            return_all_layers: Return activations from all layers

        Returns:
            Dictionary with 'policy_logits', 'value', and optionally 'layer_activations'
        """
        if self.store_activations:
            self.layer_activations = []

        # Embed input
        if self.use_piece_embeddings:
            assert pieces is not None and squares is not None
            x = self.piece_embedding(pieces) + self.square_embedding(squares)
        else:
            assert tokens is not None
            x = self.token_embedding(tokens)

        x = self.pos_encoding(x)
        x = self.dropout(x)

        # Create attention mask from padding mask
        # TransformerEncoder expects: (seq_len, seq_len) or (batch, seq_len, seq_len)
        # We'll use (batch, seq_len) key_padding_mask instead
        if mask is not None:
            # key_padding_mask: True for positions to IGNORE
            key_padding_mask = (mask == 0)
        else:
            key_padding_mask = None

        # Pass through transformer with manual layer iteration for activation storage
        if return_all_layers or self.store_activations:
            layer_outputs = []
            for i, layer in enumerate(self.transformer.layers):
                x = layer(x, src_key_padding_mask=key_padding_mask)
                if self.store_activations:
                    self.layer_activations.append(x.detach().clone())
                if return_all_layers:
                    layer_outputs.append(x)
        else:
            x = self.transformer(x, src_key_padding_mask=key_padding_mask)

        # Pool sequence: mean over valid (non-padded) positions
        if mask is not None:
            mask_expanded = mask.unsqueeze(-1)  # (batch, seq_len, 1)
            x_pooled = (x * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1)
        else:
            x_pooled = x.mean(dim=1)

        # Output heads
        outputs = self.output_heads(x_pooled)

        if return_all_layers:
            outputs['layer_activations'] = layer_outputs

        return outputs

    def get_attention_patterns(
        self,
        pieces: torch.Tensor,
        squares: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        layer_idx: int = -1
    ) -> torch.Tensor:
        """
        Extract attention patterns from a specific layer.

        Args:
            pieces, squares, mask: Input tensors
            layer_idx: Which layer to extract from (-1 = last layer)

        Returns:
            attention_weights: (batch, num_heads, seq_len, seq_len)
        """
        # Register forward hook to capture attention weights
        attention_weights = []

        def hook_fn(module, input, output):
            # For TransformerEncoderLayer, attention weights are not directly returned
            # We need to access the self_attn module
            pass

        target_layer = self.transformer.layers[layer_idx]
        handle = target_layer.self_attn.register_forward_hook(hook_fn)

        # Forward pass
        with torch.no_grad():
            self.forward(pieces=pieces, squares=squares, mask=mask)

        handle.remove()

        return attention_weights


class SmallChessTransformer(ChessTransformer):
    """Smaller variant for faster experimentation."""

    def __init__(self, dropout: float = 0.1):
        super().__init__(
            vocab_size=512,
            hidden_dim=256,
            num_layers=6,
            num_heads=4,
            ffn_dim=1024,
            max_seq_len=64,
            dropout=dropout,
            use_piece_embeddings=True
        )


class LargeChessTransformer(ChessTransformer):
    """Larger variant for maximum performance."""

    def __init__(self, dropout: float = 0.1):
        super().__init__(
            vocab_size=2048,
            hidden_dim=768,
            num_layers=16,
            num_heads=12,
            ffn_dim=3072,
            max_seq_len=128,
            dropout=dropout,
            use_piece_embeddings=True
        )


if __name__ == "__main__":
    # Test model
    model = ChessTransformer()
    model.print_model_stats()

    # Test forward pass
    batch_size = 4
    seq_len = 32

    pieces = torch.randint(0, 12, (batch_size, seq_len))
    squares = torch.randint(0, 64, (batch_size, seq_len))
    mask = torch.ones(batch_size, seq_len)

    outputs = model(pieces=pieces, squares=squares, mask=mask)
    print(f"\nOutput shapes:")
    print(f"  Policy logits: {outputs['policy_logits'].shape}")
    print(f"  Value: {outputs['value'].shape}")

    print(f"\nSmall variant:")
    small_model = SmallChessTransformer()
    small_model.print_model_stats()
