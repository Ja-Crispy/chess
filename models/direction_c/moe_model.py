"""
Mixture-of-Experts for chess with position-type routing.

Architecture:
- Shared transformer trunk for encoding
- Expert bank with N specialized experts
- Learned router that selects top-k experts per position
- Hypothesis: Experts specialize by position type (tactical, endgame, etc.)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from ..base import ChessModelBase, ChessOutputHeads, PositionalEncoding


class ExpertFFN(nn.Module):
    """Single expert: Feed-forward network."""

    def __init__(
        self,
        hidden_dim: int,
        ffn_dim: int,
        dropout: float = 0.1
    ):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(hidden_dim, ffn_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, hidden_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Router(nn.Module):
    """
    Learned router for expert selection.

    Uses softmax over expert logits with optional load balancing.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_experts: int,
        top_k: int = 2
    ):
        super().__init__()

        self.num_experts = num_experts
        self.top_k = top_k

        self.router_linear = nn.Linear(hidden_dim, num_experts)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Route input to top-k experts.

        Args:
            x: (batch, seq_len, hidden_dim) or (batch, hidden_dim)

        Returns:
            weights: (batch, top_k) normalized routing weights
            indices: (batch, top_k) expert indices
        """
        # Pool if needed
        if x.dim() == 3:
            x_pooled = x.mean(dim=1)  # (batch, hidden_dim)
        else:
            x_pooled = x

        # Compute router logits
        router_logits = self.router_linear(x_pooled)  # (batch, num_experts)

        # Top-k expert selection
        top_k_logits, top_k_indices = router_logits.topk(self.top_k, dim=-1)

        # Normalize weights (softmax over top-k)
        top_k_weights = F.softmax(top_k_logits, dim=-1)

        return top_k_weights, top_k_indices


class HorizontalMoERouter(nn.Module):
    """
    Horizontal MoE: Shared expert bank across layers.

    Each layer has:
    - Its own routing adapter
    - Access to shared expert bank
    - Expert adapters for layer-specific behavior
    """

    def __init__(
        self,
        hidden_dim: int,
        ffn_dim: int,
        num_experts: int,
        num_layers: int,
        top_k: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()

        self.num_experts = num_experts
        self.num_layers = num_layers
        self.top_k = top_k

        # Shared expert bank
        self.expert_bank = nn.ModuleList([
            ExpertFFN(hidden_dim, ffn_dim, dropout)
            for _ in range(num_experts)
        ])

        # Per-layer routers
        self.layer_routers = nn.ModuleList([
            Router(hidden_dim, num_experts, top_k)
            for _ in range(num_layers)
        ])

        # Per-layer expert adapters (modulate expert behavior per layer)
        self.expert_adapters = nn.ParameterList([
            nn.Parameter(torch.ones(num_experts, hidden_dim))
            for _ in range(num_layers)
        ])

    def forward(
        self,
        x: torch.Tensor,
        layer_idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Route through experts for a specific layer.

        Args:
            x: (batch, seq_len, hidden_dim)
            layer_idx: Which layer we're in

        Returns:
            output: (batch, seq_len, hidden_dim) expert outputs
            routing_weights: (batch, num_experts) full routing distribution
        """
        batch_size, seq_len, hidden_dim = x.shape

        # Get routing weights and indices
        router = self.layer_routers[layer_idx]
        top_k_weights, top_k_indices = router(x)  # (batch, top_k), (batch, top_k)

        # Initialize output
        output = torch.zeros_like(x)

        # Compute expert outputs and combine
        for k in range(self.top_k):
            # Get expert indices and weights for this k
            expert_idx_batch = top_k_indices[:, k]  # (batch,)
            weight_batch = top_k_weights[:, k]  # (batch,)

            # Process each unique expert
            unique_experts = expert_idx_batch.unique()

            for expert_idx in unique_experts:
                # Find which batch elements use this expert
                expert_mask = (expert_idx_batch == expert_idx)  # (batch,)

                if not expert_mask.any():
                    continue

                # Get expert
                expert = self.expert_bank[expert_idx]

                # Process batch elements using this expert
                x_expert = x[expert_mask]  # (num_using_expert, seq_len, hidden_dim)
                expert_out = expert(x_expert)

                # Apply layer-specific adapter
                adapter = self.expert_adapters[layer_idx][expert_idx]
                expert_out = expert_out * adapter

                # Weight and accumulate
                weights = weight_batch[expert_mask].view(-1, 1, 1)
                output[expert_mask] += weights * expert_out

        # Compute full routing distribution (for load balancing loss)
        router_logits = router.router_linear(x.mean(dim=1))  # (batch, num_experts)
        routing_weights = F.softmax(router_logits, dim=-1)

        return output, routing_weights


class ChessMoE(ChessModelBase):
    """
    Chess model with Mixture-of-Experts.

    Architecture:
    - Shared transformer trunk (4 layers)
    - MoE layers with expert bank
    - Router learns to specialize experts by position type
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        num_trunk_layers: int = 4,
        num_experts: int = 8,
        ffn_dim: int = 512,
        top_k: int = 2,
        num_heads: int = 8,
        max_seq_len: int = 64,
        dropout: float = 0.1
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.num_trunk_layers = num_trunk_layers

        # Input embeddings
        self.piece_embedding = nn.Embedding(12, hidden_dim)
        self.square_embedding = nn.Embedding(64, hidden_dim)
        self.pos_encoding = PositionalEncoding(hidden_dim, max_seq_len)
        self.dropout = nn.Dropout(dropout)

        # Shared trunk (standard transformer)
        trunk_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=True
        )

        self.trunk = nn.TransformerEncoder(trunk_layer, num_trunk_layers)

        # MoE router (replaces FFN in additional layers)
        num_moe_layers = 4
        self.moe_router = HorizontalMoERouter(
            hidden_dim=hidden_dim,
            ffn_dim=ffn_dim,
            num_experts=num_experts,
            num_layers=num_moe_layers,
            top_k=top_k,
            dropout=dropout
        )

        # Layer norms for MoE layers
        self.moe_layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_moe_layers)
        ])

        # Attention layers for MoE section
        self.moe_attentions = nn.ModuleList([
            nn.MultiheadAttention(
                hidden_dim,
                num_heads,
                dropout=dropout,
                batch_first=True
            )
            for _ in range(num_moe_layers)
        ])

        # Output heads
        self.output_heads = ChessOutputHeads(hidden_dim, dropout)

        # Track routing statistics
        self.routing_weights_history = []

    def forward(
        self,
        pieces: torch.Tensor,
        squares: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_routing_info: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            pieces: (batch, seq_len) piece IDs
            squares: (batch, seq_len) square IDs
            mask: (batch, seq_len) padding mask
            return_routing_info: Return expert routing information

        Returns:
            Dictionary with 'policy_logits', 'value', and optionally routing info
        """
        # Embed input
        x = self.piece_embedding(pieces) + self.square_embedding(squares)
        x = self.pos_encoding(x)
        x = self.dropout(x)

        # Create attention mask
        if mask is not None:
            key_padding_mask = (mask == 0)
        else:
            key_padding_mask = None

        # Shared trunk
        x = self.trunk(x, src_key_padding_mask=key_padding_mask)

        # MoE layers
        all_routing_weights = []

        for layer_idx in range(len(self.moe_attentions)):
            # Self-attention
            attn_out, _ = self.moe_attentions[layer_idx](
                x, x, x,
                key_padding_mask=key_padding_mask
            )
            x = self.moe_layer_norms[layer_idx](x + attn_out)

            # MoE (replaces FFN)
            moe_out, routing_weights = self.moe_router(x, layer_idx)
            x = x + moe_out

            all_routing_weights.append(routing_weights)

        # Pool sequence
        if mask is not None:
            mask_expanded = mask.unsqueeze(-1)
            x_pooled = (x * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1)
        else:
            x_pooled = x.mean(dim=1)

        # Output heads
        outputs = self.output_heads(x_pooled)

        if return_routing_info:
            outputs['routing_weights'] = torch.stack(all_routing_weights, dim=1)
            # (batch, num_moe_layers, num_experts)

        return outputs

    def get_expert_usage_stats(self) -> Dict[int, float]:
        """
        Analyze which experts are used most frequently.

        Returns:
            Dictionary: expert_idx -> usage_frequency
        """
        if len(self.routing_weights_history) == 0:
            return {}

        # Stack all routing weights
        all_weights = torch.cat(self.routing_weights_history, dim=0)
        # (num_samples, num_experts)

        # Compute usage frequency
        usage = all_weights.mean(dim=0)  # (num_experts,)

        return {i: usage[i].item() for i in range(self.num_experts)}


def compute_load_balancing_loss(
    routing_weights: torch.Tensor,
    alpha: float = 0.01
) -> torch.Tensor:
    """
    Load balancing loss to encourage uniform expert usage.

    Args:
        routing_weights: (batch, num_layers, num_experts)
        alpha: Loss weight

    Returns:
        Scalar load balancing loss
    """
    # Average routing weights across batch and layers
    expert_usage = routing_weights.mean(dim=(0, 1))  # (num_experts,)

    # Variance of usage (want this to be low)
    variance = expert_usage.var()

    return alpha * variance


if __name__ == "__main__":
    print("Direction C: Mixture-of-Experts\n")

    model = ChessMoE(
        hidden_dim=256,
        num_trunk_layers=4,
        num_experts=8,
        top_k=2
    )

    model.print_model_stats()

    # Test forward pass
    batch_size = 4
    seq_len = 32

    pieces = torch.randint(0, 12, (batch_size, seq_len))
    squares = torch.randint(0, 64, (batch_size, seq_len))
    mask = torch.ones(batch_size, seq_len)

    outputs = model(
        pieces=pieces,
        squares=squares,
        mask=mask,
        return_routing_info=True
    )

    print(f"\nOutput shapes:")
    print(f"  Policy logits: {outputs['policy_logits'].shape}")
    print(f"  Value: {outputs['value'].shape}")
    print(f"  Routing weights: {outputs['routing_weights'].shape}")

    # Analyze expert usage
    print(f"\nExpert usage:")
    routing = outputs['routing_weights'][0, 0, :]  # First sample, first layer
    for i, weight in enumerate(routing):
        print(f"  Expert {i}: {weight.item():.3f}")
