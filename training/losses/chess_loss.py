"""
Loss functions for chess model training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


def compute_policy_loss(
    policy_logits: torch.Tensor,
    best_moves: torch.Tensor,
    legal_mask: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Cross-entropy loss for policy (move prediction).

    Args:
        policy_logits: (batch, 4096) unnormalized logits
        best_moves: (batch,) target move indices
        legal_mask: (batch, 4096) binary mask of legal moves

    Returns:
        Scalar loss
    """
    # Mask illegal moves
    if legal_mask is not None:
        policy_logits = policy_logits.masked_fill(legal_mask == 0, -1e9)

    return F.cross_entropy(policy_logits, best_moves)


def compute_value_loss(
    predicted_values: torch.Tensor,
    target_values: torch.Tensor,
    loss_type: str = 'mse'
) -> torch.Tensor:
    """
    Loss for position evaluation.

    Args:
        predicted_values: (batch,) predicted values
        target_values: (batch,) target values
        loss_type: 'mse' or 'huber'

    Returns:
        Scalar loss
    """
    if loss_type == 'mse':
        return F.mse_loss(predicted_values, target_values)
    elif loss_type == 'huber':
        return F.huber_loss(predicted_values, target_values)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


class ChessLoss(nn.Module):
    """
    Combined loss for chess models.

    Combines:
    - Policy loss (cross-entropy on best move)
    - Value loss (MSE on position evaluation)
    """

    def __init__(
        self,
        policy_weight: float = 0.5,
        value_weight: float = 0.5,
        value_loss_type: str = 'mse'
    ):
        super().__init__()

        self.policy_weight = policy_weight
        self.value_weight = value_weight
        self.value_loss_type = value_loss_type

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        batch: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Compute combined loss.

        Args:
            outputs: Model outputs with 'policy_logits' and 'value'
            batch: Batch data with 'best_move', 'eval', 'legal_mask'

        Returns:
            Dictionary with 'loss', 'policy_loss', 'value_loss'
        """
        policy_loss = compute_policy_loss(
            outputs['policy_logits'],
            batch['best_move'],
            batch.get('legal_mask', None)
        )

        value_loss = compute_value_loss(
            outputs['value'],
            batch['eval'],
            loss_type=self.value_loss_type
        )

        total_loss = (
            self.policy_weight * policy_loss +
            self.value_weight * value_loss
        )

        return {
            'loss': total_loss,
            'policy_loss': policy_loss,
            'value_loss': value_loss
        }


class MoELoss(ChessLoss):
    """Loss for MoE models with load balancing."""

    def __init__(
        self,
        policy_weight: float = 0.5,
        value_weight: float = 0.5,
        balance_weight: float = 0.01,
        value_loss_type: str = 'mse'
    ):
        super().__init__(policy_weight, value_weight, value_loss_type)
        self.balance_weight = balance_weight

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        batch: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Add load balancing loss for MoE."""
        base_losses = super().forward(outputs, batch)

        # Load balancing loss
        if 'routing_weights' in outputs:
            routing_weights = outputs['routing_weights']
            # (batch, num_layers, num_experts)

            # Average routing weights
            expert_usage = routing_weights.mean(dim=(0, 1))  # (num_experts,)

            # Variance of usage (want uniform distribution)
            balance_loss = expert_usage.var()

            total_loss = base_losses['loss'] + self.balance_weight * balance_loss

            base_losses['loss'] = total_loss
            base_losses['balance_loss'] = balance_loss

        return base_losses


class ParallelVerificationLoss(ChessLoss):
    """Loss for parallel verification model with process rewards."""

    def __init__(
        self,
        policy_weight: float = 0.5,
        value_weight: float = 0.5,
        process_reward_weight: float = 0.3,
        value_loss_type: str = 'mse'
    ):
        super().__init__(policy_weight, value_weight, value_loss_type)
        self.process_reward_weight = process_reward_weight

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        batch: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Add process reward loss."""
        base_losses = super().forward(outputs, batch)

        # Process reward loss
        if 'candidate_moves' in outputs and 'branch_confidence' in outputs:
            process_loss = self._compute_process_reward_loss(outputs, batch)

            total_loss = base_losses['loss'] + self.process_reward_weight * process_loss

            base_losses['loss'] = total_loss
            base_losses['process_loss'] = process_loss

        return base_losses

    def _compute_process_reward_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        batch: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Compute process reward loss."""
        candidate_moves = outputs['candidate_moves']
        branch_confidence = outputs['branch_confidence']
        best_moves = batch['best_move']

        # Check if best move is in candidates
        best_moves_expanded = best_moves.unsqueeze(1)
        is_candidate = (candidate_moves == best_moves_expanded).any(dim=1).float()

        candidate_loss = -is_candidate.mean()

        # Confidence should be highest for correct branch
        correct_branch_mask = (candidate_moves == best_moves_expanded)
        has_correct = correct_branch_mask.any(dim=1)

        if has_correct.any():
            correct_branch_indices = correct_branch_mask.float().argmax(dim=1)
            confidence_loss = F.cross_entropy(
                branch_confidence[has_correct],
                correct_branch_indices[has_correct]
            )
        else:
            confidence_loss = torch.tensor(0.0, device=best_moves.device)

        return candidate_loss + confidence_loss


if __name__ == "__main__":
    # Test losses
    batch_size = 4

    outputs = {
        'policy_logits': torch.randn(batch_size, 4096),
        'value': torch.randn(batch_size)
    }

    batch = {
        'best_move': torch.randint(0, 4096, (batch_size,)),
        'eval': torch.randn(batch_size),
        'legal_mask': torch.randint(0, 2, (batch_size, 4096)).float()
    }

    loss_fn = ChessLoss()
    losses = loss_fn(outputs, batch)

    print("Chess Loss:")
    for k, v in losses.items():
        print(f"  {k}: {v.item():.4f}")
