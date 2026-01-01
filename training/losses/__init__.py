"""Loss functions for chess training."""

from .chess_loss import ChessLoss, compute_policy_loss, compute_value_loss

__all__ = ['ChessLoss', 'compute_policy_loss', 'compute_value_loss']
