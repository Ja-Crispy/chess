"""
NNUE-style models with learned components.

Three variants:
B1: Dense NNUE (no search, pure evaluation)
B2: NNUE + Policy Network (for move ordering)
B3: NNUE + Search Extension Predictor
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
from ..base import ChessModelBase
import chess


class DenseNNUE(ChessModelBase):
    """
    Dense NNUE model for position evaluation.

    Input: Piece-square features (768-dim binary)
    Architecture: Deep MLP with ReLU activations
    Output: Centipawn evaluation

    ~5M parameters
    """

    def __init__(
        self,
        embedding_dim: int = 128,
        hidden_layers: Tuple[int, ...] = (1024, 512, 256, 32),
        dropout: float = 0.1,
        use_king_relative: bool = True
    ):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.use_king_relative = use_king_relative

        # Piece-square embeddings (12 piece types × 64 squares)
        self.piece_square_embed = nn.Embedding(768, embedding_dim)

        # King-relative transformation (like HalfKAv2)
        if use_king_relative:
            self.king_relative_transform = nn.Linear(embedding_dim, embedding_dim)

        # Build hidden layers
        layers = []
        in_dim = embedding_dim * 32  # Max 32 pieces

        for hidden_dim in hidden_layers:
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            in_dim = hidden_dim

        # Final output layer
        layers.append(nn.Linear(in_dim, 1))

        self.mlp = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            features: (batch, 768) binary piece-square features

        Returns:
            Dictionary with 'eval' (centipawn evaluation)
        """
        batch_size = features.shape[0]

        # Get piece-square indices
        piece_square_indices = features.nonzero(as_tuple=False)  # (num_pieces, 2)

        # Group by batch
        embeddings = []
        for b in range(batch_size):
            # Get pieces for this position
            batch_mask = piece_square_indices[:, 0] == b
            ps_indices = piece_square_indices[batch_mask, 1]

            # Embed
            embeds = self.piece_square_embed(ps_indices)  # (num_pieces_in_pos, embed_dim)

            # Apply king-relative transformation if enabled
            if self.use_king_relative:
                embeds = self.king_relative_transform(embeds)

            # Pad to 32 pieces
            num_pieces = embeds.shape[0]
            if num_pieces < 32:
                padding = torch.zeros(
                    32 - num_pieces,
                    self.embedding_dim,
                    device=embeds.device
                )
                embeds = torch.cat([embeds, padding], dim=0)
            elif num_pieces > 32:
                embeds = embeds[:32]  # Truncate (shouldn't happen)

            embeddings.append(embeds.flatten())

        # Stack batch
        x = torch.stack(embeddings, dim=0)  # (batch, embedding_dim * 32)

        # MLP
        eval_output = self.mlp(x).squeeze(-1)

        return {
            'eval': eval_output,
            'value': torch.tanh(eval_output / 1000.0)  # Normalize to [-1, 1]
        }


class PolicyNetwork(nn.Module):
    """
    Small policy network for move ordering.

    Used in B2 variant to guide search.
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        num_layers: int = 3,
        dropout: float = 0.1
    ):
        super().__init__()

        layers = []
        in_dim = 768  # Piece-square features

        for _ in range(num_layers - 1):
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            in_dim = hidden_dim

        layers.append(nn.Linear(hidden_dim, 4096))  # Policy logits

        self.net = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: (batch, 768) piece-square features

        Returns:
            policy_logits: (batch, 4096)
        """
        return self.net(features)


class HybridEngine(nn.Module):
    """
    B2: NNUE eval + policy network for move ordering.

    Combines:
    - DenseNNUE for position evaluation
    - PolicyNetwork for move ordering
    - Optional alpha-beta search integration
    """

    def __init__(
        self,
        eval_net: Optional[DenseNNUE] = None,
        policy_net: Optional[PolicyNetwork] = None
    ):
        super().__init__()

        self.eval_net = eval_net or DenseNNUE()
        self.policy_net = policy_net or PolicyNetwork()

    def forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Returns:
            Dictionary with 'eval', 'value', 'policy_logits'
        """
        eval_outputs = self.eval_net(features)
        policy_logits = self.policy_net(features)

        return {
            **eval_outputs,
            'policy_logits': policy_logits
        }

    def get_move_ordering(
        self,
        board: chess.Board,
        features: torch.Tensor
    ) -> list:
        """
        Get moves ordered by policy network scores.

        Args:
            board: Chess board
            features: (1, 768) feature tensor

        Returns:
            List of moves sorted by policy score (best first)
        """
        with torch.no_grad():
            policy_logits = self.policy_net(features).squeeze(0)  # (4096,)

        legal_moves = list(board.legal_moves)

        # Score each legal move
        move_scores = []
        for move in legal_moves:
            move_idx = move.from_square * 64 + move.to_square
            score = policy_logits[move_idx].item()
            move_scores.append((move, score))

        # Sort by score (descending)
        move_scores.sort(key=lambda x: x[1], reverse=True)

        return [move for move, _ in move_scores]


class SearchExtensionPredictor(nn.Module):
    """
    B3: Predicts whether position needs deep search.

    Binary classifier: tactical (needs deep search) vs quiet (shallow OK)
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        dropout: float = 0.1
    ):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(768, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: (batch, 768) piece-square features

        Returns:
            probabilities: (batch,) P(needs_deep_search)
        """
        return self.net(features).squeeze(-1)


class NNUEWithExtension(nn.Module):
    """
    Complete B3 variant: NNUE + extension predictor.

    Dynamically adjusts search depth based on position type.
    """

    def __init__(
        self,
        eval_net: Optional[DenseNNUE] = None,
        extension_predictor: Optional[SearchExtensionPredictor] = None,
        shallow_threshold: float = 0.3,
        deep_threshold: float = 0.7
    ):
        super().__init__()

        self.eval_net = eval_net or DenseNNUE()
        self.extension_predictor = extension_predictor or SearchExtensionPredictor()

        self.shallow_threshold = shallow_threshold
        self.deep_threshold = deep_threshold

    def forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Returns:
            Dictionary with 'eval', 'value', 'extension_prob'
        """
        eval_outputs = self.eval_net(features)
        extension_prob = self.extension_predictor(features)

        return {
            **eval_outputs,
            'extension_prob': extension_prob
        }

    def get_recommended_depth(
        self,
        features: torch.Tensor,
        shallow_depth: int = 4,
        medium_depth: int = 6,
        deep_depth: int = 10
    ) -> int:
        """
        Get recommended search depth based on extension predictor.

        Args:
            features: (1, 768) feature tensor
            shallow_depth: Depth for quiet positions
            medium_depth: Depth for borderline positions
            deep_depth: Depth for tactical positions

        Returns:
            Recommended search depth
        """
        with torch.no_grad():
            extension_prob = self.extension_predictor(features).item()

        if extension_prob < self.shallow_threshold:
            return shallow_depth
        elif extension_prob < self.deep_threshold:
            return medium_depth
        else:
            return deep_depth


if __name__ == "__main__":
    # Test models
    print("Direction B: NNUE Variants\n")

    # B1: Dense NNUE
    model_b1 = DenseNNUE()
    model_b1.print_model_stats()

    # Test forward pass
    batch_size = 4
    features = torch.randint(0, 2, (batch_size, 768), dtype=torch.float32)

    outputs = model_b1(features)
    print(f"Output shapes:")
    print(f"  Eval: {outputs['eval'].shape}")
    print(f"  Value: {outputs['value'].shape}")

    print(f"\n" + "="*50)

    # B2: Hybrid engine
    model_b2 = HybridEngine()
    print(f"\nB2: Hybrid Engine")
    print(f"  Eval net params: {model_b2.eval_net.count_parameters():,}")
    print(f"  Policy net params: {model_b2.policy_net.count_parameters():,}")

    print(f"\n" + "="*50)

    # B3: With extension predictor
    model_b3 = NNUEWithExtension()
    print(f"\nB3: NNUE with Extension Predictor")
    print(f"  Eval net params: {model_b3.eval_net.count_parameters():,}")
    print(f"  Extension predictor params: {model_b3.extension_predictor.count_parameters():,}")
