"""
Parallel Verification Architecture for Chess.

Inspired by DeepMind's parallel thinking:
1. Shared encoder understands position
2. Generate N candidate moves
3. Each branch analyzes one candidate in parallel
4. Cross-verification layer compares branches
5. Aggregate with learned confidence weights

Novel aspect: Process reward training for intermediate reasoning correctness.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from ..base import ChessModelBase, PositionalEncoding


class BranchAnalyzer(nn.Module):
    """
    Analyzes consequences of a candidate move.

    Input: Position encoding + candidate move embedding
    Output: Branch-specific representation for verification
    """

    def __init__(
        self,
        hidden_dim: int,
        num_layers: int = 2,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True
        )

        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers)

    def forward(
        self,
        branch_input: torch.Tensor,
        memory: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            branch_input: (batch, seq_len + 1, hidden_dim) [position + candidate move]
            memory: (batch, seq_len, hidden_dim) encoded position

        Returns:
            branch_repr: (batch, hidden_dim) branch representation
        """
        output = self.decoder(branch_input, memory)
        return output.mean(dim=1)  # Pool to (batch, hidden_dim)


class ParallelVerificationChess(ChessModelBase):
    """
    Parallel Verification Chess Model.

    Architecture:
    1. Shared encoder (4-layer transformer)
    2. Candidate generator (policy head → top-k moves)
    3. N parallel branches (each analyzes one candidate)
    4. Cross-verification (branches attend to each other)
    5. Confidence aggregator (learned weights per branch)
    6. Final policy + value heads
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        num_encoder_layers: int = 4,
        num_branches: int = 5,
        num_heads: int = 8,
        max_seq_len: int = 64,
        dropout: float = 0.1
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_branches = num_branches
        self.num_encoder_layers = num_encoder_layers

        # Input embeddings
        self.piece_embedding = nn.Embedding(12, hidden_dim)
        self.square_embedding = nn.Embedding(64, hidden_dim)
        self.pos_encoding = PositionalEncoding(hidden_dim, max_seq_len)
        self.dropout = nn.Dropout(dropout)

        # Shared encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True
        )

        self.encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers)

        # Candidate move generator (preliminary policy head)
        self.candidate_generator = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 4096)
        )

        # Move embedding (for branch conditioning)
        self.move_embedding = nn.Embedding(4096, hidden_dim)

        # Branch analyzer (shared across branches)
        self.branch_analyzer = BranchAnalyzer(
            hidden_dim=hidden_dim,
            num_layers=2,
            num_heads=num_heads,
            dropout=dropout
        )

        # Cross-verification layer
        self.cross_verify = nn.MultiheadAttention(
            hidden_dim,
            num_heads,
            dropout=dropout,
            batch_first=True
        )

        self.verify_norm = nn.LayerNorm(hidden_dim)

        # Confidence aggregator
        self.confidence_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1)
        )

        # Final output heads
        self.final_policy = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 4096)
        )

        self.final_value = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Tanh()
        )

    def forward(
        self,
        pieces: torch.Tensor,
        squares: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        legal_mask: Optional[torch.Tensor] = None,
        return_branch_info: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            pieces: (batch, seq_len) piece IDs
            squares: (batch, seq_len) square IDs
            mask: (batch, seq_len) padding mask
            legal_mask: (batch, 4096) legal moves mask
            return_branch_info: Return branch analysis details

        Returns:
            Dictionary with 'policy_logits', 'value', and optionally branch info
        """
        batch_size = pieces.shape[0]

        # Encode position
        x = self.piece_embedding(pieces) + self.square_embedding(squares)
        x = self.pos_encoding(x)
        x = self.dropout(x)

        # Create attention mask
        if mask is not None:
            key_padding_mask = (mask == 0)
        else:
            key_padding_mask = None

        encoded = self.encoder(x, src_key_padding_mask=key_padding_mask)
        # (batch, seq_len, hidden_dim)

        # Pool for candidate generation
        if mask is not None:
            mask_expanded = mask.unsqueeze(-1)
            encoded_pooled = (encoded * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1)
        else:
            encoded_pooled = encoded.mean(dim=1)

        # Generate top-k candidate moves
        candidate_logits = self.candidate_generator(encoded_pooled)
        # (batch, 4096)

        # Mask illegal moves if provided
        if legal_mask is not None:
            candidate_logits = candidate_logits.masked_fill(legal_mask == 0, -1e9)

        # Select top-k candidates
        top_k_logits, top_k_indices = candidate_logits.topk(self.num_branches, dim=-1)
        # (batch, num_branches)

        # Analyze each branch
        branch_outputs = []

        for k in range(self.num_branches):
            # Get candidate move index for this branch
            move_indices = top_k_indices[:, k]  # (batch,)

            # Create "what-if" representation: [position + candidate_move]
            move_embeds = self.move_embedding(move_indices).unsqueeze(1)
            # (batch, 1, hidden_dim)

            branch_input = torch.cat([encoded, move_embeds], dim=1)
            # (batch, seq_len + 1, hidden_dim)

            # Analyze this branch
            branch_repr = self.branch_analyzer(branch_input, encoded)
            # (batch, hidden_dim)

            branch_outputs.append(branch_repr)

        branch_stack = torch.stack(branch_outputs, dim=1)
        # (batch, num_branches, hidden_dim)

        # Cross-verification: branches attend to each other
        verified, attention_weights = self.cross_verify(
            branch_stack,
            branch_stack,
            branch_stack
        )
        # (batch, num_branches, hidden_dim)

        verified = self.verify_norm(verified + branch_stack)

        # Compute confidence for each branch
        confidence_logits = self.confidence_head(verified).squeeze(-1)
        # (batch, num_branches)

        confidence = F.softmax(confidence_logits, dim=-1)
        # (batch, num_branches)

        # Aggregate branches with confidence weighting
        aggregated = (verified * confidence.unsqueeze(-1)).sum(dim=1)
        # (batch, hidden_dim)

        # Final outputs
        policy_logits = self.final_policy(aggregated)
        value = self.final_value(aggregated).squeeze(-1)

        outputs = {
            'policy_logits': policy_logits,
            'value': value,
        }

        if return_branch_info:
            outputs.update({
                'candidate_moves': top_k_indices,
                'candidate_logits': top_k_logits,
                'branch_confidence': confidence,
                'branch_representations': verified,
                'attention_weights': attention_weights
            })

        return outputs

    def forward_fast(
        self,
        pieces: torch.Tensor,
        squares: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        legal_mask: Optional[torch.Tensor] = None,
        num_branches: int = 2
    ) -> Dict[str, torch.Tensor]:
        """
        Fast inference with fewer branches and no verification.

        Args:
            num_branches: Number of branches to use (< self.num_branches)

        Returns:
            Same as forward() but faster
        """
        # Temporarily reduce branches
        original_num_branches = self.num_branches
        self.num_branches = num_branches

        outputs = self.forward(
            pieces=pieces,
            squares=squares,
            mask=mask,
            legal_mask=legal_mask,
            return_branch_info=False
        )

        # Restore
        self.num_branches = original_num_branches

        return outputs


def compute_process_reward_loss(
    outputs: Dict[str, torch.Tensor],
    best_moves: torch.Tensor,
    alpha: float = 0.5
) -> torch.Tensor:
    """
    Process reward loss: reward correct intermediate reasoning.

    Two components:
    1. Candidate quality: Is best move in top-k candidates?
    2. Confidence accuracy: Is confidence highest for correct branch?

    Args:
        outputs: Model outputs with branch_info
        best_moves: (batch,) ground truth best move indices
        alpha: Weight for confidence loss

    Returns:
        Scalar process reward loss
    """
    candidate_moves = outputs['candidate_moves']  # (batch, num_branches)
    branch_confidence = outputs['branch_confidence']  # (batch, num_branches)

    batch_size = best_moves.shape[0]

    # 1. Candidate quality loss
    # Check if best_move is in candidates
    best_moves_expanded = best_moves.unsqueeze(1)  # (batch, 1)
    is_candidate = (candidate_moves == best_moves_expanded).any(dim=1).float()
    # (batch,) binary: 1 if best move in candidates

    candidate_loss = -is_candidate.mean()  # Negative reward

    # 2. Confidence accuracy loss
    # Find which branch has the correct move
    correct_branch_mask = (candidate_moves == best_moves_expanded)
    # (batch, num_branches)

    # If correct move is in candidates, confidence should be highest there
    has_correct = correct_branch_mask.any(dim=1)  # (batch,)

    if has_correct.any():
        # Get correct branch indices
        correct_branch_indices = correct_branch_mask.float().argmax(dim=1)
        # (batch,) - index of correct branch (0 if not present)

        # Cross-entropy loss: encourage high confidence for correct branch
        confidence_loss = F.cross_entropy(
            branch_confidence[has_correct],
            correct_branch_indices[has_correct]
        )
    else:
        confidence_loss = torch.tensor(0.0, device=best_moves.device)

    return candidate_loss + alpha * confidence_loss


if __name__ == "__main__":
    print("Direction D: Parallel Verification Architecture\n")

    model = ParallelVerificationChess(
        hidden_dim=256,
        num_encoder_layers=4,
        num_branches=5,
        num_heads=8
    )

    model.print_model_stats()

    # Test forward pass
    batch_size = 4
    seq_len = 32

    pieces = torch.randint(0, 12, (batch_size, seq_len))
    squares = torch.randint(0, 64, (batch_size, seq_len))
    mask = torch.ones(batch_size, seq_len)
    legal_mask = torch.randint(0, 2, (batch_size, 4096)).float()

    outputs = model(
        pieces=pieces,
        squares=squares,
        mask=mask,
        legal_mask=legal_mask,
        return_branch_info=True
    )

    print(f"\nOutput shapes:")
    print(f"  Policy logits: {outputs['policy_logits'].shape}")
    print(f"  Value: {outputs['value'].shape}")
    print(f"  Candidate moves: {outputs['candidate_moves'].shape}")
    print(f"  Branch confidence: {outputs['branch_confidence'].shape}")

    print(f"\nBranch confidence for first sample:")
    for i, conf in enumerate(outputs['branch_confidence'][0]):
        print(f"  Branch {i}: {conf.item():.3f}")

    # Test process reward loss
    best_moves = torch.randint(0, 4096, (batch_size,))
    process_loss = compute_process_reward_loss(outputs, best_moves)
    print(f"\nProcess reward loss: {process_loss.item():.4f}")

    # Test fast inference
    print(f"\nTesting fast inference (2 branches):")
    fast_outputs = model.forward_fast(
        pieces=pieces,
        squares=squares,
        mask=mask,
        num_branches=2
    )
    print(f"  Policy logits: {fast_outputs['policy_logits'].shape}")
