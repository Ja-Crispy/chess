"""
Mechanistic interpretability and pruning tools for chess transformers.

Tools for:
1. Linear probing to understand learned representations
2. Attention pattern analysis
3. Structured pruning (heads, neurons, layers)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Callable
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import chess


class LinearProbe(nn.Module):
    """
    Linear probe to test if activations encode specific chess concepts.

    Examples:
    - Does layer X encode piece positions?
    - Does layer Y detect checks?
    - Does layer Z understand material balance?
    """

    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)

    def fit_and_evaluate(
        self,
        train_activations: torch.Tensor,
        train_labels: torch.Tensor,
        val_activations: torch.Tensor,
        val_labels: torch.Tensor,
        epochs: int = 100,
        lr: float = 0.001
    ) -> Dict[str, float]:
        """
        Train probe and evaluate accuracy.

        Args:
            train_activations: (N, hidden_dim)
            train_labels: (N,) integer labels
            val_activations: (M, hidden_dim)
            val_labels: (M,) integer labels

        Returns:
            Dictionary with train/val accuracy
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        for epoch in range(epochs):
            self.train()
            optimizer.zero_grad()

            logits = self.forward(train_activations)
            loss = criterion(logits, train_labels)

            loss.backward()
            optimizer.step()

        # Evaluate
        self.eval()
        with torch.no_grad():
            train_logits = self.forward(train_activations)
            val_logits = self.forward(val_activations)

            train_preds = train_logits.argmax(dim=1)
            val_preds = val_logits.argmax(dim=1)

            train_acc = (train_preds == train_labels).float().mean().item()
            val_acc = (val_preds == val_labels).float().mean().item()

        return {
            'train_accuracy': train_acc,
            'val_accuracy': val_acc
        }


class ProbingSuite:
    """
    Suite of probes to analyze what the model has learned.

    Probes:
    1. piece_position: Does it know where pieces are?
    2. material_count: Does it track material balance?
    3. check_detection: Does it detect checks?
    4. legal_moves: Does it know legal moves?
    5. king_safety: Does it evaluate king safety?
    """

    def __init__(self, model: nn.Module, device: str = 'cuda'):
        self.model = model
        self.device = device
        self.probes = {}

    def create_labels_piece_position(self, fens: List[str], target_square: int) -> torch.Tensor:
        """
        Create labels: which piece is on target_square?
        Labels: 0-11 for pieces, 12 for empty
        """
        labels = []
        for fen in fens:
            board = chess.Board(fen)
            piece = board.piece_at(target_square)

            if piece is None:
                label = 12  # Empty square
            else:
                piece_id = {
                    chess.PAWN: 0,
                    chess.KNIGHT: 1,
                    chess.BISHOP: 2,
                    chess.ROOK: 3,
                    chess.QUEEN: 4,
                    chess.KING: 5
                }[piece.piece_type]

                if piece.color == chess.BLACK:
                    piece_id += 6

                label = piece_id

            labels.append(label)

        return torch.tensor(labels, dtype=torch.long)

    def create_labels_material_balance(self, fens: List[str]) -> torch.Tensor:
        """
        Create labels: material balance category.
        0: equal, 1: white up, 2: black up
        """
        labels = []
        for fen in fens:
            board = chess.Board(fen)

            # Count material
            piece_values = {
                chess.PAWN: 1,
                chess.KNIGHT: 3,
                chess.BISHOP: 3,
                chess.ROOK: 5,
                chess.QUEEN: 9,
                chess.KING: 0
            }

            white_material = 0
            black_material = 0

            for square in chess.SQUARES:
                piece = board.piece_at(square)
                if piece:
                    value = piece_values[piece.piece_type]
                    if piece.color == chess.WHITE:
                        white_material += value
                    else:
                        black_material += value

            diff = white_material - black_material
            if diff > 2:
                label = 1  # White up
            elif diff < -2:
                label = 2  # Black up
            else:
                label = 0  # Equal

            labels.append(label)

        return torch.tensor(labels, dtype=torch.long)

    def create_labels_check_detection(self, fens: List[str]) -> torch.Tensor:
        """
        Create labels: is the current side in check?
        0: not in check, 1: in check
        """
        labels = []
        for fen in fens:
            board = chess.Board(fen)
            labels.append(1 if board.is_check() else 0)

        return torch.tensor(labels, dtype=torch.long)

    def extract_activations(
        self,
        data_loader,
        layer_idx: int
    ) -> Tuple[torch.Tensor, List[str]]:
        """
        Extract activations from a specific layer.

        Returns:
            activations: (N, hidden_dim)
            fens: List of FEN strings
        """
        self.model.eval()
        self.model.enable_activation_storage()

        all_activations = []
        all_fens = []

        with torch.no_grad():
            for batch in data_loader:
                # Forward pass
                self.model(
                    pieces=batch['pieces'].to(self.device),
                    squares=batch['squares'].to(self.device),
                    mask=batch['mask'].to(self.device)
                )

                # Get activations from target layer
                layer_acts = self.model.get_layer_activations()[layer_idx]

                # Pool over sequence dimension
                mask_expanded = batch['mask'].unsqueeze(-1).to(self.device)
                pooled_acts = (layer_acts * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1)

                all_activations.append(pooled_acts.cpu())
                all_fens.extend(batch['fen'])

        self.model.disable_activation_storage()

        return torch.cat(all_activations, dim=0), all_fens

    def run_probe_suite(
        self,
        train_loader,
        val_loader,
        layer_indices: Optional[List[int]] = None
    ) -> Dict[str, Dict[str, float]]:
        """
        Run full suite of probes across layers.

        Returns:
            results: Dict[probe_name][layer_idx] = accuracy
        """
        if layer_indices is None:
            layer_indices = list(range(self.model.num_layers))

        results = {}

        for layer_idx in layer_indices:
            print(f"\nProbing layer {layer_idx}...")

            # Extract activations
            train_acts, train_fens = self.extract_activations(train_loader, layer_idx)
            val_acts, val_fens = self.extract_activations(val_loader, layer_idx)

            # Run probes
            probes_config = {
                'check_detection': (
                    self.create_labels_check_detection,
                    2  # Binary
                ),
                'material_balance': (
                    self.create_labels_material_balance,
                    3  # 3 classes
                ),
                'piece_on_e4': (
                    lambda fens: self.create_labels_piece_position(fens, chess.E4),
                    13  # 12 pieces + empty
                )
            }

            for probe_name, (label_fn, num_classes) in probes_config.items():
                print(f"  Running {probe_name} probe...")

                # Create labels
                train_labels = label_fn(train_fens).to(self.device)
                val_labels = label_fn(val_fens).to(self.device)

                # Train probe
                probe = LinearProbe(train_acts.shape[1], num_classes).to(self.device)
                probe_results = probe.fit_and_evaluate(
                    train_acts.to(self.device),
                    train_labels,
                    val_acts.to(self.device),
                    val_labels
                )

                # Store results
                key = f"{probe_name}_layer_{layer_idx}"
                results[key] = probe_results

                print(f"    Val accuracy: {probe_results['val_accuracy']:.3f}")

        return results


class StructuredPruning:
    """
    Structured pruning for transformers.

    Methods:
    1. Attention head pruning (remove entire heads)
    2. FFN neuron pruning (remove neurons)
    3. Layer pruning (remove entire layers)
    """

    def __init__(self, model: nn.Module):
        self.model = model

    def compute_head_importance(
        self,
        data_loader,
        num_samples: int = 1000
    ) -> torch.Tensor:
        """
        Compute importance score for each attention head.

        Uses gradient-based importance: how much does head contribute to loss?

        Returns:
            importance: (num_layers, num_heads) tensor
        """
        self.model.eval()

        importance = torch.zeros(
            self.model.num_layers,
            self.model.transformer.layers[0].self_attn.num_heads
        )

        samples_seen = 0

        for batch in data_loader:
            if samples_seen >= num_samples:
                break

            # Forward pass with gradient
            outputs = self.model(
                pieces=batch['pieces'],
                squares=batch['squares'],
                mask=batch['mask']
            )

            # Dummy loss (just for gradient computation)
            loss = outputs['policy_logits'].sum() + outputs['value'].sum()
            loss.backward()

            # Accumulate head gradients
            for layer_idx, layer in enumerate(self.model.transformer.layers):
                # Get attention weight gradients
                if hasattr(layer.self_attn, 'in_proj_weight') and \
                   layer.self_attn.in_proj_weight.grad is not None:
                    grad = layer.self_attn.in_proj_weight.grad
                    # Average gradient magnitude per head
                    num_heads = layer.self_attn.num_heads
                    head_dim = grad.shape[1] // num_heads

                    for head_idx in range(num_heads):
                        start_idx = head_idx * head_dim
                        end_idx = (head_idx + 1) * head_dim
                        head_grad = grad[:, start_idx:end_idx]
                        importance[layer_idx, head_idx] += head_grad.abs().mean().item()

            samples_seen += batch['pieces'].shape[0]
            self.model.zero_grad()

        return importance / samples_seen

    def prune_attention_heads(
        self,
        keep_ratio: float = 0.5,
        importance_scores: Optional[torch.Tensor] = None
    ):
        """
        Prune attention heads based on importance.

        Args:
            keep_ratio: Fraction of heads to keep (0.5 = remove 50%)
            importance_scores: Pre-computed importance (num_layers, num_heads)
        """
        if importance_scores is None:
            raise ValueError("Must provide importance_scores")

        # Determine which heads to keep
        total_heads = importance_scores.numel()
        num_keep = int(total_heads * keep_ratio)

        # Flatten and get top-k
        flat_importance = importance_scores.flatten()
        _, top_indices = torch.topk(flat_importance, num_keep)

        # Convert back to (layer, head) indices
        num_heads_per_layer = importance_scores.shape[1]
        keep_mask = torch.zeros_like(importance_scores, dtype=torch.bool)

        for idx in top_indices:
            layer_idx = idx // num_heads_per_layer
            head_idx = idx % num_heads_per_layer
            keep_mask[layer_idx, head_idx] = True

        print(f"Keeping {keep_mask.sum().item()} / {total_heads} attention heads")

        # TODO: Actually prune the model (requires modifying attention modules)
        # For now, just return the mask
        return keep_mask

    def prune_ffn_neurons(
        self,
        keep_ratio: float = 0.7
    ):
        """
        Prune FFN neurons based on activation variance.

        Args:
            keep_ratio: Fraction of neurons to keep
        """
        # TODO: Implement FFN pruning
        pass

    def merge_layers(
        self,
        layer_pairs: List[Tuple[int, int]]
    ):
        """
        Merge similar adjacent layers.

        Args:
            layer_pairs: List of (layer_i, layer_j) to merge
        """
        # TODO: Implement layer merging
        pass


if __name__ == "__main__":
    print("Interpretability tools for Direction A")
    print("Run probing suite after training model")
