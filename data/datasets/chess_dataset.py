"""
PyTorch datasets for loading chess training data.
"""

import torch
import chess
import json
import h5py
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from torch.utils.data import Dataset


class ChessPositionEncoder:
    """
    Encodes chess positions as tensors for neural network input.

    Supports multiple encoding schemes:
    - piece_centric: Each piece as token with positional encoding
    - board_plane: 8x8 planes for each piece type (12 planes total)
    - fen_tokens: Tokenized FEN string
    """

    # Piece to index mapping
    PIECE_TO_IDX = {
        chess.PAWN: 0,
        chess.KNIGHT: 1,
        chess.BISHOP: 2,
        chess.ROOK: 3,
        chess.QUEEN: 4,
        chess.KING: 5
    }

    def __init__(self, encoding_type: str = "piece_centric"):
        self.encoding_type = encoding_type

    def encode_piece_centric(self, board: chess.Board) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode position as sequence of (piece_id, square_id) pairs.

        Returns:
            pieces: (max_pieces,) tensor of piece IDs [0-11] (6 types × 2 colors)
            squares: (max_pieces,) tensor of square IDs [0-63]
            mask: (max_pieces,) tensor indicating valid pieces (1) vs padding (0)
        """
        max_pieces = 32
        pieces = []
        squares = []

        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece is not None:
                # Piece ID: 0-5 for white, 6-11 for black
                piece_id = self.PIECE_TO_IDX[piece.piece_type]
                if piece.color == chess.BLACK:
                    piece_id += 6

                pieces.append(piece_id)
                squares.append(square)

        # Pad to max_pieces
        num_pieces = len(pieces)
        mask = [1] * num_pieces + [0] * (max_pieces - num_pieces)
        pieces = pieces + [0] * (max_pieces - num_pieces)
        squares = squares + [0] * (max_pieces - num_pieces)

        return (
            torch.tensor(pieces, dtype=torch.long),
            torch.tensor(squares, dtype=torch.long),
            torch.tensor(mask, dtype=torch.float32)
        )

    def encode_board_planes(self, board: chess.Board) -> torch.Tensor:
        """
        Encode position as 12 8x8 planes (one per piece type/color).

        Returns:
            tensor: (12, 8, 8) binary planes
        """
        planes = torch.zeros(12, 8, 8, dtype=torch.float32)

        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece is not None:
                piece_id = self.PIECE_TO_IDX[piece.piece_type]
                if piece.color == chess.BLACK:
                    piece_id += 6

                rank = chess.square_rank(square)
                file = chess.square_file(square)
                planes[piece_id, rank, file] = 1.0

        return planes

    def encode_nnue_style(self, board: chess.Board) -> torch.Tensor:
        """
        Encode for NNUE-style networks: piece-square indices.

        Returns:
            tensor: (768,) binary vector for piece-square combinations
                   12 piece types × 64 squares = 768 features
        """
        features = torch.zeros(768, dtype=torch.float32)

        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece is not None:
                piece_id = self.PIECE_TO_IDX[piece.piece_type]
                if piece.color == chess.BLACK:
                    piece_id += 6

                # Feature index: piece_id * 64 + square
                feature_idx = piece_id * 64 + square
                features[feature_idx] = 1.0

        return features

    def encode(self, board: chess.Board) -> Dict[str, torch.Tensor]:
        """Encode position based on encoding type."""
        if self.encoding_type == "piece_centric":
            pieces, squares, mask = self.encode_piece_centric(board)
            return {
                'pieces': pieces,
                'squares': squares,
                'mask': mask
            }
        elif self.encoding_type == "board_planes":
            return {
                'planes': self.encode_board_planes(board)
            }
        elif self.encoding_type == "nnue":
            return {
                'features': self.encode_nnue_style(board)
            }
        else:
            raise ValueError(f"Unknown encoding type: {self.encoding_type}")


class MoveEncoder:
    """Encode chess moves as indices."""

    @staticmethod
    def move_to_index(move: chess.Move) -> int:
        """
        Encode move as index in [0, 4095].
        Index = from_square * 64 + to_square
        """
        return move.from_square * 64 + move.to_square

    @staticmethod
    def index_to_move(index: int) -> Tuple[int, int]:
        """Decode move index to (from_square, to_square)."""
        from_square = index // 64
        to_square = index % 64
        return from_square, to_square

    @staticmethod
    def encode_legal_moves_mask(board: chess.Board) -> torch.Tensor:
        """
        Create binary mask of legal moves.

        Returns:
            tensor: (4096,) binary mask where 1 = legal move
        """
        mask = torch.zeros(4096, dtype=torch.float32)

        for move in board.legal_moves:
            move_idx = MoveEncoder.move_to_index(move)
            mask[move_idx] = 1.0

        return mask


class ChessDataset(Dataset):
    """PyTorch dataset for chess positions."""

    def __init__(
        self,
        data_path: str,
        encoding_type: str = "piece_centric",
        max_samples: Optional[int] = None
    ):
        """
        Args:
            data_path: Path to .jsonl or .h5 file
            encoding_type: Position encoding scheme
            max_samples: Maximum number of samples to load (None = all)
        """
        self.data_path = Path(data_path)
        self.encoding_type = encoding_type
        self.encoder = ChessPositionEncoder(encoding_type)
        self.move_encoder = MoveEncoder()

        self.data = self._load_data(max_samples)

    def _load_data(self, max_samples: Optional[int]) -> List[Dict]:
        """Load data from file."""
        data = []

        if self.data_path.suffix == '.jsonl':
            with open(self.data_path, 'r') as f:
                for i, line in enumerate(f):
                    if max_samples and i >= max_samples:
                        break
                    data.append(json.loads(line))

        elif self.data_path.suffix == '.h5':
            with h5py.File(self.data_path, 'r') as f:
                num_samples = len(f['fens'])
                if max_samples:
                    num_samples = min(num_samples, max_samples)

                for i in range(num_samples):
                    data.append({
                        'fen': f['fens'][i].decode(),
                        'best_move': f['best_moves'][i].decode(),
                        'eval': float(f['evals'][i]),
                        'action_values': json.loads(f['action_values'][i].decode())
                    })
        else:
            raise ValueError(f"Unsupported file format: {self.data_path.suffix}")

        return data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single training example."""
        item = self.data[idx]

        # Parse position
        board = chess.Board(item['fen'])

        # Encode position
        position_encoding = self.encoder.encode(board)

        # Encode best move
        best_move = chess.Move.from_uci(item['best_move'])
        move_idx = self.move_encoder.move_to_index(best_move)

        # Legal moves mask
        legal_mask = self.move_encoder.encode_legal_moves_mask(board)

        # Normalize evaluation to [-1, 1]
        eval_normalized = np.tanh(item['eval'] / 1000.0)

        # Build action values tensor
        action_values = torch.zeros(4096, dtype=torch.float32)
        for move_uci, cp_value in item['action_values'].items():
            move = chess.Move.from_uci(move_uci)
            move_idx_av = self.move_encoder.move_to_index(move)
            # Normalize to [-1, 1]
            action_values[move_idx_av] = np.tanh(cp_value / 1000.0)

        return {
            **position_encoding,  # Position encoding (varies by type)
            'best_move': torch.tensor(move_idx, dtype=torch.long),
            'eval': torch.tensor(eval_normalized, dtype=torch.float32),
            'legal_mask': legal_mask,
            'action_values': action_values,
            'fen': item['fen']  # Keep for debugging
        }


def create_dataloaders(
    train_path: str,
    val_path: Optional[str] = None,
    batch_size: int = 256,
    encoding_type: str = "piece_centric",
    num_workers: int = 4,
    max_train_samples: Optional[int] = None,
    max_val_samples: Optional[int] = None
):
    """
    Create train and validation dataloaders.

    Args:
        train_path: Path to training data
        val_path: Path to validation data (optional)
        batch_size: Batch size
        encoding_type: Position encoding scheme
        num_workers: Number of dataloader workers
        max_train_samples: Limit training samples
        max_val_samples: Limit validation samples

    Returns:
        (train_loader, val_loader) or just train_loader if val_path is None
    """
    from torch.utils.data import DataLoader

    train_dataset = ChessDataset(
        train_path,
        encoding_type=encoding_type,
        max_samples=max_train_samples
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    if val_path:
        val_dataset = ChessDataset(
            val_path,
            encoding_type=encoding_type,
            max_samples=max_val_samples
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )

        return train_loader, val_loader

    return train_loader


if __name__ == "__main__":
    # Test encoding
    board = chess.Board()
    encoder = ChessPositionEncoder("piece_centric")

    encoding = encoder.encode(board)
    print("Piece-centric encoding:")
    print(f"  Pieces shape: {encoding['pieces'].shape}")
    print(f"  Squares shape: {encoding['squares'].shape}")
    print(f"  Mask shape: {encoding['mask'].shape}")

    encoder = ChessPositionEncoder("board_planes")
    encoding = encoder.encode(board)
    print("\nBoard planes encoding:")
    print(f"  Planes shape: {encoding['planes'].shape}")

    encoder = ChessPositionEncoder("nnue")
    encoding = encoder.encode(board)
    print("\nNNUE encoding:")
    print(f"  Features shape: {encoding['features'].shape}")
