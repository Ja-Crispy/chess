"""
Stockfish-based data generation for training chess models.

This module generates training data by:
1. Sampling diverse chess positions
2. Analyzing them with Stockfish at specified depth
3. Extracting evaluations, best moves, and action values
"""

import chess
import chess.engine
import chess.pgn
import numpy as np
import json
import h5py
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import random


class PositionSampler:
    """Generate diverse chess positions for training."""

    def __init__(self, seed: int = 42):
        self.rng = np.random.RandomState(seed)
        random.seed(seed)

    def sample_from_random_game(self, max_moves: int = 80) -> chess.Board:
        """Sample position from a random game."""
        board = chess.Board()
        num_moves = self.rng.randint(10, max_moves)

        for _ in range(num_moves):
            if board.is_game_over():
                break

            legal_moves = list(board.legal_moves)
            if not legal_moves:
                break

            # Weighted random: prefer captures and checks
            weights = []
            for move in legal_moves:
                weight = 1.0
                if board.is_capture(move):
                    weight += 2.0
                board.push(move)
                if board.is_check():
                    weight += 1.0
                board.pop()
                weights.append(weight)

            weights = np.array(weights)
            weights = weights / weights.sum()

            move = self.rng.choice(legal_moves, p=weights)
            board.push(move)

        return board

    def sample_from_opening_book(self) -> chess.Board:
        """Sample position from common opening lines."""
        # Common opening moves for diversity
        openings = [
            ["e2e4", "e7e5", "g1f3", "b8c6"],  # Italian/Spanish
            ["e2e4", "c7c5"],  # Sicilian
            ["d2d4", "d7d5"],  # Queen's Gambit
            ["d2d4", "g8f6", "c2c4", "e7e6"],  # Nimzo-Indian
            ["e2e4", "e7e5", "g1f3", "b8c6", "f1b5"],  # Ruy Lopez
            ["g1f3", "d7d5", "c2c4"],  # Reti
        ]

        board = chess.Board()
        opening = random.choice(openings)

        for move_uci in opening:
            move = chess.Move.from_uci(move_uci)
            if move in board.legal_moves:
                board.push(move)

        # Play a few more random moves
        for _ in range(self.rng.randint(0, 10)):
            if board.is_game_over():
                break
            legal_moves = list(board.legal_moves)
            if legal_moves:
                board.push(random.choice(legal_moves))

        return board

    def sample_endgame(self) -> chess.Board:
        """Generate endgame positions."""
        board = chess.Board()
        board.clear()

        # Place kings randomly (ensuring legal position)
        while True:
            wk_square = self.rng.randint(0, 64)
            bk_square = self.rng.randint(0, 64)

            if abs(chess.square_file(wk_square) - chess.square_file(bk_square)) > 1 or \
               abs(chess.square_rank(wk_square) - chess.square_rank(bk_square)) > 1:
                board.set_piece_at(wk_square, chess.Piece(chess.KING, chess.WHITE))
                board.set_piece_at(bk_square, chess.Piece(chess.KING, chess.BLACK))
                break

        # Add random pieces (2-8 total pieces including kings)
        num_pieces = self.rng.randint(2, 6)
        piece_types = [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT, chess.PAWN]

        for _ in range(num_pieces):
            square = self.rng.randint(0, 64)
            if board.piece_at(square) is None:
                piece_type = self.rng.choice(piece_types)
                color = chess.WHITE if self.rng.random() > 0.5 else chess.BLACK

                # Avoid pawns on back ranks
                if piece_type == chess.PAWN and (chess.square_rank(square) in [0, 7]):
                    continue

                board.set_piece_at(square, chess.Piece(piece_type, color))

        board.turn = chess.WHITE if self.rng.random() > 0.5 else chess.BLACK

        # Ensure position is legal
        if board.is_valid():
            return board
        else:
            # Fallback to simple K+Q vs K
            board = chess.Board()
            board.clear()
            board.set_piece_at(chess.E1, chess.Piece(chess.KING, chess.WHITE))
            board.set_piece_at(chess.E8, chess.Piece(chess.KING, chess.BLACK))
            board.set_piece_at(chess.D1, chess.Piece(chess.QUEEN, chess.WHITE))
            return board

    def sample(self, position_type: str = "mixed") -> chess.Board:
        """Sample a position based on type."""
        if position_type == "random_game":
            return self.sample_from_random_game()
        elif position_type == "opening":
            return self.sample_from_opening_book()
        elif position_type == "endgame":
            return self.sample_endgame()
        elif position_type == "mixed":
            p = self.rng.random()
            if p < 0.5:
                return self.sample_from_random_game()
            elif p < 0.75:
                return self.sample_from_opening_book()
            else:
                return self.sample_endgame()
        else:
            raise ValueError(f"Unknown position type: {position_type}")


class StockfishGenerator:
    """Generate training data using Stockfish engine."""

    def __init__(
        self,
        engine_path: str = "/usr/games/stockfish",
        depth: int = 20,
        multipv: int = 5,
        time_limit: Optional[float] = None
    ):
        self.engine_path = engine_path
        self.depth = depth
        self.multipv = multipv
        self.time_limit = time_limit

    def analyze_position(
        self,
        board: chess.Board,
        engine: chess.engine.SimpleEngine
    ) -> Dict:
        """Analyze a single position with Stockfish."""
        try:
            # Get multiple principal variations
            limit = chess.engine.Limit(depth=self.depth, time=self.time_limit)
            info = engine.analyse(
                board,
                limit,
                multipv=min(self.multipv, board.legal_moves.count())
            )

            # Extract action values
            action_values = {}
            for pv_info in info:
                if 'pv' in pv_info and len(pv_info['pv']) > 0:
                    move = pv_info['pv'][0]
                    score = pv_info['score'].relative

                    # Convert to centipawns
                    if score.is_mate():
                        # Mate in N moves
                        mate_in = score.mate()
                        cp_value = 10000 if mate_in > 0 else -10000
                    else:
                        cp_value = score.score()

                    action_values[move.uci()] = cp_value

            # Best move and evaluation
            best_move = info[0]['pv'][0].uci() if info[0]['pv'] else None
            best_eval = action_values.get(best_move, 0) if best_move else 0

            return {
                'fen': board.fen(),
                'action_values': action_values,
                'best_move': best_move,
                'eval': best_eval,
                'legal_moves': [m.uci() for m in board.legal_moves]
            }

        except Exception as e:
            print(f"Error analyzing position: {e}")
            return None

    def generate_batch(
        self,
        num_positions: int,
        position_type: str = "mixed",
        seed: int = 42
    ) -> List[Dict]:
        """Generate a batch of annotated positions."""
        sampler = PositionSampler(seed=seed)
        results = []

        with chess.engine.SimpleEngine.popen_uci(self.engine_path) as engine:
            engine.configure({"Threads": 1})

            for _ in tqdm(range(num_positions), desc="Generating positions"):
                board = sampler.sample(position_type)

                # Skip if game over
                if board.is_game_over():
                    continue

                result = self.analyze_position(board, engine)
                if result is not None:
                    results.append(result)

        return results


def generate_dataset(
    output_path: str,
    num_positions: int = 10000,
    depth: int = 20,
    multipv: int = 5,
    engine_path: str = "/usr/games/stockfish",
    position_type: str = "mixed",
    num_workers: int = 1,
    seed: int = 42
):
    """
    Generate a complete dataset of Stockfish-annotated positions.

    Args:
        output_path: Path to save the dataset (.jsonl or .h5)
        num_positions: Number of positions to generate
        depth: Stockfish search depth
        multipv: Number of principal variations to analyze
        engine_path: Path to Stockfish binary
        position_type: Type of positions to generate (mixed, random_game, opening, endgame)
        num_workers: Number of parallel workers
        seed: Random seed
    """
    generator = StockfishGenerator(
        engine_path=engine_path,
        depth=depth,
        multipv=multipv
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if num_workers == 1:
        # Single-threaded generation
        results = generator.generate_batch(num_positions, position_type, seed)
    else:
        # Multi-threaded generation
        positions_per_worker = num_positions // num_workers
        results = []

        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = []
            for i in range(num_workers):
                worker_seed = seed + i
                future = executor.submit(
                    generator.generate_batch,
                    positions_per_worker,
                    position_type,
                    worker_seed
                )
                futures.append(future)

            for future in tqdm(as_completed(futures), total=num_workers, desc="Workers"):
                results.extend(future.result())

    # Save results
    if output_path.suffix == '.jsonl':
        with open(output_path, 'w') as f:
            for result in results:
                f.write(json.dumps(result) + '\n')
        print(f"Saved {len(results)} positions to {output_path}")

    elif output_path.suffix == '.h5':
        # Save in HDF5 format for efficient loading
        with h5py.File(output_path, 'w') as f:
            f.create_dataset('fens', data=[r['fen'].encode() for r in results])
            f.create_dataset('best_moves', data=[r['best_move'].encode() for r in results])
            f.create_dataset('evals', data=[r['eval'] for r in results])

            # Store action values as JSON strings (variable length)
            f.create_dataset(
                'action_values',
                data=[json.dumps(r['action_values']).encode() for r in results]
            )
        print(f"Saved {len(results)} positions to {output_path}")

    else:
        raise ValueError(f"Unsupported file format: {output_path.suffix}")

    # Print statistics
    evals = [r['eval'] for r in results]
    print(f"\nDataset Statistics:")
    print(f"  Total positions: {len(results)}")
    print(f"  Eval mean: {np.mean(evals):.1f} cp")
    print(f"  Eval std: {np.std(evals):.1f} cp")
    print(f"  Eval range: [{np.min(evals):.1f}, {np.max(evals):.1f}] cp")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate Stockfish-annotated chess dataset")
    parser.add_argument("--output", type=str, required=True, help="Output path (.jsonl or .h5)")
    parser.add_argument("--num_positions", type=int, default=10000, help="Number of positions")
    parser.add_argument("--depth", type=int, default=20, help="Stockfish depth")
    parser.add_argument("--multipv", type=int, default=5, help="Number of PVs")
    parser.add_argument("--engine_path", type=str, default="/usr/games/stockfish")
    parser.add_argument("--position_type", type=str, default="mixed",
                       choices=["mixed", "random_game", "opening", "endgame"])
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    generate_dataset(
        output_path=args.output,
        num_positions=args.num_positions,
        depth=args.depth,
        multipv=args.multipv,
        engine_path=args.engine_path,
        position_type=args.position_type,
        num_workers=args.num_workers,
        seed=args.seed
    )
