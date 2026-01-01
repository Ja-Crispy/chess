"""
Play games with trained chess models against Stockfish or other engines.
"""

import chess
import chess.engine
import torch
import torch.nn.functional as F
from typing import Dict, Optional, List, Tuple
from pathlib import Path
import json
from tqdm import tqdm

from data.datasets import ChessPositionEncoder, MoveEncoder


class ChessModelPlayer:
    """Wrapper for chess model to play games."""

    def __init__(
        self,
        model: torch.nn.Module,
        encoding_type: str = 'piece_centric',
        device: str = 'cuda',
        temperature: float = 1.0
    ):
        self.model = model.to(device)
        self.model.eval()
        self.encoder = ChessPositionEncoder(encoding_type)
        self.move_encoder = MoveEncoder()
        self.device = device
        self.temperature = temperature
        self.encoding_type = encoding_type

    @torch.no_grad()
    def get_move(self, board: chess.Board) -> chess.Move:
        """
        Get best move from model for current position.

        Args:
            board: Current chess board

        Returns:
            Selected move
        """
        # Encode position
        encoding = self.encoder.encode(board)

        # Move to device and add batch dimension
        if self.encoding_type == 'piece_centric':
            pieces = encoding['pieces'].unsqueeze(0).to(self.device)
            squares = encoding['squares'].unsqueeze(0).to(self.device)
            mask = encoding['mask'].unsqueeze(0).to(self.device)

            outputs = self.model(pieces=pieces, squares=squares, mask=mask)

        elif self.encoding_type == 'nnue':
            features = encoding['features'].unsqueeze(0).to(self.device)
            outputs = self.model(features=features)

        else:
            raise ValueError(f"Unknown encoding type: {self.encoding_type}")

        # Get policy logits
        policy_logits = outputs['policy_logits'].squeeze(0)  # (4096,)

        # Mask illegal moves
        legal_moves = list(board.legal_moves)
        legal_mask = torch.zeros(4096, device=self.device)

        for move in legal_moves:
            move_idx = self.move_encoder.move_to_index(move)
            legal_mask[move_idx] = 1.0

        policy_logits = policy_logits.masked_fill(legal_mask == 0, -1e9)

        # Apply temperature
        policy_logits = policy_logits / self.temperature

        # Sample move (or take argmax for greedy)
        if self.temperature > 0:
            probs = F.softmax(policy_logits, dim=0)
            move_idx = torch.multinomial(probs, 1).item()
        else:
            move_idx = policy_logits.argmax().item()

        # Convert to move
        from_square, to_square = self.move_encoder.index_to_move(move_idx)

        # Find matching legal move
        for move in legal_moves:
            if move.from_square == from_square and move.to_square == to_square:
                return move

        # Fallback: return first legal move (shouldn't happen)
        return legal_moves[0] if legal_moves else None


def play_game(
    player1: ChessModelPlayer,
    player2,  # ChessModelPlayer or chess.engine.SimpleEngine
    max_moves: int = 200,
    verbose: bool = False
) -> Dict:
    """
    Play a single game between two players.

    Args:
        player1: First player (white)
        player2: Second player (black) - can be model or Stockfish
        max_moves: Maximum number of moves before draw
        verbose: Print game moves

    Returns:
        Dictionary with game result and metadata
    """
    board = chess.Board()
    moves = []

    for move_num in range(max_moves):
        if board.is_game_over():
            break

        # Determine current player
        if board.turn == chess.WHITE:
            current_player = player1
            player_name = "Player1"
        else:
            current_player = player2
            player_name = "Player2"

        # Get move
        if isinstance(current_player, ChessModelPlayer):
            move = current_player.get_move(board)
        elif isinstance(current_player, chess.engine.SimpleEngine):
            # Stockfish engine
            result = current_player.play(
                board,
                chess.engine.Limit(depth=10)
            )
            move = result.move
        else:
            raise ValueError(f"Unknown player type: {type(current_player)}")

        if move is None:
            break

        # Make move
        if verbose:
            print(f"{move_num + 1}. {player_name}: {move.uci()}")

        moves.append(move.uci())
        board.push(move)

    # Determine result
    if board.is_checkmate():
        winner = "white" if board.turn == chess.BLACK else "black"
        result = "1-0" if winner == "white" else "0-1"
    elif board.is_stalemate() or board.is_insufficient_material() or \
         board.is_fifty_moves() or board.is_repetition():
        winner = "draw"
        result = "1/2-1/2"
    else:
        # Max moves reached
        winner = "draw"
        result = "1/2-1/2"

    return {
        'result': result,
        'winner': winner,
        'moves': moves,
        'num_moves': len(moves),
        'final_fen': board.fen()
    }


def play_tournament(
    model_path: str,
    opponent: str = 'stockfish',
    opponent_path: str = '/usr/games/stockfish',
    opponent_depth: int = 10,
    num_games: int = 100,
    encoding_type: str = 'piece_centric',
    device: str = 'cuda',
    output_path: Optional[str] = None
) -> Dict:
    """
    Play a tournament between model and opponent.

    Args:
        model_path: Path to trained model checkpoint
        opponent: Opponent type ('stockfish' or 'random')
        opponent_path: Path to Stockfish binary
        opponent_depth: Stockfish search depth
        num_games: Number of games to play
        encoding_type: Position encoding type
        device: Device for model
        output_path: Optional path to save results

    Returns:
        Tournament results
    """
    # Load model
    print(f"Loading model from {model_path}...")
    checkpoint = torch.load(model_path, map_location=device)

    # Determine model type from checkpoint
    # TODO: Store model config in checkpoint
    from models.direction_a import ChessTransformer
    model = ChessTransformer()  # Default to transformer
    model.load_state_dict(checkpoint['model_state_dict'])

    player1 = ChessModelPlayer(
        model=model,
        encoding_type=encoding_type,
        device=device,
        temperature=0.1  # Low temperature for stronger play
    )

    # Create opponent
    if opponent == 'stockfish':
        print(f"Loading Stockfish from {opponent_path}...")
        opponent_engine = chess.engine.SimpleEngine.popen_uci(opponent_path)
        opponent_engine.configure({"Threads": 1})
        player2 = opponent_engine
    else:
        raise ValueError(f"Unknown opponent: {opponent}")

    # Play games
    results = {
        'wins': 0,
        'losses': 0,
        'draws': 0,
        'games': []
    }

    print(f"\nPlaying {num_games} games...")

    for game_num in tqdm(range(num_games)):
        # Alternate colors
        if game_num % 2 == 0:
            # Model plays white
            game_result = play_game(player1, player2)
            if game_result['winner'] == 'white':
                results['wins'] += 1
            elif game_result['winner'] == 'black':
                results['losses'] += 1
            else:
                results['draws'] += 1
        else:
            # Model plays black
            game_result = play_game(player2, player1)
            if game_result['winner'] == 'black':
                results['wins'] += 1
            elif game_result['winner'] == 'white':
                results['losses'] += 1
            else:
                results['draws'] += 1

        results['games'].append(game_result)

    # Close engine
    if opponent == 'stockfish':
        opponent_engine.quit()

    # Compute statistics
    total_games = results['wins'] + results['losses'] + results['draws']
    results['win_rate'] = results['wins'] / total_games
    results['loss_rate'] = results['losses'] / total_games
    results['draw_rate'] = results['draws'] / total_games

    print(f"\nTournament Results:")
    print(f"  Wins: {results['wins']}")
    print(f"  Losses: {results['losses']}")
    print(f"  Draws: {results['draws']}")
    print(f"  Win rate: {results['win_rate']:.2%}")

    # Save results
    if output_path:
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved results to {output_path}")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Play chess games with trained model")
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--opponent', type=str, default='stockfish')
    parser.add_argument('--opponent_path', type=str, default='/usr/games/stockfish')
    parser.add_argument('--opponent_depth', type=int, default=10)
    parser.add_argument('--num_games', type=int, default=100)
    parser.add_argument('--encoding_type', type=str, default='piece_centric')
    parser.add_argument('--output', type=str, default='tournament_results.json')

    args = parser.parse_args()

    results = play_tournament(
        model_path=args.model_path,
        opponent=args.opponent,
        opponent_path=args.opponent_path,
        opponent_depth=args.opponent_depth,
        num_games=args.num_games,
        encoding_type=args.encoding_type,
        output_path=args.output
    )
