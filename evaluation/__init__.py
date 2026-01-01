"""Evaluation and benchmarking for chess models."""

from .play_games import play_game, play_tournament
from .elo_calculator import compute_elo

__all__ = ['play_game', 'play_tournament', 'compute_elo']
