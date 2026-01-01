"""
Elo rating calculator for chess models.
"""

import math
from typing import List, Dict


def expected_score(rating_a: float, rating_b: float) -> float:
    """
    Calculate expected score for player A against player B.

    Args:
        rating_a: Elo rating of player A
        rating_b: Elo rating of player B

    Returns:
        Expected score (0 to 1)
    """
    return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400.0))


def compute_elo(
    wins: int,
    losses: int,
    draws: int,
    opponent_elo: float = 3000,  # Stockfish depth 10 ~ 3000 Elo
    k_factor: float = 32.0,
    initial_elo: float = 1500.0
) -> Dict[str, float]:
    """
    Compute Elo rating based on game results.

    Args:
        wins: Number of wins
        losses: Number of losses
        draws: Number of draws
        opponent_elo: Opponent's Elo rating
        k_factor: K-factor for Elo calculation
        initial_elo: Starting Elo rating

    Returns:
        Dictionary with Elo rating and statistics
    """
    total_games = wins + losses + draws

    if total_games == 0:
        return {
            'elo': initial_elo,
            'expected_score': 0.5,
            'actual_score': 0.0,
            'confidence_interval': (initial_elo, initial_elo)
        }

    # Actual score
    actual_score = (wins + 0.5 * draws) / total_games

    # Expected score
    expected = expected_score(initial_elo, opponent_elo)

    # Compute Elo change
    elo_change = k_factor * (actual_score - expected)
    new_elo = initial_elo + elo_change

    # Iteratively refine (simple approach)
    for _ in range(10):
        expected = expected_score(new_elo, opponent_elo)
        elo_change = k_factor * (actual_score - expected)
        new_elo = new_elo + elo_change

    # Confidence interval (rough estimate)
    # Standard error of proportion
    se = math.sqrt(actual_score * (1 - actual_score) / total_games)
    se_elo = se * 400 * math.log(10)  # Convert to Elo

    ci_lower = new_elo - 1.96 * se_elo
    ci_upper = new_elo + 1.96 * se_elo

    return {
        'elo': new_elo,
        'expected_score': expected,
        'actual_score': actual_score,
        'confidence_interval': (ci_lower, ci_upper),
        'performance': actual_score - expected
    }


if __name__ == "__main__":
    # Example: Model with 30 wins, 60 losses, 10 draws vs Stockfish depth 10
    results = compute_elo(
        wins=30,
        losses=60,
        draws=10,
        opponent_elo=3000
    )

    print("Elo Calculation:")
    print(f"  Elo: {results['elo']:.0f}")
    print(f"  95% CI: [{results['confidence_interval'][0]:.0f}, "
          f"{results['confidence_interval'][1]:.0f}]")
    print(f"  Actual score: {results['actual_score']:.2%}")
    print(f"  Expected score: {results['expected_score']:.2%}")
    print(f"  Performance: {results['performance']:+.2%}")
