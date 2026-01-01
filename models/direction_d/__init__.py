"""
Direction D: Parallel Verification Architecture

Maps DeepMind's parallel thinking approach to chess:
- Spawn multiple candidate move analyzers
- Cross-verify via attention
- Aggregate with learned confidence
"""

from .parallel_verification import ParallelVerificationChess, BranchAnalyzer

__all__ = ['ParallelVerificationChess', 'BranchAnalyzer']
