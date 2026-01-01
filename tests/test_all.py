"""
Comprehensive test suite for all chess model components.
"""

import sys
import traceback
from pathlib import Path

# Test results storage
test_results = {
    'passed': [],
    'failed': []
}


def test(name):
    """Decorator to register and run tests."""
    def decorator(func):
        def wrapper():
            try:
                print(f"\n{'='*60}")
                print(f"Testing: {name}")
                print('='*60)
                func()
                print(f"✓ PASSED: {name}")
                test_results['passed'].append(name)
                return True
            except Exception as e:
                print(f"✗ FAILED: {name}")
                print(f"Error: {e}")
                traceback.print_exc()
                test_results['failed'].append((name, str(e)))
                return False
        return wrapper
    return decorator


@test("Import core modules")
def test_imports():
    """Test all module imports."""
    import torch
    import chess
    import numpy as np
    import yaml

    print(f"  PyTorch: {torch.__version__}")
    print(f"  python-chess: {chess.__version__}")
    print(f"  NumPy: {np.__version__}")
    print(f"  CUDA available: {torch.cuda.is_available()}")


@test("Chess position encoder")
def test_position_encoder():
    """Test position encoding."""
    import chess
    from data.datasets.chess_dataset import ChessPositionEncoder

    board = chess.Board()

    # Test piece-centric encoding
    encoder = ChessPositionEncoder("piece_centric")
    encoding = encoder.encode(board)
    print(f"  Piece-centric shapes:")
    print(f"    pieces: {encoding['pieces'].shape}")
    print(f"    squares: {encoding['squares'].shape}")
    print(f"    mask: {encoding['mask'].shape}")
    assert encoding['pieces'].shape[0] == 32
    assert encoding['squares'].shape[0] == 32

    # Test board planes encoding
    encoder = ChessPositionEncoder("board_planes")
    encoding = encoder.encode(board)
    print(f"  Board planes shape: {encoding['planes'].shape}")
    assert encoding['planes'].shape == (12, 8, 8)

    # Test NNUE encoding
    encoder = ChessPositionEncoder("nnue")
    encoding = encoder.encode(board)
    print(f"  NNUE features shape: {encoding['features'].shape}")
    assert encoding['features'].shape == (768,)


@test("Move encoder")
def test_move_encoder():
    """Test move encoding."""
    import chess
    from data.datasets.chess_dataset import MoveEncoder

    encoder = MoveEncoder()

    # Test move encoding
    move = chess.Move.from_uci("e2e4")
    idx = encoder.move_to_index(move)
    print(f"  e2e4 -> index {idx}")
    assert 0 <= idx < 4096

    # Test legal moves mask
    board = chess.Board()
    mask = encoder.encode_legal_moves_mask(board)
    print(f"  Legal moves mask shape: {mask.shape}")
    print(f"  Num legal moves: {mask.sum().item()}")
    assert mask.shape == (4096,)
    assert mask.sum() == 20  # Starting position has 20 legal moves


@test("Direction A: Transformer model")
def test_transformer():
    """Test transformer model."""
    import torch
    from models.direction_a import ChessTransformer

    model = ChessTransformer(
        hidden_dim=256,
        num_layers=4,
        num_heads=4,
        ffn_dim=512
    )

    print(f"  Model parameters: {model.count_parameters():,}")
    print(f"  Model size: {model.get_model_size_mb():.2f} MB")

    # Test forward pass
    batch_size = 2
    seq_len = 20

    pieces = torch.randint(0, 12, (batch_size, seq_len))
    squares = torch.randint(0, 64, (batch_size, seq_len))
    mask = torch.ones(batch_size, seq_len)

    outputs = model(pieces=pieces, squares=squares, mask=mask)

    print(f"  Output shapes:")
    print(f"    policy_logits: {outputs['policy_logits'].shape}")
    print(f"    value: {outputs['value'].shape}")

    assert outputs['policy_logits'].shape == (batch_size, 4096)
    assert outputs['value'].shape == (batch_size,)


@test("Direction B1: Dense NNUE")
def test_nnue():
    """Test NNUE model."""
    import torch
    from models.direction_b import DenseNNUE

    model = DenseNNUE(
        embedding_dim=64,
        hidden_layers=(256, 128, 32)
    )

    print(f"  Model parameters: {model.count_parameters():,}")
    print(f"  Model size: {model.get_model_size_mb():.2f} MB")

    # Test forward pass
    batch_size = 2
    features = torch.randint(0, 2, (batch_size, 768), dtype=torch.float32)

    outputs = model(features=features)

    print(f"  Output shapes:")
    print(f"    eval: {outputs['eval'].shape}")
    print(f"    value: {outputs['value'].shape}")

    assert outputs['eval'].shape == (batch_size,)
    assert outputs['value'].shape == (batch_size,)


@test("Direction B2: Hybrid NNUE")
def test_hybrid_engine():
    """Test hybrid NNUE + policy model."""
    import torch
    from models.direction_b import HybridEngine

    model = HybridEngine()

    print(f"  Total parameters: {model.eval_net.count_parameters() + model.policy_net.count_parameters():,}")

    # Test forward pass
    batch_size = 2
    features = torch.randint(0, 2, (batch_size, 768), dtype=torch.float32)

    outputs = model(features=features)

    print(f"  Output shapes:")
    print(f"    eval: {outputs['eval'].shape}")
    print(f"    value: {outputs['value'].shape}")
    print(f"    policy_logits: {outputs['policy_logits'].shape}")

    assert outputs['policy_logits'].shape == (batch_size, 4096)


@test("Direction C: MoE model")
def test_moe():
    """Test MoE model."""
    import torch
    from models.direction_c import ChessMoE

    model = ChessMoE(
        hidden_dim=128,
        num_trunk_layers=2,
        num_experts=4,
        ffn_dim=256,
        top_k=2
    )

    print(f"  Model parameters: {model.count_parameters():,}")
    print(f"  Num experts: {model.num_experts}")

    # Test forward pass
    batch_size = 2
    seq_len = 20

    pieces = torch.randint(0, 12, (batch_size, seq_len))
    squares = torch.randint(0, 64, (batch_size, seq_len))
    mask = torch.ones(batch_size, seq_len)

    outputs = model(
        pieces=pieces,
        squares=squares,
        mask=mask,
        return_routing_info=True
    )

    print(f"  Output shapes:")
    print(f"    policy_logits: {outputs['policy_logits'].shape}")
    print(f"    value: {outputs['value'].shape}")
    print(f"    routing_weights: {outputs['routing_weights'].shape}")

    assert outputs['routing_weights'].shape[2] == 4  # num_experts


@test("Direction D: Parallel verification")
def test_parallel_verification():
    """Test parallel verification model."""
    import torch
    from models.direction_d import ParallelVerificationChess

    model = ParallelVerificationChess(
        hidden_dim=128,
        num_encoder_layers=2,
        num_branches=3,
        num_heads=4
    )

    print(f"  Model parameters: {model.count_parameters():,}")
    print(f"  Num branches: {model.num_branches}")

    # Test forward pass
    batch_size = 2
    seq_len = 20

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

    print(f"  Output shapes:")
    print(f"    policy_logits: {outputs['policy_logits'].shape}")
    print(f"    value: {outputs['value'].shape}")
    print(f"    candidate_moves: {outputs['candidate_moves'].shape}")
    print(f"    branch_confidence: {outputs['branch_confidence'].shape}")

    assert outputs['candidate_moves'].shape == (batch_size, 3)
    assert outputs['branch_confidence'].shape == (batch_size, 3)


@test("Loss functions")
def test_losses():
    """Test loss functions."""
    import torch
    from training.losses import ChessLoss, MoELoss, ParallelVerificationLoss

    batch_size = 2

    # Test ChessLoss
    loss_fn = ChessLoss()
    outputs = {
        'policy_logits': torch.randn(batch_size, 4096),
        'value': torch.randn(batch_size)
    }
    batch = {
        'best_move': torch.randint(0, 4096, (batch_size,)),
        'eval': torch.randn(batch_size),
        'legal_mask': torch.ones(batch_size, 4096)
    }

    losses = loss_fn(outputs, batch)
    print(f"  ChessLoss components:")
    print(f"    loss: {losses['loss'].item():.4f}")
    print(f"    policy_loss: {losses['policy_loss'].item():.4f}")
    print(f"    value_loss: {losses['value_loss'].item():.4f}")

    assert 'loss' in losses

    # Test MoELoss
    loss_fn = MoELoss()
    outputs['routing_weights'] = torch.randn(batch_size, 2, 4)
    losses = loss_fn(outputs, batch)
    print(f"  MoELoss has balance_loss: {'balance_loss' in losses}")

    # Test ParallelVerificationLoss
    loss_fn = ParallelVerificationLoss()
    outputs['candidate_moves'] = torch.randint(0, 4096, (batch_size, 3))
    outputs['branch_confidence'] = torch.randn(batch_size, 3)
    losses = loss_fn(outputs, batch)
    print(f"  ParallelVerificationLoss has process_loss: {'process_loss' in losses}")


@test("Chess dataset")
def test_dataset():
    """Test chess dataset (mock data)."""
    import torch
    import json
    import tempfile
    from pathlib import Path
    from data.datasets import ChessDataset

    # Create mock data
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        for i in range(5):
            data = {
                'fen': 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1',
                'best_move': 'e2e4',
                'eval': 20,
                'action_values': {'e2e4': 20, 'd2d4': 15}
            }
            f.write(json.dumps(data) + '\n')
        temp_path = f.name

    try:
        # Test dataset
        dataset = ChessDataset(
            data_path=temp_path,
            encoding_type='piece_centric',
            max_samples=5
        )

        print(f"  Dataset size: {len(dataset)}")

        # Test __getitem__
        sample = dataset[0]
        print(f"  Sample keys: {list(sample.keys())}")
        print(f"  pieces shape: {sample['pieces'].shape}")
        print(f"  best_move: {sample['best_move'].item()}")

        assert len(dataset) == 5
        assert 'pieces' in sample
        assert 'best_move' in sample

    finally:
        Path(temp_path).unlink()


@test("Config loading")
def test_configs():
    """Test configuration loading."""
    import yaml
    from pathlib import Path

    config_dir = Path('configs')
    config_files = list(config_dir.glob('*.yaml'))

    print(f"  Found {len(config_files)} config files")

    for config_file in config_files:
        with open(config_file) as f:
            config = yaml.safe_load(f)
        print(f"  ✓ {config_file.name}: {config['model']['type']}")

        assert 'model' in config
        assert 'data' in config
        assert 'training' in config


@test("Elo calculator")
def test_elo_calculator():
    """Test Elo rating calculator."""
    from evaluation.elo_calculator import compute_elo, expected_score

    # Test expected score
    exp = expected_score(1500, 1500)
    print(f"  Expected score (equal): {exp:.3f}")
    assert abs(exp - 0.5) < 0.01

    # Test Elo computation
    results = compute_elo(
        wins=30,
        losses=60,
        draws=10,
        opponent_elo=3000
    )

    print(f"  Estimated Elo: {results['elo']:.0f}")
    print(f"  Actual score: {results['actual_score']:.2%}")
    print(f"  95% CI: [{results['confidence_interval'][0]:.0f}, {results['confidence_interval'][1]:.0f}]")

    assert results['elo'] < 3000  # Should be lower than opponent


def run_all_tests():
    """Run all tests and print summary."""
    print("\n" + "="*60)
    print("MINIMAL CHESS MODELS - COMPREHENSIVE TEST SUITE")
    print("="*60)

    # Run all tests
    test_imports()
    test_position_encoder()
    test_move_encoder()
    test_transformer()
    test_nnue()
    test_hybrid_engine()
    test_moe()
    test_parallel_verification()
    test_losses()
    test_dataset()
    test_configs()
    test_elo_calculator()

    # Print summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    print(f"\n✓ PASSED: {len(test_results['passed'])} tests")
    for test_name in test_results['passed']:
        print(f"  • {test_name}")

    if test_results['failed']:
        print(f"\n✗ FAILED: {len(test_results['failed'])} tests")
        for test_name, error in test_results['failed']:
            print(f"  • {test_name}: {error}")

    print("\n" + "="*60)
    total = len(test_results['passed']) + len(test_results['failed'])
    success_rate = len(test_results['passed']) / total * 100 if total > 0 else 0
    print(f"SUCCESS RATE: {success_rate:.1f}% ({len(test_results['passed'])}/{total})")
    print("="*60 + "\n")

    return len(test_results['failed']) == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
