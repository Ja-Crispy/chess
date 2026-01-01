# Quick Start Guide

Get started with minimal chess models research in 5 minutes.

## 1. Install Dependencies

```bash
pip install -r requirements.txt
# or
pip install -e .
```

## 2. Generate Sample Training Data

```bash
# Generate 1,000 positions for quick testing
bash scripts/generate_sample_data.sh

# Or generate full dataset (10M positions)
python -m data.generators.stockfish_generator \
    --output data/stockfish_annotations/train.h5 \
    --num_positions 10000000 \
    --depth 20 \
    --multipv 5 \
    --num_workers 8
```

## 3. Train a Model

### Option A: Transformer (Direction A)

```bash
python -m training.train --config configs/direction_a.yaml
```

### Option B: Dense NNUE (Direction B1)

```bash
python -m training.train --config configs/direction_b1.yaml
```

### Option C: MoE (Direction C)

```bash
python -m training.train --config configs/direction_c.yaml
```

### Option D: Parallel Verification (Direction D)

```bash
python -m training.train --config configs/direction_d.yaml
```

### Train All Directions in Parallel

```bash
bash scripts/train_all_directions.sh
```

## 4. Evaluate Model

```bash
# Play 100 games against Stockfish
python -m evaluation.play_games \
    --model_path checkpoints/direction_a/best_model.pt \
    --opponent stockfish \
    --opponent_depth 10 \
    --num_games 100 \
    --output results/tournament.json
```

## 5. Analyze Results

### Check Elo Rating

```python
from evaluation.elo_calculator import compute_elo

results = compute_elo(
    wins=30,
    losses=60,
    draws=10,
    opponent_elo=3000  # Stockfish depth 10
)

print(f"Estimated Elo: {results['elo']:.0f}")
```

### Mechanistic Interpretability (Direction A)

```python
from models.direction_a import ChessTransformer
from models.direction_a.interpretability import ProbingSuite

model = ChessTransformer()
# Load checkpoint...

probe_suite = ProbingSuite(model)
results = probe_suite.run_probe_suite(train_loader, val_loader)
```

### Expert Analysis (Direction C)

```python
from models.direction_c import ChessMoE

model = ChessMoE()
# Load checkpoint...

# Analyze expert specialization
expert_stats = model.get_expert_usage_stats()
for expert_id, usage in expert_stats.items():
    print(f"Expert {expert_id}: {usage:.2%} usage")
```

## Project Structure

```
chess/
├── data/                      # Data generation and processing
│   ├── generators/           # Stockfish annotation generators
│   └── datasets/             # PyTorch datasets
├── models/                    # Model architectures
│   ├── direction_a/          # Transformer + interpretability
│   ├── direction_b/          # NNUE variants
│   ├── direction_c/          # MoE routing
│   └── direction_d/          # Parallel verification
├── training/                  # Training infrastructure
├── evaluation/                # Evaluation and benchmarking
├── configs/                   # Experiment configurations
└── scripts/                   # Utility scripts
```

## Key Features by Direction

### Direction A: Transformer + Interpretability
- 50M → 15-20M params via pruning
- Mechanistic interpretability probes
- Structured pruning (heads, neurons, layers)

### Direction B: NNUE Variants
- B1: Dense NNUE (5M params)
- B2: + Policy network for move ordering (7M params)
- B3: + Search extension predictor

### Direction C: MoE Routing
- 8 experts with top-2 routing
- 25M total, 12M active params
- Expert specialization analysis

### Direction D: Parallel Verification
- 5 parallel branches
- Cross-verification with attention
- Process reward training
- Inference-time scaling

## Next Steps

1. **Experiment**: Try different hyperparameters in config files
2. **Analyze**: Use interpretability tools to understand what models learned
3. **Compare**: Benchmark all directions against each other
4. **Optimize**: Apply pruning and quantization for deployment

## Troubleshooting

### Out of Memory
- Reduce batch_size in config files
- Use gradient accumulation
- Try smaller model variants (SmallChessTransformer)

### Slow Training
- Enable mixed precision training (add to config)
- Increase num_workers in dataloader
- Use multiple GPUs

### Poor Performance
- Increase training data size
- Try different learning rates
- Experiment with loss weights

## Citation

Based on minimal chess models research specifications.
