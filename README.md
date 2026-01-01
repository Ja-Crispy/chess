# Minimal Chess Models Research

Four parallel research directions for building minimal, high-performance chess models.
Target: highest Elo achievable with smallest parameter count.

## Research Directions

- **Direction A**: End-to-End Transformer with Mechanistic Interpretability + Pruning (~50M → 15-20M params)
- **Direction B**: NNUE-Style + Learned Components (~5-7M params)
- **Direction C**: MoE with Position-Type Routing (~25M params, 12M active)
- **Direction D**: Parallel Verification Architecture (~24M params)

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
│   ├── trainers/             # Training loops
│   └── losses/               # Loss functions
├── evaluation/                # Evaluation and benchmarking
├── analysis/                  # Interpretability and analysis tools
└── configs/                   # Experiment configurations
```

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Generate Training Data
```bash
# Generate Stockfish-annotated positions
python -m data.generators.stockfish_generator \
    --num_positions 100000 \
    --depth 20 \
    --output_dir data/stockfish_annotations

# Generate specific datasets for each direction
python -m data.generators.generate_all_datasets
```

### 3. Train Models

```bash
# Direction A: Transformer
python -m training.train --config configs/direction_a.yaml

# Direction B1: Dense NNUE
python -m training.train --config configs/direction_b1.yaml

# Direction C: MoE
python -m training.train --config configs/direction_c.yaml

# Direction D: Parallel Verification
python -m training.train --config configs/direction_d.yaml
```

### 4. Evaluate

```bash
python -m evaluation.play_games --model_path checkpoints/model.pt --opponent stockfish
```

## Compute Requirements

- Direction A: ~1 week on single A100
- Direction B: ~2-3 days on single GPU
- Direction C: ~3-5 days on single A100
- Direction D: ~4-6 days on single A100

## Citation

Based on minimal chess models research specifications.
