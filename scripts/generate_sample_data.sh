#!/bin/bash
# Generate sample training data for quick testing

echo "Generating sample dataset (1000 positions)..."

python -m data.generators.stockfish_generator \
    --output data/stockfish_annotations/sample.h5 \
    --num_positions 1000 \
    --depth 15 \
    --multipv 3 \
    --engine_path /usr/games/stockfish \
    --position_type mixed \
    --num_workers 1

echo "Done! Sample data saved to data/stockfish_annotations/sample.h5"
