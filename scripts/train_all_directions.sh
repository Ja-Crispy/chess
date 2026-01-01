#!/bin/bash
# Train all research directions

echo "Training all research directions..."

# Direction A: Transformer
echo "Starting Direction A: Transformer..."
python -m training.train --config configs/direction_a.yaml &
PID_A=$!

# Direction B1: Dense NNUE
echo "Starting Direction B1: Dense NNUE..."
python -m training.train --config configs/direction_b1.yaml &
PID_B1=$!

# Direction B2: Hybrid NNUE
echo "Starting Direction B2: Hybrid NNUE..."
python -m training.train --config configs/direction_b2.yaml &
PID_B2=$!

# Direction C: MoE
echo "Starting Direction C: MoE..."
python -m training.train --config configs/direction_c.yaml &
PID_C=$!

# Direction D: Parallel Verification
echo "Starting Direction D: Parallel Verification..."
python -m training.train --config configs/direction_d.yaml &
PID_D=$!

echo "All training jobs started!"
echo "PIDs: A=$PID_A, B1=$PID_B1, B2=$PID_B2, C=$PID_C, D=$PID_D"

# Wait for all jobs
wait

echo "All training complete!"
