#!/bin/bash
# Start TQBridge decode node
# Usage: bash start_decode_node.sh [port] [device]

PORT="${1:-9473}"
DEVICE="${2:-NV}"
TQDIR="${HOME}/tqbridge"

echo "Starting TQBridge decode node on port $PORT (device=$DEVICE)..."
cd "$TQDIR"
PYTHONPATH=tinygrad:src python3 -m tqbridge.serve_decode \
    --port "$PORT" \
    --device "$DEVICE" \
    --head-dim 128 \
    --seed 42
