#!/bin/bash
# GX10 decode node deployment script
# Run on each GX10 node (192.168.68.60, .61)
#
# Prerequisites: Python 3.11+, CUDA toolkit, pip
# Usage: bash gx10_setup.sh

set -eo pipefail

echo "=== TQBridge Decode Node Setup ==="
echo "Host: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo ""

# 1. Clone/update tqbridge
TQDIR="${HOME}/tqbridge"
if [ -d "$TQDIR" ]; then
    echo "[1/4] Updating tqbridge..."
    cd "$TQDIR" && git pull
else
    echo "[1/4] Cloning tqbridge..."
    git clone https://github.com/TheTom/turboquant-tinygrad-bridge "$TQDIR"
    cd "$TQDIR"
fi

# 2. Install Python dependencies
echo "[2/4] Installing dependencies..."
pip install -e ".[dev]" 2>/dev/null || pip install numpy scipy

# 3. Build C library
echo "[3/4] Building libtqbridge..."
cd llama-cpp-turboquant
git submodule update --init
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target tqbridge -j$(nproc)
cd ..

# 4. Verify
echo "[4/4] Verifying..."
python3 -c "
from tqbridge.native import NativeCompressor, _find_library
lib = _find_library()
print(f'  libtqbridge: {lib}')

nc = NativeCompressor(head_dim=128, seed=42)
import numpy as np
data = np.random.randn(8, 128).astype(np.float32)
from tqbridge.wire import Format
comp = nc.compress(data, Format.TURBO3)
result = nc.decompress(comp)
mse = np.mean((result - data)**2)
print(f'  Compress/decompress OK, MSE={mse:.6f}')
nc.close()
"

echo ""
echo "=== Setup complete ==="
echo ""
echo "Start the decode server:"
echo "  python3 -m tqbridge.serve_decode --port 9473 --device NV"
echo ""
echo "Or for CPU-only (no GPU):"
echo "  python3 -m tqbridge.serve_decode --port 9473 --device CPU"
