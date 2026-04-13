#!/bin/bash
# llmfit download — downloads directly to central 18TB-Mirror
#
# Usage: llmfit-download.sh <model> [llmfit args...]
#
# Models save to /Volumes/18TB-Mirror/models/gguf/ via LLMFIT_MODELS_DIR env var.
# If running on GX10, downloads to ~/models/ then copies to mirror.

MIRROR_PATH="/Volumes/18TB-Mirror/models/gguf"

if [ -d "$MIRROR_PATH" ]; then
    # Running on M3 or M1 with mirror mounted
    export LLMFIT_MODELS_DIR="$MIRROR_PATH"
    echo "Downloading to: $MIRROR_PATH"
    llmfit download "$@"
else
    # Running remotely — download local then copy
    export LLMFIT_MODELS_DIR="${HOME}/models"
    mkdir -p "$LLMFIT_MODELS_DIR"
    echo "Downloading to: $LLMFIT_MODELS_DIR (will copy to mirror after)"
    llmfit download "$@"
    EXIT=$?
    if [ $EXIT -eq 0 ]; then
        echo ""
        echo "Copying to 18TB-Mirror..."
        for f in "$LLMFIT_MODELS_DIR"/*.gguf; do
            [ -f "$f" ] || continue
            name=$(basename "$f")
            scp "$f" "james@192.168.68.50:$MIRROR_PATH/$name" && echo "  ✓ $name"
        done
    fi
    exit $EXIT
fi
