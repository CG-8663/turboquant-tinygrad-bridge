#!/bin/bash
# Full TurboQuant model benchmark: Qwen3-8B on M3 Ultra
# Tests baseline vs turbo at multiple context depths
# Output: JSON lines for each config

MODEL="/Volumes/Chronara-Storage/Models/Qwen3-8B-Q8_0.gguf"
BENCH="./bin/llama-bench"
OUT="../benchmarks/model_bench_qwen3_8b.jsonl"

echo "=== Qwen3-8B TurboQuant Benchmark ==="
echo "Model: $MODEL"
echo "Output: $OUT"
echo ""

# Clear output
> "$OUT"

# Configs: (label, ctk, ctv)
configs=(
    "f16,f16"
    "q8_0,q8_0"
    "q8_0,turbo3"
    "q8_0,turbo4"
    "turbo3,turbo3"
    "turbo4,turbo4"
)

# Context depths to test
depths="0 512 2048 8192"

for config in "${configs[@]}"; do
    IFS=',' read -r ctk ctv <<< "$config"
    echo "--- KV cache: K=$ctk V=$ctv ---"

    for depth in $depths; do
        echo -n "  depth=$depth: "
        $BENCH -m "$MODEL" \
            -ctk "$ctk" -ctv "$ctv" \
            -p 512 -n 128 -d "$depth" \
            -r 1 -o jsonl \
            2>/dev/null >> "$OUT"

        # Extract tok/s from last 2 lines (pp and tg)
        tail -2 "$OUT" | python3 -c "
import sys,json
for line in sys.stdin:
    d=json.loads(line)
    t='pp' if d['n_prompt']>0 else 'tg'
    print(f\"{t}={d['avg_ts']:.1f}\", end='  ')
"
        echo ""
    done
done

echo ""
echo "Results saved to $OUT"
