"""vLLM vs llama.cpp latency benchmark — runs against a local vLLM server.

Usage:
    python3 vllm_bench.py --url http://localhost:8000 --model Qwen/Qwen3-8B
"""
import argparse
import json
import time
import urllib.request

def bench_completions(url, model, prompt_tokens, max_tokens, runs=5):
    prompt = " ".join(["word"] * prompt_tokens)
    endpoint = f"{url}/v1/completions"
    results = []

    for i in range(runs):
        payload = json.dumps({
            "model": model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0.0,
        }).encode()

        req = urllib.request.Request(endpoint, data=payload,
                                     headers={"Content-Type": "application/json"})
        t0 = time.perf_counter()
        with urllib.request.urlopen(req, timeout=300) as resp:
            data = json.loads(resp.read())
        t1 = time.perf_counter()

        usage = data["usage"]
        dt = t1 - t0
        pp = usage["prompt_tokens"]
        tg = usage["completion_tokens"]
        results.append({"run": i+1, "dt_s": dt, "pp": pp, "tg": tg})
        print(f"  Run {i+1}: {dt*1000:.0f}ms  pp={pp} tg={tg}  total={pp+tg}  ~{(pp+tg)/dt:.0f} tok/s")

    # Drop first run (compilation warmup)
    steady = results[1:] if len(results) > 1 else results
    avg_dt = sum(r["dt_s"] for r in steady) / len(steady)
    avg_pp = sum(r["pp"] for r in steady) / len(steady)
    avg_tg = sum(r["tg"] for r in steady) / len(steady)
    return {
        "prompt_tokens": int(avg_pp),
        "completion_tokens": int(avg_tg),
        "avg_wall_ms": round(avg_dt * 1000, 1),
        "avg_total_tps": round((avg_pp + avg_tg) / avg_dt, 1),
        "avg_tg_tps": round(avg_tg / avg_dt, 1),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--output", "-o", default=None)
    args = parser.parse_args()

    results = {}

    print(f"\n{'='*60}")
    print(f"  vLLM Latency Benchmark: {args.model}")
    print(f"{'='*60}")

    # pp512 + tg128
    print("\n[1/2] pp512 + tg128 (5 runs, drop first)...")
    results["pp512_tg128"] = bench_completions(args.url, args.model, 450, 128, runs=5)

    # pp8192 + tg128
    print("\n[2/2] pp8192 + tg128 (4 runs, drop first)...")
    results["pp8192_tg128"] = bench_completions(args.url, args.model, 7500, 128, runs=4)

    print(f"\n{'='*60}")
    print("  Summary")
    print(f"{'='*60}")
    for k, v in results.items():
        print(f"  {k}: {v['avg_wall_ms']:.0f}ms  tg={v['avg_tg_tps']:.1f} tok/s  total={v['avg_total_tps']:.1f} tok/s")

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
