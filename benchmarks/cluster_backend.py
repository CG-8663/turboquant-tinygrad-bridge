#!/opt/homebrew/bin/python3.12
"""Chronara Cluster Backend — multi-endpoint inference server.

Routes requests to the right GPU node based on use case:
  - Hyvia:       GX10-001 (:8080) — planning approval prediction
  - Remittance:  GX10-002 (:8081) — WhatsApp transfer agent

Each endpoint runs llama.cpp server with CUDA on GB10.
TQBridge compresses KV for distribution to decode nodes.

Usage:
    python benchmarks/cluster_backend.py                    # test both endpoints
    python benchmarks/cluster_backend.py --endpoint hyvia   # test Hyvia only
    python benchmarks/cluster_backend.py --serve 9090       # start proxy server
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
import urllib.error

BOLD = "\033[1m"
DIM = "\033[2m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
CYAN = "\033[96m"
RESET = "\033[0m"

ENDPOINTS = {
    "hyvia": {
        "name": "Hyvia — Planning Approval",
        "host": "192.168.68.61",
        "port": 8080,
        "gpu": "GX10-001 GB10 CUDA",
        "system_prompt": (
            "You are Hyvia, an AI planning approval advisor for UK housing and renovation applications. "
            "You analyze planning applications against local authority policies and predict approval probability. "
            "Always provide: 1) Approval probability (0-100%), 2) Key risk factors, 3) Recommended changes to improve chances. "
            "Be specific about which planning policies apply. Reference UK planning law where relevant."
        ),
    },
    "remittance": {
        "name": "Remittance — Transfer Agent",
        "host": "192.168.68.62",
        "port": 8081,
        "gpu": "GX10-002 GB10 CUDA",
        "system_prompt": (
            "You are a remittance transfer assistant for Chronara. You help users send money internationally, "
            "primarily UK to Philippines corridor. You can: check exchange rates, estimate fees, explain transfer "
            "timelines, verify compliance requirements (KYC/AML), and guide users through the transfer process. "
            "Always be clear about fees and exchange rates. Never process transfers without identity verification. "
            "Respond concisely — this is a WhatsApp conversation."
        ),
    },
}


def query_endpoint(endpoint_key, user_prompt, max_tokens=200):
    """Send a prompt to the specified endpoint with its system prompt."""
    ep = ENDPOINTS[endpoint_key]
    url = f"http://{ep['host']}:{ep['port']}/v1/chat/completions"

    messages = [
        {"role": "system", "content": ep["system_prompt"]},
        {"role": "user", "content": user_prompt},
    ]

    payload = json.dumps({
        "model": "qwen3-8b",
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.7,
        "stream": False,
    }).encode()

    req = urllib.request.Request(url, data=payload,
                                 headers={"Content-Type": "application/json"})

    t0 = time.perf_counter()
    try:
        resp = urllib.request.urlopen(req, timeout=60)
        data = json.loads(resp.read())
        elapsed = time.perf_counter() - t0

        content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        usage = data.get("usage", {})

        return {
            "status": "ok",
            "content": content,
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
            "elapsed": elapsed,
            "tok_s": usage.get("completion_tokens", 0) / elapsed if elapsed > 0 else 0,
        }
    except Exception as e:
        return {"status": "error", "error": str(e), "elapsed": time.perf_counter() - t0}


def check_endpoints():
    """Check which endpoints are live."""
    results = {}
    for key, ep in ENDPOINTS.items():
        try:
            req = urllib.request.urlopen(
                f"http://{ep['host']}:{ep['port']}/health", timeout=3)
            data = json.loads(req.read())
            results[key] = data.get("status") == "ok"
        except Exception:
            results[key] = False
    return results


def test_endpoints(max_tokens=200):
    """Test both endpoints with domain-specific prompts."""
    print(f"\n  {BOLD}{CYAN}Chronara Cluster Backend — Domain Endpoints{RESET}")
    print(f"  {DIM}{'═' * 60}{RESET}")
    print()

    status = check_endpoints()
    for key, ep in ENDPOINTS.items():
        icon = f"{GREEN}●{RESET}" if status.get(key) else f"{RED}●{RESET}"
        state = "READY" if status.get(key) else "OFFLINE"
        print(f"  {icon} {ep['name']:30s} {ep['gpu']:20s} {state}")
    print()

    test_cases = {
        "hyvia": [
            "I want to build a single-storey rear extension on my Victorian terrace in Camden, London. The extension would be 4 metres deep and 3 metres wide. What's my approval probability?",
            "My client wants to convert a detached garage into a home office in a Conservation Area in Bath. What are the key planning risks?",
        ],
        "remittance": [
            "I want to send £500 to Manila. What's the current rate and how long will it take?",
            "What documents do I need to send money to the Philippines for the first time?",
        ],
    }

    for key, prompts in test_cases.items():
        if not status.get(key):
            print(f"  {RED}Skipping {ENDPOINTS[key]['name']} — endpoint offline{RESET}")
            print()
            continue

        ep = ENDPOINTS[key]
        print(f"  {BOLD}{ep['name']}{RESET}")
        print(f"  {DIM}GPU: {ep['gpu']} | System prompt active{RESET}")
        print()

        for i, prompt in enumerate(prompts):
            print(f"  {BOLD}User:{RESET} {prompt[:80]}...")
            r = query_endpoint(key, prompt, max_tokens=max_tokens)

            if r["status"] == "ok":
                print(f"  {GREEN}●{RESET} {r['completion_tokens']} tokens in {r['elapsed']:.1f}s ({r['tok_s']:.1f} tok/s)")
                # Show response, wrapped
                lines = r["content"].split("\n")
                for line in lines[:8]:
                    print(f"  {DIM}{line[:100]}{RESET}")
                if len(lines) > 8:
                    print(f"  {DIM}... ({len(lines)-8} more lines){RESET}")
            else:
                print(f"  {RED}●{RESET} Error: {r['error']}")
            print()

        print(f"  {DIM}{'─' * 60}{RESET}")
        print()

    print(f"  {BOLD}All inference on real GPUs via llama.cpp CUDA.{RESET}")
    print(f"  {DIM}GX10-001: Hyvia (:8080) | GX10-002: Remittance (:8081){RESET}")
    print()


def main():
    parser = argparse.ArgumentParser(description="Chronara cluster backend")
    parser.add_argument("--endpoint", choices=["hyvia", "remittance"], default=None)
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--max-tokens", type=int, default=200)
    args = parser.parse_args()

    if args.prompt and args.endpoint:
        r = query_endpoint(args.endpoint, args.prompt, args.max_tokens)
        if r["status"] == "ok":
            print(r["content"])
        else:
            print(f"Error: {r['error']}", file=sys.stderr)
    else:
        test_endpoints(args.max_tokens)


if __name__ == "__main__":
    main()
