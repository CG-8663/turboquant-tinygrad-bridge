#!/opt/homebrew/bin/python3.12
"""Hyvia Interactive Demo — distributed cluster inference.

Fans out each question to ALL cluster nodes in parallel:
  - GX10-001 (CUDA, Qwen3-8B Q8_0)
  - GX10-002 (CUDA, Qwen3-8B Q8_0)
  - M3 Ultra (MLX, Qwen2.5-7B 4bit)

Merges the best response, or shows all for comparison.
Logs everything to 18TB-Mirror for training data extraction.
"""

import json
import os
import re
import sys
import threading
import time
import urllib.request
import urllib.error

BOLD = "\033[1m"
DIM = "\033[2m"
GREEN = "\033[92m"
CYAN = "\033[96m"
YELLOW = "\033[93m"
WHITE = "\033[97m"
RED = "\033[91m"
MAGENTA = "\033[95m"
R = "\033[0m"

SYSTEM = (
    "You are Hyvia, a UK planning approval advisor. "
    "Answer directly and thoroughly. Never use <think> tags. "
    "Provide: 1) Key points 2) Risk factors 3) Recommendations where applicable. "
    "Cite specific policies, paragraph numbers, and legislation. "
    "Format with markdown headers and bullet points."
)

# Cluster nodes
NODES = [
    {
        "name": "GX10-001",
        "host": "192.168.68.61",
        "port": 8080,
        "gpu": "GB10 CUDA",
        "color": GREEN,
        "type": "llama.cpp",
    },
    {
        "name": "GX10-002",
        "host": "192.168.68.62",
        "port": 8081,
        "gpu": "GB10 CUDA",
        "color": YELLOW,
        "color_name": "yellow",
        "type": "llama.cpp",
    },
    {
        "name": "M3 Ultra",
        "host": "127.0.0.1",
        "port": 0,  # MLX direct, no server
        "gpu": "Metal MLX",
        "color": CYAN,
        "type": "mlx",
    },
]

# Logging
LOGDIR = "/Volumes/18TB-Mirror/HYVIA-DEMO-TRAINING/baselines"
if not os.path.isdir(LOGDIR):
    LOGDIR = os.path.expanduser("~/hyvia-baselines")
    os.makedirs(LOGDIR, exist_ok=True)
LOGFILE = os.path.join(LOGDIR, f"cluster-{time.strftime('%Y%m%d-%H%M%S')}.md")


def fmt_line(line, indent="  "):
    """Render one markdown line with ANSI formatting."""
    if not line.rstrip():
        print(flush=True)
        return
    line = re.sub(r"\*\*(.+?)\*\*", f"{BOLD}{WHITE}\\1{R}", line.rstrip())
    raw = re.sub(r"\033\[[^m]*m", "", line)
    stripped = raw.lstrip()

    if stripped.startswith("### "):
        print(f"{indent}{BOLD}{CYAN}{line.lstrip()[4:]}{R}", flush=True)
    elif stripped.startswith("## "):
        print(f"{indent}{BOLD}{CYAN}{line.lstrip()[3:]}{R}", flush=True)
    elif stripped.startswith("# "):
        print(f"{indent}{BOLD}{CYAN}{line.lstrip()[2:]}{R}", flush=True)
    elif stripped.startswith("- ") or stripped.startswith("* "):
        print(f"{indent}{GREEN}●{R} {line.lstrip()[2:]}", flush=True)
    elif re.match(r"\d+[\.\)]\s", stripped):
        m = re.match(r"(\d+[\.\)])\s*(.*)", stripped)
        if m:
            print(f"{indent}{YELLOW}{m.group(1)}{R} {line.lstrip()[len(m.group(1))+1:].lstrip()}", flush=True)
    else:
        print(f"{indent}{line.lstrip()}", flush=True)


def strip_think(text):
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"<think>.*", "", text, flags=re.DOTALL)
    return text.strip()


def query_llama_server(host, port, prompt, max_tokens=2000):
    """Query a llama.cpp server, return (text, tps, pp_tps, tokens)."""
    payload = json.dumps({
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.7,
        "stop": ["<|im_end|>", "<|im_start|>"],
    }).encode()

    req = urllib.request.Request(
        f"http://{host}:{port}/completion",
        data=payload,
        headers={"Content-Type": "application/json"},
    )

    try:
        t0 = time.perf_counter()
        resp = urllib.request.urlopen(req, timeout=90)
        data = json.loads(resp.read())
        elapsed = time.perf_counter() - t0

        text = strip_think(data.get("content", ""))
        tps = data.get("timings", {}).get("predicted_per_second", 0)
        pp = data.get("timings", {}).get("prompt_per_second", 0)
        tokens = data.get("tokens_predicted", 0)
        return text, tps, pp, tokens, elapsed
    except Exception as e:
        return f"Error: {e}", 0, 0, 0, 0


def query_mlx(prompt, max_tokens=500):
    """Query local MLX model."""
    try:
        from mlx_lm import load, generate
        if not hasattr(query_mlx, "_model"):
            query_mlx._model, query_mlx._tokenizer = load(
                "mlx-community/Qwen2.5-7B-Instruct-4bit")

        # Extract just the user question from the prompt
        user_q = prompt.split("<|im_start|>user\n")[-1].split("<|im_end|>")[0]
        mlx_prompt = f"You are Hyvia, a UK planning advisor. Answer directly, cite policies.\n\nQuestion: {user_q}"

        t0 = time.perf_counter()
        response = generate(
            query_mlx._model, query_mlx._tokenizer,
            prompt=mlx_prompt, max_tokens=max_tokens, verbose=False)
        elapsed = time.perf_counter() - t0
        tokens = len(query_mlx._tokenizer.encode(response))
        tps = tokens / elapsed if elapsed > 0 else 0

        return strip_think(response), tps, 0, tokens, elapsed
    except Exception as e:
        return f"Error: {e}", 0, 0, 0, 0


def query_node(node, prompt, results, index):
    """Query a single node — used in threading."""
    if node["type"] == "mlx":
        results[index] = query_mlx(prompt)
    else:
        results[index] = query_llama_server(node["host"], node["port"], prompt)


def check_nodes():
    """Check which nodes are available."""
    available = []
    for node in NODES:
        if node["type"] == "mlx":
            try:
                from mlx_lm import load
                available.append(node)
            except ImportError:
                pass
        else:
            try:
                req = urllib.request.urlopen(
                    f"http://{node['host']}:{node['port']}/health", timeout=2)
                data = json.loads(req.read())
                if data.get("status") == "ok":
                    available.append(node)
            except Exception:
                pass
    return available


def merge_responses(results, nodes):
    """Pick the best response — longest substantive answer wins."""
    best_idx = 0
    best_len = 0
    for i, (text, tps, pp, tokens, elapsed) in enumerate(results):
        if text and not text.startswith("Error"):
            # Score by content length (not token count — actual substance)
            clean = re.sub(r"\s+", " ", text).strip()
            if len(clean) > best_len:
                best_len = len(clean)
                best_idx = i
    return best_idx


def stream_from_node(node, prompt):
    """Stream response from a llama.cpp server with live rendering."""
    payload = json.dumps({
        "prompt": prompt,
        "max_tokens": 2000,
        "temperature": 0.7,
        "stream": True,
        "stop": ["<|im_end|>", "<|im_start|>"],
    }).encode()

    req = urllib.request.Request(
        f"http://{node['host']}:{node['port']}/completion",
        data=payload,
        headers={"Content-Type": "application/json"},
    )

    try:
        resp = urllib.request.urlopen(req, timeout=90)
    except Exception as e:
        print(f"  {RED}Error: {e}{R}")
        return ""

    buf = ""
    full_text = ""
    n_tokens = 0
    in_think = False
    tps = pp = n_final = 0

    for raw_line in resp:
        raw_line = raw_line.decode("utf-8").strip()
        if raw_line.startswith("data: "):
            raw_line = raw_line[6:]
        if not raw_line or raw_line == "[DONE]":
            continue

        try:
            d = json.loads(raw_line)
        except json.JSONDecodeError:
            continue

        tok = d.get("content", "")
        stop = d.get("stop", False)

        if stop:
            if buf.strip():
                fmt_line(buf)
            tps = d.get("timings", {}).get("predicted_per_second", 0)
            pp = d.get("timings", {}).get("prompt_per_second", 0)
            n_final = d.get("tokens_predicted", n_tokens)
            break

        if "<think>" in tok:
            in_think = True
            continue
        if "</think>" in tok:
            in_think = False
            continue
        if in_think:
            continue

        n_tokens += 1
        buf += tok
        full_text += tok

        while "\n" in buf:
            line, buf = buf.split("\n", 1)
            fmt_line(line)

    if buf.strip():
        fmt_line(buf)

    tokens = n_final if n_final else n_tokens
    if tokens > 0:
        print(f"\n  {DIM}[{tokens} tokens | prefill {pp:.0f} tok/s | decode {tps:.1f} tok/s | {node['name']} {node['gpu']}]{R}", flush=True)

    return full_text


def main():
    print(f"\n{BOLD}{CYAN}╔══════════════════════════════════════════════════════════╗{R}")
    print(f"{BOLD}{CYAN}║  Hyvia — UK Planning Approval Advisor                    ║{R}")
    print(f"{BOLD}{CYAN}║  Distributed Cluster Inference — Chronara Group           ║{R}")
    print(f"{BOLD}{CYAN}╚══════════════════════════════════════════════════════════════╝{R}")

    available = check_nodes()
    print(f"\n  {BOLD}Cluster nodes:{R}")
    for node in NODES:
        if node in available:
            print(f"  {GREEN}●{R} {node['name']:12s} {node['gpu']:15s} ready")
        else:
            print(f"  {RED}●{R} {node['name']:12s} {node['gpu']:15s} offline")

    mode = "stream" if len(available) == 1 else "parallel"
    print(f"\n{DIM}  Mode: {'parallel fan-out' if mode == 'parallel' else 'single node streaming'}")
    print(f"  Commands: 'compare' (show all), 'stream' (fastest node), 'new' (reset)")
    print(f"  Logging to: {LOGFILE}{R}\n")

    with open(LOGFILE, "w") as f:
        f.write(f"# Hyvia Cluster Baseline — {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Nodes: {', '.join(n['name'] + ' ' + n['gpu'] for n in available)}\n\n")

    messages = []

    while True:
        try:
            prompt = input(f"{GREEN}  You: {R}").strip()
        except (EOFError, KeyboardInterrupt):
            print(f"\n{DIM}  Goodbye.{R}")
            break

        if not prompt:
            continue
        if prompt.lower() in ("new", "reset"):
            messages = []
            print(f"\n{DIM}  Conversation reset.{R}\n")
            continue
        if prompt.lower() == "compare":
            mode = "compare"
            print(f"{DIM}  Mode: compare (show all node responses){R}\n")
            continue
        if prompt.lower() == "stream":
            mode = "stream"
            print(f"{DIM}  Mode: stream (fastest node, live tokens){R}\n")
            continue

        messages.append(("user", prompt))

        # Build prompt
        parts = [f"<|im_start|>system\n{SYSTEM}<|im_end|>"]
        for role, content in messages:
            parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")
        parts.append("<|im_start|>assistant\nHere is my analysis:\n")
        full_prompt = "\n".join(parts)

        if mode == "stream":
            # Stream from first available llama.cpp node
            cuda_nodes = [n for n in available if n["type"] == "llama.cpp"]
            if cuda_nodes:
                node = cuda_nodes[0]
                print(f"\n{CYAN}  Hyvia ({node['name']} {node['gpu']}):{R}")
                response = stream_from_node(node, full_prompt)
                if response:
                    messages.append(("assistant", "Here is my analysis:\n" + response))
                    with open(LOGFILE, "a") as f:
                        f.write(f"## Q: {prompt}\n\n")
                        f.write(f"### {node['name']} ({node['gpu']})\n\n{response}\n\n---\n\n")
            print()

        else:
            # Parallel fan-out to all nodes
            print(f"\n{DIM}  Querying {len(available)} nodes in parallel...{R}")
            results = [None] * len(available)
            threads = []

            t0 = time.perf_counter()
            for i, node in enumerate(available):
                t = threading.Thread(target=query_node, args=(node, full_prompt, results, i))
                t.start()
                threads.append(t)

            for t in threads:
                t.join(timeout=90)

            total_elapsed = time.perf_counter() - t0

            # Display all responses
            print()
            log_entry = f"## Q: {prompt}\n\n"

            for i, node in enumerate(available):
                if results[i] is None:
                    continue
                text, tps, pp, tokens, elapsed = results[i]
                if not text or text.startswith("Error"):
                    print(f"  {RED}● {node['name']}: {text}{R}\n")
                    continue

                color = node.get("color", WHITE)
                print(f"  {BOLD}{color}━━━ {node['name']} ({node['gpu']}) ━━━{R}")
                for line in text.split("\n"):
                    fmt_line(line)
                print(f"  {DIM}[{tokens} tok | {tps:.1f} tok/s | {elapsed:.1f}s]{R}")
                print()

                log_entry += f"### {node['name']} ({node['gpu']}) — {tokens} tok, {tps:.1f} tok/s\n\n{text}\n\n"

            # Pick best for conversation history
            best_idx = merge_responses(results, available)
            if results[best_idx] and not results[best_idx][0].startswith("Error"):
                best_text = results[best_idx][0]
                best_node = available[best_idx]["name"]
                print(f"  {DIM}Best response: {best_node} ({len(best_text)} chars){R}")
                messages.append(("assistant", "Here is my analysis:\n" + best_text))

            print(f"  {DIM}Total: {total_elapsed:.1f}s across {len(available)} nodes{R}")

            log_entry += "---\n\n"
            with open(LOGFILE, "a") as f:
                f.write(log_entry)

            print()


if __name__ == "__main__":
    main()
