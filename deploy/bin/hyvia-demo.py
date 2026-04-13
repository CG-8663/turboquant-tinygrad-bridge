#!/opt/homebrew/bin/python3.12
"""Hyvia Interactive Demo — streaming, formatted, with conversation history and logging."""

import json
import os
import re
import sys
import time
import urllib.request

HOST = "192.168.68.61"
PORT = 8080

BOLD = "\033[1m"
DIM = "\033[2m"
GREEN = "\033[92m"
CYAN = "\033[96m"
YELLOW = "\033[93m"
WHITE = "\033[97m"
R = "\033[0m"

SYSTEM = (
    "You are Hyvia, a UK planning approval advisor. "
    "Answer directly and thoroughly. Never use <think> tags. "
    "Provide: 1) Key points 2) Risk factors 3) Recommendations where applicable. "
    "Cite specific policies, paragraph numbers, and legislation. "
    "Format with markdown headers and bullet points."
)

# Logging
LOGDIR = "/Volumes/18TB-Mirror/HYVIA-DEMO-TRAINING/baselines"
if not os.path.isdir(LOGDIR):
    LOGDIR = os.path.expanduser("~/hyvia-baselines")
    os.makedirs(LOGDIR, exist_ok=True)
LOGFILE = os.path.join(LOGDIR, f"baseline-{time.strftime('%Y%m%d-%H%M%S')}.md")


def fmt_line(line):
    """Render one markdown line with ANSI formatting."""
    if not line.rstrip():
        print(flush=True)
        return
    # Bold **text**
    line = re.sub(r"\*\*(.+?)\*\*", f"{BOLD}{WHITE}\\1{R}", line.rstrip())
    raw = re.sub(r"\033\[[^m]*m", "", line)
    stripped = raw.lstrip()

    if stripped.startswith("### "):
        print(f"  {BOLD}{CYAN}{line.lstrip()[4:]}{R}", flush=True)
    elif stripped.startswith("## "):
        print(f"  {BOLD}{CYAN}{line.lstrip()[3:]}{R}", flush=True)
    elif stripped.startswith("# "):
        print(f"  {BOLD}{CYAN}{line.lstrip()[2:]}{R}", flush=True)
    elif stripped.startswith("- ") or stripped.startswith("* "):
        print(f"  {GREEN}●{R} {line.lstrip()[2:]}", flush=True)
    elif re.match(r"\d+[\.\)]\s", stripped):
        m = re.match(r"(\d+[\.\)])\s*(.*)", stripped)
        if m:
            print(f"  {YELLOW}{m.group(1)}{R} {line.lstrip()[len(m.group(1))+1:].lstrip()}", flush=True)
    else:
        print(f"  {line.lstrip()}", flush=True)


def stream_response(prompt_text):
    """Send prompt, stream response, return full text."""
    payload = json.dumps({
        "prompt": prompt_text,
        "max_tokens": 2000,
        "temperature": 0.7,
        "stream": True,
        "stop": ["<|im_end|>", "<|im_start|>"],
    }).encode()

    req = urllib.request.Request(
        f"http://{HOST}:{PORT}/completion",
        data=payload,
        headers={"Content-Type": "application/json"},
    )

    try:
        resp = urllib.request.urlopen(req, timeout=120)
    except Exception as e:
        print(f"  {DIM}Error: {e}{R}")
        return ""

    buf = ""
    full_text = ""
    n_tokens = 0
    in_think = False
    tps = 0
    pp = 0
    n_final = 0

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

        # Skip think blocks
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

        # Render complete lines as they arrive
        while "\n" in buf:
            line, buf = buf.split("\n", 1)
            fmt_line(line)

    # Flush remaining
    if buf.strip():
        fmt_line(buf)

    tokens = n_final if n_final else n_tokens
    if tokens > 0:
        print(f"\n  {DIM}[{tokens} tokens | prefill {pp:.0f} tok/s | decode {tps:.1f} tok/s | GB10 CUDA]{R}", flush=True)

    return full_text


def main():
    print(f"\n{BOLD}{CYAN}╔══════════════════════════════════════════════════╗{R}")
    print(f"{BOLD}{CYAN}║  Hyvia — UK Planning Approval Advisor            ║{R}")
    print(f"{BOLD}{CYAN}║  Chronara Cluster (GX10 GB10 CUDA)               ║{R}")
    print(f"{BOLD}{CYAN}╚══════════════════════════════════════════════════╝{R}")
    print(f"\n{DIM}  Type a planning question. 'continue' for more. 'new' to reset.{R}")
    print(f"{DIM}  Ctrl+C to exit.{R}")
    print(f"{DIM}  Logging to: {LOGFILE}{R}\n")

    with open(LOGFILE, "w") as f:
        f.write(f"# Hyvia Baseline Capture — {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Model: Qwen3-8B Q8_0 | Node: GX10-001 GB10 CUDA\n\n")

    messages = []  # conversation history as list of (role, content)

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

        messages.append(("user", prompt))

        # Build full prompt with history
        parts = [f"<|im_start|>system\n{SYSTEM}<|im_end|>"]
        for role, content in messages:
            parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")
        # Seed assistant to prevent think mode
        parts.append("<|im_start|>assistant\nHere is my analysis:\n")
        full_prompt = "\n".join(parts)

        print(f"\n{CYAN}  Hyvia:{R}")

        response = stream_response(full_prompt)

        if response:
            messages.append(("assistant", "Here is my analysis:\n" + response))

            # Log to file
            with open(LOGFILE, "a") as f:
                f.write(f"## Q: {prompt}\n\n")
                f.write(f"{response}\n\n---\n\n")

        print()


if __name__ == "__main__":
    main()
