#!/bin/bash
# Hyvia Interactive Demo — Formatted markdown output
# Runs on GX10-001 GB10 CUDA via llama.cpp

HOST="192.168.68.61"
PORT="8080"

BOLD="\033[1m"
DIM="\033[2m"
GREEN="\033[92m"
CYAN="\033[96m"
YELLOW="\033[93m"
RESET="\033[0m"

echo -e "\n${BOLD}${CYAN}╔══════════════════════════════════════════════════╗${RESET}"
echo -e "${BOLD}${CYAN}║  Hyvia — UK Planning Approval Advisor            ║${RESET}"
echo -e "${BOLD}${CYAN}║  Chronara Cluster (GX10 GB10 CUDA)               ║${RESET}"
echo -e "${BOLD}${CYAN}╚══════════════════════════════════════════════════╝${RESET}"
echo -e "\n${DIM}  Type a planning question. Ctrl+C to exit.${RESET}\n"

SYSTEM="You are Hyvia, a UK planning approval advisor. Provide: 1) Approval probability 2) Risk factors 3) Recommendations. Be specific about policies. Respond directly, no internal reasoning. Format with markdown headers and bullet points."

while true; do
    printf "${GREEN}  You: ${RESET}"
    read -r PROMPT
    [ -z "$PROMPT" ] && continue

    echo -e "\n${CYAN}  Hyvia:${RESET}\n"

    # Stream + render: collect tokens, render markdown progressively
    curl -sN "http://${HOST}:${PORT}/completion" \
        -H "Content-Type: application/json" \
        -d "{
            \"prompt\": \"<|im_start|>system\n${SYSTEM}<|im_end|>\n<|im_start|>user\n/no_think ${PROMPT}<|im_end|>\n<|im_start|>assistant\n\",
            \"max_tokens\": 500,
            \"temperature\": 0.7,
            \"stream\": true,
            \"stop\": [\"<|im_end|>\", \"<|im_start|>\"]
        }" 2>/dev/null | python3 -c "
import json, re, sys, time, textwrap

BOLD = '\033[1m'
DIM = '\033[2m'
GREEN = '\033[92m'
CYAN = '\033[96m'
YELLOW = '\033[93m'
WHITE = '\033[97m'
R = '\033[0m'

def render_line(line):
    if not line.rstrip():
        print()
        return
    line = re.sub(r'\*\*(.+?)\*\*', f'{BOLD}{WHITE}\\1{R}', line.rstrip())
    if line.lstrip().startswith('### '):
        print(f'  {BOLD}{CYAN}{line.lstrip()[4:]}{R}')
    elif line.lstrip().startswith('## '):
        print(f'  {BOLD}{CYAN}{line.lstrip()[3:]}{R}')
    elif line.lstrip().startswith('# '):
        print(f'  {BOLD}{CYAN}{line.lstrip()[2:]}{R}')
    elif line.lstrip().startswith('- ') or line.lstrip().startswith('* '):
        indent = '  ' + ' ' * ((len(line) - len(line.lstrip())) // 2)
        content = line.lstrip()[2:]
        wrapped = textwrap.fill(content, width=72, subsequent_indent=indent + '  ')
        print(f'{indent}{GREEN}●{R} {wrapped}')
    elif re.match(r'\s*\d+[\.\)]\s', line):
        m = re.match(r'(\s*)(\d+[\.\)])\s*(.*)', line)
        if m:
            pad = '  ' + ' ' * (len(m.group(1)) // 2)
            wrapped = textwrap.fill(m.group(3), width=72, subsequent_indent=pad + '   ')
            print(f'{pad}{YELLOW}{m.group(2)}{R} {wrapped}')
    else:
        wrapped = textwrap.fill(line, width=76, initial_indent='  ', subsequent_indent='  ')
        print(wrapped)

t0 = time.time()
n_tokens = 0
in_think = False
line_buf = ''

for raw in sys.stdin:
    raw = raw.strip()
    if raw.startswith('data: '): raw = raw[6:]
    if not raw or raw == '[DONE]': continue

    try:
        d = json.loads(raw)
        tok = d.get('content', '')
        stop = d.get('stop', False)
    except:
        continue

    if stop:
        # Final token — render remaining buffer
        if line_buf.strip():
            render_line(line_buf)
        # Get timing from the stop message
        tps = d.get('timings', {}).get('predicted_per_second', 0)
        pp = d.get('timings', {}).get('prompt_per_second', 0)
        n_tok = d.get('tokens_predicted', n_tokens)
        print(f'\n  {DIM}[{n_tok} tokens | prefill {pp:.0f} tok/s | decode {tps:.1f} tok/s | GB10 CUDA]{R}')
        break

    n_tokens += 1

    # Skip <think> blocks
    if '<think>' in tok:
        in_think = True
        continue
    if '</think>' in tok:
        in_think = False
        continue
    if in_think:
        continue

    line_buf += tok

    # Render complete lines as they arrive
    while '\n' in line_buf:
        line, line_buf = line_buf.split('\n', 1)
        render_line(line)
        sys.stdout.flush()

# Flush any remaining buffer
if line_buf.strip():
    render_line(line_buf)

if n_tokens > 0 and not stop:
    elapsed = time.time() - t0
    tps = n_tokens / elapsed if elapsed > 0 else 0
    print(f'\n  {DIM}[{n_tokens} tokens | {tps:.1f} tok/s | GB10 CUDA]{R}')
"
    echo
done
