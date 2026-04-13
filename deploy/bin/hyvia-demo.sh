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

    echo -e "\n${DIM}  Generating on GX10-001 GB10 CUDA...${RESET}"

    # Non-streaming request — simpler, gets timing from server
    RESPONSE=$(curl -s "http://${HOST}:${PORT}/completion" \
        -H "Content-Type: application/json" \
        -d "{
            \"prompt\": \"<|im_start|>system\n${SYSTEM}<|im_end|>\n<|im_start|>user\n/no_think ${PROMPT}<|im_end|>\n<|im_start|>assistant\n\",
            \"max_tokens\": 500,
            \"temperature\": 0.7,
            \"stop\": [\"<|im_end|>\", \"<|im_start|>\"]
        }" 2>/dev/null)

    # Render with markdown formatting
    echo "$RESPONSE" | python3 -c "
import json, re, sys, textwrap

BOLD = '\033[1m'
DIM = '\033[2m'
GREEN = '\033[92m'
CYAN = '\033[96m'
YELLOW = '\033[93m'
WHITE = '\033[97m'
R = '\033[0m'

try:
    d = json.load(sys.stdin)
    text = d.get('content', '')
    tps = d.get('timings', {}).get('predicted_per_second', 0)
    tokens = d.get('tokens_predicted', 0)
    prompt_tps = d.get('timings', {}).get('prompt_per_second', 0)

    # Strip think blocks
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    text = re.sub(r'<think>.*', '', text, flags=re.DOTALL)
    text = text.strip()

    print(f'\n{CYAN}  Hyvia:{R}')
    print()

    for line in text.split('\n'):
        line = line.rstrip()
        if not line:
            print()
            continue

        # Bold markers **text**
        line = re.sub(r'\*\*(.+?)\*\*', f'{BOLD}{WHITE}\\1{R}', line)

        # Headers
        if line.lstrip().startswith('### '):
            print(f'  {BOLD}{CYAN}{line.lstrip()[4:]}{R}')
        elif line.lstrip().startswith('## '):
            print(f'  {BOLD}{CYAN}{line.lstrip()[3:]}{R}')
        elif line.lstrip().startswith('# '):
            print(f'  {BOLD}{CYAN}{line.lstrip()[2:]}{R}')
        # Bullet points
        elif line.lstrip().startswith('- ') or line.lstrip().startswith('* '):
            indent = len(line) - len(line.lstrip())
            pad = '  ' + ' ' * (indent // 2)
            content = line.lstrip()[2:]
            # Wrap long lines
            wrapped = textwrap.fill(content, width=72, subsequent_indent=pad + '  ')
            print(f'{pad}{GREEN}●{R} {wrapped}')
        # Numbered lists
        elif re.match(r'\s*\d+[\.\)]\s', line):
            m = re.match(r'(\s*)(\d+[\.\)])\s*(.*)', line)
            if m:
                pad = '  ' + ' ' * (len(m.group(1)) // 2)
                num = m.group(2)
                content = m.group(3)
                wrapped = textwrap.fill(content, width=72, subsequent_indent=pad + '   ')
                print(f'{pad}{YELLOW}{num}{R} {wrapped}')
        else:
            wrapped = textwrap.fill(line, width=76, initial_indent='  ', subsequent_indent='  ')
            print(wrapped)

    print(f'\n  {DIM}[{tokens} tokens | prefill {prompt_tps:.0f} tok/s | decode {tps:.1f} tok/s | GB10 CUDA]{R}')

except Exception as e:
    print(f'  Error: {e}')
"
    echo
done
