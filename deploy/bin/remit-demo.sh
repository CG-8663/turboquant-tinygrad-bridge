#!/bin/bash
# Remittance Interactive Demo — Formatted WhatsApp-style
# Runs on GX10-002 GB10 CUDA via llama.cpp

HOST="192.168.68.62"
PORT="8081"

BOLD="\033[1m"
DIM="\033[2m"
GREEN="\033[92m"
CYAN="\033[96m"
YELLOW="\033[93m"
RESET="\033[0m"

echo -e "\n${BOLD}${CYAN}╔══════════════════════════════════════════════════╗${RESET}"
echo -e "${BOLD}${CYAN}║  Chronara Remittance — Transfer Assistant         ║${RESET}"
echo -e "${BOLD}${CYAN}║  Chronara Cluster (GX10 GB10 CUDA)               ║${RESET}"
echo -e "${BOLD}${CYAN}╚══════════════════════════════════════════════════╝${RESET}"
echo -e "\n${DIM}  WhatsApp-style. Type your question. Ctrl+C to exit.${RESET}\n"

SYSTEM="You are a Chronara remittance assistant. UK to Philippines transfers. Check rates, fees, timelines, compliance. Be concise for WhatsApp. Respond directly, no reasoning. Use bullet points for clarity."

while true; do
    printf "${GREEN}  You: ${RESET}"
    read -r PROMPT
    [ -z "$PROMPT" ] && continue

    echo -e "\n${DIM}  Generating on GX10-002 GB10 CUDA...${RESET}"

    RESPONSE=$(curl -s "http://${HOST}:${PORT}/completion" \
        -H "Content-Type: application/json" \
        -d "{
            \"prompt\": \"<|im_start|>system\n${SYSTEM}<|im_end|>\n<|im_start|>user\n/no_think ${PROMPT}<|im_end|>\n<|im_start|>assistant\n\",
            \"max_tokens\": 250,
            \"temperature\": 0.7,
            \"stop\": [\"<|im_end|>\", \"<|im_start|>\"]
        }" 2>/dev/null)

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

    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    text = re.sub(r'<think>.*', '', text, flags=re.DOTALL)
    text = text.strip()

    print(f'\n{CYAN}  Agent:{R}')
    print()

    for line in text.split('\n'):
        line = line.rstrip()
        if not line:
            print()
            continue
        line = re.sub(r'\*\*(.+?)\*\*', f'{BOLD}{WHITE}\\1{R}', line)
        if line.lstrip().startswith('- ') or line.lstrip().startswith('* '):
            content = line.lstrip()[2:]
            wrapped = textwrap.fill(content, width=72, subsequent_indent='    ')
            print(f'  {GREEN}●{R} {wrapped}')
        elif re.match(r'\s*\d+[\.\)]\s', line):
            m = re.match(r'(\s*)(\d+[\.\)])\s*(.*)', line)
            if m:
                wrapped = textwrap.fill(m.group(3), width=72, subsequent_indent='     ')
                print(f'  {YELLOW}{m.group(2)}{R} {wrapped}')
        else:
            wrapped = textwrap.fill(line, width=76, initial_indent='  ', subsequent_indent='  ')
            print(wrapped)

    print(f'\n  {DIM}[{tokens} tokens | {tps:.1f} tok/s | GB10 CUDA]{R}')

except Exception as e:
    print(f'  Error: {e}')
"
    echo
done
