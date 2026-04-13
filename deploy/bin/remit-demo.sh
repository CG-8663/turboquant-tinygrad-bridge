#!/bin/bash
# Remittance Interactive Demo — Streaming WhatsApp-style
# Runs on GX10-002 GB10 CUDA via llama.cpp

BOLD="\033[1m"
DIM="\033[2m"
GREEN="\033[92m"
CYAN="\033[96m"
RESET="\033[0m"

HOST="192.168.68.62"
PORT="8081"

echo -e "\n${BOLD}${CYAN}╔══════════════════════════════════════════════════╗${RESET}"
echo -e "${BOLD}${CYAN}║  Chronara Remittance — Transfer Assistant         ║${RESET}"
echo -e "${BOLD}${CYAN}║  Powered by Chronara Cluster (GX10 GB10 CUDA)    ║${RESET}"
echo -e "${BOLD}${CYAN}╚══════════════════════════════════════════════════╝${RESET}"
echo -e "\n${DIM}  WhatsApp-style. Type your question. Ctrl+C to exit.${RESET}\n"

SYSTEM="You are a Chronara remittance assistant. UK to Philippines transfers. Check rates, fees, timelines, compliance. Be concise for WhatsApp. Respond directly, no reasoning."

while true; do
    printf "${GREEN}  You: ${RESET}"
    read -r PROMPT
    [ -z "$PROMPT" ] && continue

    echo -e "\n${CYAN}  Agent:${RESET} \c"

    curl -sN "http://${HOST}:${PORT}/completion" \
        -H "Content-Type: application/json" \
        -d "{
            \"prompt\": \"<|im_start|>system\n${SYSTEM}<|im_end|>\n<|im_start|>user\n/no_think ${PROMPT}<|im_end|>\n<|im_start|>assistant\n\",
            \"max_tokens\": 200,
            \"temperature\": 0.7,
            \"stream\": true,
            \"stop\": [\"<|im_end|>\", \"<|im_start|>\"]
        }" 2>/dev/null | while IFS= read -r line; do
        line="${line#data: }"
        [ -z "$line" ] && continue
        [ "$line" = "[DONE]" ] && break

        TOKEN=$(echo "$line" | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    tok = d.get('content', '')
    if '<think>' in tok or '</think>' in tok:
        pass
    else:
        print(tok, end='', flush=True)
except:
    pass
" 2>/dev/null)
        printf "%s" "$TOKEN"
    done

    echo -e "\n"
done
