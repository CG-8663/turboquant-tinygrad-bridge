#!/bin/bash
# Hyvia Interactive Demo — Streaming responses
# Runs on GX10-001 GB10 CUDA via llama.cpp

BOLD="\033[1m"
DIM="\033[2m"
GREEN="\033[92m"
CYAN="\033[96m"
RESET="\033[0m"

HOST="192.168.68.61"
PORT="8080"

echo -e "\n${BOLD}${CYAN}╔══════════════════════════════════════════════════╗${RESET}"
echo -e "${BOLD}${CYAN}║  Hyvia — UK Planning Approval Advisor            ║${RESET}"
echo -e "${BOLD}${CYAN}║  Powered by Chronara Cluster (GX10 GB10 CUDA)    ║${RESET}"
echo -e "${BOLD}${CYAN}╚══════════════════════════════════════════════════╝${RESET}"
echo -e "\n${DIM}  Type your planning question. Ctrl+C to exit.${RESET}\n"

SYSTEM="You are Hyvia, a UK planning approval advisor. Provide: 1) Approval probability 2) Risk factors 3) Recommendations. Be specific about policies. Respond directly, no internal reasoning."

while true; do
    printf "${GREEN}  You: ${RESET}"
    read -r PROMPT
    [ -z "$PROMPT" ] && continue

    echo -e "\n${CYAN}  Hyvia:${RESET} \c"

    # Stream response — tokens appear as generated
    curl -sN "http://${HOST}:${PORT}/completion" \
        -H "Content-Type: application/json" \
        -d "{
            \"prompt\": \"<|im_start|>system\n${SYSTEM}<|im_end|>\n<|im_start|>user\n/no_think ${PROMPT}<|im_end|>\n<|im_start|>assistant\n\",
            \"max_tokens\": 400,
            \"temperature\": 0.7,
            \"stream\": true,
            \"stop\": [\"<|im_end|>\", \"<|im_start|>\"]
        }" 2>/dev/null | while IFS= read -r line; do
        # Each line is "data: {json}" in SSE format
        line="${line#data: }"
        [ -z "$line" ] && continue
        [ "$line" = "[DONE]" ] && break

        # Extract token, skip <think> blocks
        TOKEN=$(echo "$line" | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    tok = d.get('content', '')
    if '<think>' in tok or '</think>' in tok:
        pass  # skip think tokens
    else:
        print(tok, end='', flush=True)
except:
    pass
" 2>/dev/null)
        printf "%s" "$TOKEN"
    done

    echo -e "\n"
done
