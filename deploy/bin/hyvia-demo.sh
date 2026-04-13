#!/bin/bash
# Hyvia Interactive Demo — wrapper for Python version
# Runs on GX10-001 GB10 CUDA via llama.cpp
exec /opt/homebrew/bin/python3.12 "$(dirname "$0")/hyvia-demo.py" "$@"

# Legacy bash version below (kept for reference, not executed)
exit 0

HOST="192.168.68.61"
PORT="8080"

BOLD="\033[1m"
DIM="\033[2m"
GREEN="\033[92m"
CYAN="\033[96m"
YELLOW="\033[93m"
WHITE="\033[97m"
RESET="\033[0m"

echo -e "\n${BOLD}${CYAN}╔══════════════════════════════════════════════════╗${RESET}"
echo -e "${BOLD}${CYAN}║  Hyvia — UK Planning Approval Advisor            ║${RESET}"
echo -e "${BOLD}${CYAN}║  Chronara Cluster (GX10 GB10 CUDA)               ║${RESET}"
echo -e "${BOLD}${CYAN}╚══════════════════════════════════════════════════╝${RESET}"
echo -e "\n${DIM}  Type a planning question. 'continue' for more. 'new' to reset.${RESET}"
echo -e "${DIM}  Ctrl+C to exit.${RESET}\n"

SYSTEM="You are Hyvia, a UK planning approval advisor. Answer directly. Never use <think> tags. Provide: 1) Approval probability if applicable 2) Key risk factors 3) Recommendations. Cite specific policies, paragraph numbers, and legislation. Format with markdown."

HISTORY=""
LOGDIR="/Volumes/18TB-Mirror/HYVIA-DEMO-TRAINING/baselines"
# Fallback if 18TB not mounted locally
if [ ! -d "/Volumes/18TB-Mirror" ]; then
    LOGDIR="/Volumes/Chronara-Storage/Projects/clients/Hyvia-Projects/Hyvia-Planning-Running-Demo/baselines"
fi
mkdir -p "$LOGDIR"
LOGFILE="${LOGDIR}/baseline-$(date +%Y%m%d-%H%M%S).md"
echo "# Hyvia Baseline Capture — $(date)" > "$LOGFILE"
echo "Model: Qwen3-8B Q8_0 | Node: GX10-001 GB10 CUDA" >> "$LOGFILE"
echo "" >> "$LOGFILE"
echo -e "${DIM}  Logging to: ${LOGFILE}${RESET}\n"

while true; do
    printf "${GREEN}  You: ${RESET}"
    read -r PROMPT
    [ -z "$PROMPT" ] && continue

    if [ "$PROMPT" = "new" ] || [ "$PROMPT" = "reset" ]; then
        HISTORY=""
        echo -e "\n${DIM}  Conversation reset.${RESET}\n"
        continue
    fi

    HISTORY="${HISTORY}<|im_start|>user\n${PROMPT}<|im_end|>\n"
    # Seed the assistant response to prevent <think> mode
    FULL_PROMPT="<|im_start|>system\n${SYSTEM}<|im_end|>\n${HISTORY}<|im_start|>assistant\nHere is my analysis:\n"

    echo -e "\n${CYAN}  Hyvia:${RESET}"

    # Stream tokens and render markdown line-by-line
    RESP_FILE=$(mktemp /tmp/hyvia_resp.XXXXXX)

    curl -sN "http://${HOST}:${PORT}/completion" \
        -H "Content-Type: application/json" \
        -d "{
            \"prompt\": \"${FULL_PROMPT}\",
            \"max_tokens\": 2000,
            \"temperature\": 0.7,
            \"stream\": true,
            \"stop\": [\"<|im_end|>\", \"<|im_start|>\"]
        }" 2>/dev/null | python3 -uc "
import json, re, sys, time

BOLD = '\033[1m'
DIM = '\033[2m'
GREEN = '\033[92m'
CYAN = '\033[96m'
YELLOW = '\033[93m'
WHITE = '\033[97m'
R = '\033[0m'

def fmt(line):
    \"\"\"Render one line with markdown formatting.\"\"\"
    if not line.rstrip():
        print(flush=True)
        return
    # Bold **text**
    line = re.sub(r'\*\*(.+?)\*\*', BOLD + WHITE + r'\g<1>' + R, line.rstrip())
    raw = re.sub(r'\033\[[^m]*m', '', line)  # strip ansi for pattern matching
    stripped = raw.lstrip()

    if stripped.startswith('### '):
        print(f'  {BOLD}{CYAN}{line.lstrip()[4:]}{R}', flush=True)
    elif stripped.startswith('## '):
        print(f'  {BOLD}{CYAN}{line.lstrip()[3:]}{R}', flush=True)
    elif stripped.startswith('# '):
        print(f'  {BOLD}{CYAN}{line.lstrip()[2:]}{R}', flush=True)
    elif stripped.startswith('- ') or stripped.startswith('* '):
        print(f'  {GREEN}●{R} {line.lstrip()[2:]}', flush=True)
    elif re.match(r'\d+[\.\)]\s', stripped):
        m = re.match(r'(\d+[\.\)])\s*(.*)', stripped)
        if m:
            print(f'  {YELLOW}{m.group(1)}{R} {line.lstrip()[len(m.group(1))+1:].lstrip()}', flush=True)
    else:
        print(f'  {line.lstrip()}', flush=True)

t0 = time.time()
n = 0
in_think = False
buf = ''
full = ''
tps_final = 0
pp_final = 0
n_final = 0

for raw in sys.stdin:
    raw = raw.strip()
    if raw.startswith('data: '): raw = raw[6:]
    if not raw or raw == '[DONE]': continue
    try:
        d = json.loads(raw)
    except:
        continue

    tok = d.get('content', '')
    stop = d.get('stop', False)

    if stop:
        if buf.strip(): fmt(buf)
        tps_final = d.get('timings', {}).get('predicted_per_second', 0)
        pp_final = d.get('timings', {}).get('prompt_per_second', 0)
        n_final = d.get('tokens_predicted', n)
        break

    # Skip think blocks
    if '<think>' in tok: in_think = True; continue
    if '</think>' in tok: in_think = False; continue
    if in_think: continue

    n += 1
    buf += tok
    full += tok

    # Render complete lines immediately
    while '\n' in buf:
        line, buf = buf.split('\n', 1)
        fmt(line)

# Flush remaining
if buf.strip(): fmt(buf)

# Footer
if n_final > 0 or n > 0:
    tokens = n_final if n_final else n
    print(f'\n  {DIM}[{tokens} tokens | prefill {pp_final:.0f} tok/s | decode {tps_final:.1f} tok/s | GB10 CUDA]{R}', flush=True)

# Save full text for conversation history
with open('$RESP_FILE', 'w') as f:
    f.write(full)
"

    # Append to conversation history
    LAST=$(cat "$RESP_FILE" 2>/dev/null)
    HISTORY="${HISTORY}<|im_start|>assistant\nHere is my analysis:\n${LAST}<|im_end|>\n"

    # Log Q&A to baseline file
    echo "## Q: ${PROMPT}" >> "$LOGFILE"
    echo "" >> "$LOGFILE"
    echo "${LAST}" >> "$LOGFILE"
    echo "" >> "$LOGFILE"
    echo "---" >> "$LOGFILE"
    echo "" >> "$LOGFILE"

    rm -f "$RESP_FILE"
    echo
done
