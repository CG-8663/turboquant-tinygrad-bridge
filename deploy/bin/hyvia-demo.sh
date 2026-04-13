#!/bin/bash
# Hyvia Interactive Demo — Streaming with conversation history
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
echo -e "\n${DIM}  Type a planning question. Say 'continue' for more detail.${RESET}"
echo -e "${DIM}  Ctrl+C to exit. 'new' to start a fresh conversation.${RESET}\n"

SYSTEM="You are Hyvia, a UK planning approval advisor. Provide: 1) Approval probability 2) Risk factors 3) Recommendations. Be specific about policies, cite paragraph numbers and legislation names. Respond directly, no internal reasoning. Format with markdown headers and bullet points."

# Conversation history — keeps context for follow-ups
HISTORY=""

while true; do
    printf "${GREEN}  You: ${RESET}"
    read -r PROMPT
    [ -z "$PROMPT" ] && continue

    # Reset conversation
    if [ "$PROMPT" = "new" ] || [ "$PROMPT" = "reset" ]; then
        HISTORY=""
        echo -e "\n${DIM}  Conversation reset.${RESET}\n"
        continue
    fi

    # Build prompt with history
    HISTORY="${HISTORY}<|im_start|>user\n${PROMPT}<|im_end|>\n"
    FULL_PROMPT="<|im_start|>system\n${SYSTEM}<|im_end|>\n${HISTORY}<|im_start|>assistant\n"

    echo -e "\n${CYAN}  Hyvia:${RESET}\n"

    # Stream + render markdown
    RESPONSE_TEXT=$(curl -sN "http://${HOST}:${PORT}/completion" \
        -H "Content-Type: application/json" \
        -d "{
            \"prompt\": \"${FULL_PROMPT}\",
            \"max_tokens\": 600,
            \"temperature\": 0.7,
            \"stream\": true,
            \"stop\": [\"<|im_end|>\", \"<|im_start|>\"]
        }" 2>/dev/null | python3 -c "
import json, re, sys, time

BOLD = '\033[1m'
DIM = '\033[2m'
GREEN = '\033[92m'
CYAN = '\033[96m'
YELLOW = '\033[93m'
WHITE = '\033[97m'
R = '\033[0m'

def fmt_bold(text):
    return re.sub(r'\*\*(.+?)\*\*', BOLD + WHITE + r'\1' + R, text)

def render_line(line):
    if not line.rstrip():
        print()
        return
    line = fmt_bold(line.rstrip())
    stripped = line.lstrip()
    indent_n = len(line) - len(line.lstrip())
    pad = '  '

    if stripped.startswith('### '):
        # strip the ansi from the header detection
        raw = re.sub(r'\033\[[^m]*m', '', stripped)
        if raw.startswith('### '):
            print(f'{pad}{BOLD}{CYAN}{stripped[4:]}{R}')
    elif stripped.startswith('## '):
        raw = re.sub(r'\033\[[^m]*m', '', stripped)
        if raw.startswith('## '):
            print(f'{pad}{BOLD}{CYAN}{stripped[3:]}{R}')
    elif stripped.startswith('- ') or stripped.startswith('* '):
        content = stripped[2:]
        print(f'{pad}{GREEN}●{R} {content}')
    elif re.match(r'\d+[\.\)]\s', re.sub(r'\033\[[^m]*m', '', stripped)):
        raw = re.sub(r'\033\[[^m]*m', '', stripped)
        m = re.match(r'(\d+[\.\)])\s*(.*)', raw)
        if m:
            # Re-apply bold formatting to the content part
            content = stripped[len(m.group(1))+1:].lstrip()
            print(f'{pad}{YELLOW}{m.group(1)}{R} {content}')
    else:
        print(f'{pad}{line.lstrip()}')

t0 = time.time()
n_tokens = 0
in_think = False
line_buf = ''
full_text = ''

for raw_line in sys.stdin:
    raw_line = raw_line.strip()
    if raw_line.startswith('data: '): raw_line = raw_line[6:]
    if not raw_line or raw_line == '[DONE]': continue

    try:
        d = json.loads(raw_line)
        tok = d.get('content', '')
        stop = d.get('stop', False)
    except:
        continue

    if stop:
        if line_buf.strip():
            render_line(line_buf)
        tps = d.get('timings', {}).get('predicted_per_second', 0)
        pp = d.get('timings', {}).get('prompt_per_second', 0)
        n_tok = d.get('tokens_predicted', n_tokens)
        print(f'\n{pad}{DIM}[{n_tok} tokens | prefill {pp:.0f} tok/s | decode {tps:.1f} tok/s | GB10 CUDA]{R}')
        # Output full text for history capture (to stderr)
        print(full_text, file=sys.stderr, end='')
        break

    n_tokens += 1

    if '<think>' in tok:
        in_think = True
        continue
    if '</think>' in tok:
        in_think = False
        continue
    if in_think:
        continue

    line_buf += tok
    full_text += tok

    while '\n' in line_buf:
        line, line_buf = line_buf.split('\n', 1)
        render_line(line)
        sys.stdout.flush()

if line_buf.strip():
    render_line(line_buf)

if n_tokens > 0 and not stop:
    elapsed = time.time() - t0
    tps = n_tokens / elapsed if elapsed > 0 else 0
    print(f'\n  {DIM}[{n_tokens} tokens | {tps:.1f} tok/s | GB10 CUDA]{R}')
    print(full_text, file=sys.stderr, end='')
" 2>/tmp/hyvia_last_response.txt)

    # Append assistant response to history for follow-ups
    LAST_RESPONSE=$(cat /tmp/hyvia_last_response.txt 2>/dev/null)
    HISTORY="${HISTORY}<|im_start|>assistant\n${LAST_RESPONSE}<|im_end|>\n"

    echo
done
