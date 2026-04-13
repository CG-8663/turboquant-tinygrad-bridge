/**
 * tqbridge-menu — Interactive demo & test launcher for TQBridge.
 *
 * Presents a menu of all benchmarks, demos, tests, and tools.
 * Arrow keys / j/k to navigate, Enter to run, q to quit.
 * Runs in alternate screen buffer — clean terminal when done.
 *
 * Build:
 *   cc -O2 -o tqbridge-menu tqbridge-menu.c -lpthread
 *
 * Usage:
 *   ./tqbridge-menu
 *
 * James Tervit, Founder Chronara Group
 */

#include <stdarg.h>
#include <arpa/inet.h>
#include <errno.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <pthread.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/socket.h>
#include <sys/wait.h>
#include <termios.h>
#include <time.h>
#include <unistd.h>

/* ── ANSI ────────────────────────────────────────────────────── */

#define BOLD     "\033[1m"
#define DIM      "\033[2m"
#define GREEN    "\033[92m"
#define YELLOW   "\033[93m"
#define RED      "\033[91m"
#define CYAN     "\033[96m"
#define MAGENTA  "\033[95m"
#define WHITE    "\033[97m"
#define BG_CYAN  "\033[46m"
#define BG_GRAY  "\033[100m"
#define RESET    "\033[0m"
#define CUR_HOME "\033[H"
#define CLR_LINE "\033[2K"
#define CLR_SCR  "\033[2J"
#define HIDE_CUR "\033[?25l"
#define SHOW_CUR "\033[?25h"
#define ALT_ON   "\033[?1049h"
#define ALT_OFF  "\033[?1049l"

/* ── Menu item ───────────────────────────────────────────────── */

typedef struct {
    const char *key;          /* shortcut key: "1", "2", ... or section header */
    const char *label;        /* display name */
    const char *desc;         /* one-line description */
    const char *cmd;          /* shell command to run (NULL = section header) */
    const char *category;     /* grouping tag */
} menu_item;

/* All items — section headers have cmd=NULL */
static const menu_item ITEMS[] = {
    /* ── Live Monitors ── */
    {NULL, "LIVE MONITORS", NULL, NULL, NULL},
    {"1", "Cluster Monitor (C)",
     "Static terminal display — GPU detection, TB5 latency, network nodes",
     "deploy/bin/tqbridge-monitor", "monitor"},
    {"2", "Cluster Monitor (Python)",
     "Live stats via tinygrad — RTX thermal, Mac stats, node status",
     "/opt/homebrew/bin/python3.12 benchmarks/monitor.py", "monitor"},

    /* ── Real GPU Inference (visible on monitors) ── */
    {NULL, "REAL GPU INFERENCE (actual model generation)", NULL, NULL, NULL},
    {"3", "Round-Robin — ALL GPUs",
     "M3 → GX10-001 → GX10-002 → RTX → M1 — prove every GPU works",
     "/opt/homebrew/bin/python3.12 benchmarks/cluster_roundrobin.py", "gen"},
    {"4", "Cluster Generative — 3 prompts",
     "GX10 CUDA + M3 MLX generate real text, side by side, watch both GPUs",
     "/opt/homebrew/bin/python3.12 benchmarks/generative_cluster.py", "gen"},
    {"5", "Hyvia — interactive demo",
     "Chat with planning advisor on GX10-001 CUDA (strips thinking, clean output)",
     "deploy/bin/hyvia-demo.sh", "demo"},
    {"6", "Remittance — interactive demo",
     "Chat with transfer agent on GX10-002 CUDA (WhatsApp style)",
     "deploy/bin/remit-demo.sh", "demo"},
    {"6", "GX10 CUDA only — llama-bench",
     "Raw CUDA benchmark on GB10: prefill + decode tok/s",
     "ssh pxcghost@192.168.68.61 '~/turboquant/llama-cpp-turboquant/build/bin/llama-bench -m /home/pxcghost/models/Qwen3-8B-Q8_0.gguf -ngl 99 -p 512 -n 128'", "gen"},
    {"7", "M3 MLX only — generate",
     "MLX Metal inference on M3 Ultra, visible in macmon",
     "/opt/homebrew/bin/python3.12 -c \"from mlx_lm import load,generate; m,t=load('mlx-community/Qwen2.5-7B-Instruct-4bit'); print(generate(m,t,prompt='Explain GPU computing in 3 sentences',max_tokens=100,verbose=True))\"", "gen"},

    /* ── TQBridge Pipeline Tests ── */
    {NULL, "TQBRIDGE PIPELINE (compress + distribute)", NULL, NULL, NULL},
    {"7", "Chat Session — 30s sustained",
     "TriAttention+TurboQuant compress, distribute to GX10+M1, real TCP",
     "/opt/homebrew/bin/python3.12 benchmarks/sustained_bridge_test.py --scenario chat", "sustained"},
    {"8", "10 Users — 30s sustained",
     "10 concurrent users, KV compress + distribute to cluster",
     "/opt/homebrew/bin/python3.12 benchmarks/sustained_bridge_test.py --scenario multi_user", "sustained"},
    {"9", "405B Pipeline — snapshot",
     "405B dims: TriAttention → TurboQuant → network, one pass",
     "/opt/homebrew/bin/python3.12 benchmarks/real_bridge_test.py --model 405B --tokens 10000 --all", "real"},

    /* ── Model Fitting ── */
    {NULL, "MODEL FITTING (find best models per node)", NULL, NULL, NULL},
    {"0", "llmfit — this Mac",
     "Find which models fit on M3 Ultra (96GB Metal) with TQ+ filter",
     "LLMFIT_MODELS_DIR=/Volumes/18TB-Mirror/models/gguf deploy/bin/llmfit", "tools"},
    {"a", "llmfit — GX10-001",
     "Find which models fit on GB10 (124GB CUDA)",
     "ssh -t pxcghost@192.168.68.61 'LLMFIT_MODELS_DIR=/home/pxcghost/models LLAMA_CPP_PATH=~/turboquant/llama-cpp-turboquant/build/bin ~/.cargo/bin/llmfit'", "tools"},
    {"b", "llmfit — GX10-002",
     "Find which models fit on GB10 (122GB CUDA)",
     "ssh -t pxcghost@192.168.68.62 'LLMFIT_MODELS_DIR=/home/pxcghost/models LLAMA_CPP_PATH=~/llama.cpp/build/bin ~/.cargo/bin/llmfit'", "tools"},
    {"r", "llmfit — RTX PRO 6000 (eGPU)",
     "Find which models fit on Blackwell 96GB CUDA (via GX10 with --memory 96G)",
     "ssh -t pxcghost@192.168.68.61 'LLMFIT_MODELS_DIR=/home/pxcghost/models LLAMA_CPP_PATH=~/turboquant/llama-cpp-turboquant/build/bin ~/.cargo/bin/llmfit --memory 96G'", "tools"},

    /* ── Visual Demos ── */
    {NULL, "VISUAL DEMOS (animated dashboards)", NULL, NULL, NULL},
    {"u", "405B x 1T Visual Demo",
     "Animated: 405B across 4 nodes, 1T context, compression dashboard",
     "/opt/homebrew/bin/python3.12 benchmarks/long_context_demo.py --model 'Llama-3.1-405B' --context 1000000000000", "demo"},
    {"v", "27B Long Context Visual",
     "Animated: 27B at 10M tokens, NIAH validation dashboard",
     "/opt/homebrew/bin/python3.12 benchmarks/long_context_demo.py", "demo"},
    {"w", "Multi-User Stress Visual",
     "Animated: 50 users, 1M+ tokens, per-user accuracy dashboard",
     "/opt/homebrew/bin/python3.12 benchmarks/multi_user_stress.py", "demo"},
    {"x", "Single vs Cluster Visual",
     "Animated: single machine limits vs cluster comparison",
     "/opt/homebrew/bin/python3.12 benchmarks/single_vs_cluster.py", "demo"},

    /* ── Benchmarks ── */
    {NULL, "BENCHMARKS", NULL, NULL, NULL},
    {"9", "C Driver Self-Test",
     "Compression round-trip validation — turbo2/3/4, Q8_0, wire protocol",
     "deploy/bin/tqbridge-server --self-test 2>/dev/null || "
     "llama-cpp-turboquant/build/tqbridge-server --self-test", "bench"},
    {"0", "C Driver Benchmark",
     "Compression throughput — compress/decompress timing per format",
     "deploy/bin/tqbridge-server --benchmark 2>/dev/null || "
     "llama-cpp-turboquant/build/tqbridge-server --benchmark", "bench"},
    {"a", "TurboQuant+ Demo",
     "Quick compress/decompress simulation at 2/3/4-bit widths",
     "cd turboquant_plus && /opt/homebrew/bin/python3.12 benchmarks/demo.py", "bench"},
    {"b", "NIAH Test (requires model)",
     "Needle-In-A-Haystack KV quality benchmark (Kamradt 2023 methodology)",
     "echo 'Usage: python3 turboquant_plus/scripts/niah_test.py /path/to/llama.cpp /path/to/model.gguf' && "
     "echo 'Run manually with your model path'", "bench"},

    /* ── Test Suites ── */
    {NULL, "TEST SUITES", NULL, NULL, NULL},
    {"c", "Full Python Test Suite",
     "All 151 tests — compression, bridge, wire, router, native, cleanup",
     "/opt/homebrew/bin/python3.12 -m pytest tests/ -v --tb=short 2>&1 | head -80", "test"},
    {"d", "Smoke Test",
     "Quick import validation — confirms package loads correctly",
     "/opt/homebrew/bin/python3.12 -m pytest tests/test_smoke.py -v", "test"},
    {"e", "Compression Tests",
     "turbo3, Q8_0, asymmetric K/V compression pipeline",
     "/opt/homebrew/bin/python3.12 -m pytest tests/test_compression.py -v", "test"},
    {"f", "Wire Protocol Tests",
     "Header serialization, CRC32, format negotiation",
     "/opt/homebrew/bin/python3.12 -m pytest tests/test_wire.py -v", "test"},
    {"g", "Native C Bridge Tests",
     "ctypes bindings, C↔Python parity, precomputed rotation",
     "/opt/homebrew/bin/python3.12 -m pytest tests/test_native.py -v", "test"},
    {"h", "Router Tests",
     "Multi-node KV distribution, TCP transport, failover",
     "/opt/homebrew/bin/python3.12 -m pytest tests/test_router.py -v", "test"},
    {"i", "Bridge Pipeline Tests",
     "End-to-end Metal↔CUDA transfer (requires hardware)",
     "/opt/homebrew/bin/python3.12 -m pytest tests/test_bridge.py -v", "test"},
    {"j", "CUDA Kernel Tests",
     "PolarQuant GPU kernels (requires NV device)",
     "/opt/homebrew/bin/python3.12 -m pytest tests/test_cuda_kernels.py -v", "test"},

    /* ── Deployment ── */
    {NULL, "DEPLOYMENT & TOOLS", NULL, NULL, NULL},
    {"k", "Check Upstream Dependencies",
     "Compare submodule pins against upstream — tinygrad, TurboQuant+",
     "/opt/homebrew/bin/python3.12 tools/check_upstream.py --fetch --verbose", "deploy"},
    {"l", "Start Decode Server",
     "Launch tqbridge-server on port 9473 (Ctrl+C to stop)",
     "deploy/bin/tqbridge-server --port 9473 --head-dim 128", "deploy"},
    {"n", "Docker Build & Run",
     "Build chronaragroup/chronara-bridge image and run self-test",
     "docker build -t chronaragroup/chronara-bridge deploy/ && "
     "docker run --rm chronaragroup/chronara-bridge --self-test", "deploy"},
    {"o", "GX10 Node Setup",
     "Deploy decode node to GX10 (run ON the GX10)",
     "cat deploy/gx10_setup.sh", "deploy"},
};

#define N_ITEMS (sizeof(ITEMS) / sizeof(ITEMS[0]))

/* ── Terminal ────────────────────────────────────────────────── */

static struct termios g_orig_termios;
static int g_raw_mode = 0;

static volatile int g_resized = 0;

static void on_winch(int sig) {
    (void)sig;
    g_resized = 1;
}

static void term_raw(void) {
    tcgetattr(STDIN_FILENO, &g_orig_termios);
    struct termios raw = g_orig_termios;
    raw.c_lflag &= ~(ECHO | ICANON | ISIG);
    raw.c_cc[VMIN] = 1;   /* block until at least 1 byte */
    raw.c_cc[VTIME] = 0;  /* no timeout — pure blocking read */
    tcsetattr(STDIN_FILENO, TCSAFLUSH, &raw);
    g_raw_mode = 1;

    /* SIGWINCH for terminal resize — interrupts the blocking read */
    struct sigaction sa;
    sa.sa_handler = on_winch;
    sa.sa_flags = 0;  /* no SA_RESTART so read() returns EINTR */
    sigemptyset(&sa.sa_mask);
    sigaction(SIGWINCH, &sa, NULL);
}

static void term_restore(void) {
    if (g_raw_mode) {
        tcsetattr(STDIN_FILENO, TCSAFLUSH, &g_orig_termios);
        g_raw_mode = 0;
    }
}

static int get_rows(void) {
    struct winsize ws;
    if (ioctl(STDOUT_FILENO, TIOCGWINSZ, &ws) == 0 && ws.ws_row > 0)
        return ws.ws_row;
    return 40;
}

static int get_cols(void) {
    struct winsize ws;
    if (ioctl(STDOUT_FILENO, TIOCGWINSZ, &ws) == 0 && ws.ws_col > 0)
        return ws.ws_col;
    return 80;
}

/* ── Selectable items index ──────────────────────────────────── */

static int g_selectable[N_ITEMS];  /* indices into ITEMS that are runnable */
static int g_n_selectable = 0;

static void build_selectable(void) {
    g_n_selectable = 0;
    for (int i = 0; i < (int)N_ITEMS; i++) {
        if (ITEMS[i].cmd != NULL)
            g_selectable[g_n_selectable++] = i;
    }
}

/* ── Render ──────────────────────────────────────────────────── */

static void render(int sel, int scroll) {
    int rows = get_rows();
    int cols = get_cols();
    (void)cols;

    /* Build frame in buffer */
    char buf[32768];
    int pos = 0;
    int remain;

    #define EMIT(...) do { \
        remain = (int)sizeof(buf) - pos; \
        if (remain > 10) pos += snprintf(buf + pos, remain, __VA_ARGS__); \
    } while(0)

    EMIT(CUR_HOME);

    /* Header — 3 lines */
    EMIT("\033[1;1H" CLR_LINE BOLD CYAN
         "  ╔══════════════════════════════════════════════════════════════╗\n");
    EMIT(CLR_LINE
         "  ║   TQBridge Demo & Test Menu                 Chronara Group  ║\n");
    EMIT(CLR_LINE
         "  ╚══════════════════════════════════════════════════════════════╝" RESET "\n");
    EMIT(CLR_LINE "\n");

    /* Menu body: rows-6 lines available (3 header + 1 blank + 2 footer) */
    int body_rows = rows - 6;
    if (body_rows < 5) body_rows = 5;

    /* Adjust scroll to keep selection visible */
    /* Map sel (index into g_selectable) to display line */
    /* We need to render ITEMS in order, with headers inline */

    /* Count display lines and find which line the selection is on */
    int display_lines = 0;
    int sel_display_line = 0;
    for (int i = 0; i < (int)N_ITEMS; i++) {
        if (ITEMS[i].cmd == NULL) {
            display_lines++;  /* section header */
        } else {
            /* Find which selectable index this is */
            for (int s = 0; s < g_n_selectable; s++) {
                if (g_selectable[s] == i && s == sel) {
                    sel_display_line = display_lines;
                }
            }
            display_lines++;
        }
    }

    /* Adjust scroll */
    if (sel_display_line < scroll) scroll = sel_display_line;
    if (sel_display_line >= scroll + body_rows) scroll = sel_display_line - body_rows + 1;
    if (scroll < 0) scroll = 0;

    /* Render visible lines */
    int line = 0;
    int rendered = 0;
    for (int i = 0; i < (int)N_ITEMS && rendered < body_rows; i++) {
        if (line < scroll) {
            line++;
            continue;
        }

        EMIT(CLR_LINE);

        if (ITEMS[i].cmd == NULL) {
            /* Section header */
            EMIT("  " BOLD CYAN "── %s ──" RESET "\n", ITEMS[i].label);
        } else {
            /* Selectable item — check if this is the selected one */
            int is_sel = 0;
            for (int s = 0; s < g_n_selectable; s++) {
                if (g_selectable[s] == i && s == sel) {
                    is_sel = 1;
                    break;
                }
            }

            if (is_sel) {
                EMIT("  " BG_CYAN BOLD WHITE " %s " RESET
                     BG_CYAN WHITE " %-30s" RESET
                     "  " DIM "%s" RESET "\n",
                     ITEMS[i].key, ITEMS[i].label, ITEMS[i].desc);
            } else {
                EMIT("   " BOLD "%s" RESET
                     "  %-30s  " DIM "%s" RESET "\n",
                     ITEMS[i].key, ITEMS[i].label, ITEMS[i].desc);
            }
        }

        line++;
        rendered++;
    }

    /* Clear remaining body lines */
    while (rendered < body_rows) {
        EMIT(CLR_LINE "\n");
        rendered++;
    }

    /* Footer — 2 lines */
    EMIT(CLR_LINE "\n");
    EMIT(CLR_LINE "  " DIM "↑↓/jk Navigate  Enter Run  q Quit  "
         "1-9/a-o Shortcut" RESET);

    /* Scroll indicator */
    if (display_lines > body_rows) {
        int pct = (scroll * 100) / (display_lines - body_rows + 1);
        EMIT("  " DIM "[%d%%]" RESET, pct);
    }

    /* Single write */
    (void)!write(STDOUT_FILENO, buf, pos);

    #undef EMIT
}

/* ── Run command ─────────────────────────────────────────────── */

static void run_item(int sel) {
    if (sel < 0 || sel >= g_n_selectable) return;
    const menu_item *item = &ITEMS[g_selectable[sel]];
    if (!item->cmd) return;

    /* Leave alternate screen, restore terminal */
    printf(SHOW_CUR ALT_OFF);
    fflush(stdout);
    term_restore();

    printf("\n" BOLD CYAN "═══ Running: %s ═══" RESET "\n", item->label);
    printf(DIM "  %s" RESET "\n\n", item->cmd);
    fflush(stdout);

    /* Run command */
    int ret = system(item->cmd);
    (void)ret;

    printf("\n" DIM "─── Press any key to return to menu ───" RESET);
    fflush(stdout);

    /* Wait for keypress in cooked mode */
    {
        struct termios t;
        tcgetattr(STDIN_FILENO, &t);
        struct termios raw = t;
        raw.c_lflag &= ~(ECHO | ICANON);
        raw.c_cc[VMIN] = 1;
        raw.c_cc[VTIME] = 0;
        tcsetattr(STDIN_FILENO, TCSAFLUSH, &raw);
        char c;
        (void)!read(STDIN_FILENO, &c, 1);
        tcsetattr(STDIN_FILENO, TCSAFLUSH, &t);
    }

    /* Re-enter alternate screen */
    printf(ALT_ON HIDE_CUR CLR_SCR);
    fflush(stdout);
    term_raw();
}

/* ── Main ────────────────────────────────────────────────────── */

int main(void) {
    build_selectable();

    printf(ALT_ON HIDE_CUR CLR_SCR);
    fflush(stdout);
    term_raw();

    /* Ensure cleanup on exit */
    atexit(term_restore);

    int sel = 0;
    int scroll = 0;
    int running = 1;
    int dirty = 1;  /* render on first iteration */

    while (running) {
        if (dirty) {
            render(sel, scroll);
            dirty = 0;
        }

        /* Block until input or SIGWINCH (resize) */
        char seq[4] = {0};
        int n = (int)read(STDIN_FILENO, seq, sizeof(seq));
        if (n <= 0) {
            /* EINTR from SIGWINCH — just redraw */
            if (g_resized) {
                g_resized = 0;
                dirty = 1;
            }
            continue;
        }
        dirty = 1;  /* any input triggers redraw */

        if (seq[0] == 'q' || seq[0] == 'Q') {
            running = 0;
        } else if (seq[0] == '\r' || seq[0] == '\n') {
            run_item(sel);
        } else if (seq[0] == 'j' || (n == 3 && seq[0] == '\033' && seq[1] == '[' && seq[2] == 'B')) {
            /* Down */
            if (sel < g_n_selectable - 1) sel++;
        } else if (seq[0] == 'k' || (n == 3 && seq[0] == '\033' && seq[1] == '[' && seq[2] == 'A')) {
            /* Up */
            if (sel > 0) sel--;
        } else if (seq[0] == 'G' || (n == 3 && seq[0] == '\033' && seq[1] == '[' && seq[2] == 'F')) {
            /* End */
            sel = g_n_selectable - 1;
        } else if (seq[0] == 'g' || (n == 3 && seq[0] == '\033' && seq[1] == '[' && seq[2] == 'H')) {
            /* Home */
            sel = 0;
        } else {
            /* Shortcut key: find matching item */
            for (int i = 0; i < g_n_selectable; i++) {
                if (ITEMS[g_selectable[i]].key &&
                    ITEMS[g_selectable[i]].key[0] == seq[0]) {
                    sel = i;
                    run_item(sel);
                    break;
                }
            }
        }
    }

    printf(SHOW_CUR ALT_OFF);
    printf("  " DIM "Menu closed." RESET "\n");
    fflush(stdout);
    return 0;
}
