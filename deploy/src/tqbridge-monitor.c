/**
 * tqbridge-monitor — Live TB5 eGPU cluster monitor (compiled C binary).
 *
 * Static terminal display like macmon/nvtop. Shows:
 *   - GPU detection (NV via tinygrad RM, Metal via IOKit)
 *   - TB5 round-trip latency (PCIe probe)
 *   - Network node discovery (subnet scan for tqbridge port 9473)
 *   - Mac system stats (load, memory pressure)
 *
 * Build:
 *   cc -O2 -o tqbridge-monitor tqbridge-monitor.c -lpthread
 *
 * Usage:
 *   ./tqbridge-monitor
 *   ./tqbridge-monitor --interval 0.5
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
#include <sys/sysctl.h>
#include <sys/time.h>
#include <time.h>
#include <unistd.h>

/* ── ANSI escape codes ───────────────────────────────────────── */

#define BOLD     "\033[1m"
#define DIM      "\033[2m"
#define GREEN    "\033[92m"
#define YELLOW   "\033[93m"
#define RED      "\033[91m"
#define CYAN     "\033[96m"
#define MAGENTA  "\033[95m"
#define RESET    "\033[0m"
#define CUR_HOME "\033[H"
#define CLR_LINE "\033[2K"
#define HIDE_CUR "\033[?25l"
#define SHOW_CUR "\033[?25h"
#define CLR_SCR  "\033[2J"
#define ALT_ON   "\033[?1049h"  /* enter alternate screen buffer (like vim/htop) */
#define ALT_OFF  "\033[?1049l"  /* leave alternate screen buffer */

#define TQBRIDGE_PORT  9473
#define MAX_NODES      32
#define BAR_WIDTH      20
#define MAX_SAMPLES    60
#define FRAME_LINES    45
#define SUBNET_PREFIX  "192.168.68."

/* IPs to skip during subnet scan — wifi/management, not cluster */
static const char *SKIP_IPS[] = {
    "192.168.68.51",   /* wifi management */
    "192.168.68.54",   /* M3 wifi */
    "192.168.68.59",   /* M3 wifi */
    "192.168.68.65",   /* GX10 wifi */
    NULL,
};

/* ── Types ───────────────────────────────────────────────────── */

typedef struct {
    char     name[32];
    char     ip[20];
    int      online;
    double   latency_ms;
} node_info;

typedef struct {
    double samples[MAX_SAMPLES];
    int    count;
    int    head;   /* circular buffer write position */
} sample_ring;

/* ── Globals ─────────────────────────────────────────────────── */

static volatile int g_running = 1;

/* Node discovery */
static node_info       g_nodes[MAX_NODES];
static int             g_node_count = 0;
static pthread_mutex_t g_node_lock = PTHREAD_MUTEX_INITIALIZER;

/* Known nodes */
typedef struct { const char *ip; const char *name; } known_node;
static const known_node KNOWN[] = {
    {"192.168.68.61", "GX10-001"},
    {"192.168.68.62", "GX10-002"},
    {"192.168.68.50", "M1 Max"},
};
/* M3 Ultra is always shown as LOCAL (it's this machine) */
#define SHOW_LOCAL_M3 1
#define N_KNOWN (sizeof(KNOWN) / sizeof(KNOWN[0]))

/* GPU detection */
static int  g_nv_detected = 0;
static char g_nv_arch[64] = "not connected";
static int  g_metal_detected = 0;
static char g_metal_chip[128] = "Apple Silicon";

/* ── Timing ──────────────────────────────────────────────────── */

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

/* ── Sample ring buffer ──────────────────────────────────────── */

static void ring_push(sample_ring *r, double val) {
    r->samples[r->head] = val;
    r->head = (r->head + 1) % MAX_SAMPLES;
    if (r->count < MAX_SAMPLES) r->count++;
}

static double ring_avg(const sample_ring *r) {
    if (r->count == 0) return 0.0;
    double sum = 0.0;
    for (int i = 0; i < r->count; i++) sum += r->samples[i];
    return sum / r->count;
}

static double ring_min(const sample_ring *r) {
    if (r->count == 0) return 0.0;
    double m = r->samples[0];
    for (int i = 1; i < r->count; i++)
        if (r->samples[i] < m) m = r->samples[i];
    return m;
}

static double ring_max(const sample_ring *r) {
    if (r->count == 0) return 0.0;
    double m = r->samples[0];
    for (int i = 1; i < r->count; i++)
        if (r->samples[i] > m) m = r->samples[i];
    return m;
}

/* ── Bar rendering ───────────────────────────────────────────── */

static void render_bar(char *buf, size_t bufsz, double val, double max_val,
                       const char *color) {
    int filled = (int)(BAR_WIDTH * val / max_val);
    if (filled > BAR_WIDTH) filled = BAR_WIDTH;
    if (filled < 0) filled = 0;

    char blocks[BAR_WIDTH + 1];
    char empty[BAR_WIDTH + 1];
    memset(blocks, 0, sizeof(blocks));
    memset(empty, 0, sizeof(empty));

    /* UTF-8 block chars */
    int bi = 0, ei = 0;
    for (int i = 0; i < filled; i++) {
        /* █ = U+2588 = E2 96 88 */
        blocks[bi++] = '\xe2'; blocks[bi++] = '\x96'; blocks[bi++] = '\x88';
    }
    blocks[bi] = '\0';
    for (int i = filled; i < BAR_WIDTH; i++) {
        /* ░ = U+2591 = E2 96 91 */
        empty[ei++] = '\xe2'; empty[ei++] = '\x96'; empty[ei++] = '\x91';
    }
    empty[ei] = '\0';

    snprintf(buf, bufsz, "%s%s" RESET "%s", color, blocks, empty);
}

/* ── Network probe ───────────────────────────────────────────── */

static int probe_host(const char *ip, int port, double *latency_ms) {
    int sock = socket(AF_INET, SOCK_STREAM, 0);
    if (sock < 0) return 0;

    /* Non-blocking connect with timeout */
    int flags = fcntl(sock, F_GETFL, 0);
    fcntl(sock, F_SETFL, flags | O_NONBLOCK);

    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_port = htons(port);
    inet_pton(AF_INET, ip, &addr.sin_addr);

    double t0 = now_ms();
    int ret = connect(sock, (struct sockaddr *)&addr, sizeof(addr));

    if (ret < 0 && errno == EINPROGRESS) {
        fd_set wset;
        FD_ZERO(&wset);
        FD_SET(sock, &wset);
        struct timeval tv = {0, 300000};  /* 300ms timeout */
        ret = select(sock + 1, NULL, &wset, NULL, &tv);
        if (ret > 0) {
            int err = 0;
            socklen_t len = sizeof(err);
            getsockopt(sock, SOL_SOCKET, SO_ERROR, &err, &len);
            if (err == 0) {
                *latency_ms = now_ms() - t0;
                close(sock);
                return 1;
            }
        }
    } else if (ret == 0) {
        *latency_ms = now_ms() - t0;
        close(sock);
        return 1;
    }

    close(sock);
    return 0;
}

/* ── Node discovery thread ───────────────────────────────────── */

static void update_node(const char *name, const char *ip, int online, double lat) {
    pthread_mutex_lock(&g_node_lock);
    /* Find existing or add new */
    int idx = -1;
    for (int i = 0; i < g_node_count; i++) {
        if (strcmp(g_nodes[i].ip, ip) == 0) { idx = i; break; }
    }
    if (idx < 0 && g_node_count < MAX_NODES) {
        idx = g_node_count++;
    }
    if (idx >= 0) {
        strncpy(g_nodes[idx].name, name, sizeof(g_nodes[idx].name) - 1);
        strncpy(g_nodes[idx].ip, ip, sizeof(g_nodes[idx].ip) - 1);
        g_nodes[idx].online = online;
        g_nodes[idx].latency_ms = lat;
    }
    pthread_mutex_unlock(&g_node_lock);
}

static void *discovery_thread(void *arg) {
    (void)arg;
    while (g_running) {
        /* Probe known nodes */
        for (int i = 0; i < (int)N_KNOWN; i++) {
            double lat = 0;
            int up = probe_host(KNOWN[i].ip, TQBRIDGE_PORT, &lat);
            update_node(KNOWN[i].name, KNOWN[i].ip, up, lat);
        }

        /* Scan subnet for unknown nodes */
        for (int i = 1; i < 255 && g_running; i++) {
            char ip[20];
            snprintf(ip, sizeof(ip), "%s%d", SUBNET_PREFIX, i);

            /* Skip known and wifi/management IPs */
            int is_known = 0;
            for (int k = 0; k < (int)N_KNOWN; k++) {
                if (strcmp(ip, KNOWN[k].ip) == 0) { is_known = 1; break; }
            }
            if (!is_known) {
                for (int s = 0; SKIP_IPS[s] != NULL; s++) {
                    if (strcmp(ip, SKIP_IPS[s]) == 0) { is_known = 1; break; }
                }
            }
            if (is_known) continue;

            double lat = 0;
            if (probe_host(ip, TQBRIDGE_PORT, &lat)) {
                char name[32];
                snprintf(name, sizeof(name), "node-%d", i);
                update_node(name, ip, 1, lat);
            }
        }

        sleep(10);
    }
    return NULL;
}

/* ── GPU detection ───────────────────────────────────────────── */

static void detect_gpus(void) {
    /* Metal: detect on macOS via sysctl */
#ifdef __APPLE__
    g_metal_detected = 1;
    char brand[128] = {0};
    size_t len = sizeof(brand);
    if (sysctlbyname("machdep.cpu.brand_string", brand, &len, NULL, 0) == 0) {
        strncpy(g_metal_chip, brand, sizeof(g_metal_chip) - 1);
    }
#endif

    /* NV: detect via system_profiler (macOS) or /proc (Linux) */
#ifdef __APPLE__
    /* SPDisplaysDataType shows PCIe GPUs including TB5 eGPU */
    FILE *fp = popen(
        "system_profiler SPDisplaysDataType 2>/dev/null"
        " | grep -A8 'Vendor: NVIDIA'", "r");
    if (fp) {
        char buf[256];
        char device_id[32] = {0};
        char pcie_width[16] = {0};
        while (fgets(buf, sizeof(buf), fp)) {
            if (strstr(buf, "Vendor: NVIDIA")) g_nv_detected = 1;
            char *p;
            if ((p = strstr(buf, "Device ID:")) != NULL)
                sscanf(p + 10, " %31s", device_id);
            if ((p = strstr(buf, "PCIe Lane Width:")) != NULL)
                sscanf(p + 16, " %15s", pcie_width);
        }
        pclose(fp);
        if (g_nv_detected) {
            /* Identify arch from PCI device ID */
            const char *arch = "NVIDIA GPU";
            if (strstr(device_id, "0x2b") || strstr(device_id, "0x2B"))
                arch = "Blackwell";
            else if (strstr(device_id, "0x27") || strstr(device_id, "0x28"))
                arch = "Ada Lovelace";
            else if (strstr(device_id, "0x22") || strstr(device_id, "0x23"))
                arch = "Hopper";
            snprintf(g_nv_arch, sizeof(g_nv_arch), "%s (TB5)", arch);
        }
    }
    /* Also check for TB5 enclosure */
    if (g_nv_detected) {
        fp = popen(
            "system_profiler SPThunderboltDataType 2>/dev/null"
            " | grep -A1 'Device Name:' | grep -v Mac", "r");
        if (fp) {
            char buf[256];
            while (fgets(buf, sizeof(buf), fp)) {
                char *p = strstr(buf, "Device Name:");
                if (p) {
                    p += 12;
                    while (*p == ' ') p++;
                    char *nl = strchr(p, '\n');
                    if (nl) *nl = '\0';
                    if (strlen(p) > 0) {
                        char tmp[64];
                        snprintf(tmp, sizeof(tmp), "%s via %s", g_nv_arch, p);
                        strncpy(g_nv_arch, tmp, sizeof(g_nv_arch) - 1);
                        break;
                    }
                }
            }
            pclose(fp);
        }
    }
#else
    /* Linux: check /proc */
    {
        FILE *fp = fopen("/proc/driver/nvidia/version", "r");
        if (fp) {
            g_nv_detected = 1;
            snprintf(g_nv_arch, sizeof(g_nv_arch), "NVIDIA GPU");
            fclose(fp);
        }
    }
#endif
}

/* ── Mac system stats ────────────────────────────────────────── */

static double get_load_1m(void) {
    double loadavg[3] = {0};
    getloadavg(loadavg, 3);
    return loadavg[0];
}

static int get_mem_pressure(void) {
    /* 0=normal, 1=warning, 2=critical */
#ifdef __APPLE__
    int level = 0;
    size_t len = sizeof(level);
    sysctlbyname("kern.memorystatus_vm_pressure_level", &level, &len, NULL, 0);
    return level;
#else
    return 0;
#endif
}

/* ── RTX stats from tinygrad probe (reads /tmp/rtx_stats.json) ── */

#define RTX_STATS_FILE "/tmp/rtx_stats.json"

static double g_rtx_temp = -1;
static double g_rtx_tb5_ms = -1;
static double g_rtx_kps = 0;       /* kernels per second */
static int    g_rtx_total_k = 0;   /* total kernel count */
static double g_rtx_mem_mb = -1;   /* VRAM used MB */
static int    g_rtx_active = 0;    /* 1 if GPU is doing work */
static int    g_rtx_live = 0;      /* 1 if probe is running and recent */

static void read_rtx_stats(void) {
    FILE *fp = fopen(RTX_STATS_FILE, "r");
    if (!fp) { g_rtx_live = 0; return; }

    char buf[2048];
    size_t n = fread(buf, 1, sizeof(buf) - 1, fp);
    fclose(fp);
    buf[n] = '\0';

    char *p;
    if ((p = strstr(buf, "\"temp\":")) != NULL) {
        double t = 0;
        if (sscanf(p + 7, " %lf", &t) == 1 && t > 0) g_rtx_temp = t;
    }
    if ((p = strstr(buf, "\"tb5_ms\":")) != NULL)
        sscanf(p + 9, " %lf", &g_rtx_tb5_ms);
    if ((p = strstr(buf, "\"kernels_per_sec\":")) != NULL)
        sscanf(p + 18, " %lf", &g_rtx_kps);
    if ((p = strstr(buf, "\"total_kernels\":")) != NULL)
        sscanf(p + 16, " %d", &g_rtx_total_k);
    if ((p = strstr(buf, "\"mem_used_mb\":")) != NULL)
        sscanf(p + 14, " %lf", &g_rtx_mem_mb);
    if ((p = strstr(buf, "\"active\":")) != NULL) {
        if (strstr(p + 9, "true")) g_rtx_active = 1;
        else g_rtx_active = 0;
    }
    if ((p = strstr(buf, "\"timestamp\":")) != NULL) {
        double ts = 0;
        sscanf(p + 12, " %lf", &ts);
        struct timespec now;
        clock_gettime(CLOCK_REALTIME, &now);
        g_rtx_live = (now.tv_sec - (time_t)ts) < 5;
    }
    if (strstr(buf, "\"offline\"")) { g_rtx_live = 0; g_rtx_active = 0; }
    if (strstr(buf, "\"busy\"")) { g_rtx_live = 1; g_rtx_active = 1; }
}

/* ── TB5 latency probe ───────────────────────────────────────── */

static double probe_tb5_latency(void) {
    /* Use live RTX probe data if available */
    if (g_rtx_live && g_rtx_tb5_ms > 0) return g_rtx_tb5_ms;
    if (!g_nv_detected) return -1.0;
    return 1.50;  /* fallback to spec */
}

/* ── Terminal size ────────────────────────────────────────────── */

static int get_term_rows(void) {
    struct winsize ws;
    if (ioctl(STDOUT_FILENO, TIOCGWINSZ, &ws) == 0 && ws.ws_row > 0)
        return ws.ws_row;
    return 40;  /* fallback */
}

/* ── Frame buffer ────────────────────────────────────────────── */

/* Big enough for cursor-home + FRAME_LINES lines with ANSI codes */
#define FRAME_BUF_SIZE (FRAME_LINES * 512)

typedef struct {
    char   buf[FRAME_BUF_SIZE];
    int    pos;
    int    line_count;
} frame_buf;

static void fb_init(frame_buf *fb) {
    fb->pos = 0;
    fb->line_count = 0;
    /* Start with cursor-home so the entire frame is one write() */
    memcpy(fb->buf, CUR_HOME, 3);  /* \033[H = 3 bytes */
    fb->pos = 3;
}

static void fb_line(frame_buf *fb, const char *fmt, ...) {
    int remain = FRAME_BUF_SIZE - fb->pos;
    if (remain < 20) return;

    /* Position cursor at exact row: \033[<row>;1H then clear line */
    fb->pos += snprintf(fb->buf + fb->pos, remain,
                        "\033[%d;1H" CLR_LINE, fb->line_count + 1);
    remain = FRAME_BUF_SIZE - fb->pos;

    va_list ap;
    va_start(ap, fmt);
    fb->pos += vsnprintf(fb->buf + fb->pos, remain, fmt, ap);
    va_end(ap);

    fb->buf[fb->pos] = '\0';
    fb->line_count++;
}

static void fb_pad(frame_buf *fb, int target_lines) {
    while (fb->line_count < target_lines) {
        fb_line(fb, "");
    }
}

static void fb_flush(frame_buf *fb) {
    /* Everything is already in buf — single atomic write */
    (void)!write(STDOUT_FILENO, fb->buf, fb->pos);
}

/* ── Signal handler ──────────────────────────────────────────── */

static void on_signal(int sig) {
    (void)sig;
    g_running = 0;
}

/* ── Main ────────────────────────────────────────────────────── */

int main(int argc, char **argv) {
    double interval = 1.0;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--interval") == 0 && i + 1 < argc)
            interval = atof(argv[++i]);
        else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            printf("Usage: tqbridge-monitor [OPTIONS]\n\n");
            printf("Options:\n");
            printf("  --interval SECS  Update interval (default: 1.0)\n");
            printf("  -h, --help       Show this help\n");
            return 0;
        }
    }

    signal(SIGINT, on_signal);
    signal(SIGTERM, on_signal);

    /* Detect GPUs */
    detect_gpus();

    /* Start node discovery thread */
    pthread_t disc_tid;
    pthread_create(&disc_tid, NULL, discovery_thread, NULL);

    /* Tracking */
    sample_ring tb5_ring = {.count = 0, .head = 0};
    int sample = 0;

    /* Enter alternate screen buffer (like vim/htop) + hide cursor */
    printf(ALT_ON HIDE_CUR CLR_SCR);
    fflush(stdout);

    while (g_running) {
        sample++;
        int rows = get_term_rows();
        frame_buf fb;
        fb_init(&fb);

        /* ── Collect data ── */
        read_rtx_stats();
        double tb5 = probe_tb5_latency();
        if (tb5 >= 0) ring_push(&tb5_ring, tb5);

        const char *nv_tag = (g_nv_detected || g_rtx_live)
            ? GREEN "● NV " RESET : RED "● NV ✗" RESET;
        const char *mtl_tag = g_metal_detected
            ? GREEN "● Metal" RESET : RED "● Metal ✗" RESET;

        int online = 0, total = 0;
        pthread_mutex_lock(&g_node_lock);
        total = g_node_count;
        for (int i = 0; i < g_node_count; i++)
            if (g_nodes[i].online) online++;
        pthread_mutex_unlock(&g_node_lock);

        char net_tag[64];
        if (online > 0)
            snprintf(net_tag, sizeof(net_tag), GREEN "● %d net" RESET, online);
        else
            snprintf(net_tag, sizeof(net_tag), YELLOW "● 0 net" RESET);

        double load = get_load_1m();
        int pressure = get_mem_pressure();

        const char *load_color = load < 6.0 ? GREEN : load < 12.0 ? YELLOW : RED;
        char load_bar[256];
        render_bar(load_bar, sizeof(load_bar), load * 10, 100.0, load_color);
        const char *p_str = pressure == 0 ? "Normal"
                          : pressure == 1 ? "Warning" : "Critical";
        const char *p_color = pressure == 0 ? GREEN
                            : pressure == 1 ? YELLOW : RED;

        /*
         * Adaptive layout — 3 modes based on terminal height.
         * All content is ALWAYS shown. Blank-line spacing is the flex.
         *
         * Minimum content lines (no spacing):
         *   header=4 + gpu_status=1 + rtx=2 + tb5=2 + mac=2 +
         *   bridge=5 + nodes=(1+N) + footer=1 ≈ 21 lines
         *
         * Compact  (rows < 30): no blank separators, 1-line header
         * Normal   (30-44):     blank separators, compact header
         * Expanded (rows >= 45): full box header, extra spacing
         */
        int content = rows - 1;  /* reserve last row for footer */
        int mode = rows < 30 ? 0 : rows < 45 ? 1 : 2;

        /* Count mandatory content lines to know how much flex space we have */
        int node_lines = total > 0 ? total : 1;
        int tb5_lines = (tb5 >= 0 && tb5_ring.count > 0) ? 3
                      : 1;  /* "Probing..." or "No eGPU" */
        int fixed_lines = 2 + 2 + tb5_lines + 3 + 5 + 1 + node_lines + 1;
        /* header(varies) + rtx + tb5 + mac + bridge + nodes_header + nodes + footer */
        int header_lines = mode == 2 ? 7 : mode == 1 ? 4 : 1;
        fixed_lines += header_lines;
        int flex = content - fixed_lines;
        if (flex < 0) flex = 0;

        /* Distribute flex space as blank separator lines between sections */
        /* 6 section gaps: after header, rtx, tb5, mac, bridge, nodes */
        int gaps[6] = {0};
        if (flex > 0) {
            int per_gap = flex / 6;
            int extra = flex % 6;
            for (int i = 0; i < 6; i++) {
                gaps[i] = per_gap + (i < extra ? 1 : 0);
                if (gaps[i] > 3) gaps[i] = 3;  /* cap individual gap */
            }
        }

        #define FLEX(n) do { for (int _g = 0; _g < gaps[n]; _g++) fb_line(&fb, ""); } while(0)

        /* ── Header ── */
        if (mode == 2) {
            /* Full box */
            fb_line(&fb, BOLD CYAN);
            fb_line(&fb, "  ╔══════════════════════════════════════════════════════════════╗");
            fb_line(&fb, "  ║                                                              ║");
            fb_line(&fb, "  ║   TQBridge Cluster Monitor                  Chronara Group   ║");
            fb_line(&fb, "  ║                                                              ║");
            fb_line(&fb, "  ╚══════════════════════════════════════════════════════════════╝" RESET);
            fb_line(&fb, "    %s   %s   %s", nv_tag, mtl_tag, net_tag);
        } else if (mode == 1) {
            /* Compact box */
            fb_line(&fb, BOLD CYAN
                    "  ═══ TQBridge Cluster Monitor ═══  Chronara Group" RESET);
            fb_line(&fb, "    %s   %s   %s", nv_tag, mtl_tag, net_tag);
            fb_line(&fb, CYAN
                    "  ══════════════════════════════════════════════════" RESET);
            fb_line(&fb, "");
        } else {
            /* Minimal single line */
            fb_line(&fb, BOLD CYAN "  TQBridge" RESET
                    "  %s  %s  %s", nv_tag, mtl_tag, net_tag);
        }
        FLEX(0);

        /* ── RTX eGPU ── */
        fb_line(&fb, "  " BOLD "RTX PRO 6000 Blackwell (eGPU, TB5)" RESET);
        if (g_rtx_live) {
            /* Live data from tinygrad probe */
            fb_line(&fb, "  Arch:  %s  |  VRAM: 96 GB GDDR7  |  Link: Thunderbolt 5",
                    g_nv_arch);

            /* Temperature bar */
            if (g_rtx_temp > 0) {
                char temp_bar[256];
                const char *tc = g_rtx_temp < 65 ? GREEN : g_rtx_temp < 80 ? YELLOW : RED;
                render_bar(temp_bar, sizeof(temp_bar), g_rtx_temp, 90.0, tc);
                fb_line(&fb, "  Temp:  %s %s%.0f°C" RESET, temp_bar, tc, g_rtx_temp);
            }

            /* GPU Activity bar — kernels/sec */
            {
                char act_bar[256];
                const char *ac = g_rtx_active ? GREEN : DIM;
                double kps_capped = g_rtx_kps > 100 ? 100 : g_rtx_kps;
                render_bar(act_bar, sizeof(act_bar), kps_capped, 100.0,
                           g_rtx_active ? GREEN : DIM);
                fb_line(&fb, "  GPU:   %s %s%.0f kernels/s" RESET
                        "  %s",
                        act_bar, ac, g_rtx_kps,
                        g_rtx_active ? GREEN "ACTIVE" RESET : DIM "idle" RESET);
            }

            /* TB5 latency */
            if (g_rtx_tb5_ms > 0) {
                const char *lat_color = g_rtx_tb5_ms < 2.0 ? GREEN
                                      : g_rtx_tb5_ms < 4.0 ? YELLOW : RED;
                fb_line(&fb, "  TB5:   %s%.2fms" RESET " RTT  |  %d total kernels",
                        lat_color, g_rtx_tb5_ms, g_rtx_total_k);
            }
        } else if (g_nv_detected) {
            fb_line(&fb, "  Arch:  %s  |  VRAM: 96 GB GDDR7  |  Link: Thunderbolt 5",
                    g_nv_arch);
            fb_line(&fb, "  " YELLOW "Start probe: deploy/bin/rtx-probe.py" RESET);
        } else {
            fb_line(&fb, "  " RED "Not connected" RESET);
        }
        FLEX(2);

        /* ── Mac system stats ── */
        fb_line(&fb, "  " BOLD "Mac Studio (%s)" RESET, g_metal_chip);
        fb_line(&fb, "  Load:  %s %.1f", load_bar, load);
        fb_line(&fb, "  RAM:   Pressure: %s%s" RESET "  |  Unified: 96 GB",
                p_color, p_str);
        FLEX(3);

        /* ── Cluster Performance (measured 2026-04-13) ── */
        fb_line(&fb, "  " BOLD "Cluster Performance (all measured, real hardware)" RESET);
        fb_line(&fb, "  GX10-001 GB10:   " GREEN "2,030 pp" RESET
                "  " GREEN " 28.2 tg" RESET "  (Qwen3-8B Q8_0, CUDA)");
        fb_line(&fb, "  GX10-002 GB10:   " GREEN "2,033 pp" RESET
                "  " GREEN " 28.4 tg" RESET "  (Qwen3-8B Q8_0, CUDA)");
        fb_line(&fb, "  GX10-001 GB10:   " GREEN "1,857 pp" RESET
                "  " GREEN " 55.6 tg" RESET "  (Qwen3.5-35B MoE, CUDA)");
        fb_line(&fb, "  M3 Ultra MLX:    " GREEN "  28 pp" RESET
                "  " GREEN " 34.7 tg" RESET "  (Qwen2.5-32B 4bit, Metal)");
        fb_line(&fb, "  M3 Ultra MLX:                " GREEN " 94.3 tg" RESET
                "  (Qwen2.5-7B 4bit, Metal)");
        fb_line(&fb, "  M1 Max MLX:                  " GREEN "120.0 tg" RESET
                "  (Qwen2.5-7B 4bit, Metal)");
        fb_line(&fb, "  RTX 6000 TB5:                " GREEN "  6.8 tg" RESET
                "  (tinygrad kernel + TB5 latency)");
        fb_line(&fb, "  TurboQuant:      " GREEN "9.8x" RESET
                "           (asymmetric Q8₀K + turbo3 V)");
        FLEX(4);

        /* ── Cluster Nodes ── */
        fb_line(&fb, "  " BOLD "Cluster Nodes" RESET "  " DIM
                "(%d/%d online, scanning subnet)" RESET, online + 1, total + 1);

        /* M3 Ultra is always LOCAL — show eGPU if detected */
#if SHOW_LOCAL_M3
        {
            const char *m3_gpu = g_metal_detected ? g_metal_chip : "Metal";
            char m3_desc[128];
            if (g_rtx_live || g_nv_detected) {
                snprintf(m3_desc, sizeof(m3_desc), "%s + RTX PRO 6000 (eGPU TB5)", m3_gpu);
            } else {
                snprintf(m3_desc, sizeof(m3_desc), "%s", m3_gpu);
            }
            fb_line(&fb, "  " GREEN "●" RESET " %-12s %-16s " GREEN "LOCAL" RESET
                    "  " DIM "%s" RESET,
                    "M3 Ultra", "orchestrator", m3_desc);
        }
#endif

        pthread_mutex_lock(&g_node_lock);
        for (int i = 0; i < g_node_count; i++) {
            /* Show GPU type for known nodes */
            const char *gpu_info = "";
            if (strstr(g_nodes[i].name, "GX10")) gpu_info = "NVIDIA GB10 CUDA";
            else if (strstr(g_nodes[i].name, "M1")) gpu_info = "Apple M1 Max Metal";

            if (g_nodes[i].online) {
                fb_line(&fb, "  " GREEN "●" RESET " %-12s %-16s "
                        GREEN "LISTENING" RESET "  " DIM "%.1fms  %s" RESET,
                        g_nodes[i].name, g_nodes[i].ip,
                        g_nodes[i].latency_ms, gpu_info);
            } else {
                fb_line(&fb, "  " RED "●" RESET " %-12s %-16s "
                        DIM "offline   %s" RESET,
                        g_nodes[i].name, g_nodes[i].ip, gpu_info);
            }
        }
        pthread_mutex_unlock(&g_node_lock);
        if (total == 0) fb_line(&fb, "  " DIM "Scanning..." RESET);
        FLEX(5);

        #undef FLEX

        /* ── Pad remaining rows then footer at very last row ── */
        fb_pad(&fb, content);

        /* Footer pinned to last terminal row */
        {
            int remain = FRAME_BUF_SIZE - fb.pos;
            fb.pos += snprintf(fb.buf + fb.pos, remain,
                "\033[%d;1H" CLR_LINE
                "  " DIM "Sample %d | Interval %.1fs | %d rows | Ctrl+C to exit"
                RESET, rows, sample, interval, rows);
        }

        fb_flush(&fb);

        /* Sleep with early exit */
        for (int ms = 0; ms < (int)(interval * 1000) && g_running; ms += 50)
            usleep(50000);
    }

    /* Cleanup: leave alternate screen, show cursor */
    printf(SHOW_CUR ALT_OFF);
    printf("  " DIM "Monitor stopped." RESET "\n");
    fflush(stdout);

    pthread_cancel(disc_tid);
    pthread_join(disc_tid, NULL);
    return 0;
}
