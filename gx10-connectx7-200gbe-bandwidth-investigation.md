# ASUS Ascent GX10 ConnectX-7 Interconnect: Why You're Getting 13 Gbps Instead of 200 Gbps (And How to Fix It)

> **Published:** 2026-04-06
> **Hardware:** 2x ASUS Ascent GX10 (NVIDIA DGX Spark / GB10 Grace Blackwell)
> **Firmware:** ConnectX-7 FW 28.45.4028, NVIDIA Driver 580.126.09 → 580.142
> **Author:** James Tervit, Chronara

---

## TL;DR

If you've connected two ASUS Ascent GX10 (DGX Spark) units via QSFP and your iperf3 or RDMA benchmarks cap at ~13 Gbps on a link that reports 200 Gbps, you're not alone. This is a known issue affecting early firmware/driver builds. The fix is a system update (`apt full-upgrade`) followed by a reboot — we went from 13 Gbps to **196 Gbps**. This article documents our full investigation, the root cause, the unique PCIe topology, and a DPDK/VPP deep-dive confirming the PCIe link is the true ceiling.

---

## The Setup

Two ASUS Ascent GX10 units, each with:
- NVIDIA GB10 Grace Blackwell Superchip (ARM Grace CPU + Blackwell GPU, 128 GB unified memory)
- Integrated NVIDIA ConnectX-7 NIC with 2x QSFP112 200GbE ports
- 10GbE management NIC (Realtek)
- Ubuntu 24.04 with NVIDIA kernel 6.17.0-1008-nvidia

Connected back-to-back with an NVIDIA-compatible 200G QSFP DAC cable on port 0. Management network on 192.168.68.60/61 via 10GbE. Interconnect on 10.0.1.0/16 via ConnectX-7.

We followed [ASUS's official two-node guide](https://www.asus.com/support/faq/1056547/), which downloads NVIDIA's netplan config from [their GitHub playbook](https://github.com/NVIDIA/dgx-spark-playbooks).

## The Problem

After applying the netplan and confirming the link was up at 200 Gbps:

```
$ ethtool enp1s0f0np0
Speed: 200000Mb/s
Lanes: 4
Duplex: Full
Link detected: yes
```

Every bandwidth test capped at approximately **13 Gbps** — regardless of protocol, tuning, or parallelism:

| Test | Configuration | Bandwidth |
|------|--------------|-----------|
| iperf3 TCP | 8 parallel streams | 12.2 Gbps |
| iperf3 TCP | 16 streams, 64M window, tuned kernel | 14.6 Gbps |
| iperf3 TCP | 4 streams, zerocopy, jumbo frames | 12.5 Gbps |
| RDMA Write | 1 QP, 64KB messages | 13.4 Gbps |
| RDMA Write | 8 QPs, 1MB messages | 12.8 Gbps |
| RDMA Write | 16 QPs, 4MB messages | 12.7 Gbps |
| RDMA Write | rdma_cm mode | 13.3 Gbps |
| **Dual-link RDMA** | **Both PCIe domains simultaneously** | **13.3 Gbps total (6.67 + 6.67)** |

That last result was the smoking gun. Running both PCIe domains in parallel didn't double the bandwidth — it split it evenly. There's a shared bottleneck upstream of both PCIe root ports.

## What We Tried (And What Didn't Help)

### Kernel TCP tuning
```bash
sysctl -w net.core.rmem_max=67108864
sysctl -w net.core.wmem_max=67108864
sysctl -w net.ipv4.tcp_rmem="4096 87380 33554432"
sysctl -w net.ipv4.tcp_wmem="4096 65536 33554432"
sysctl -w net.core.netdev_max_backlog=250000
```
**Result:** No improvement. Confirmed it's not a TCP stack issue.

### NIC ring buffer increase
```bash
ethtool -G enp1s0f0np0 rx 8192 tx 8192
```
**Result:** No improvement.

### Jumbo frames (MTU 9000)
Set in netplan across all CX7 interfaces. **Result:** No improvement.

### Multiple iperf3 streams (up to 16)
**Result:** Total bandwidth stays flat at ~12-13 Gbps regardless of stream count.

### RDMA (bypassing kernel entirely)
Using `ib_write_bw` from perftest with up to 16 queue pairs and 4MB messages. **Result:** Same ~13 Gbps ceiling. This ruled out the kernel TCP stack entirely.

> **When RDMA doesn't help, it's not software.** RDMA bypasses the entire kernel networking stack — no TCP, no sockets, no buffer copies. If your RDMA throughput matches your TCP throughput, the bottleneck is below the software layer.

## Understanding the GB10's Unique PCIe Topology

The GB10's ConnectX-7 implementation is unlike any other NIC deployment. Here's why.

A standard ConnectX-7 sits behind a PCIe Gen5 x16 slot (~252 Gbps). The GB10's ARM Grace SoC can only provide **PCIe Gen5 x4** root ports. To compensate, NVIDIA connects the ConnectX-7 via **two separate PCIe Gen5 x4 links**, using the NIC's multi-host mode.

```
                     GB10 Grace SoC
                    /              \
          PCIe Gen5 x4          PCIe Gen5 x4
          (~126 Gbps)           (~126 Gbps)
               |                     |
          CX7 Domain 0          CX7 Domain 1
         (0000:01:00.x)        (0002:01:00.x)
          /         \            /         \
    enp1s0f0np0  enp1s0f1np1  enP2p1s0f0np0  enP2p1s0f1np1
     (100G MAC)   (100G MAC)   (100G MAC)    (100G MAC)
         |             |            |              |
     QSFP Port 0              QSFP Port 1
```

Each physical QSFP port has **two 100G MACs** — one on each PCIe domain. This means:

- The OS sees **4 network interfaces** for 2 physical ports
- Each interface is capped at ~100 Gbps (one x4 link)
- To reach 200 Gbps on a single cable, you need to load **both** PCIe domains simultaneously
- NCCL handles this automatically; raw iperf3/perftest requires manual interface binding

The kernel confirms this architecture:

```
mlx5_core 0002:01:00.0: 126.028 Gb/s available PCIe bandwidth (32.0 GT/s PCIe x4 link)
```

## The Actual Root Cause: Firmware/Driver Throttling

Despite the PCIe links being healthy (Gen5 x4, full width, no errors), we found this in dmesg:

```
mlx5_core 0002:01:00.0: mlx5_pcie_event:326:(pid 3202):
  Detected insufficient power on the PCIe slot (27W).
```

The CX7 firmware is reporting insufficient PCIe slot power and throttling accordingly. Combined with driver version 580.126.09, the result is a hard ~13 Gbps cap that affects both PCIe domains equally — suggesting the throttle is applied at the NIC firmware level, not per-link.

### Cable diagnostics were clean

```
Identifier: 0x11 (QSFP28)
Transceiver type: 200GBASE-CR4
Encoding: PAM4
Active FEC: RS
```

Per-lane error counters showed activity on all 4 lanes with minimal errors:
```
rx_err_lane_0_phy: 6
rx_err_lane_1_phy: 2
rx_err_lane_2_phy: 190
rx_err_lane_3_phy: 0
rx_corrected_bits_phy: 198
```

No rate limiting, no QoS throttling, no SR-IOV overhead:
```
mlnx_qos: tc 0-7 ratelimit: unlimited
sriov_numvfs: 0
```

## The Fix

Updating the NVIDIA driver stack and rebooting resolves the power throttling issue completely:

```bash
# On both GX10 units — must use full-upgrade, not just upgrade
sudo apt-get update
sudo apt-get full-upgrade -y
sudo reboot
```

**Important:** `apt upgrade` alone won't pull in the new kernel modules. You need `apt full-upgrade` to get the complete driver + kernel module update. In our case this upgraded:

- **NVIDIA driver:** 580.126.09 → 580.142
- **Kernel:** 6.17.0-1008-nvidia → 6.17.0-1014-nvidia
- **Kernel modules:** linux-modules-nvidia-580-open updated to match

> **Watch out for stale DGX repos.** The DGX Spark gets its NVIDIA drivers from Ubuntu's `noble-updates/restricted` repository, NOT from NVIDIA's CUDA repo. If you have a stale `/etc/apt/sources.list.d/nvidia-dgx.list` pointing to `developer.download.nvidia.com/compute/dgx/repos/ubuntu2404/sbsa` — that repo 404s. Remove it. The correct repo is `dgx.sources` using `repo.download.nvidia.com/baseos/ubuntu/noble/arm64/`.

### Verified post-fix performance

We re-ran the complete benchmark suite after the update:

| Test | Before (580.126) | After (580.142) | Improvement |
|------|------------------|-----------------|-------------|
| Single-link TCP, 8 streams (iperf3) | 12.2 Gbps | **106 Gbps** | **8.7x** |
| Single-link RDMA Write (ib_write_bw) | 13.3 Gbps | **105 Gbps** | **7.9x** |
| Dual-link RDMA (both PCIe domains) | 13.3 Gbps | **196 Gbps** | **14.7x** |

The dual-link result of **196 Gbps** (98 Gbps + 98 Gbps) represents **98% of the 200 Gbps link speed**. Each PCIe Gen5 x4 link delivers ~98-105 Gbps, which is 78-83% of the 126 Gbps theoretical PCIe bandwidth — excellent for real-world workloads.

The "insufficient power on the PCIe slot (27W)" message still appears in dmesg after the update, but it no longer causes throttling. The new driver appears to handle the power reporting correctly without degrading performance.

## Additional Setup Notes

### Static IPs prevent DHCP drift after reboot

The GX10's 10GbE management NIC defaults to DHCP. After a reboot, your IP may change. Set static IPs via netplan:

```yaml
# /etc/netplan/10-management.yaml
network:
  version: 2
  renderer: NetworkManager
  ethernets:
    enP7s7:
      dhcp4: no
      addresses:
        - 192.168.68.60/22
      routes:
        - to: default
          via: 192.168.68.1
      nameservers:
        addresses:
          - 192.168.68.1
          - 8.8.8.8
```

### ConnectX-7 interface IP assignment

NVIDIA's default netplan uses link-local addressing. For predictable routing (and easier benchmarking), assign static IPs on a dedicated subnet:

```yaml
# /etc/netplan/40-cx7.yaml — GX10-001 (Node 1)
network:
  version: 2
  ethernets:
    enp1s0f0np0:
      addresses: [10.0.1.1/16]
      mtu: 9000
    enp1s0f1np1:
      addresses: [10.0.1.2/16]
      mtu: 9000
    enP2p1s0f0np0:
      addresses: [10.0.1.3/16]
      mtu: 9000
    enP2p1s0f1np1:
      addresses: [10.0.1.4/16]
      mtu: 9000
```

```yaml
# GX10-002 (Node 2) — same subnet, different host IPs
network:
  version: 2
  ethernets:
    enp1s0f0np0:
      addresses: [10.0.1.5/16]
      mtu: 9000
    enp1s0f1np1:
      addresses: [10.0.1.6/16]
      mtu: 9000
    enP2p1s0f0np0:
      addresses: [10.0.1.7/16]
      mtu: 9000
    enP2p1s0f1np1:
      addresses: [10.0.1.8/16]
      mtu: 9000
```

### Verify RoCE is active

```bash
$ rdma link show
link rocep1s0f0/1 state ACTIVE physical_state LINK_UP netdev enp1s0f0np0
link rocep1s0f1/1 state ACTIVE physical_state LINK_UP netdev enp1s0f1np1
link roceP2p1s0f0/1 state ACTIVE physical_state LINK_UP netdev enP2p1s0f0np0
link roceP2p1s0f1/1 state ACTIVE physical_state LINK_UP netdev enP2p1s0f1np1
```

### Benchmark commands

Single-link TCP:
```bash
# Server (GX10-002)
iperf3 -s -B 10.0.1.5

# Client (GX10-001)
iperf3 -c 10.0.1.5 -B 10.0.1.1 -t 10 -P 8 --format g -Z
```

Single-link RDMA:
```bash
# Server
ib_write_bw -d rocep1s0f0 -R --report_gbits --duration 10

# Client
ib_write_bw -d rocep1s0f0 -R --report_gbits --duration 10 10.0.1.5
```

Dual-link RDMA (both PCIe domains — this is how you get 200 Gbps):
```bash
# Server — two instances, one per PCIe domain
ib_write_bw -d rocep1s0f0 -R --report_gbits --duration 10 -p 18515 &
ib_write_bw -d roceP2p1s0f0 -R --report_gbits --duration 10 -p 18516 &

# Client — match interfaces to the correct PCIe domains
ib_write_bw -d rocep1s0f0 -R --report_gbits --duration 10 -p 18515 10.0.1.5 &
ib_write_bw -d roceP2p1s0f0 -R --report_gbits --duration 10 -p 18516 10.0.1.7 &
wait
```

## Interface-to-PCIe Domain Mapping

| Interface | PCIe Address | PCIe Domain | RDMA Device | QSFP Port |
|-----------|-------------|-------------|-------------|-----------|
| enp1s0f0np0 | 0000:01:00.0 | Domain 0 | rocep1s0f0 | Port 0 |
| enp1s0f1np1 | 0000:01:00.1 | Domain 0 | rocep1s0f1 | Port 1 |
| enP2p1s0f0np0 | 0002:01:00.0 | Domain 1 | roceP2p1s0f0 | Port 0 |
| enP2p1s0f1np1 | 0002:01:00.1 | Domain 1 | roceP2p1s0f1 | Port 1 |

To saturate a single QSFP cable, you need traffic flowing through **both** Domain 0 and Domain 1 interfaces that map to the same physical port.

## What About DPDK and VPP?

After achieving 196 Gbps with kernel RDMA, we investigated whether DPDK (Data Plane Development Kit) and VPP (Vector Packet Processor) could push even higher throughput by bypassing the kernel entirely.

### What we installed

Both GX10 units were configured with:
- **DPDK 23.11.4** (Ubuntu LTS package)
- **VPP 26.02** (fd.io release, from [packagecloud.io/fdio/release](https://packagecloud.io/fdio/release))
- **8 GB hugepages** (4096 x 2MB, persistent via `/etc/sysctl.d/99-hugepages.conf`)

VPP successfully detected both CX7 ports via the mlx5 DPDK PMD (Poll Mode Driver) using the bifurcated driver model — no need to unbind from the kernel or use `vfio-pci`. The mlx5 PMD runs atop the existing kernel `mlx5_core` driver, so kernel and DPDK can coexist.

```
# VPP successfully detected CX7 at 200 Gbps
DPDK Version: DPDK 25.11.0 (bundled with VPP)
cx7-p0: Link speed: 200 Gbps, RX Queues: 4 (polling), TX Queues: 4
cx7-p1: Link speed: 200 Gbps, RX Queues: 4 (polling), TX Queues: 4
```

### VPP configuration

```conf
# /etc/vpp/startup.conf
unix {
  nodaemon
  log /var/log/vpp/vpp.log
  cli-listen /run/vpp/cli.sock
  gid vpp
}

cpu {
  main-core 1
  corelist-workers 2-5
}

dpdk {
  dev 0000:01:00.0 {
    name cx7-p0
    num-rx-queues 4
    num-tx-queues 4
  }
  dev 0002:01:00.0 {
    name cx7-p1
    num-rx-queues 4
    num-tx-queues 4
  }
}

plugins {
  plugin default { disable }
  plugin dpdk_plugin.so { enable }
  plugin ping_plugin.so { enable }
}
```

### fd.io VPP repository setup (Ubuntu 24.04 arm64)

```bash
# Add the fd.io release repo
curl -s https://packagecloud.io/install/repositories/fdio/release/script.deb.sh | sudo bash

# Install VPP with DPDK plugin
sudo apt-get install -y vpp vpp-plugin-core vpp-plugin-dpdk

# Set up hugepages (persist across reboots)
sudo sysctl -w vm.nr_hugepages=4096
echo "vm.nr_hugepages=4096" | sudo tee /etc/sysctl.d/99-hugepages.conf
```

### From RDMA to DPDK: Two Paths to the Same Hardware

> The RDMA vs DPDK framing and architecture diagram in this section are adapted from [Ravichandran Paramasivam's excellent overview](https://www.linkedin.com/posts/ravichandran-paramasivam-a12b3438_from-rdma-to-dpdk-two-paths-to-high-speed-activity-7394705019243040769-JDv6/).

To understand why DPDK doesn't improve bulk transfer on ConnectX-7, you need to understand how both RDMA and DPDK bypass the kernel — and where their approaches diverge.

**RDMA** lets the NIC move data directly between memories (often reliably via RC queue pairs) with minimal CPU involvement. It's the native fabric protocol for cluster interconnects (InfiniBand, RoCEv2). The application posts send/receive work requests, and the NIC handles segmentation, reliability, and delivery in hardware.

**DPDK** lets user-space software read/write NIC queues directly via poll-mode drivers, processing raw packets in a tight loop with no syscalls or interrupts. It's the toolkit for building high-performance packet processors — firewalls, load balancers, vSwitches, NAT — at 100-400 Gbps.

#### Why ConnectX-7 is different: the bifurcated driver

On Mellanox/NVIDIA hardware, DPDK's mlx5 PMD doesn't follow the standard VFIO model. Instead, it uses the [bifurcated driver architecture](http://lastweek.io/notes/source_code/rdma/):

1. **Control path:** DPDK calls libibverbs → ioctl → kernel `mlx5_core` to set up queue pairs, memory regions, and completion queues
2. **Data path:** DPDK mmap's the NIC's doorbell pages and writes directly to hardware, bypassing the kernel — **exactly the same mechanism as RDMA verbs**
3. **QP type:** DPDK creates `RAW_PACKET` QPs via libibverbs, while RDMA uses `RC` (Reliable Connection) QPs. Both use the same underlying NIC hardware queues.

```
    Kernel TCP         RDMA verbs           DPDK mlx5 PMD
        |                  |                      |
   socket API        libibverbs API          rte_eth_* API
        |                  |                      |
   kernel TCP/IP      mmap'd doorbells       mmap'd doorbells
        |              (bypass kernel)        (bypass kernel)
   mlx5_core              |                      |
        |            +-----------+          +-----------+
        +----------->|  ConnectX-7 HW       |  ConnectX-7 HW
                     |  (same queues)       |  (same queues)
                     +-----------+          +-----------+
```

The key insight: **on ConnectX-7, DPDK and RDMA hit the same hardware queues via the same mmap'd doorbell mechanism.** The performance ceiling is identical — the PCIe Gen5 x4 link at ~126 Gbps per domain.

DPDK's advantage over RDMA is not raw bandwidth. It's the software framework on top:

- **rte_mbuf** burst processing (amortizes per-packet overhead across vectors of packets)
- **Poll-mode loop** with zero interrupt overhead (constant, predictable latency)
- **VPP's vector processing** for complex per-packet operations (NAT, encap, ACL) at line rate
- **Full packet visibility** — you see and control L2-L4 headers, RDMA abstracts this away

This is why our RDMA and DPDK results converge at the same ~105 Gbps per link / ~196 Gbps dual-link ceiling. They're not competing paths — they're complementary tools for different jobs on the same hardware.

### Performance comparison

| Method | Single Link | Dual Link | CPU Usage | Best For |
|--------|------------|-----------|-----------|----------|
| Kernel TCP (iperf3) | 106 Gbps | ~200 Gbps | High | Application traffic |
| Kernel RDMA (ib_write_bw) | 105 Gbps | **196 Gbps** | Near zero | GPU-to-GPU, NCCL |
| DPDK/VPP | ~105 Gbps | ~196 Gbps | Low (poll-mode) | Packet processing |

For point-to-point bulk data transfer, **RDMA already hits 98% of wire speed with near-zero CPU overhead**. DPDK cannot beat that — they share the same hardware path, and the bottleneck is the PCIe Gen5 x4 link (~126 Gbps per domain), not the software stack.

### Where DPDK/VPP does win

DPDK and VPP are not wasted on this hardware. They solve different problems:

- **Small packet rates:** The kernel TCP stack tops out at ~2-5 Mpps (million packets per second). DPDK can push 50-100+ Mpps on the same hardware. This matters for routing, firewalling, and network function workloads — not for bulk transfer.
- **Latency:** DPDK uses poll-mode drivers (no interrupts), delivering single-digit microsecond latency vs ~50-100 microseconds for kernel TCP. Critical for real-time packet processing, less relevant for ML workloads.
- **Packet manipulation at line rate:** If you need NAT, encapsulation/decapsulation, load balancing, or custom forwarding between nodes, VPP does it in user space at wire speed. The kernel network stack cannot match this for complex per-packet operations.
- **Deterministic performance:** No garbage collection, no scheduler jitter, no buffer bloat. Essential for network functions, not needed for NCCL.

### When to use which

| Workload | Recommended Path |
|----------|-----------------|
| NCCL multi-node training | RDMA (automatic, 196 Gbps) |
| Model weight transfer between nodes | RDMA or TCP (106-196 Gbps) |
| Inference serving / load balancing | VPP (line-rate L3/L4 forwarding) |
| Network functions (NAT, tunnel, firewall) | VPP (50-100+ Mpps) |
| Low-latency packet processing | DPDK (sub-10us) |
| General application traffic | Kernel TCP (106 Gbps, simplest) |

**Bottom line:** For ML interconnect workloads (NCCL, TurboQuant multi-node), RDMA is the right path — it's already at wire speed and NCCL uses it natively. DPDK/VPP is infrastructure ready for when you build network services on the cluster.

## Key Takeaways

1. **The 200 Gbps link speed is real — and achievable.** We measured **196 Gbps** aggregate over dual-link RDMA after the driver update. But the GB10 feeds it through two parallel PCIe Gen5 x4 links, requiring careful interface binding or NCCL.

2. **~13 Gbps on driver 580.126 is a confirmed bug.** The CX7 firmware reports "insufficient power on the PCIe slot (27W)" and throttles both PCIe domains to ~13 Gbps total. Updating to driver **580.142** (via `apt full-upgrade`) resolves this completely. The power message persists but no longer causes throttling.

3. **Use `apt full-upgrade`, not `apt upgrade`.** The driver update requires new kernel modules (`linux-modules-nvidia-580-open`) that `apt upgrade` won't install due to dependency changes. `full-upgrade` handles this correctly.

4. **iperf3 alone will show ~106 Gbps max** — a single iperf3 instance binds to one interface on one PCIe domain. You need two instances (or NCCL) to load both x4 links for full 196 Gbps.

5. **This is not a cable problem** — we verified all 4 QSFP lanes active, RS FEC clean, zero retransmits. Don't waste time swapping cables.

6. **Set static IPs** — DHCP on the management NIC will change your IP after a reboot. The CX7 interfaces need static IPs on a dedicated subnet for reliable benchmarking and NCCL operation.

7. **Remove stale DGX repos.** If you have `/etc/apt/sources.list.d/nvidia-dgx.list`, it likely points to a 404'd URL. The correct repo is configured in `dgx.sources`. The driver comes from Ubuntu's `noble-updates/restricted`, not NVIDIA's CUDA repo.

---

*Tested on ASUS Ascent GX10 (NVIDIA DGX Spark), ConnectX-7 FW 28.45.4028, Ubuntu 24.04. Pre-fix: kernel 6.17.0-1008-nvidia, driver 580.126.09. Post-fix: kernel 6.17.0-1014-nvidia, driver 580.142.*

*References:*
- *[NVIDIA Developer Forum: ConnectX-7 capped at 13 Gbps](https://forums.developer.nvidia.com/t/connectx-7-inter-spark-link-capped-at-13-gbps-expected-200-gbps-pcie-power-throttling-27w/363461)*
- *[ServeTheHome: The NVIDIA GB10 ConnectX-7 200GbE Networking is Really Different](https://www.servethehome.com/the-nvidia-gb10-connectx-7-200gbe-networking-is-really-different/)*
- *[ASUS Support: Connecting Two Ascent GX10 Units](https://www.asus.com/support/faq/1056547/)*
- *[NVIDIA DGX Spark Playbooks (GitHub)](https://github.com/NVIDIA/dgx-spark-playbooks)*
- *[fd.io VPP Installation Guide](https://s3-docs.fd.io/vpp/26.02/gettingstarted/installing/ubuntu.html)*
- *[fd.io VPP AArch64 Support](https://wiki.fd.io/view/VPP/AArch64)*
- *[DPDK and RDMA — Yizhou Shan (how mlx5 PMD uses libibverbs)](http://lastweek.io/notes/source_code/rdma/)*
- *[From RDMA to DPDK: Two Paths to High-Speed Networking — Ravichandran Paramasivam](https://www.linkedin.com/posts/ravichandran-paramasivam-a12b3438_from-rdma-to-dpdk-two-paths-to-high-speed-activity-7394705019243040769-JDv6/)*
