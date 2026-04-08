"""Thermal monitor: temperature reading and throttle gating for bridge devices.

Supports:
- Metal (M3 Ultra): powermetrics GPU power + thermal pressure (requires sudo)
- NV eGPU: direct GPU die temp via tinygrad RM control (GSP thermal sensor)
- NV fallback: nvidia-smi (Linux only)
- Polling: background thread samples at configurable interval
- Gating: pause transfers when temperature exceeds threshold
"""

from __future__ import annotations

import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from enum import Enum


class ThermalPressure(Enum):
    """macOS thermal pressure levels (from powermetrics)."""
    NOMINAL = "Nominal"
    FAIR = "Fair"
    SERIOUS = "Serious"
    CRITICAL = "Critical"
    UNKNOWN = "Unknown"


@dataclass
class ThermalSnapshot:
    """Point-in-time thermal reading from one or more devices."""
    timestamp: float  # time.monotonic()
    metal_gpu_power_mw: float | None = None
    metal_pressure: ThermalPressure = ThermalPressure.UNKNOWN
    nv_temp_c: float | None = None
    nv_gpu_power_mw: float | None = None

    @property
    def is_throttled(self) -> bool:
        """True if any device reports thermal concern."""
        if self.metal_pressure in (ThermalPressure.SERIOUS, ThermalPressure.CRITICAL):
            return True
        if self.nv_temp_c is not None and self.nv_temp_c >= 85.0:
            return True
        return False

    def summary(self) -> str:
        parts = []
        if self.metal_gpu_power_mw is not None:
            parts.append(f"Metal GPU: {self.metal_gpu_power_mw:.0f} mW ({self.metal_pressure.value})")
        if self.nv_temp_c is not None:
            parts.append(f"NV: {self.nv_temp_c:.0f}°C")
        elif self.nv_gpu_power_mw is not None:
            parts.append(f"NV GPU: {self.nv_gpu_power_mw:.0f} mW")
        if not parts:
            return "No thermal data"
        return " | ".join(parts)


# ---------------------------------------------------------------------------
# Metal thermal (powermetrics)
# ---------------------------------------------------------------------------

def _read_powermetrics(timeout_s: float = 3.0) -> ThermalSnapshot:
    """Read Metal GPU power and thermal pressure via powermetrics (requires sudo)."""
    snap = ThermalSnapshot(timestamp=time.monotonic())
    try:
        result = subprocess.run(
            ["sudo", "-n", "powermetrics", "--samplers", "gpu_power,thermal", "-i", "1", "-n", "1"],
            capture_output=True, text=True, timeout=timeout_s,
        )
        for line in result.stdout.splitlines():
            if line.startswith("GPU Power:"):
                try:
                    snap.metal_gpu_power_mw = float(line.split(":")[1].strip().split()[0])
                except (ValueError, IndexError):
                    pass
            elif "pressure level:" in line:
                level_str = line.split(":")[-1].strip()
                try:
                    snap.metal_pressure = ThermalPressure(level_str)
                except ValueError:
                    snap.metal_pressure = ThermalPressure.UNKNOWN
    except (subprocess.TimeoutExpired, FileNotFoundError, PermissionError):
        pass
    return snap


# ---------------------------------------------------------------------------
# NV thermal (tinygrad RM control via GSP)
# ---------------------------------------------------------------------------

_nv_thermal_lock = threading.Lock()
_nv_dev_cache = None
_nv_last_temp: float | None = None


def _read_nv_rm_thermal() -> float | None:
    """Read NV GPU die temperature via tinygrad RM control (GSP thermal sensor).

    Uses NV2080_CTRL_CMD_THERMAL_SYSTEM_EXECUTE_V2_PHYSICAL to query sensor 0
    (GPU die) through the GPU System Processor. Works on macOS eGPU where
    nvidia-smi is unavailable.

    Non-blocking: if the NV device is busy (e.g. during a pipelined transfer),
    returns the last known temperature instead of blocking.

    Returns temperature in Celsius, or None if unavailable.
    """
    global _nv_dev_cache, _nv_last_temp

    # Non-blocking acquire — if device is busy with transfers, skip this read
    if not _nv_thermal_lock.acquire(blocking=False):
        return _nv_last_temp

    try:
        from tinygrad import Device
        from tinygrad.runtime.autogen import nv_580 as nv_gpu

        if _nv_dev_cache is None:
            _nv_dev_cache = Device["NV"]
        dev = _nv_dev_cache

        params = nv_gpu.NV2080_CTRL_THERMAL_SYSTEM_EXECUTE_V2_PARAMS(
            clientAPIVersion=2,
            clientAPIRevision=0,
            clientInstructionSizeOf=44,
            executeFlags=nv_gpu.NV2080_CTRL_THERMAL_SYSTEM_EXECUTE_FLAGS_IGNORE_FAIL,
            instructionListSize=1,
        )
        params.instructionList[0].opcode = (
            nv_gpu.NV2080_CTRL_THERMAL_SYSTEM_GET_STATUS_SENSOR_READING_OPCODE
        )
        params.instructionList[0].operands.getStatusSensorReading.sensorIndex = 0

        result = dev.iface.rm_control(
            dev.subdevice,
            nv_gpu.NV2080_CTRL_CMD_THERMAL_SYSTEM_EXECUTE_V2_PHYSICAL,
            params,
        )

        if result.successfulInstructions >= 1:
            _nv_last_temp = float(result.instructionList[0].operands.getStatusSensorReading.value)
            return _nv_last_temp
    except Exception:
        pass
    finally:
        _nv_thermal_lock.release()
    return _nv_last_temp


def _read_nvidia_smi(timeout_s: float = 3.0) -> tuple[float | None, float | None]:
    """Read NV GPU temp and power via nvidia-smi (Linux fallback)."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=temperature.gpu,power.draw", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=timeout_s,
        )
        if result.returncode == 0:
            parts = result.stdout.strip().split(",")
            temp = float(parts[0].strip())
            power = float(parts[1].strip()) * 1000  # W -> mW
            return temp, power
    except (FileNotFoundError, subprocess.TimeoutExpired, ValueError, IndexError):
        pass
    return None, None


# ---------------------------------------------------------------------------
# Combined read
# ---------------------------------------------------------------------------

def read_thermal() -> ThermalSnapshot:
    """Read thermal state from all available sources."""
    snap = _read_powermetrics()

    # Try tinygrad RM control first (works on macOS eGPU)
    nv_temp = _read_nv_rm_thermal()
    if nv_temp is not None:
        snap.nv_temp_c = nv_temp
    else:
        # Fallback to nvidia-smi (Linux)
        nv_temp, nv_power = _read_nvidia_smi()
        snap.nv_temp_c = nv_temp
        snap.nv_gpu_power_mw = nv_power

    return snap


# ---------------------------------------------------------------------------
# Background monitor
# ---------------------------------------------------------------------------

class ThermalMonitor:
    """Background thermal polling with throttle gating.

    Usage:
        monitor = ThermalMonitor(interval_s=2.0, temp_limit_c=85.0)
        monitor.start()

        # During transfer loop:
        monitor.wait_if_throttled()  # blocks until thermal is OK
        snap = monitor.latest        # non-blocking read

        monitor.stop()
    """

    def __init__(
        self,
        interval_s: float = 2.0,
        temp_limit_c: float = 85.0,
        power_limit_mw: float = 50_000.0,
        on_snapshot: callable | None = None,
    ):
        self.interval_s = interval_s
        self.temp_limit_c = temp_limit_c
        self.power_limit_mw = power_limit_mw
        self.on_snapshot = on_snapshot
        self._latest: ThermalSnapshot = ThermalSnapshot(timestamp=time.monotonic())
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._throttle_event = threading.Event()
        self._throttle_event.set()  # not throttled initially

    @property
    def latest(self) -> ThermalSnapshot:
        with self._lock:
            return self._latest

    @property
    def is_throttled(self) -> bool:
        return not self._throttle_event.is_set()

    def start(self) -> None:
        """Start background polling thread."""
        if self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop background polling."""
        self._stop.set()
        self._throttle_event.set()  # unblock any waiters
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None

    def wait_if_throttled(self, timeout_s: float = 30.0) -> bool:
        """Block until thermal is OK. Returns False if timed out."""
        return self._throttle_event.wait(timeout=timeout_s)

    def _poll_loop(self) -> None:
        while not self._stop.is_set():
            snap = read_thermal()
            should_throttle = self._check_limits(snap)

            with self._lock:
                self._latest = snap

            if should_throttle:
                self._throttle_event.clear()
            else:
                self._throttle_event.set()

            if self.on_snapshot is not None:
                self.on_snapshot(snap)

            self._stop.wait(timeout=self.interval_s)

    def _check_limits(self, snap: ThermalSnapshot) -> bool:
        """Return True if we should throttle."""
        if snap.metal_pressure in (ThermalPressure.SERIOUS, ThermalPressure.CRITICAL):
            return True
        if snap.nv_temp_c is not None and snap.nv_temp_c >= self.temp_limit_c:
            return True
        if snap.metal_gpu_power_mw is not None and snap.metal_gpu_power_mw >= self.power_limit_mw:
            return True
        if snap.nv_gpu_power_mw is not None and snap.nv_gpu_power_mw >= self.power_limit_mw:
            return True
        return False


# ---------------------------------------------------------------------------
# CLI thermal display
# ---------------------------------------------------------------------------

def print_thermal_header() -> None:
    """Print CLI thermal display header."""
    print(f"{'Layer':>6} {'Compress':>10} {'Transfer':>10} {'Decomp':>10} "
          f"{'Total':>8} {'Ratio':>6} {'Metal':>12} {'NV':>8} {'Status':>10}")
    print("─" * 90)


def print_thermal_row(
    layer_idx: int,
    compress_ms: float,
    transfer_ms: float,
    decompress_ms: float,
    ratio: float,
    snap: ThermalSnapshot | None,
) -> None:
    """Print one row of CLI thermal display."""
    total_ms = compress_ms + transfer_ms + decompress_ms
    metal_str = f"{snap.metal_gpu_power_mw:.0f} mW" if snap and snap.metal_gpu_power_mw is not None else "—"
    nv_str = f"{snap.nv_temp_c:.0f}°C" if snap and snap.nv_temp_c is not None else "—"
    status = "THROTTLED" if snap and snap.is_throttled else "OK"
    print(f"{layer_idx:>6} {compress_ms:>9.1f}ms {transfer_ms:>9.1f}ms {decompress_ms:>9.1f}ms "
          f"{total_ms:>7.1f}ms {ratio:>5.1f}x {metal_str:>12} {nv_str:>8} {status:>10}")


def print_thermal_footer(
    pipeline_summary: str,
    snap: ThermalSnapshot | None,
) -> None:
    """Print CLI thermal display footer."""
    print("─" * 90)
    print(f"  Pipeline: {pipeline_summary}")
    if snap:
        print(f"  Thermal:  {snap.summary()}")
