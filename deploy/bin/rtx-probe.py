#!/opt/homebrew/bin/python3.12
"""Background probe — reads RTX eGPU temp and utilization via tinygrad.
Writes to /tmp/rtx_stats.txt every second for the C monitor to read."""

import sys, os, time, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "tinygrad"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

STATS_FILE = "/tmp/rtx_stats.json"

def read_rtx_temp():
    try:
        from tinygrad import Device
        from tinygrad.runtime.autogen import nv_580 as nv_gpu
        dev = Device["NV"]
        params = nv_gpu.NV2080_CTRL_THERMAL_SYSTEM_EXECUTE_V2_PARAMS(
            clientAPIVersion=2, clientAPIRevision=0, clientInstructionSizeOf=44,
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
            return float(result.instructionList[0].operands.getStatusSensorReading.value)
    except Exception:
        pass
    return None

def probe_loop():
    """Non-blocking probe: connect, sample temp, disconnect.
    Doesn't hold the TinyGPU lock between samples so inference can run."""
    arch = "unknown"

    while True:
        temp = None
        tb5_ms = -1
        active = False

        try:
            # Brief connection — acquire lock, read temp, release
            from tinygrad import Device, Tensor
            dev = Device["NV"]
            arch = dev.compiler.arch

            temp = read_rtx_temp()

            # Quick TB5 probe
            t0 = time.perf_counter()
            t = Tensor.ones(64, device="NV").realize()
            _ = t.numpy()
            tb5_ms = (time.perf_counter() - t0) * 1000

            # Check if something else ran kernels (inference running)
            from tinygrad import GlobalCounters
            active = GlobalCounters.kernel_count > 0

            stats = {
                "status": "online",
                "arch": arch,
                "temp": temp,
                "tb5_ms": round(tb5_ms, 2),
                "active": active,
                "timestamp": time.time(),
            }
        except Exception:
            # Lock held by inference or eGPU disconnected
            stats = {
                "status": "busy" if arch != "unknown" else "offline",
                "arch": arch,
                "temp": temp,
                "tb5_ms": -1,
                "active": True,  # if locked, something is using the GPU
                "timestamp": time.time(),
            }

        try:
            with open(STATS_FILE + ".tmp", "w") as f:
                json.dump(stats, f)
            os.replace(STATS_FILE + ".tmp", STATS_FILE)
        except Exception:
            pass

        time.sleep(2)  # sample every 2s to minimize lock contention

if __name__ == "__main__":
    probe_loop()
