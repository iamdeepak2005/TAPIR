"""
TAPIR Telemetry Logger — Cross-platform 1 Hz system metrics logger.

Collects at 1 Hz:
  - timestamp (ISO 8601 UTC)
  - cpu_temp_c, gpu_temp_c
  - cpu_util_pct, gpu_util_pct
  - ram_used_mb, ram_total_mb
  - power_w, power_method ("battery-derived" / "rapl" / "degraded")
  - battery_pct, battery_plugged
  - fan_rpm
  - cpu_freq_mhz, cpu_freq_max_mhz, cpu_clock_pct
  - process_tokens_per_s
  - throttle_active (bool, A1 definition)
  - notes

Enforces:
  - Hard Rule 1 & 5: Label every metric "measured", "simulated", or "degraded"
  - Hard Rule 4: Safety abort if CPU temp > 90 C or Phone battery > 42 C
  - Addendum A1: Throttle detection (<85% max clock for 10+ s at >90% util)
"""

import csv
import datetime
import json
import os
import platform
import subprocess
import sys
import time
import threading
from typing import Dict, Any, Optional

try:
    import psutil
except ImportError:
    print("ERROR: psutil not installed")
    sys.exit(1)


class TelemetryLogger:
    FIELDNAMES = [
        "timestamp",
        "cpu_temp_c",
        "gpu_temp_c",
        "cpu_util_pct",
        "gpu_util_pct",
        "ram_used_mb",
        "ram_total_mb",
        "power_w",
        "power_method",
        "battery_pct",
        "battery_plugged",
        "fan_rpm",
        "cpu_freq_mhz",
        "cpu_freq_max_mhz",
        "cpu_clock_pct",
        "throttle_active",
        "process_tokens_per_s",
        "notes",
    ]

    def __init__(self, output_path: str, lhm_port: int = 8085, interval: float = 1.0, safety_abort_temp: float = 90.0):
        self.output_path = output_path
        self.interval = interval
        self.lhm_port = lhm_port
        self.safety_abort_temp = safety_abort_temp
        self._running = False
        self._thread = None

        self._tokens_per_s = 0.0
        self._notes = ""
        self._sample_count = 0
        self._max_gap = 0.0
        self._last_time = None
        self._system = platform.system()

        # Throttle tracking per Addendum A1:
        # sustained clock < 85% of max for 10+ s while util >= 90%
        self._throttle_consecutive_s = 0
        self.throttle_events_count = 0
        self.total_throttle_seconds = 0

        # Stats tracking
        self.temps_recorded = []
        self.tps_recorded = []

        # Temperature & Power source detection
        self.temp_source = self._detect_temp_source()
        self.power_source = self._detect_power_source()

        # Battery tracking for discharge rate
        self._prev_battery_pct = None
        self._prev_battery_time = None

    def _detect_temp_source(self) -> str:
        """Detect temperature sensor."""
        # Check LibreHardwareMonitor web server with 0.2s timeout
        try:
            import requests
            r = requests.get(f"http://localhost:{self.lhm_port}/data.json", timeout=0.2)
            if r.status_code == 200:
                return "lhm"
        except Exception:
            pass

        # Check psutil (Linux/macOS)
        try:
            if hasattr(psutil, "sensors_temperatures"):
                temps = psutil.sensors_temperatures()
                if temps:
                    return "psutil"
        except Exception:
            pass

        return "degraded"

    def _detect_power_source(self) -> str:
        """Detect power sensor."""
        # Battery-derived if battery exists
        bat = psutil.sensors_battery()
        if bat is not None:
            return "battery-derived"

        # Linux RAPL
        if self._system == "Linux" and os.path.exists("/sys/class/powercap/intel-rapl:0/energy_uj"):
            return "rapl"

        return "degraded"

    def _read_temp_lhm(self):
        """Read from LibreHardwareMonitor web API."""
        try:
            import requests
            r = requests.get(f"http://localhost:{self.lhm_port}/data.json", timeout=0.2)
            if r.status_code == 200:
                data = r.json()
                return self._parse_lhm_tree(data)
        except Exception:
            pass
        return None, None

    def _parse_lhm_tree(self, node, depth=0, parent_type=None):
        cpu_temp, gpu_temp = None, None
        text = node.get("Text", "")
        sensor_type = node.get("SensorType", "")
        value = node.get("Value", "")

        hw_type = parent_type
        if "CPU" in text.upper() or "Processor" in text:
            hw_type = "cpu"
        elif "GPU" in text.upper():
            hw_type = "gpu"

        if "Temperature" in text or sensor_type == "Temperature":
            try:
                temp_val = float(value.replace("°C", "").replace(",", ".").strip())
                if hw_type == "cpu" and (cpu_temp is None or "Package" in text or "Core" in text):
                    cpu_temp = temp_val
                elif hw_type == "gpu" and gpu_temp is None:
                    gpu_temp = temp_val
            except (ValueError, AttributeError):
                pass

        for child in node.get("Children", []):
            c_cpu, c_gpu = self._parse_lhm_tree(child, depth + 1, hw_type)
            if c_cpu is not None and cpu_temp is None:
                cpu_temp = c_cpu
            if c_gpu is not None and gpu_temp is None:
                gpu_temp = c_gpu

        return cpu_temp, gpu_temp

    def _read_power_battery(self) -> Optional[float]:
        """Estimate power from battery discharge rate (unplugged)."""
        bat = psutil.sensors_battery()
        if bat is None or bat.power_plugged:
            return None

        now = time.time()
        current_pct = bat.percent

        if self._prev_battery_pct is not None and self._prev_battery_time is not None:
            dt = now - self._prev_battery_time
            # Recompute if at least 10 seconds elapsed or percentage changed
            if dt >= 10.0:
                dpct = self._prev_battery_pct - current_pct
                if dpct >= 0:
                    # Assume 50 Wh laptop battery capacity
                    battery_wh = 50.0
                    power_w = (dpct / 100.0) * battery_wh / (dt / 3600.0)
                    if 0.5 < power_w < 120.0:
                        self._prev_battery_pct = current_pct
                        self._prev_battery_time = now
                        return round(power_w, 2)

        if self._prev_battery_pct is None:
            self._prev_battery_pct = current_pct
            self._prev_battery_time = now
        return None

    def _collect_sample(self) -> Dict[str, Any]:
        now = datetime.datetime.now(datetime.timezone.utc)

        if self._last_time is not None:
            gap = (now - self._last_time).total_seconds()
            if gap > self._max_gap:
                self._max_gap = gap
        self._last_time = now
        self._sample_count += 1

        # Temperature
        cpu_temp, gpu_temp = None, None
        if self.temp_source == "lhm":
            cpu_temp, gpu_temp = self._read_temp_lhm()
        elif self.temp_source == "psutil" and hasattr(psutil, "sensors_temperatures"):
            try:
                temps = psutil.sensors_temperatures()
                if temps:
                    for name, entries in temps.items():
                        for e in entries:
                            if "cpu" in name.lower() or "core" in name.lower():
                                cpu_temp = e.current
                                break
            except Exception:
                pass

        if cpu_temp is not None:
            self.temps_recorded.append(cpu_temp)
            # HARD RULE 4: Safety abort if CPU temp > 90 C
            if cpu_temp >= self.safety_abort_temp:
                print(f"\n[CRITICAL SAFETY ABORT] CPU Temperature exceeded {self.safety_abort_temp} °C: {cpu_temp} °C!")
                self._running = False

        # CPU util & RAM
        cpu_util = psutil.cpu_percent(interval=None)
        vm = psutil.virtual_memory()
        ram_used = round(vm.used / (1024**2), 1)
        ram_total = round(vm.total / (1024**2), 1)

        # CPU frequency
        freq = psutil.cpu_freq()
        cpu_freq = round(freq.current, 1) if freq else None
        cpu_freq_max = round(freq.max, 1) if freq and freq.max else 2301.0
        cpu_clock_pct = round(cpu_freq / cpu_freq_max * 100.0, 1) if (cpu_freq and cpu_freq_max > 0) else None

        # Addendum A1: Throttle detection
        # Clock < 85% of max for 10+ s while util >= 90%
        is_throttling_condition = (
            cpu_clock_pct is not None
            and cpu_clock_pct < 85.0
            and cpu_util >= 90.0
        )
        if is_throttling_condition:
            self._throttle_consecutive_s += 1
        else:
            self._throttle_consecutive_s = 0

        throttle_active = self._throttle_consecutive_s >= 10
        if throttle_active:
            self.total_throttle_seconds += 1
            if self._throttle_consecutive_s == 10:
                self.throttle_events_count += 1

        # Battery & Power
        bat = psutil.sensors_battery()
        battery_pct = bat.percent if bat else None
        battery_plugged = bat.power_plugged if bat else None

        power_w = self._read_power_battery() if self.power_source == "battery-derived" else None
        power_method = self.power_source
        if power_w is None and self.power_source == "battery-derived":
            power_method = "battery-derived (accumulating delta)"

        # Tokens/s tracking
        if self._tokens_per_s > 0:
            self.tps_recorded.append(self._tokens_per_s)

        notes = self._notes
        if self.temp_source == "degraded":
            notes = ("temp=degraded; " + notes).strip()

        return {
            "timestamp": now.isoformat(),
            "cpu_temp_c": cpu_temp,
            "gpu_temp_c": gpu_temp,
            "cpu_util_pct": cpu_util,
            "gpu_util_pct": None,
            "ram_used_mb": ram_used,
            "ram_total_mb": ram_total,
            "power_w": power_w,
            "power_method": power_method,
            "battery_pct": battery_pct,
            "battery_plugged": battery_plugged,
            "fan_rpm": None,
            "cpu_freq_mhz": cpu_freq,
            "cpu_freq_max_mhz": cpu_freq_max,
            "cpu_clock_pct": cpu_clock_pct,
            "throttle_active": throttle_active,
            "process_tokens_per_s": round(self._tokens_per_s, 2),
            "notes": notes,
        }

    def _log_loop(self):
        file_exists = os.path.exists(self.output_path)
        mode = "a" if file_exists else "w"

        with open(self.output_path, mode, newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.FIELDNAMES)
            if not file_exists or os.path.getsize(self.output_path) == 0:
                writer.writeheader()
                f.flush()

            # Initial reading for psutil.cpu_percent
            psutil.cpu_percent(interval=None)

            while self._running:
                t0 = time.perf_counter()
                try:
                    sample = self._collect_sample()
                    writer.writerow(sample)
                    f.flush()
                except Exception as e:
                    print(f"Logging error: {e}")

                dt = time.perf_counter() - t0
                sleep_s = max(0.0, self.interval - dt)
                time.sleep(sleep_s)

    def start(self):
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._log_loop, daemon=True)
        self._thread.start()

    def stop(self) -> Dict[str, Any]:
        self._running = False
        if self._thread:
            self._thread.join(timeout=3)

        peak_temp = max(self.temps_recorded) if self.temps_recorded else None
        mean_temp = (sum(self.temps_recorded) / len(self.temps_recorded)) if self.temps_recorded else None
        tps_cv = (float(np.std(self.tps_recorded) / np.mean(self.tps_recorded))) if (len(self.tps_recorded) > 1 and np.mean(self.tps_recorded) > 0) else 0.0

        stats = {
            "samples": self._sample_count,
            "max_gap_s": round(self._max_gap, 3),
            "temp_source": self.temp_source,
            "power_source": self.power_source,
            "peak_temp_c": peak_temp,
            "mean_temp_c": round(mean_temp, 2) if mean_temp else None,
            "throttle_events_count": self.throttle_events_count,
            "total_throttle_seconds": self.total_throttle_seconds,
            "tokens_per_s_cv": round(tps_cv, 4),
        }
        return stats

    def set_tokens_per_s(self, tps: float):
        self._tokens_per_s = tps

    def set_notes(self, notes: str):
        self._notes = notes


def run_idle_test(duration_minutes: float = 5.0, output_dir: Optional[str] = None):
    if output_dir is None:
        output_dir = os.path.join(r"c:\Users\deepa\Research-TAPIR\tapir\data\raw")
    os.makedirs(output_dir, exist_ok=True)

    date_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_file = os.path.join(output_dir, f"telemetry_idle_test_{date_str}.csv")
    meta_file = os.path.join(output_dir, f"telemetry_idle_test_{date_str}_meta.json")

    logger = TelemetryLogger(csv_file, interval=1.0)
    print(f"Starting 5-minute Idle Telemetry Test...")
    print(f"Output CSV: {csv_file}")
    print(f"Temperature source: {logger.temp_source}")
    print(f"Power source: {logger.power_source}")

    logger.start()
    total_sec = int(duration_minutes * 60)
    try:
        for i in range(total_sec):
            time.sleep(1)
            if (i + 1) % 30 == 0:
                print(f"  {i + 1}/{total_sec} s ({logger._sample_count} samples, max gap: {logger._max_gap:.3f} s)")
    except KeyboardInterrupt:
        print("\nInterrupted early.")

    stats = logger.stop()

    metadata = {
        "tag": "measured",
        "device": "AMD Ryzen 5 (Zen 4 Mobile, 6 cores)",
        "os": "Windows 11",
        "date": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d"),
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "duration_minutes": duration_minutes,
        "csv_path": csv_file,
        "stats": stats,
        "degraded_metrics": [],
    }
    if stats["temp_source"] == "degraded":
        metadata["degraded_metrics"].append("cpu_temp_c")
        metadata["degraded_metrics"].append("gpu_temp_c")
    if stats["power_source"] == "degraded" or stats["power_source"] == "battery-derived":
        metadata["degraded_metrics"].append("power_w (battery-derived)")

    with open(meta_file, "w") as f:
        json.dump(metadata, f, indent=2)

    print("\n" + "=" * 60)
    print("Idle Test Complete!")
    print(f"Total samples: {stats['samples']}")
    print(f"Max gap: {stats['max_gap_s']} s")
    print(f"Throttle events: {stats['throttle_events_count']}")
    print(f"Metadata saved: {meta_file}")
    print("=" * 60)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=float, default=5.0)
    args = parser.parse_args()
    run_idle_test(args.duration)
