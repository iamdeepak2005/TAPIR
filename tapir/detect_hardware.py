"""
Phase 0: Detect hardware and system capabilities.
Writes data/raw/device_info.json.
"""
import json
import os
import platform
import datetime
import subprocess
import sys

def get_device_info():
    info = {
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "python_version": platform.python_version(),
        },
        "cpu": {},
        "memory": {},
        "gpu": {},
        "battery": {},
        "thermal_sensors": {},
        "power_sensors": {},
        "available_tools": {},
    }

    # CPU info
    try:
        import psutil
        info["cpu"]["logical_cores"] = psutil.cpu_count(logical=True)
        info["cpu"]["physical_cores"] = psutil.cpu_count(logical=False)
        freq = psutil.cpu_freq()
        if freq:
            info["cpu"]["max_freq_mhz"] = freq.max
            info["cpu"]["min_freq_mhz"] = freq.min
            info["cpu"]["current_freq_mhz"] = freq.current
    except Exception as e:
        info["cpu"]["error"] = str(e)

    # Processor identifier from env
    proc_id = os.environ.get("PROCESSOR_IDENTIFIER", "unknown")
    info["cpu"]["processor_identifier"] = proc_id

    # Memory
    try:
        import psutil
        vm = psutil.virtual_memory()
        info["memory"]["total_gb"] = round(vm.total / (1024**3), 2)
        info["memory"]["available_gb"] = round(vm.available / (1024**3), 2)
        info["memory"]["used_percent"] = vm.percent
    except Exception as e:
        info["memory"]["error"] = str(e)

    # GPU
    # Check nvidia-smi
    try:
        result = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
                                  "--format=csv,noheader"], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            info["gpu"]["nvidia"] = result.stdout.strip()
            info["gpu"]["type"] = "nvidia_discrete"
        else:
            info["gpu"]["nvidia"] = "not_available"
    except FileNotFoundError:
        info["gpu"]["nvidia"] = "nvidia-smi not found"
    except Exception as e:
        info["gpu"]["nvidia_error"] = str(e)

    # Check for AMD integrated GPU (common with Ryzen mobile)
    if "AMD" in proc_id:
        info["gpu"]["note"] = "AMD integrated GPU likely present (Ryzen mobile). No nvidia-smi."
        info["gpu"]["type"] = "amd_integrated"

    # Battery
    try:
        import psutil
        bat = psutil.sensors_battery()
        if bat:
            info["battery"]["percent"] = bat.percent
            info["battery"]["power_plugged"] = bat.power_plugged
            info["battery"]["secs_left"] = bat.secsleft if bat.secsleft != psutil.POWER_TIME_UNLIMITED else "unlimited"
            info["battery"]["present"] = True
        else:
            info["battery"]["present"] = False
    except Exception as e:
        info["battery"]["error"] = str(e)

    # Thermal sensors
    try:
        import psutil
        temps = psutil.sensors_temperatures()
        if temps:
            info["thermal_sensors"]["available"] = True
            for name, entries in temps.items():
                info["thermal_sensors"][name] = [
                    {"label": e.label, "current": e.current, "high": e.high, "critical": e.critical}
                    for e in entries
                ]
        else:
            info["thermal_sensors"]["available"] = False
            info["thermal_sensors"]["note"] = "psutil.sensors_temperatures() returned empty on Windows. Will need WMI or LibreHardwareMonitor."
    except AttributeError:
        info["thermal_sensors"]["available"] = False
        info["thermal_sensors"]["note"] = "sensors_temperatures not available on this platform"
    except Exception as e:
        info["thermal_sensors"]["error"] = str(e)

    # Power sensors
    info["power_sensors"]["rapl"] = False
    info["power_sensors"]["note"] = "No RAPL on Windows. Power measurement via battery discharge rate (unplugged) or external tool."
    if info["battery"].get("present"):
        info["power_sensors"]["battery_derived"] = True
        info["power_sensors"]["method"] = "battery-derived (discharge rate when unplugged)"
    else:
        info["power_sensors"]["battery_derived"] = False

    # Check for WMI temperature access
    try:
        result = subprocess.run(
            ["powershell", "-Command",
             "Get-CimInstance -Namespace root/WMI -ClassName MSAcpi_ThermalZoneTemperature -ErrorAction Stop | Select-Object InstanceName, CurrentTemperature | ConvertTo-Json"],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0 and result.stdout.strip():
            info["thermal_sensors"]["wmi_thermal_zone"] = json.loads(result.stdout.strip())
            info["thermal_sensors"]["wmi_available"] = True
        else:
            info["thermal_sensors"]["wmi_available"] = False
            info["thermal_sensors"]["wmi_error"] = result.stderr.strip()[:200] if result.stderr else "empty output"
    except Exception as e:
        info["thermal_sensors"]["wmi_available"] = False
        info["thermal_sensors"]["wmi_error"] = str(e)

    # Available tools
    for tool in ["llama-cli", "llama-bench", "git", "cmake", "gcc", "cl"]:
        try:
            result = subprocess.run(["where", tool], capture_output=True, text=True, timeout=5)
            info["available_tools"][tool] = result.stdout.strip() if result.returncode == 0 else "not found"
        except Exception:
            info["available_tools"][tool] = "not found"

    # Check adb for phone phase
    try:
        result = subprocess.run(["adb", "devices"], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            info["available_tools"]["adb"] = result.stdout.strip()
        else:
            info["available_tools"]["adb"] = "not found"
    except FileNotFoundError:
        info["available_tools"]["adb"] = "not found"

    # Installed Python packages
    try:
        result = subprocess.run([sys.executable, "-m", "pip", "list", "--format=json"],
                                capture_output=True, text=True, timeout=15)
        if result.returncode == 0:
            packages = json.loads(result.stdout)
            info["python_packages"] = {p["name"]: p["version"] for p in packages}
    except Exception as e:
        info["python_packages_error"] = str(e)

    return info


if __name__ == "__main__":
    info = get_device_info()
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "raw", "device_info.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(info, f, indent=2, default=str)
    print(f"Device info written to: {out_path}")

    # Summary
    print(f"\n=== SYSTEM SUMMARY ===")
    print(f"OS: {info['platform']['system']} {info['platform']['release']}")
    print(f"CPU: {info['cpu'].get('processor_identifier', 'unknown')}")
    print(f"Cores: {info['cpu'].get('physical_cores', '?')} physical, {info['cpu'].get('logical_cores', '?')} logical")
    print(f"CPU Freq: {info['cpu'].get('current_freq_mhz', '?')} MHz (max: {info['cpu'].get('max_freq_mhz', '?')})")
    print(f"RAM: {info['memory'].get('total_gb', '?')} GB total, {info['memory'].get('available_gb', '?')} GB available")
    print(f"GPU: {info['gpu'].get('type', 'unknown')}")
    print(f"Battery: present={info['battery'].get('present', '?')}, percent={info['battery'].get('percent', '?')}")
    print(f"Thermal sensors via psutil: {info['thermal_sensors'].get('available', False)}")
    print(f"WMI thermal: {info['thermal_sensors'].get('wmi_available', False)}")
    print(f"Power method: {info['power_sensors'].get('method', 'none')}")
    print(f"llama-cli: {info['available_tools'].get('llama-cli', 'not found')}")

