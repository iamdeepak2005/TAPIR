"""
Probe for thermal sensors on Windows.
Checks:
1. LibreHardwareMonitor Web Server (default port 8085)
2. WMI MSAcpi_ThermalZoneTemperature
3. OpenHardwareMonitor WMI / HTTP
4. psutil sensors
"""
import requests
import subprocess
import json
import psutil

print("=== Probing Thermal Sensors ===")

# 1. LHM HTTP
lhm_found = False
for port in [8085, 8080]:
    try:
        r = requests.get(f"http://localhost:{port}/data.json", timeout=1)
        if r.status_code == 200:
            print(f"[OK] LibreHardwareMonitor found on port {port}!")
            data = r.json()
            print(f"     Root node: {data.get('Text')}")
            lhm_found = True
            break
    except Exception as e:
        pass
if not lhm_found:
    print("[FAIL] LibreHardwareMonitor web server not detected on localhost:8085 or 8080.")

# 2. WMI
try:
    cmd = ["powershell", "-Command", "Get-CimInstance -Namespace root/WMI -ClassName MSAcpi_ThermalZoneTemperature -ErrorAction SilentlyContinue | Select-Object InstanceName, CurrentTemperature | ConvertTo-Json"]
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
    if res.returncode == 0 and res.stdout.strip():
        print(f"[OK] WMI MSAcpi_ThermalZoneTemperature: {res.stdout.strip()}")
    else:
        print("[FAIL] WMI MSAcpi_ThermalZoneTemperature not accessible or returned empty.")
except Exception as e:
    print(f"[ERROR] WMI probe: {e}")

# 3. psutil
try:
    temps = psutil.sensors_temperatures()
    if temps:
        print(f"[OK] psutil sensors_temperatures: {temps}")
    else:
        print("[FAIL] psutil.sensors_temperatures() returned empty/None.")
except Exception as e:
    print(f"[ERROR] psutil temps: {e}")

# 4. Battery
try:
    bat = psutil.sensors_battery()
    print(f"Battery: {bat}")
except Exception as e:
    print(f"Battery error: {e}")
