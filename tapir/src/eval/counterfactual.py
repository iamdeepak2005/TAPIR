"""
TAPIR Counterfactual Evaluation Protocol (Addendum A10).

Constructs the Full-Feedback Dataset by collecting (or simulating from measured anchor profiles)
the performance of every candidate action:
  - {local_q8, local_q5, local_q4, local_early_exit, cloud}
for each request in the benchmark traces across 5 random seeds.

Logs per action:
  - Quality (accuracy / agreement score in [0.0, 1.0])
  - Dollar cost ($)
  - Latency (seconds)
  - Energy (Joules)
  - Power (Watts)

Simulates live environment state transitions using the identified thermal RC model:
  T[k+1] = a * T[k] + b * P[k] + (1 - a) * T_amb
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple

BASE = r"c:\Users\deepa\Research-TAPIR\tapir"
RAW_DIR = os.path.join(BASE, "data", "raw")
PROCESSED_DIR = os.path.join(BASE, "data", "processed")
RESULTS_DIR = os.path.join(BASE, "results", "tables")

os.makedirs(PROCESSED_DIR, exist_ok=True)


class CounterfactualEnvironment:
    def __init__(
        self,
        quant_profile: Dict[str, Dict[str, float]],
        thermal_a: float = 0.985,
        thermal_b: float = 0.045,
        T_amb: float = 25.0,
        T_max: float = 90.0,
        T_soft: float = 80.0,
        network_rtt_s: float = 0.05,
    ):
        self.quant_profile = quant_profile
        self.a = thermal_a
        self.b = thermal_b
        self.T_amb = T_amb
        self.T_max = T_max
        self.T_soft = T_soft
        self.network_rtt_s = network_rtt_s

        # Current dynamic device state
        self.current_temp = T_amb + 15.0  # Initial warm state ~40C
        self.prev_temp = self.current_temp
        self.battery_pct = 75.0
        self.total_energy_j = 0.0
        self.total_cost_usd = 0.0
        self.throttle_events = 0
        self.time_above_tmax_s = 0
        self.privacy_violations = 0

    def step(self, action: str, request: Dict[str, Any], sim_time_s: float) -> Dict[str, Any]:
        """
        Execute an action counterfactually.
        Returns: quality, cost_usd, latency_s, energy_j, temp_after, throttled
        """
        is_sensitive = request.get("is_sensitive", False)
        diff_score = request.get("difficulty", 0.4)
        prompt_len = request.get("prompt_length", 100)
        task = request.get("task", "open")

        # 1. Privacy Check
        if is_sensitive and action == "cloud":
            self.privacy_violations += 1

        # 2. Performance & Resource Profiles per Action
        # Ground truths / characteristics
        if action == "cloud":
            # Cloud has top quality across all tasks
            quality = 1.00
            cost = 0.00015 * (prompt_len / 4.0) / 1000.0 + 0.0006 * (60.0 / 4.0) / 1000.0
            latency = self.network_rtt_s + 0.45  # Network RTT + cloud generation
            power_w = 2.0  # Idle wifi power
            exec_time = latency
        elif action == "local_q8":
            # Q8 has highest local quality
            quality = max(0.4, 0.96 - 0.25 * diff_score)
            cost = 0.0
            tps = self.quant_profile.get("Q8_0", {}).get("decode_tok_per_s_mean", 14.5)
            gen_tokens = 32
            exec_time = (prompt_len / 4.0) / self.quant_profile.get("Q8_0", {}).get("prefill_tok_per_s_mean", 40.0) + (gen_tokens / tps)
            latency = exec_time
            power_w = self.quant_profile.get("Q8_0", {}).get("mean_power_w", 24.5)
        elif action == "local_q5":
            quality = max(0.38, 0.94 - 0.28 * diff_score)
            cost = 0.0
            tps = self.quant_profile.get("Q5_K_M", {}).get("decode_tok_per_s_mean", 17.8)
            gen_tokens = 32
            exec_time = (prompt_len / 4.0) / self.quant_profile.get("Q5_K_M", {}).get("prefill_tok_per_s_mean", 48.0) + (gen_tokens / tps)
            latency = exec_time
            power_w = self.quant_profile.get("Q5_K_M", {}).get("mean_power_w", 21.0)
        elif action == "local_q4":
            quality = max(0.35, 0.90 - 0.32 * diff_score)
            cost = 0.0
            tps = self.quant_profile.get("Q4_K_M", {}).get("decode_tok_per_s_mean", 21.2)
            gen_tokens = 32
            exec_time = (prompt_len / 4.0) / self.quant_profile.get("Q4_K_M", {}).get("prefill_tok_per_s_mean", 55.0) + (gen_tokens / tps)
            latency = exec_time
            power_w = self.quant_profile.get("Q4_K_M", {}).get("mean_power_w", 18.5)
        elif action == "local_early_exit":
            # Early exit: ~85% layers executed, ~1.25x speedup, slight quality delta
            quality = max(0.32, 0.88 - 0.35 * diff_score)
            cost = 0.0
            tps = 26.0  # Measured prototype speedup
            gen_tokens = 32
            exec_time = (gen_tokens / tps)
            latency = exec_time
            power_w = 14.0
        else:
            raise ValueError(f"Unknown action: {action}")

        # 3. Energy computation
        energy_j = power_w * exec_time
        self.total_energy_j += energy_j
        self.total_cost_usd += cost

        # 4. Thermal Model Update (discrete step across exec_time seconds)
        # T[k+1] = a * T[k] + b * P[k] + (1 - a) * T_amb
        n_steps = max(1, int(round(exec_time)))
        temp = self.current_temp
        for _ in range(n_steps):
            temp = self.a * temp + self.b * power_w + (1.0 - self.a) * self.T_amb

        dT_dt = (temp - self.current_temp) / exec_time if exec_time > 0 else 0.0
        self.prev_temp = self.current_temp
        self.current_temp = temp

        # Check thermal limits
        throttled = False
        if self.current_temp >= self.T_max:
            self.time_above_tmax_s += n_steps
            throttled = True
            self.throttle_events += 1

        # Battery discharge update
        # 50 Wh = 180,000 J
        battery_drain_pct = (energy_j / 180000.0) * 100.0
        self.battery_pct = max(0.0, self.battery_pct - battery_drain_pct)

        return {
            "action": action,
            "quality": round(float(quality), 4),
            "cost_usd": round(float(cost), 6),
            "latency_s": round(float(latency), 4),
            "energy_j": round(float(energy_j), 3),
            "power_w": round(float(power_w), 2),
            "current_temp": round(float(self.current_temp), 2),
            "dT_dt": round(float(dT_dt), 4),
            "throttled": throttled,
            "battery_pct": round(float(self.battery_pct), 2),
        }
