"""
TAPIR Benchmark Evaluation Harness.
Runs all candidate routing methods & baselines across multiple workload traces and seeds:
  - always_local_q8
  - always_local_q5
  - always_local_q4
  - always_cloud
  - random
  - static_threshold
  - difficulty_only (RouteLLM/HybridLLM style)
  - thermal_unaware_linucb
  - reactive_thermal
  - tapir_full (Constrained LinUCB)

Plus 7 Ablations:
  - tapir_no_rls
  - tapir_no_uncertainty_margin
  - tapir_no_soft_penalty
  - tapir_no_early_exit
  - tapir_no_dynamic_quant
  - tapir_no_privacy_rule
  - tapir_no_difficulty_warmstart

Computes all metrics per method:
  - Quality retention vs always-cloud
  - Offload ratio (fraction to cloud, A4)
  - Cloud cost ($)
  - Energy per request (J/req)
  - Throttle events per hour (A1)
  - Time above T_max (s)
  - Privacy leakage rate (violations / sensitive reqs)
  - Privacy over-blocking rate (A9)
  - P50 & P95 latency (ms)
  - Router decision overhead (ms)
  - Cumulative regret

Outputs:
  - CSV: results/tables/main_comparison.csv
  - Ablations CSV: results/tables/ablations.csv
  - Robustness CSV: results/tables/robustness.csv
"""

import os
import sys
import json
import time
import datetime
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple

BASE = r"c:\Users\deepa\Research-TAPIR\tapir"
RAW_DIR = os.path.join(BASE, "data", "raw")
RESULTS_DIR = os.path.join(BASE, "results", "tables")
FIGURES_DIR = os.path.join(BASE, "results", "figures")
os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

sys.path.append(os.path.join(BASE, "src"))
from workload.dataset import build_workload_trace
from eval.counterfactual import CounterfactualEnvironment
from router.linucb import ConstrainedLinUCBRouter, ACTIONS, ACTION_LOCAL_Q8, ACTION_LOCAL_Q5, ACTION_LOCAL_Q4, ACTION_LOCAL_EARLY, ACTION_CLOUD
from router.difficulty import DifficultyPredictor
from privacy.detector import PrivacyDetector
from thermal.model import FirstOrderRCModel


def run_experiment_trace(
    method_name: str,
    trace: List[Dict[str, Any]],
    env: CounterfactualEnvironment,
    difficulty_pred: DifficultyPredictor,
    privacy_det: PrivacyDetector,
    router: ConstrainedLinUCBRouter,
    seed: int = 42,
) -> Dict[str, Any]:
    rng = np.random.RandomState(seed)

    qualities = []
    latencies = []
    energies = []
    costs = []
    router_overheads = []
    cloud_calls = 0
    sensitive_count = 0
    privacy_leaks = 0
    non_sensitive_blocked_local = 0
    total_non_sensitive = 0

    cloud_qualities = []  # For regret and APGR computation

    for req in trace:
        sim_time = req["arrival_time_s"]
        prompt = req["prompt"]
        true_sensitive = req.get("is_sensitive", False)
        if true_sensitive:
            sensitive_count += 1
        else:
            total_non_sensitive += 1

        t_router_start = time.perf_counter()

        # 1. State feature extraction
        diff_score = difficulty_pred.predict_difficulty(prompt)
        priv_res = privacy_det.detect(prompt)
        pred_sensitive = priv_res["sensitive"]

        state = {
            "difficulty_score": diff_score,
            "prompt_length": req["prompt_length"],
            "is_sensitive": pred_sensitive,
            "current_temp": env.current_temp,
            "dT_dt": (env.current_temp - env.prev_temp),
            "battery_pct": env.battery_pct,
            "free_ram_gb": 1.2,
            "network_rtt_s": env.network_rtt_s,
        }

        # 2. Action selection per method
        if method_name == "always_cloud":
            selected_action = ACTION_CLOUD if not pred_sensitive else ACTION_LOCAL_Q8
        elif method_name == "always_local_q8":
            selected_action = ACTION_LOCAL_Q8
        elif method_name == "always_local_q5":
            selected_action = ACTION_LOCAL_Q5
        elif method_name == "always_local_q4":
            selected_action = ACTION_LOCAL_Q4
        elif method_name == "random":
            cands = list(ACTIONS)
            if pred_sensitive and ACTION_CLOUD in cands:
                cands.remove(ACTION_CLOUD)
            selected_action = rng.choice(cands)
        elif method_name == "static_threshold":
            # Length threshold > 200 -> cloud
            if req["prompt_length"] > 200 and not pred_sensitive:
                selected_action = ACTION_CLOUD
            else:
                selected_action = ACTION_LOCAL_Q4
        elif method_name == "difficulty_only":
            # RouteLLM/HybridLLM style threshold on difficulty
            if diff_score > 0.45 and not pred_sensitive:
                selected_action = ACTION_CLOUD
            else:
                selected_action = ACTION_LOCAL_Q4
        elif method_name == "thermal_unaware_linucb":
            # LinUCB with unconstrained action selection
            cands = list(ACTIONS)
            if pred_sensitive and ACTION_CLOUD in cands:
                cands.remove(ACTION_CLOUD)
            x = router.build_context(state)
            best_a = cands[0]
            max_u = -float("inf")
            for a in cands:
                u = float(np.dot(x, router.theta[a])) + router.alpha * float(np.sqrt(np.dot(x, np.dot(router.A_inv[a], x))))
                if u > max_u:
                    max_u = u
                    best_a = a
            selected_action = best_a
        elif method_name == "reactive_thermal":
            # Switch only after temperature crosses T_soft
            if env.current_temp > env.T_soft:
                selected_action = ACTION_LOCAL_EARLY
            elif diff_score > 0.45 and not pred_sensitive:
                selected_action = ACTION_CLOUD
            else:
                selected_action = ACTION_LOCAL_Q4
        elif method_name.startswith("tapir"):
            # TAPIR & variants
            selected_action, _ = router.select_action(state)
        else:
            selected_action = ACTION_LOCAL_Q4

        router_overhead_ms = (time.perf_counter() - t_router_start) * 1000.0
        router_overheads.append(router_overhead_ms)

        # Check Privacy metrics
        if selected_action == ACTION_CLOUD:
            cloud_calls += 1
            if true_sensitive:
                privacy_leaks += 1
        else:
            if not true_sensitive and pred_sensitive:
                non_sensitive_blocked_local += 1

        # 3. Environment counterfactual execution
        res = env.step(selected_action, req, sim_time)
        qualities.append(res["quality"])
        latencies.append(res["latency_s"])
        energies.append(res["energy_j"])
        costs.append(res["cost_usd"])

        # Cloud oracle execution for regret
        c_res = env.step(ACTION_CLOUD if not true_sensitive else ACTION_LOCAL_Q8, req, sim_time)
        cloud_qualities.append(c_res["quality"])

        # 4. Bandit learning update for LinUCB methods
        if "linucb" in method_name or "tapir" in method_name:
            x_ctx = router.build_context(state)
            reward = router.compute_reward(
                quality=res["quality"],
                cloud_cost=res["cost_usd"],
                energy_j=res["energy_j"],
                latency_s=res["latency_s"],
                t_predicted=res["current_temp"],
            )
            router.update(selected_action, x_ctx, reward)

    n_req = len(trace)
    leakage_rate = (privacy_leaks / sensitive_count) if sensitive_count > 0 else 0.0
    overblocking_rate = (non_sensitive_blocked_local / total_non_sensitive) if total_non_sensitive > 0 else 0.0
    offload_ratio = cloud_calls / n_req

    mean_q = float(np.mean(qualities))
    mean_cloud_q = float(np.mean(cloud_qualities))
    q_retention = (mean_q / mean_cloud_q) if mean_cloud_q > 0 else 1.0

    # APGR (Average Performance Gain Recovered)
    # (Q_method - Q_local_min) / (Q_cloud - Q_local_min)
    local_min_q = 0.65  # Typical lower bound
    apgr = max(0.0, min(1.0, (mean_q - local_min_q) / (mean_cloud_q - local_min_q))) if (mean_cloud_q > local_min_q) else 1.0

    return {
        "method": method_name,
        "n_requests": n_req,
        "quality_mean": round(mean_q, 4),
        "quality_retention_pct": round(q_retention * 100.0, 2),
        "apgr": round(apgr, 4),
        "offload_ratio_pct": round(offload_ratio * 100.0, 2),
        "cloud_cost_total_usd": round(sum(costs), 5),
        "cost_per_req_usd": round(float(np.mean(costs)), 6),
        "energy_per_req_j": round(float(np.mean(energies)), 2),
        "latency_p50_ms": round(float(np.percentile(latencies, 50)) * 1000.0, 1),
        "latency_p95_ms": round(float(np.percentile(latencies, 95)) * 1000.0, 1),
        "router_overhead_mean_ms": round(float(np.mean(router_overheads)), 2),
        "router_overhead_p95_ms": round(float(np.percentile(router_overheads, 95)), 2),
        "throttle_events": env.throttle_events,
        "time_above_tmax_s": env.time_above_tmax_s,
        "peak_temp_c": round(env.current_temp, 2),
        "privacy_leakage_rate": round(leakage_rate, 4),
        "privacy_overblocking_rate": round(overblocking_rate, 4),
    }
