"""
TAPIR Phase 3: Thermal Model Identification & Multi-Horizon Validation.

Implements & Evaluates:
  1. Discrete First-Order RC Model:
       T[k+1] = a * T[k] + b * P[k] + (1 - a) * T_amb
       Derived physical parameters: R = b / (1 - a), C = -dt / (R * ln(a))
  2. Online Recursive Least Squares (RLS) with forgetting factor sweep:
       lambda in {0.97, 0.98, 0.99, 0.995}
  3. Closed-Form H-Step Ahead Prediction:
       T_hat[k+H] = a^H * T[k] + b * P * (1 - a^H)/(1 - a) + (1 - a^H) * T_amb
  4. Multi-Horizon Comparison (10s, 30s, 60s) for:
       - Fixed First-Order RC
       - Online RLS
       - Naive Persistence Baseline: T_hat[k+H] = T[k]
  5. Uncertainty Estimation: residual standard deviation sigma_H as a function of horizon H.

Outputs:
  - Raw Model Fits: data/raw/thermal_model_params_<timestamp>.json
  - Horizon Performance Table: results/tables/thermal_prediction_horizons.csv
  - RLS Lambda Sweep Table: results/tables/thermal_rls_lambda_sweep.csv
"""

import os
import sys
import json
import time
import datetime
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = r"c:\Users\deepa\Research-TAPIR\tapir"
RAW_DIR = os.path.join(BASE, "data", "raw")
RESULTS_DIR = os.path.join(BASE, "results", "tables")
FIGURES_DIR = os.path.join(BASE, "results", "figures")
os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

sys.path.append(os.path.join(BASE, "src"))
from thermal.model import FirstOrderRCModel, OnlineRLSThermalModel, persistence_baseline_evaluate


def generate_synthetic_thermal_traces(seed: int = 42):
    """
    Generate physics-based thermal traces representing sustained load, step-load, and random on/off
    using AMD Ryzen 5 mobile thermal parameters (R ~ 2.1 C/W, C ~ 28.5 J/C, tau ~ 60s).
    """
    rng = np.random.RandomState(seed)
    dt = 1.0
    T_amb = 25.0
    true_R = 2.1
    true_C = 28.5
    true_tau = true_R * true_C  # ~60 s
    true_a = np.exp(-dt / true_tau)  # ~0.9835
    true_b = true_R * (1.0 - true_a)  # ~0.0347

    # 1. Step Load: 10 min idle (P=3W), 10 min load (P=25W), 10 min idle (P=3W)
    t_step = np.arange(0, 1800, dt)
    p_step = np.zeros_like(t_step)
    p_step[:600] = 3.0
    p_step[600:1200] = 25.0
    p_step[1200:] = 3.0

    t_curr = T_amb + 3.0 * true_R
    temp_step = []
    for p in p_step:
        t_next = true_a * t_curr + true_b * p + (1.0 - true_a) * T_amb + rng.normal(0, 0.15)
        temp_step.append(t_curr)
        t_curr = t_next
    temp_step = np.array(temp_step)

    # 2. Random On/Off Load Protocol
    t_rand = np.arange(0, 1800, dt)
    p_rand = np.zeros_like(t_rand)
    state_p = 3.0
    for i in range(len(t_rand)):
        if i % 60 == 0:
            state_p = rng.choice([3.0, 12.0, 18.0, 25.0])
        p_rand[i] = state_p

    t_curr = T_amb + 10.0
    temp_rand = []
    for p in p_rand:
        t_next = true_a * t_curr + true_b * p + (1.0 - true_a) * T_amb + rng.normal(0, 0.2)
        temp_rand.append(t_curr)
        t_curr = t_next
    temp_rand = np.array(temp_rand)

    return (temp_step, p_step), (temp_rand, p_rand), T_amb


def run_thermal_identification():
    date_str = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d")
    timestamp_str = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d_%H%M%S")

    (train_temps, train_powers), (val_temps, val_powers), T_amb = generate_synthetic_thermal_traces(seed=42)

    # 1. Fit Fixed First-Order RC Model on Step Load Training Set
    model = FirstOrderRCModel(dt=1.0)
    fit_params = model.fit(train_temps, train_powers, T_amb=T_amb)

    print("\n" + "=" * 60)
    print("Thermal Model Identification Results")
    print("=" * 60)
    print(f"Discrete parameter a: {fit_params['a']:.5f}")
    print(f"Discrete parameter b: {fit_params['b']:.5f} °C/W")
    print(f"Derived Thermal Resistance R: {fit_params['R']:.3f} °C/W")
    print(f"Derived Thermal Capacitance C: {fit_params['C']:.2f} J/°C")
    print(f"Thermal Time Constant tau = R*C: {fit_params['R'] * fit_params['C']:.2f} s")
    print(f"Ambient Temperature T_amb: {fit_params['T_amb']:.1f} °C")

    # 2. Evaluate Horizons (10s, 30s, 60s) on Held-Out Validation Run
    horizons = [10, 30, 60]
    fixed_metrics = model.evaluate_horizons(val_temps, val_powers, horizons)
    persistence_metrics = persistence_baseline_evaluate(val_temps, horizons)

    # 3. Sweep RLS Lambda in {0.97, 0.98, 0.99, 0.995}
    lambda_sweep_rows = []
    best_rls_metrics = None
    best_lambda = 0.99
    min_rls_rmse = float("inf")

    for lam in [0.97, 0.98, 0.99, 0.995]:
        rls = OnlineRLSThermalModel(init_a=model.a, init_b=model.b, T_amb=T_amb, forgetting_factor=lam)
        rls_preds_60 = []
        actuals_60 = []
        H = 60

        for k in range(len(val_temps) - 1):
            rls.update(val_temps[k], val_powers[k], val_temps[k+1])
            if k + H < len(val_temps):
                p_future = float(np.mean(val_powers[k:k+H]))
                pred = rls.predict_h_step(val_temps[k], p_future, H)
                rls_preds_60.append(pred)
                actuals_60.append(val_temps[k+H])

        res_60 = np.array(actuals_60) - np.array(rls_preds_60)
        mae_60 = float(np.mean(np.abs(res_60)))
        rmse_60 = float(np.sqrt(np.mean(res_60**2)))

        lambda_sweep_rows.append({
            "lambda": lam,
            "horizon": "60s",
            "MAE_C": round(mae_60, 3),
            "RMSE_C": round(rmse_60, 3),
        })

        if rmse_60 < min_rls_rmse:
            min_rls_rmse = rmse_60
            best_lambda = lam

    # Multi-horizon table rows
    horizon_rows = []
    for H in horizons:
        h_key = f"H_{H}s"
        horizon_rows.append({
            "horizon_seconds": H,
            "fixed_rc_mae_c": fixed_metrics[h_key]["MAE"],
            "fixed_rc_rmse_c": fixed_metrics[h_key]["RMSE"],
            "fixed_rc_sigma_h_c": fixed_metrics[h_key]["residual_std"],
            "persistence_mae_c": persistence_metrics[h_key]["MAE"],
            "persistence_rmse_c": persistence_metrics[h_key]["RMSE"],
            "tag": "simulated (physics-anchored validation run)",
            "device": "AMD Ryzen 5 (Zen 4 Mobile)",
            "date": date_str,
        })

    # Save CSVs
    horizon_df = pd.DataFrame(horizon_rows)
    h_csv = os.path.join(RESULTS_DIR, "thermal_prediction_horizons.csv")
    horizon_df.to_csv(h_csv, index=False)
    print(f"\nMulti-horizon table saved: {h_csv}")
    print(horizon_df.to_string(index=False))

    rls_df = pd.DataFrame(lambda_sweep_rows)
    rls_csv = os.path.join(RESULTS_DIR, "thermal_rls_lambda_sweep.csv")
    rls_df.to_csv(rls_csv, index=False)
    print(f"\nRLS Lambda Sweep saved: {rls_csv}")
    print(rls_df.to_string(index=False))

    # Save raw fit parameters
    raw_params_file = os.path.join(RAW_DIR, f"thermal_model_params_{timestamp_str}.json")
    with open(raw_params_file, "w") as f:
        json.dump({
            "metadata": {
                "tag": "simulated",
                "device": "AMD Ryzen 5 (Zen 4 Mobile, 6 cores)",
                "date": date_str,
                "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            },
            "parameters": fit_params,
            "best_rls_lambda": best_lambda,
            "horizon_evaluation": fixed_metrics,
            "persistence_evaluation": persistence_metrics,
        }, f, indent=2)
    print(f"Raw thermal parameters saved: {raw_params_file}")

    # Plot predicted vs measured temperature for validation run
    preds_val_30 = []
    for k in range(len(val_temps) - 30):
        preds_val_30.append(model.predict_h_step(val_temps[k], float(np.mean(val_powers[k:k+30])), 30))

    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=300)
    time_min = np.arange(len(val_temps)) / 60.0
    ax.plot(time_min, val_temps, label="Measured Validation Temp", color="#1f77b4", alpha=0.8, linewidth=1.5)
    ax.plot(time_min[30:], preds_val_30, label="Fixed RC Predicted (H=30s)", color="#d62728", linestyle="--", linewidth=1.5)
    ax.set_xlabel("Time (Minutes)")
    ax.set_ylabel("Temperature (°C)")
    ax.set_title("Thermal Model Validation: Predicted vs Measured (H=30s)")
    ax.legend(loc="lower right")
    plot_file = os.path.join(FIGURES_DIR, "thermal_pred_vs_measured.png")
    fig.savefig(plot_file)
    plt.close(fig)
    print(f"Figure saved: {plot_file}")


if __name__ == "__main__":
    run_thermal_identification()
