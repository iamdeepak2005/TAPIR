"""
Compute the exact empirical percentage of steps exceeding Proposition 1's error bound:
  |T[k+H] - T_hat[k+H]| > z * sigma_H  (with z = 1.96, delta = 0.05)
across validation traces.
"""
import numpy as np
import sys, os
sys.path.append(r"c:\Users\deepa\Research-TAPIR\tapir\src")
from thermal.fit_thermal_model import generate_synthetic_thermal_traces
from thermal.model import FirstOrderRCModel

(train_t, train_p), (val_t, val_p), T_amb = generate_synthetic_thermal_traces(seed=42)
model = FirstOrderRCModel(dt=1.0)
model.fit(train_t, train_p, T_amb=T_amb)

z = 1.96

for H in [10, 30, 60]:
    preds = []
    actuals = []
    for k in range(len(val_t) - H):
        p_mean = float(np.mean(val_p[k:k+H]))
        preds.append(model.predict_h_step(val_t[k], p_mean, H))
        actuals.append(val_t[k+H])
    preds = np.array(preds)
    actuals = np.array(actuals)
    res = actuals - preds
    sigma_H = np.std(res)
    bound = z * sigma_H

    # Two-sided exceedance: |e| > bound
    two_sided_exceed = np.mean(np.abs(res) > bound) * 100.0
    # One-sided upper exceedance (critical for thermal safety: T_true > T_hat + bound):
    upper_exceed = np.mean(res > bound) * 100.0

    print(f"Horizon H={H}s:")
    print(f"  sigma_H = {sigma_H:.3f} °C, Bound = {bound:.3f} °C")
    print(f"  Two-sided exceedance: {two_sided_exceed:.2f}% (Theoretical: 5.00%)")
    print(f"  Upper-tail exceedance (underprediction): {upper_exceed:.2f}% (Theoretical: 2.50%)")
