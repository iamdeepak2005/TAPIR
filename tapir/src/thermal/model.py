"""
TAPIR Thermal Model Identification & Multi-Horizon Predictor.

Implements:
1. First-order discrete RC model:
     T[k+1] = a * T[k] + b * P[k] + (1 - a) * T_amb
     Derived parameters: R = b / (1 - a), C = -dt / (R * ln(a))
2. Recursive Least Squares (RLS) with forgetting factor lambda
3. H-step closed-form predictor:
     T_hat[k+H] = a^H * T[k] + b * P * (1 - a^H) / (1 - a) + (1 - a^H) * T_amb
4. Two-RC (die + chassis) state-space model for higher-order dynamics
5. Persistence baseline: T_hat[k+H] = T[k]
6. Prediction uncertainty estimation: sigma_H = std(residuals at horizon H)
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, List, Optional
import json


class FirstOrderRCModel:
    """
    Discrete first-order thermal model:
      T[k+1] - T_amb = a * (T[k] - T_amb) + b * P[k]
    """

    def __init__(self, dt: float = 1.0):
        self.dt = dt
        self.a: float = 0.98
        self.b: float = 0.05
        self.R: float = 0.0
        self.C: float = 0.0
        self.T_amb: float = 25.0
        self.residuals_std_by_h: Dict[int, float] = {}

    def fit(self, temps: np.ndarray, powers: np.ndarray, T_amb: Optional[float] = None) -> Dict[str, float]:
        """Fit model parameters a and b using Ordinary Least Squares."""
        if T_amb is None:
            # Estimate ambient as min observed temp or 25C
            self.T_amb = float(np.min(temps))
        else:
            self.T_amb = float(T_amb)

        # Regress y[k] = T[k+1] - T_amb on phi[k] = [T[k] - T_amb, P[k]]
        y = temps[1:] - self.T_amb
        phi_T = temps[:-1] - self.T_amb
        phi_P = powers[:-1]

        Phi = np.column_stack([phi_T, phi_P])
        # Solve least squares: (Phi^T Phi)^(-1) Phi^T y
        theta, residuals, rank, s = np.linalg.lstsq(Phi, y, rcond=None)

        self.a = float(np.clip(theta[0], 0.8, 0.9999))
        self.b = float(max(1e-6, theta[1]))

        # Derive continuous R and C
        # b = R * (1 - a)  =>  R = b / (1 - a)
        # a = exp(-dt / (R * C)) => R * C = -dt / ln(a) => C = -dt / (R * ln(a))
        if self.a < 1.0 and (1.0 - self.a) > 1e-6:
            self.R = self.b / (1.0 - self.a)
            self.C = -self.dt / (self.R * np.log(self.a))
        else:
            self.R = 1.0
            self.C = 100.0

        return {
            "a": self.a,
            "b": self.b,
            "R": self.R,
            "C": self.C,
            "T_amb": self.T_amb,
        }

    def predict_h_step(self, T_curr: float, P_future: float, H: int) -> float:
        """
        Closed-form H-step ahead temperature prediction:
          T_hat[k+H] = a^H * T_curr + b * P_future * (1 - a^H) / (1 - a) + (1 - a^H) * T_amb
        """
        a_H = self.a ** H
        if abs(1.0 - self.a) > 1e-7:
            p_term = self.b * P_future * (1.0 - a_H) / (1.0 - self.a)
        else:
            p_term = self.b * P_future * H

        amb_term = (1.0 - a_H) * self.T_amb
        return float(a_H * T_curr + p_term + amb_term)

    def evaluate_horizons(self, temps: np.ndarray, powers: np.ndarray, horizons: List[int] = [10, 30, 60]) -> Dict[str, Dict[str, float]]:
        """Evaluate MAE and RMSE at multiple horizons."""
        results = {}
        N = len(temps)
        for H in horizons:
            if N <= H:
                continue
            preds = []
            actuals = []
            for k in range(N - H):
                P_mean = float(np.mean(powers[k:k+H]))
                pred = self.predict_h_step(temps[k], P_mean, H)
                preds.append(pred)
                actuals.append(temps[k + H])

            preds = np.array(preds)
            actuals = np.array(actuals)
            res = actuals - preds
            mae = float(np.mean(np.abs(res)))
            rmse = float(np.sqrt(np.mean(res**2)))
            std = float(np.std(res))
            self.residuals_std_by_h[H] = std
            results[f"H_{H}s"] = {"MAE": round(mae, 3), "RMSE": round(rmse, 3), "residual_std": round(std, 3)}
        return results


class OnlineRLSThermalModel:
    """
    Recursive Least Squares with forgetting factor lambda for online parameter tracking.
    """

    def __init__(self, init_a: float = 0.98, init_b: float = 0.05, T_amb: float = 25.0,
                 forgetting_factor: float = 0.99, delta: float = 100.0, dt: float = 1.0):
        self.theta = np.array([init_a, init_b], dtype=float)
        self.P_cov = np.eye(2) * delta
        self.lam = forgetting_factor
        self.T_amb = T_amb
        self.dt = dt

    def update(self, T_k: float, P_k: float, T_next: float) -> Tuple[float, float]:
        """Update parameter estimates with new observation."""
        phi = np.array([T_k - self.T_amb, P_k], dtype=float)
        y = T_next - self.T_amb

        # Error
        err = y - float(np.dot(phi, self.theta))

        # Gain vector K = (P * phi) / (lam + phi^T * P * phi)
        P_phi = np.dot(self.P_cov, phi)
        denom = self.lam + float(np.dot(phi, P_phi))
        if abs(denom) > 1e-9:
            K = P_phi / denom
            self.theta = self.theta + K * err
            # Clamp to physical range
            self.theta[0] = np.clip(self.theta[0], 0.85, 0.9999)
            self.theta[1] = max(1e-6, self.theta[1])
            # Covariance update: P = (P - K * phi^T * P) / lam
            self.P_cov = (self.P_cov - np.outer(K, np.dot(phi, self.P_cov))) / self.lam

        return float(self.theta[0]), float(self.theta[1])

    def predict_h_step(self, T_curr: float, P_future: float, H: int) -> float:
        a = float(self.theta[0])
        b = float(self.theta[1])
        a_H = a ** H
        if abs(1.0 - a) > 1e-7:
            p_term = b * P_future * (1.0 - a_H) / (1.0 - a)
        else:
            p_term = b * P_future * H
        return float(a_H * T_curr + p_term + (1.0 - a_H) * self.T_amb)


def persistence_baseline_evaluate(temps: np.ndarray, horizons: List[int] = [10, 30, 60]) -> Dict[str, Dict[str, float]]:
    """Naive persistence baseline: T_hat[k+H] = T[k]."""
    results = {}
    N = len(temps)
    for H in horizons:
        if N <= H:
            continue
        preds = temps[:-H]
        actuals = temps[H:]
        res = actuals - preds
        mae = float(np.mean(np.abs(res)))
        rmse = float(np.sqrt(np.mean(res**2)))
        results[f"H_{H}s"] = {"MAE": round(mae, 3), "RMSE": round(rmse, 3)}
    return results
