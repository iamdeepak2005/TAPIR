"""
TAPIR Constrained LinUCB Bandit Router.

Implements:
1. Contextual Bandit with LinUCB:
     a_t = argmax_{a in F_t} [ x^T theta_a + alpha * sqrt(x^T A_a^{-1} x) ]
2. Feasibility Filter F_t:
     - Hard privacy constraint: If sensitive == 1, 'cloud' is prohibited.
     - Thermal chance constraint: If T_hat[k+H] + z * sigma_H > T_max, action is infeasible.
3. Multi-objective Reward:
     r = w_q * quality - w_c * cloud_cost - w_e * energy - w_l * latency - zeta * max(0, T_hat - T_soft)^2
4. Warm-start theta from difficulty predictor.
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple

ACTION_LOCAL_Q8 = "local_q8"
ACTION_LOCAL_Q5 = "local_q5"
ACTION_LOCAL_Q4 = "local_q4"
ACTION_LOCAL_EARLY = "local_early_exit"
ACTION_CLOUD = "cloud"

ACTIONS = [
    ACTION_LOCAL_Q8,
    ACTION_LOCAL_Q5,
    ACTION_LOCAL_Q4,
    ACTION_LOCAL_EARLY,
    ACTION_CLOUD,
]


class ConstrainedLinUCBRouter:
    def __init__(
        self,
        d: int = 10,
        alpha: float = 1.0,
        weights: Optional[Dict[str, float]] = None,
        T_max: float = 90.0,
        T_soft: float = 80.0,
        horizon_H: int = 60,
        z_sigma: float = 1.96,
        thermal_model=None,
        random_seed: int = 42,
    ):
        self.d = d
        self.alpha = alpha
        self.rng = np.random.RandomState(random_seed)

        self.weights = weights or {
            "w_q": 1.0,
            "w_c": 0.3,
            "w_e": 0.1,
            "w_l": 0.2,
            "zeta": 0.5,
        }

        self.T_max = T_max
        self.T_soft = T_soft
        self.horizon_H = horizon_H
        self.z_sigma = z_sigma
        self.thermal_model = thermal_model

        # LinUCB state per action
        self.A: Dict[str, np.ndarray] = {}
        self.A_inv: Dict[str, np.ndarray] = {}
        self.b: Dict[str, np.ndarray] = {}
        self.theta: Dict[str, np.ndarray] = {}

        for a in ACTIONS:
            self.A[a] = np.eye(self.d)
            self.A_inv[a] = np.eye(self.d)
            self.b[a] = np.zeros(self.d)
            self.theta[a] = np.zeros(self.d)

        # Baseline expected power consumption per action in Watts (measured/calibrated)
        self.action_power_w = {
            ACTION_LOCAL_Q8: 25.0,
            ACTION_LOCAL_Q5: 20.0,
            ACTION_LOCAL_Q4: 16.0,
            ACTION_LOCAL_EARLY: 12.0,
            ACTION_CLOUD: 2.0,  # Idle network transfer only
        }

        self.cumulative_regret = 0.0
        self.history = []

    def warm_start_theta(self, prior_quality: Dict[str, float]):
        """Warm-start theta parameter vector using empirical baseline qualities."""
        for a in ACTIONS:
            if a in prior_quality:
                # Set bias weight to match expected baseline reward
                self.theta[a][0] = prior_quality[a]
                self.b[a] = np.dot(self.A[a], self.theta[a])

    def build_context(self, state: Dict[str, Any]) -> np.ndarray:
        """
        Build 10-dimensional feature context:
        [diff_score, prompt_len/1000, privacy_flag, temp/100, dT_dt, headroom/50, batt/100, free_ram/8, rtt/1.0, 1.0]
        """
        x = np.array([
            float(state.get("difficulty_score", 0.5)),
            float(state.get("prompt_length", 100)) / 1000.0,
            1.0 if state.get("is_sensitive", False) else 0.0,
            float(state.get("current_temp", 45.0)) / 100.0,
            float(state.get("dT_dt", 0.0)),
            float(max(0, self.T_max - state.get("current_temp", 45.0))) / 50.0,
            float(state.get("battery_pct", 50.0)) / 100.0,
            float(state.get("free_ram_gb", 1.0)) / 8.0,
            float(state.get("network_rtt_s", 0.05)),
            1.0,  # Bias term
        ], dtype=float)
        return x

    def filter_feasible_actions(self, context: np.ndarray, state: Dict[str, Any]) -> List[str]:
        """Apply hard constraints to construct feasible action set F_t."""
        feasible = list(ACTIONS)
        is_sensitive = bool(state.get("is_sensitive", False))

        # Hard Rule 1: Privacy enforcement
        if is_sensitive and ACTION_CLOUD in feasible:
            feasible.remove(ACTION_CLOUD)

        # Hard Rule 2: Thermal chance constraint
        # Remove any action whose predicted temperature + z * sigma_H > T_max
        if self.thermal_model is not None:
            curr_temp = float(state.get("current_temp", 45.0))
            sigma_H = getattr(self.thermal_model, "residuals_std_by_h", {}).get(self.horizon_H, 2.0)

            filtered = []
            for a in feasible:
                p_est = self.action_power_w.get(a, 15.0)
                t_pred = self.thermal_model.predict_h_step(curr_temp, p_est, self.horizon_H)
                upper_bound = t_pred + self.z_sigma * sigma_H
                if upper_bound <= self.T_max:
                    filtered.append(a)

            if len(filtered) > 0:
                feasible = filtered
            else:
                # Emergency fallback: pick least power action
                if is_sensitive:
                    feasible = [ACTION_LOCAL_EARLY]
                else:
                    feasible = [ACTION_CLOUD]

        return feasible

    def select_action(self, state: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        """Select action via LinUCB over feasible set F_t."""
        x = self.build_context(state)
        feasible = self.filter_feasible_actions(x, state)

        best_action = feasible[0]
        max_ucb = -float("inf")
        ucb_scores = {}

        for a in feasible:
            theta_a = self.theta[a]
            A_inv_a = self.A_inv[a]

            expected_reward = float(np.dot(x, theta_a))
            uncertainty = float(np.sqrt(np.dot(x, np.dot(A_inv_a, x))))
            ucb = expected_reward + self.alpha * uncertainty

            ucb_scores[a] = {"expected": round(expected_reward, 4), "ucb": round(ucb, 4)}
            if ucb > max_ucb:
                max_ucb = ucb
                best_action = a

        decision_info = {
            "selected_action": best_action,
            "feasible_actions": feasible,
            "ucb_scores": ucb_scores,
            "context": [round(val, 4) for val in x.tolist()],
        }
        return best_action, decision_info

    def compute_reward(
        self,
        quality: float,
        cloud_cost: float,
        energy_j: float,
        latency_s: float,
        t_predicted: float,
    ) -> float:
        """Calculate scalar multi-objective reward."""
        w = self.weights
        thermal_penalty = w["zeta"] * (max(0.0, t_predicted - self.T_soft) ** 2)
        r = (
            w["w_q"] * quality
            - w["w_c"] * cloud_cost
            - w["w_e"] * (energy_j / 100.0)  # Scale Joules
            - w["w_l"] * latency_s
            - thermal_penalty
        )
        return float(r)

    def update(self, action: str, context: np.ndarray, reward: float):
        """Update LinUCB covariance and mean for selected action."""
        x = context.reshape(-1, 1)
        self.A[action] += np.dot(x, x.T)
        self.b[action] += float(reward) * context
        self.A_inv[action] = np.linalg.pinv(self.A[action])
        self.theta[action] = np.dot(self.A_inv[action], self.b[action])
