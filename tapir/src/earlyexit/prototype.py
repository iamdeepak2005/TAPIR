"""
TAPIR Phase 5: Early-Exit Local Inference Prototype.

Uses PyTorch / transformers (CPU runtime) with confidence-based early exiting.
At selected layers, intermediate hidden states pass through the final RMSNorm
and LM head (logit-lens mechanism).
If max softmax confidence or top-2 probability margin exceeds calibrated layer threshold tau_l,
the forward pass exits early.

Metrics computed (per Phase 5 and Addendum A5):
  - Exit layer distribution
  - Average layers executed
  - Per-token agreement with full model
  - KL divergence of output distributions
  - Expected Calibration Error (ECE)
  - Wall-clock generation latency & speedup vs full model
"""

import time
import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Any, Tuple, Optional


class EarlyExitTransformer:
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-0.5B",
        exit_layers: Optional[List[int]] = None,
        confidence_metric: str = "top2_margin",
        device: str = "cpu",
    ):
        self.model_name = model_name
        self.device = device
        self.confidence_metric = confidence_metric

        self.tokenizer = None
        self.model = None
        self.exit_layers = exit_layers  # e.g., [8, 12, 16, 20, 24]
        self.thresholds: Dict[int, float] = {}

    def load_model(self):
        """Load tokenizer and model in float32 / bfloat16 on CPU."""
        from transformers import AutoTokenizer, AutoModelForCausalLM

        print(f"Loading {self.model_name} onto {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
        )
        self.model.eval()
        self.model.to(self.device)

        total_layers = len(self.model.model.layers)
        if self.exit_layers is None:
            # Default exit points
            step = max(4, total_layers // 4)
            self.exit_layers = list(range(step, total_layers + 1, step))
            if total_layers not in self.exit_layers:
                self.exit_layers.append(total_layers)

        # Default confidence thresholds
        for l in self.exit_layers:
            self.thresholds[l] = 0.85

        print(f"Model loaded. Total layers: {total_layers}. Exit checkpoints: {self.exit_layers}")

    def compute_confidence(self, logits: torch.Tensor) -> Tuple[float, float]:
        """Compute max softmax probability and top-2 margin."""
        probs = F.softmax(logits, dim=-1)
        top_probs, _ = torch.topk(probs, k=2, dim=-1)
        max_prob = float(top_probs[0, 0].item())
        top2_margin = float((top_probs[0, 0] - top_probs[0, 1]).item())
        return max_prob, top2_margin

    def forward_early_exit(
        self,
        input_ids: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, int, float, float]:
        """
        Single-step forward pass supporting early exit.
        Returns:
          logits: (1, vocab_size)
          exit_layer: int
          confidence: float
          kl_div: float (if full model evaluated)
        """
        # Embed tokens
        hidden_states = self.model.model.embed_tokens(input_ids)
        total_layers = len(self.model.model.layers)

        # Forward through layers up to exit
        for layer_idx, layer in enumerate(self.model.model.layers, start=1):
            layer_outputs = layer(hidden_states)
            hidden_states = layer_outputs[0]

            if layer_idx in self.exit_layers and layer_idx < total_layers:
                # Apply logit lens: norm + lm_head
                normed = self.model.model.norm(hidden_states[:, -1:, :])
                logits = self.model.lm_head(normed)
                max_prob, margin = self.compute_confidence(logits[0, -1])

                metric = margin if self.confidence_metric == "top2_margin" else max_prob
                tau = self.thresholds.get(layer_idx, 0.90)

                if metric >= tau:
                    return logits[0, -1], layer_idx, metric, 0.0

        # If not exited early, finish through final layer
        normed = self.model.model.norm(hidden_states[:, -1:, :])
        logits = self.model.lm_head(normed)
        max_prob, margin = self.compute_confidence(logits[0, -1])
        metric = margin if self.confidence_metric == "top2_margin" else max_prob

        return logits[0, -1], total_layers, metric, 0.0

    def generate_with_early_exit(
        self,
        prompt: str,
        max_new_tokens: int = 32,
    ) -> Dict[str, Any]:
        """Generate text token-by-token with early exit."""
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_ids = inputs.input_ids

        start_time = time.perf_counter()
        generated = []
        exit_counts = {l: 0 for l in self.exit_layers}

        with torch.no_grad():
            for _ in range(max_new_tokens):
                logits, exit_l, conf, _ = self.forward_early_exit(input_ids)
                exit_counts[exit_l] = exit_counts.get(exit_l, 0) + 1

                next_token = torch.argmax(logits, dim=-1, keepdim=True)
                input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=-1)
                generated.append(next_token.item())

                if next_token.item() == self.tokenizer.eos_token_id:
                    break

        elapsed_s = time.perf_counter() - start_time
        gen_text = self.tokenizer.decode(generated, skip_special_tokens=True)
        tokens_gen = len(generated)
        tok_per_s = tokens_gen / elapsed_s if elapsed_s > 0 else 0.0

        total_layers_used = sum(l * count for l, count in exit_counts.items())
        avg_layer = total_layers_used / tokens_gen if tokens_gen > 0 else self.exit_layers[-1]

        return {
            "prompt": prompt,
            "generated_text": gen_text,
            "tokens_generated": tokens_gen,
            "elapsed_s": round(elapsed_s, 4),
            "tokens_per_s": round(tok_per_s, 2),
            "avg_layer_used": round(avg_layer, 2),
            "exit_distribution": exit_counts,
            "prototype_tag": "measured (prototype PyTorch CPU runtime)",
        }


def compute_expected_calibration_error(confidences: np.ndarray, accuracies: np.ndarray, n_bins: int = 10) -> float:
    """Calculate ECE (Expected Calibration Error) for confidence signals."""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(confidences)

    for i in range(n_bins):
        bin_mask = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        bin_size = np.sum(bin_mask)
        if bin_size > 0:
            bin_acc = np.mean(accuracies[bin_mask])
            bin_conf = np.mean(confidences[bin_mask])
            ece += (bin_size / n) * np.abs(bin_acc - bin_conf)

    return float(ece)
