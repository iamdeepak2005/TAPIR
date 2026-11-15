# TAPIR: Constraint-Aware Adaptive Hybrid LLM Routing with Early-Exit Local Inference

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.12](https://img.shields.io/badge/Python-3.12-green.svg)](https://python.org)
[![llama.cpp](https://img.shields.io/badge/llama.cpp-b12399-orange.svg)](https://github.com/ggml-org/llama.cpp)

Official open-source implementation and evaluation suite for **TAPIR** (*Thermal-aware, Adaptive, Privacy-constrained Inference Routing*).

---

## Overview

TAPIR formulates hybrid LLM routing as a constrained contextual bandit problem. Rather than treating local inference as a monolithic fallback, TAPIR dynamically routes user requests across:
- **Quantized Local Variants**: Q8_0, Q5_K_M, Q4_K_M (served via `llama.cpp`)
- **Confidence-Driven Early Exit**: Layer-pruned inference (PyTorch runtime)
- **High-Capacity Cloud LLM**: Google Gemini / OpenAI APIs

Routing decisions are governed by three primary constraint mechanisms:
1. **Hard Privacy Enforcement**: Requests matching PII / sensitive criteria are prohibited from cloud offloading.
2. **Thermal Chance Constraints**: Discrete first-order RC thermal prediction ($H$-step horizon) with uncertainty margins prevents hardware thermal throttling.
3. **Constrained LinUCB Bandit**: Exploits dynamic device contexts (difficulty, prompt length, temperature, thermal headroom, battery, memory, and network RTT) with warm-start initialization.

---

## Project Structure

```text
.
├── LICENSE                           # MIT License
├── README.md                         # Reproduction guide
├── requirements.txt                  # Python dependencies
├── .gitignore                        # Git exclusion rules (ignores secrets & large models)
└── tapir/
    ├── .env.template                 # Cloud API credentials template
    ├── compute_prop1_exceedance.py   # Proposition 1 empirical verification script
    ├── detect_hardware.py            # Hardware discovery script
    ├── download_deps.py              # Model & binary download automation
    ├── config/
    │   ├── devices.yaml              # System hardware and throttle definitions
    │   ├── prices.yaml               # API token rates for cloud models
    │   └── experiment.yaml           # Hyperparameters, random seeds, bandit weights
    ├── data/
    │   ├── raw/                      # Measured telemetry logs and raw JSON benchmark traces
    │   └── processed/                # Response caches and preprocessed datasets
    ├── src/
    │   ├── logger/                   # 1 Hz Telemetry logger (cross-platform system monitor)
    │   ├── thermal/                  # RC model identification, RLS parameter tracking
    │   ├── privacy/                  # Hybrid regex, NER, and sensitivity classifier
    │   ├── earlyexit/                # Logit-lens early-exit transformer prototype
    │   ├── router/                   # Constrained LinUCB, difficulty predictor, cloud client
    │   ├── workload/                 # Poisson and bursty trace generation
    │   └── eval/                     # Benchmark harnesses, plotting routines, report generators
    └── results/
        ├── figures/                  # 300 DPI publication plots (Pareto curves, thermal traces)
        ├── tables/                   # Benchmark outputs in CSV and LaTeX (.tex) format
        └── report.md                 # Comprehensive experimental results and limitations
```

---

## Installation & Prerequisites

### 1. Python Environment
```bash
python -m venv venv
# Windows:
.\venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

pip install -r requirements.txt
```

Core dependencies: `torch`, `transformers`, `psutil`, `numpy`, `pandas`, `scipy`, `scikit-learn`, `matplotlib`, `pyyaml`, `requests`, `spacy`.

### 2. llama.cpp Binary & Model Quantizations
Download the official CPU release of `llama.cpp` and GGUF quantizations of `Qwen2.5-1.5B-Instruct`:
```bash
python tapir/download_deps.py
```
This downloads and verifies SHA-256 checksums for:
- `qwen2.5-1.5b-instruct-q8_0.gguf`
- `qwen2.5-1.5b-instruct-q5_k_m.gguf`
- `qwen2.5-1.5b-instruct-q4_k_m.gguf`

### 3. Cloud API Configuration
Copy `.env.template` to `.env` and configure your API key:
```bash
cp tapir/.env.template tapir/.env
```
*(Note: `.env` is git-ignored and never committed).*

---

## Reproduction Commands

### Step 1: System Telemetry Verification (Phase 1)
Verify 1 Hz telemetry logging regularity under idle conditions:
```bash
python tapir/src/logger/telemetry.py --duration 5
```

### Step 2: Quantization Benchmarking (Phase 2)
Benchmark decode throughput, prefill rates, TTFT, and energy consumption:
```bash
python tapir/src/eval/benchmark.py
```

### Step 3: Thermal Identification & Multi-Horizon Validation (Phase 3)
Fit the lumped thermal RC model and sweep RLS forgetting factors:
```bash
python tapir/src/thermal/fit_thermal_model.py
```

### Step 4: Privacy Detector Evaluation (Phase 4)
Evaluate deterministic regex rules, entity detection, and sensitivity classification:
```bash
python tapir/src/privacy/eval_privacy.py
```

### Step 5: Model Quality Profiling (Addendum A2)
Evaluate task accuracy on MMLU, GSM8K, and held-out perplexity across quantizations:
```bash
python tapir/src/eval/quality_profile.py
```

### Step 6: Early-Exit Calibration (Addendum A5)
Calibrate confidence exit thresholds and compute token agreement and ECE:
```bash
python tapir/src/earlyexit/eval_earlyexit.py
```

### Step 7: End-to-End Routing Comparison & Ablations (Phases 6 & 7)
Execute full multi-seed evaluation across all 10 routing methods, 7 ablations, and robustness conditions:
```bash
python tapir/src/router/difficulty.py
python tapir/src/eval/run_full_evaluation.py
python tapir/src/eval/generate_plots.py
```

---

## Key Experimental Results

| Metric | Always-Cloud | Always-Local (Q8) | Reactive Thermal | TAPIR (Ours) |
|---|---|---|---|---|
| **Quality Retention** | 100.0% | 88.41% | 79.49% | **96.31%** |
| **Offload Ratio** | 82.0% | 0.0% | 0.0% | **69.5%** |
| **Energy / Request** | 10.88 J | 56.90 J | 42.93 J | **12.02 J** |
| **Cost / Request** | $12.0 \times 10^{-6}$ | $0.00 | $0.00 | **$10.0 \times 10^{-6}$** |
| **Throttle Events/hr** | 0.0 | 0.0 | 0.0 | **0.0** |
| **Privacy Leakage** | 0.0% | 0.0% | 0.0% | **0.0%** |
| **Router Latency** | 0.00 ms | 0.00 ms | 0.45 ms | **0.85 ms** |

Detailed ablation studies, latency distributions, and statistical tests are reported in [`tapir/results/report.md`](tapir/results/report.md).

---

## Citation

```bibtex
@inproceedings{deepak2026tapir,
  title={{TAPIR: Constraint-Aware Adaptive Hybrid LLM Routing with Early-Exit Local Inference}},
  author={Deepak, B. O. and Desai, Gannamaraju Rohitha},
  booktitle={IEEE Conference Proceedings},
  year={2026}
}
```

---

## License

This project is licensed under the [MIT License](LICENSE).
