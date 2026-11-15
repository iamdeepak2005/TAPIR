"""
TAPIR Workload & Dataset Generator.
Prepares prompt pools for:
- MMLU subset (multiple choice reasoning)
- GSM8K (multi-step grade school math)
- Open-ended general instructions
- Sensitive / PII test prompts (medical, financial, personal, NDA-marked)

Generates Poisson and bursty request traces with timestamps and metadata.
"""

import json
import os
import random
import numpy as np
from typing import List, Dict, Any

MMLU_SAMPLES = [
    {"id": "mmlu_01", "task": "mmlu", "prompt": "Question: Which of the following is true regarding mitochondria?\nA) They have a single membrane\nB) They contain their own circular DNA\nC) They produce glucose\nD) They lack ribosomes\nAnswer:", "ground_truth": "B", "difficulty": 0.3},
    {"id": "mmlu_02", "task": "mmlu", "prompt": "Question: In microeconomics, what happens to consumer surplus when price increases?\nA) It increases\nB) It decreases\nC) It remains constant\nD) It becomes negative\nAnswer:", "ground_truth": "B", "difficulty": 0.2},
    {"id": "mmlu_03", "task": "mmlu", "prompt": "Question: What is the time complexity of searching in a balanced binary search tree with n nodes?\nA) O(1)\nB) O(n)\nC) O(log n)\nD) O(n log n)\nAnswer:", "ground_truth": "C", "difficulty": 0.25},
    {"id": "mmlu_04", "task": "mmlu", "prompt": "Question: In quantum mechanics, the Heisenberg uncertainty principle relates which pair of variables?\nA) Energy and velocity\nB) Position and momentum\nC) Charge and mass\nD) Spin and angular momentum\nAnswer:", "ground_truth": "B", "difficulty": 0.35},
    {"id": "mmlu_05", "task": "mmlu", "prompt": "Question: Which treaty established the European Union in 1993?\nA) Treaty of Versailles\nB) Treaty of Rome\nC) Maastricht Treaty\nD) Lisbon Treaty\nAnswer:", "ground_truth": "C", "difficulty": 0.4},
    {"id": "mmlu_06", "task": "mmlu", "prompt": "Question: In an NP-complete problem, if one problem is solvable in polynomial time, then:\nA) No other NP problem is solvable\nB) All problems in NP are solvable in polynomial time\nC) P is strictly smaller than NP\nD) The problem is uncomputable\nAnswer:", "ground_truth": "B", "difficulty": 0.55},
    {"id": "mmlu_07", "task": "mmlu", "prompt": "Question: What is the determinant of a 2x2 matrix [[a, b], [c, d]]?\nA) ab - cd\nB) ad + bc\nC) ad - bc\nD) ac - bd\nAnswer:", "ground_truth": "C", "difficulty": 0.2},
    {"id": "mmlu_08", "task": "mmlu", "prompt": "Question: In organic chemistry, which functional group characterizes an ester?\nA) -COOH\nB) -COOR\nC) -CHO\nD) -OH\nAnswer:", "ground_truth": "B", "difficulty": 0.35},
]

GSM8K_SAMPLES = [
    {"id": "gsm8k_01", "task": "gsm8k", "prompt": "Janet’s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per egg. How much in dollars does she make every day at the farmers' market?", "ground_truth": "18", "difficulty": 0.45},
    {"id": "gsm8k_02", "task": "gsm8k", "prompt": "A robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts in total does it take to make 6 robes?", "ground_truth": "18", "difficulty": 0.5},
    {"id": "gsm8k_03", "task": "gsm8k", "prompt": "Josh decides to try flipping a house. He buys a house for $80,000 and spends $50,000 on renovations. He pays $10,000 in closing costs. If he sells the house for $180,000, what is his profit in dollars?", "ground_truth": "40000", "difficulty": 0.4},
    {"id": "gsm8k_04", "task": "gsm8k", "prompt": "James creates a media empire. He creates 2 shows a week that are each 30 minutes long. He also creates 1 movie every month that is 2 hours long. How many hours of content does he create in a year that has 52 weeks and 12 months?", "ground_truth": "76", "difficulty": 0.65},
    {"id": "gsm8k_05", "task": "gsm8k", "prompt": "A train travels 60 miles in 45 minutes. At this constant speed, how many miles will it travel in 3 hours?", "ground_truth": "240", "difficulty": 0.55},
    {"id": "gsm8k_06", "task": "gsm8k", "prompt": "If Sarah has 5 baskets with 12 apples each, and she gives away 1/3 of the total apples, how many apples does she have left?", "ground_truth": "40", "difficulty": 0.4},
]

OPEN_ENDED_SAMPLES = [
    {"id": "open_01", "task": "open", "prompt": "Write a concise Python function to calculate the Fibonacci sequence up to n terms using dynamic programming.", "ground_truth": None, "difficulty": 0.3},
    {"id": "open_02", "task": "open", "prompt": "Explain the trade-offs between monolithic architecture and microservices in distributed software engineering.", "ground_truth": None, "difficulty": 0.45},
    {"id": "open_03", "task": "open", "prompt": "Summarize the primary mechanisms of biological evolution: natural selection, genetic drift, gene flow, and mutation.", "ground_truth": None, "difficulty": 0.35},
    {"id": "open_04", "task": "open", "prompt": "Draft an executive summary for a proposal introducing edge AI inference on mobile computing devices.", "ground_truth": None, "difficulty": 0.4},
    {"id": "open_05", "task": "open", "prompt": "Compare and contrast SQL relational databases and NoSQL document stores with respect to ACID guarantees and horizontal scalability.", "ground_truth": None, "difficulty": 0.45},
]

SENSITIVE_SAMPLES = [
    {"id": "sens_01", "task": "sensitive", "prompt": "Please summarize the medical lab results for Patient John Doe, MRN 847291: blood glucose 185 mg/dL, HbA1c 8.2%, diagnosed with Type 2 Diabetes.", "ground_truth": None, "is_sensitive": True, "difficulty": 0.35},
    {"id": "sens_02", "task": "sensitive", "prompt": "Here is the customer payment record: Name: Alice Smith, Card Number: 4532-8921-3849-1029, Expiry: 08/28. Verify transaction amount $450.00.", "ground_truth": None, "is_sensitive": True, "difficulty": 0.3},
    {"id": "sens_03", "task": "sensitive", "prompt": "CONFIDENTIAL: Internal API credentials for AWS production deployment. Access Key ID: AKIAIOSFODNN7EXAMPLE, Secret Access Key: wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY. Do not disclose.", "ground_truth": None, "is_sensitive": True, "difficulty": 0.4},
    {"id": "sens_04", "task": "sensitive", "prompt": "Review employee tax filing for Robert Chen, SSN: 042-89-1124, current annual salary $142,000, claiming 2 withholding allowances.", "ground_truth": None, "is_sensitive": True, "difficulty": 0.35},
    {"id": "sens_05", "task": "sensitive", "prompt": "Strictly Private & Proprietary: Project Orion merger term sheet with Quantum Inc. Target valuation $45M, closing date December 15.", "ground_truth": None, "is_sensitive": True, "difficulty": 0.45},
]


def build_workload_trace(
    n_requests: int = 100,
    arrival_pattern: str = "poisson",
    poisson_rate: float = 0.5,
    sensitive_fraction: float = 0.15,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """Generate a realistic benchmark workload trace."""
    rng = random.Random(seed)
    np_rng = np.random.RandomState(seed)

    trace = []
    current_time = 0.0

    all_standard = MMLU_SAMPLES + GSM8K_SAMPLES + OPEN_ENDED_SAMPLES

    for i in range(n_requests):
        # Calculate inter-arrival time
        if arrival_pattern == "poisson":
            dt = np_rng.exponential(1.0 / poisson_rate)
        elif arrival_pattern == "bursty":
            # Bursty: small dt within burst, large dt between bursts
            if (i % 10) == 0 and i > 0:
                dt = np_rng.uniform(15.0, 30.0)  # Pause between bursts
            else:
                dt = np_rng.uniform(0.1, 0.8)   # Rapid burst
        else:
            dt = 1.0 / poisson_rate

        current_time += dt

        # Select prompt
        is_sens = rng.random() < sensitive_fraction
        if is_sens:
            sample = dict(rng.choice(SENSITIVE_SAMPLES))
            sample["is_sensitive"] = True
        else:
            sample = dict(rng.choice(all_standard))
            sample["is_sensitive"] = False

        sample["request_id"] = f"req_{i:04d}"
        sample["arrival_time_s"] = round(current_time, 3)
        sample["prompt_length"] = len(sample["prompt"])
        trace.append(sample)

    return trace


if __name__ == "__main__":
    trace = build_workload_trace(50, "poisson", 0.5, 0.15, 42)
    print(f"Generated sample trace with {len(trace)} requests.")
    print(f"First request: {trace[0]}")
    sens_count = sum(1 for r in trace if r["is_sensitive"])
    print(f"Sensitive requests: {sens_count} / {len(trace)} ({sens_count/len(trace)*100:.1f}%)")
