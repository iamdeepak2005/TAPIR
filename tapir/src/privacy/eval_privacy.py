"""
TAPIR Phase 4: Privacy Detector Training & Evaluation.

Combines:
  1. Regex patterns for deterministic PII
  2. spaCy NER for person/org/date/monetary entities
  3. Scikit-learn TF-IDF + Logistic Regression sensitivity classifier

Evaluates:
  - Precision, Recall, F1 score
  - False Negative Rate (FNR) -> leakage risk
  - False Positive Rate (Over-blocking rate, Addendum A9)
  - Latency per request (mean, P50, P95, P99 in ms)

Outputs:
  - Raw evaluation: data/raw/privacy_eval_<timestamp>.json
  - Results table: results/tables/privacy_metrics.csv
"""

import os
import sys
import json
import time
import datetime
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix

BASE = r"c:\Users\deepa\Research-TAPIR\tapir"
RAW_DIR = os.path.join(BASE, "data", "raw")
RESULTS_DIR = os.path.join(BASE, "results", "tables")
os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# Add src to sys.path
sys.path.append(os.path.join(BASE, "src"))
from privacy.detector import PrivacyDetector


def generate_benchmark_dataset(seed: int = 42):
    """
    Generate balanced labeled dataset (sensitive=1, non_sensitive=0)
    mixing PII categories and diverse open-ended instructions.
    """
    rng = np.random.RandomState(seed)

    # 1. Diverse Sensitive Samples (PII, Financial, Medical, Confidential)
    sensitive_samples = [
        "Patient Alice Walker, MRN 481920: diagnosis Acute Appendicitis. Prescribed Cefazolin 1g IV.",
        "Tax return documentation for taxpayer SSN: 219-45-8821, gross reported income $89,500.",
        "Credit card authorization: card number 4128-9821-3412-8890, exp 05/27, CVV 391.",
        "CONFIDENTIAL: Internal engineering architecture for upcoming product launch code-named Aurora. Do not disclose.",
        "Customer email contact: support@secureclient-holdings.co.uk, phone +1-555-839-2019.",
        "Database connection URI: postgresql://admin:SuperSecretPass992@db.internal.corp:5432/finance",
        "Employee payroll review: Robert Vance, compensation package $165,000 + 20% bonus.",
        "Patient medical history: Johnathan Doe, DOB 1978-04-12, diagnosis Stage II Hypertension.",
        "NDA agreement between Horizon Robotics and Vertex AI: strictly confidential valuation terms.",
        "API Authorization token: Bearer sk-ant-live98438291048120391283019284901284",
        "Home mortgage application: applicant Michael Brown, social security number 092-11-4829.",
        "Corporate acquisition term sheet: buyout price $120,000,000 for equity stake in Alpha Bio.",
        "Billing address: Sarah Connor, 742 Evergreen Terrace, Springfield, OR 97477, USA.",
        "Patient blood test record: fasting glucose 142 mg/dL, cholesterol 240 mg/dL, Dr. Evans presiding.",
        "AWS root access keys: AKIAIOSFODNN7EXAMP12, secret key secret_key_value_here_corporate_root",
        "Employee disciplinary hearing record: strictly private HR file regarding conduct on Nov 2.",
        "Wire transfer confirmation: routing number 021000021, account 9481029482, beneficiary Jane Doe.",
        "Passport verification scan: United States Passport #982103918 issued to David Miller.",
        "Internal vulnerability report: zero-day exploit in authentication handler, do not leak externally.",
        "Medical insurance claim #839102-A: reimbursement for surgical procedure billed at $12,450.",
    ]

    # 2. Diverse Non-Sensitive Samples (Math, Code, Science, General Instructions)
    non_sensitive_samples = [
        "Write a Python function to compute the greatest common divisor using Euclid's algorithm.",
        "Explain the difference between supervised, unsupervised, and reinforcement learning.",
        "What is the capital of Australia and what are its primary administrative territories?",
        "Solve the quadratic equation 2x^2 + 5x - 3 = 0 using the quadratic formula.",
        "Describe the process of photosynthesis and the role of chlorophyll in light absorption.",
        "Write an essay outlining the economic causes of the 1929 stock market crash.",
        "How do transformers handle positional information through rotary position embeddings (RoPE)?",
        "What are Kepler's three laws of planetary motion and how did Newton generalize them?",
        "Implement a merge sort algorithm in C++ with O(n log n) time complexity.",
        "Explain the thermodynamic concept of entropy and the second law of thermodynamics.",
        "Translate the following phrase into conversational French: 'Where is the train station?'",
        "What is the function of the human hippocampus in memory consolidation and spatial navigation?",
        "Compare synchronous and asynchronous input/output handling in high-throughput web servers.",
        "How does a hash map resolve collisions using open addressing versus separate chaining?",
        "Explain the significance of the Rosetta Stone in deciphering ancient Egyptian hieroglyphics.",
        "What are the main differences between TCP and UDP transport protocols in computer networks?",
        "Describe the life cycle of a massive star from main sequence to core-collapse supernova.",
        "Write a SQL query to calculate the rolling 7-day average of daily active users.",
        "How does gradient descent with momentum help navigate ill-conditioned loss surfaces?",
        "What is the difference between inductive reasoning and deductive reasoning in formal logic?",
    ]

    # Expand dataset with realistic variations
    texts = []
    labels = []

    for s in sensitive_samples:
        texts.append(s)
        labels.append(1)
        # Variation
        texts.append(f"Please review this: {s}")
        labels.append(1)

    for ns in non_sensitive_samples:
        texts.append(ns)
        labels.append(0)
        # Variation
        texts.append(f"Can you explain this topic: {ns}")
        labels.append(0)

    # Shuffle
    indices = np.arange(len(texts))
    rng.shuffle(indices)

    texts = [texts[i] for i in indices]
    labels = [labels[i] for i in indices]

    # Train/Test Split (50% train, 50% test)
    split_idx = len(texts) // 2
    return texts[:split_idx], labels[:split_idx], texts[split_idx:], labels[split_idx:]


def run_privacy_evaluation():
    date_str = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d")
    timestamp_str = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d_%H%M%S")

    train_texts, train_labels, test_texts, test_labels = generate_benchmark_dataset(seed=42)
    print(f"Dataset split: {len(train_texts)} train, {len(test_texts)} test")

    detector = PrivacyDetector(use_spacy=True, use_classifier=True)
    # Fit classifier on training set
    print("Training TF-IDF sensitivity classifier on training split...")
    detector.fit_classifier(train_texts, train_labels)

    # Evaluate on test set
    preds = []
    latencies = []
    details = []

    for text, label in zip(test_texts, test_labels):
        res = detector.detect(text)
        pred_sensitive = int(res["sensitive"])
        preds.append(pred_sensitive)
        latencies.append(res["latency_ms"])
        details.append({
            "text": text,
            "true_label": label,
            "pred_label": pred_sensitive,
            "latency_ms": res["latency_ms"],
            "regex_entities": res["regex_entities"],
            "ner_entities": res["ner_entities"],
            "classifier_score": res["classifier_score"],
        })

    preds = np.array(preds)
    test_labels = np.array(test_labels)

    # Compute metrics
    precision, recall, f1, _ = precision_recall_fscore_support(test_labels, preds, average="binary", pos_label=1)
    tn, fp, fn, tp = confusion_matrix(test_labels, preds, labels=[0, 1]).ravel()

    # False Negative Rate (Leakage risk): FN / (FN + TP)
    fnr = float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0

    # False Positive Rate (Over-blocking rate per Addendum A9): FP / (FP + TN)
    # Fraction of non-sensitive requests wrongly kept local
    overblocking_rate = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0

    lat_mean = float(np.mean(latencies))
    lat_p50 = float(np.percentile(latencies, 50))
    lat_p95 = float(np.percentile(latencies, 95))
    lat_p99 = float(np.percentile(latencies, 99))

    print("\n" + "=" * 60)
    print("Phase 4: Privacy Detector Evaluation Results")
    print("=" * 60)
    print(f"Accuracy:              {(tp + tn) / len(test_labels) * 100:.2f}%")
    print(f"Precision:             {precision * 100:.2f}%")
    print(f"Recall:                {recall * 100:.2f}%")
    print(f"F1-Score:              {f1:.4f}")
    print(f"False Negative Rate:   {fnr * 100:.2f}% (Leakage Rate)")
    print(f"Over-blocking Rate:    {overblocking_rate * 100:.2f}% (Addendum A9)")
    print(f"Mean Latency:          {lat_mean:.2f} ms")
    print(f"P95 Latency:           {lat_p95:.2f} ms")
    print("=" * 60)

    # Save raw log
    raw_file = os.path.join(RAW_DIR, f"privacy_eval_{timestamp_str}.json")
    with open(raw_file, "w") as f:
        json.dump({
            "metadata": {
                "tag": "measured",
                "device": "AMD Ryzen 5 (Zen 4 Mobile, 6 cores)",
                "date": date_str,
                "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "test_samples": len(test_labels),
            },
            "metrics": {
                "precision": round(float(precision), 4),
                "recall": round(float(recall), 4),
                "f1": round(float(f1), 4),
                "false_negative_rate": round(float(fnr), 4),
                "overblocking_rate": round(float(overblocking_rate), 4),
                "latency_mean_ms": round(lat_mean, 2),
                "latency_p50_ms": round(lat_p50, 2),
                "latency_p95_ms": round(lat_p95, 2),
                "latency_p99_ms": round(lat_p99, 2),
                "true_positives": int(tp),
                "false_positives": int(fp),
                "true_negatives": int(tn),
                "false_negatives": int(fn),
            },
            "detailed_predictions": details,
        }, f, indent=2)
    print(f"Raw log saved: {raw_file}")

    # Save summary table
    summary_df = pd.DataFrame([{
        "detector": "TAPIR Hybrid Privacy (Regex+NER+LogReg)",
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1_score": round(float(f1), 4),
        "fnr_leakage_rate": round(float(fnr), 4),
        "overblocking_rate_a9": round(float(overblocking_rate), 4),
        "latency_mean_ms": round(lat_mean, 2),
        "latency_p95_ms": round(lat_p95, 2),
        "tag": "measured",
        "device": "AMD Ryzen 5 (Zen 4 Mobile)",
        "date": date_str,
    }])
    csv_file = os.path.join(RESULTS_DIR, "privacy_metrics.csv")
    summary_df.to_csv(csv_file, index=False)
    print(f"Summary table saved: {csv_file}")


if __name__ == "__main__":
    run_privacy_evaluation()
