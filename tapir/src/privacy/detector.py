"""
TAPIR Privacy Detector.
Combines:
1. Fast regex patterns for high-risk PII (SSN, credit cards, emails, phone numbers, API keys, passwords)
2. spaCy NER for named entities (PERSON, ORG, GPE, DATE, MONEY)
3. Scikit-learn sensitivity classifier (TF-IDF + LogisticRegression) trained on sensitive vs non-sensitive prompts.

Outputs:
  - sensitive: bool (enforced as hard constraint in router)
  - confidence: float
  - detected_entities: list of matched rules/entities
  - latency_ms: float
"""

import re
import time
from typing import Dict, Any, List, Tuple
import numpy as np

# Fast Regex rules for deterministic PII detection
PII_REGEX_RULES = {
    "email": re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),
    "ssn": re.compile(r'\b\d{3}-\d{2}-\d{4}\b|\b\d{9}\b'),
    "credit_card": re.compile(r'\b(?:\d{4}[-\s]?){3}\d{4}\b'),
    "phone": re.compile(r'\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b'),
    "api_key": re.compile(r'(?i)\b(?:api[_-]?key|secret|token|bearer|auth|password)\s*[:=]\s*["\']?[A-Za-z0-9_\-\.]{12,}'),
    "ip_address": re.compile(r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b'),
    "medical_record": re.compile(r'(?i)\b(?:patient\s*(?:id|number)|mrn|diagnosis|prescription|medical\s*record)\s*[:#]?\s*\w+'),
    "confidential_marking": re.compile(r'(?i)\b(?:confidential|strictly\s*private|internal\s*only|proprietary|do\s*not\s*share|nda)\b'),
}


class PrivacyDetector:
    def __init__(self, use_spacy: bool = True, use_classifier: bool = True):
        self.use_spacy = use_spacy
        self.use_classifier = use_classifier
        self.nlp = None
        self.vectorizer = None
        self.classifier = None
        self._init_components()

    def _init_components(self):
        if self.use_spacy:
            try:
                import spacy
                try:
                    self.nlp = spacy.load("en_core_web_sm")
                except Exception:
                    # If model not downloaded, try to create blank or disable ner gracefully
                    try:
                        self.nlp = spacy.blank("en")
                    except Exception:
                        self.nlp = None
            except ImportError:
                self.nlp = None

    def fit_classifier(self, texts: List[str], labels: List[int]):
        """Train a lightweight TF-IDF + Logistic Regression sensitivity classifier."""
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression

        self.vectorizer = TfidfVectorizer(max_features=2000, ngram_range=(1, 2), stop_words='english')
        X = self.vectorizer.fit_transform(texts)
        self.classifier = LogisticRegression(class_weight='balanced', C=1.0, random_state=42)
        self.classifier.fit(X, labels)

    def detect(self, text: str) -> Dict[str, Any]:
        """
        Evaluate sensitivity of input text.
        Returns detailed telemetry including latency in ms.
        """
        start_time = time.perf_counter()
        entities = []
        is_sensitive = False

        # 1. Regex check (microsecond latency)
        for rule_name, pattern in PII_REGEX_RULES.items():
            matches = pattern.findall(text)
            if matches:
                entities.append(f"regex:{rule_name}({len(matches)})")
                is_sensitive = True

        # 2. spaCy NER check if not already triggered by critical regex
        ner_entities = []
        if self.nlp is not None:
            doc = self.nlp(text)
            for ent in doc.ents:
                if ent.label_ in ["PERSON", "ORG", "GPE", "DATE", "MONEY"]:
                    ner_entities.append(f"{ent.label_}:{ent.text}")
                    # Certain high-sensitivity entities trigger sensitive flag
                    if ent.label_ in ["PERSON"] and len(ent.text.split()) >= 2:
                        # Full names considered sensitive
                        is_sensitive = True

        # 3. Sensitivity classifier
        clf_prob = 0.0
        if self.classifier is not None and self.vectorizer is not None:
            vec = self.vectorizer.transform([text])
            probs = self.classifier.predict_proba(vec)[0]
            clf_prob = float(probs[1]) if len(probs) > 1 else 0.0
            if clf_prob > 0.5:
                is_sensitive = True
                entities.append(f"classifier:sensitive({clf_prob:.2f})")

        latency_ms = (time.perf_counter() - start_time) * 1000.0

        return {
            "sensitive": bool(is_sensitive),
            "classifier_score": round(clf_prob, 4),
            "regex_entities": entities,
            "ner_entities": ner_entities,
            "latency_ms": round(latency_ms, 3),
        }
