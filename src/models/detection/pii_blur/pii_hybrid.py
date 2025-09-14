"""
Utility to load the tiny PII classifier and expose a simple function.
(This module is optional; main script already handles classifier loading.)
"""
import joblib

def load_classifier(path="pii_clf.joblib"):
    bundle = joblib.load(path)
    return bundle["vec"], bundle["clf"], float(bundle.get("thr", 0.5))
