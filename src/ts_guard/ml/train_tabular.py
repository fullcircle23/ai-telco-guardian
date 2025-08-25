import json
import os
import time

import joblib
import numpy as np
import pandas as pd
from features import make_features, make_labels
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split

DATA_PATH = os.path.join(
    os.path.dirname(__file__), "..", "data", "sample_call_logs.csv"
)
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.joblib")


def maybe_generate_sample(path):
    if os.path.exists(path):
        return
    rng = np.random.default_rng(7)
    n = 4000
    df = pd.DataFrame(
        {
            "caller": [f"+60{rng.integers(100000000,999999999)}" for _ in range(n)],
            "callee": [f"+603{rng.integers(1000000,9999999)}" for _ in range(n)],
            "duration_sec": rng.integers(5, 600, n),
            "hour_of_day": rng.integers(0, 24, n),
            "country_code": ["MY"] * n,
            "is_outbound": rng.integers(0, 2, n),
            "recent_calls_from_caller_24h": rng.poisson(5, n),
            "pct_answered_last_7d": rng.uniform(0, 1, n),
            "complaints_last_7d": rng.poisson(0.3, n),
        }
    )
    risk = (
        (df["duration_sec"] < 20).astype(int) * 0.25
        + ((df["hour_of_day"] >= 22) | (df["hour_of_day"] <= 6)).astype(int) * 0.25
        + (df["recent_calls_from_caller_24h"] >= 10).astype(int) * 0.25
        + (df["complaints_last_7d"] >= 1).astype(int) * 0.25
    )
    prob = np.clip(risk, 0, 1)
    df["is_scam"] = (rng.uniform(0, 1, n) < prob).astype(int)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)


def main():
    maybe_generate_sample(DATA_PATH)
    df = pd.read_csv(DATA_PATH)
    X = make_features(df)
    y = make_labels(df)
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    clf = RandomForestClassifier(
        n_estimators=200, max_depth=10, random_state=42, class_weight="balanced"
    )
    clf.fit(Xtr, ytr)
    proba = clf.predict_proba(Xte)[:, 1]
    auc = roc_auc_score(yte, proba)
    print("AUC:", round(auc, 3))
    print(classification_report(yte, (proba > 0.5).astype(int)))
    joblib.dump(clf, MODEL_PATH)
    meta = {
        "timestamp": int(time.time()),
        "features": list(X.columns),
        "roc_auc": float(auc),
        "thresholds": {"low": 0.4, "high": 0.7},
    }
    with open(os.path.join(os.path.dirname(MODEL_PATH), "model_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print("Saved:", MODEL_PATH)


if __name__ == "__main__":
    main()
