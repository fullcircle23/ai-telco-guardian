
import pandas as pd

FEATURES = [
    "duration_sec","hour_of_day","is_outbound",
    "recent_calls_from_caller_24h","pct_answered_last_7d","complaints_last_7d"
]

def make_features(df: pd.DataFrame) -> pd.DataFrame:
    X = df[FEATURES].copy()
    X["is_outbound"] = X["is_outbound"].astype(int)
    return X

def make_labels(df: pd.DataFrame):
    return df["is_scam"].astype(int)
