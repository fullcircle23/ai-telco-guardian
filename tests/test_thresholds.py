from ts_guard.api.main import risk_label_from_proba


def test_thresholds():
    assert risk_label_from_proba(0.39) == "low"
    assert risk_label_from_proba(0.40) == "medium"
    assert risk_label_from_proba(0.6999) == "medium"
    assert risk_label_from_proba(0.70) == "high"
