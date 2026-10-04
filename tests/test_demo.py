"""The demo must score real transactions sensibly. Skipped when the model/sample files are absent (e.g. CI)."""
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
BUNDLE, SAMPLE = ROOT / "outputs/real/models/bundle.joblib", ROOT / "demo/sample_transactions.parquet"
pytestmark = pytest.mark.skipif(not (BUNDLE.exists() and SAMPLE.exists()), reason="model or demo sample missing")


def test_demo_sample_scores_vary_and_rank_fraud_higher():
    sys.path.insert(0, str(ROOT / "src"))
    b = joblib.load(BUNDLE)
    s = pd.read_parquet(SAMPLE)
    assert len(s) >= 100 and set(s.isFraud.unique()) == {0, 1}
    p = b["model"].predict_proba(b["builder"].transform(s))[:, 1]
    flags = p >= b["threshold"]
    assert 0.05 < flags.mean() < 0.95                       # not "flag everything" / "flag nothing"
    assert p[s.isFraud.to_numpy() == 1].mean() > 3 * p[s.isFraud.to_numpy() == 0].mean()
    assert np.isfinite(p).all()
