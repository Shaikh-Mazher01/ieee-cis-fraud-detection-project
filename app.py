"""Streamlit demo: score real historical transactions with the trained fraud model.

Why not a small form? The model reads ~430 fields (card history counters, device info, ...).
A form with 4 fields leaves ~99% of them empty, which never happens in real data, so the score
would be meaningless. Instead the demo replays real transactions from the held-out test period.
"""
from __future__ import annotations

import sys
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
import xgboost as xgb

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))  # joblib needs the `fraud` package to unpickle the model

BUNDLE_PATH = ROOT / "outputs" / "real" / "models" / "bundle.joblib"
SAMPLE_PATH = ROOT / "demo" / "sample_transactions.parquet"

st.set_page_config(page_title="Fraud detection demo", page_icon="🛡️", layout="centered")


@st.cache_resource
def load_bundle():
    return joblib.load(BUNDLE_PATH) if BUNDLE_PATH.exists() else None


@st.cache_data
def load_sample() -> pd.DataFrame:
    return pd.read_parquet(SAMPLE_PATH) if SAMPLE_PATH.exists() else pd.DataFrame()


def score(bundle: dict, row: pd.DataFrame):
    X = bundle["builder"].transform(row)
    model = bundle["model"]
    prob = float(model.predict_proba(X)[:, 1][0])
    contrib = model.get_booster().predict(xgb.DMatrix(X), pred_contribs=True)[0, :-1]
    reasons = pd.DataFrame({"feature": X.columns, "value": X.iloc[0].to_numpy(), "impact": contrib})
    return prob, reasons


def _random_pick(n: int):
    st.session_state["pick"] = int(pd.Series(range(n)).sample(1).iloc[0])


bundle, sample = load_bundle(), load_sample()
st.title("🛡️ Fraud detection demo")

if bundle is None or sample.empty:
    st.error("Missing `outputs/real/models/bundle.joblib` or `demo/sample_transactions.parquet`.")
    st.stop()

threshold = float(bundle["threshold"])
cats = bundle["builder"].cat_maps_

st.info(
    "**Demo only.** This replays 200 real transactions from the model's held-out *test* period "
    "(the model never saw them). The sample is **30% fraud** so you see interesting cases; in real "
    "traffic only about 3.5% is fraud. The model flags a transaction for human **REVIEW** when its "
    "score is at or above the threshold, chosen to minimise money lost at an assumed $5 per review."
)

labels = [f"#{i + 1:03d}  |  ${r.TransactionAmt:,.2f}  |  product {r.ProductCD}" for i, r in sample.iterrows()]
st.session_state.setdefault("pick", 0)
c1, c2 = st.columns([4, 1])
choice = c1.selectbox("Pick a transaction", range(len(sample)), format_func=lambda i: labels[i], key="pick")
c2.write("")
c2.button("🎲 Random", on_click=_random_pick, args=(len(sample),), width="stretch")
row = sample.iloc[choice]

st.subheader("What-if (optional)")
st.caption("Edit a field; the other 400+ fields stay as in the original transaction.")
e1, e2, e3 = st.columns(3)
amt = e1.number_input("Amount ($)", min_value=0.01, value=float(row.TransactionAmt), step=10.0, key=f"a{choice}")
prod_opts = cats["ProductCD"]
prod = e2.selectbox("ProductCD", prod_opts, index=prod_opts.index(row.ProductCD) if row.ProductCD in prod_opts else 0,
                    key=f"p{choice}")
card_opts = cats["card4"]
card = e3.selectbox("Card network", card_opts,
                    index=card_opts.index(row.card4) if row.card4 in card_opts else 0, key=f"c{choice}")

edited = row.drop(labels=["isFraud"]).to_frame().T
edited["TransactionAmt"], edited["ProductCD"], edited["card4"] = amt, prod, card

if st.button("Analyze transaction", type="primary"):
    prob, reasons = score(bundle, edited)
    st.divider()
    m1, m2 = st.columns(2)
    m1.metric("Fraud score", f"{prob:.3f}")
    m2.metric("Review threshold", f"{threshold:.3f}")
    if prob >= threshold:
        st.error("🚨 **REVIEW**: score is at or above the threshold")
    else:
        st.success("✅ **APPROVE**: score is below the threshold")

    st.markdown("**What pushed the score** (positive = toward fraud)")
    top = pd.concat([reasons.nlargest(5, "impact"), reasons.nsmallest(3, "impact")])
    st.dataframe(top.assign(impact=top.impact.round(3), value=top.value.round(3)), hide_index=True,
                 width="stretch")
    st.caption("These are patterns the model uses, not proven causes of fraud.")

    if st.checkbox("Reveal what actually happened"):
        st.write("Historical label: " + ("**FRAUD**" if row.isFraud == 1 else "**legitimate**"))

with st.expander("About this model"):
    st.markdown(
        "- XGBoost trained on the IEEE-CIS Fraud Detection data (Kaggle), time-ordered split.\n"
        "- Held-out test period: PR-AUC 0.54, ROC-AUC 0.90. At the threshold it catches ~82% of fraud, "
        "and ~16% of flags are real fraud.\n"
        "- A learning project, not a production fraud system."
    )
