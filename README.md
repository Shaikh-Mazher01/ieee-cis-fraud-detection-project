# IEEE-CIS Fraud Detection

End-to-end card-fraud project on the [IEEE-CIS Fraud Detection](https://www.kaggle.com/competitions/ieee-fraud-detection)
data (590,540 transactions, 3.5% fraud): SQL analysis, EDA, time-ordered training, a cost-based decision
threshold, SHAP explanations, drift monitoring, a scoring API and a Streamlit demo.

**Live demo:** _add your Streamlit link here_

## Results (held-out most-recent 15% of transactions, never used for any choice)

| Metric | Value |
|---|---|
| ROC-AUC / PR-AUC | 0.904 / 0.541 |
| Fraud caught at the chosen threshold | 82% (16% of flags are real fraud) |
| Fraud found by reviewing only the riskiest 5% | 59% |
| Cost with model vs. approving everything | $160,894 vs. $469,609 |

The saving is an estimate and depends on the assumed cost of a manual review ($5 here). At $50 per review
it falls from about $309k to about $54k; see `outputs/real/reports/executive_summary.md`.
It also assumes every flagged fraud is stopped and flagging a good customer costs nothing beyond the review.

## Honest caveats

- The shipped model was trained in quick mode (`--fast`: 400 trees, no tuning) and was still improving when it
  stopped. Retrain without `--fast` for the configured model.
- Performance fell over time in the validation/test windows (PR-AUC about 0.74 down to 0.50), and some top
  features (`D1_start`, `D15_start`, `freq_uid2`) drift strongly. Plan for regular retraining.
- This is a learning project on public data, not a production fraud system.

## Methodology

- **Time-ordered split** (70/15/15): train on the past, validate and test on later periods.
- Feature statistics (frequency encodings, per-group amount stats) are learned on **train only**.
- Early stopping, model choice and the decision threshold use **validation only**; the test period is scored once.
- Models: dummy baseline, logistic regression, XGBoost (plain and class-weighted), optional random search.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt

bash scripts/download_data.sh          # needs a Kaggle account + API token (see the script header)
python -m fraud all                    # prepare, sql, eda, train, evaluate, drift, report
# python -m fraud all --fast           # quicker, lower quality (laptop check)
# python -m fraud synth --rows 100000  # no Kaggle account? generate fake data, add --profile synthetic

pytest                                 # tests
streamlit run app.py                   # demo app
python -m fraud serve                  # scoring API at http://127.0.0.1:8000/docs
```

Outputs are written to `outputs/real/` (figures, tables, `reports/executive_summary.md`).
`outputs/` and `data/` are git-ignored; the model file used by the demo was added with `git add -f`
(`outputs/real/models/bundle.joblib`), so re-add it the same way after retraining.

## Layout

```
app.py                 Streamlit demo (replays real held-out transactions)
demo/                  200 real test-period transactions used by the demo
configs/config.yaml    every setting in one place
sql/                   analysis queries (run via `python -m fraud sql`)
src/fraud/             data, features, train, evaluate, drift, api, report, ...
tests/                 pytest suite
scripts/               data download helper
```

## Notes on the demo and API

The model reads about 430 fields per transaction. A form with a handful of fields would leave almost all of
them empty, which never happens in real data, so the demo scores **real historical transactions** instead
(with an optional what-if edit of amount, product and card network). The API accepts sparse requests but returns
a `warning` when most fields are missing, because such a score is unreliable.

The model file is a pickle, so library versions are pinned in `requirements.txt` to match training.
