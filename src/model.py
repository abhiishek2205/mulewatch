"""Models: rules-only baseline, logistic regression, gradient boosting.

Tuning is time-ordered: fit on inner_train (Sep 1-4), judge on validation (Sep 5-6).
The chosen settings are refit on the full training window (Sep 1-6) and the test window
(Sep 7-10) is scored exactly once.

Run:  python -m src.model
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.features import window_days
from src.load import load_config, window_bounds
from src.rules import RULES, fire_rules

OUT = Path("data/processed")
ID_AND_LABELS = ["account_id", "is_mule", "is_grey"]

# Non-negative and heavily skewed (a few accounts are enormous): log1p squeezes the giants
# so they don't dominate a linear model. Trees only use the order of values, so they don't need it.
LOG_COLS = ["in_count_per_day", "out_count_per_day", "in_usd_per_day", "out_usd_per_day",
            "fan_in_per_day", "fan_out_per_day", "pass_through_ratio", "amount_cv",
            "pagerank_scaled", "short_cycles_per_day"]

# The six payment-format shares always add up to 1, so any one is fully determined by the others
# (the "dummy-variable trap"). A linear model then cannot decide which share deserves the weight and
# gives them large, unstable, partly cancelling weights. Dropping one share makes it the baseline:
# the other weights then read as "compared with paying by cheque".
# POST-HOC FIX: this was found only after the first test run, and both versions are reported.
LR_BASELINE_SHARE = "fmt_cheque_share"

# Small grids on purpose: with 88 validation mules, a big search would fit noise
LR_GRID = [{"C": c} for c in (0.01, 0.1, 1.0, 10.0)]
HGB_GRID = [{"learning_rate": lr, "max_depth": d, "max_iter": n}
            for lr in (0.05, 0.1) for d in (3, 6) for n in (200, 400)]


def load_split(split):
    """Phase 2 features + Phase 4 graph features + labels, one row per account."""
    feats = pd.read_parquet(OUT / f"features_{split}.parquet")
    graph = pd.read_parquet(OUT / f"graph_features_{split}.parquet")
    labels = pd.read_parquet(OUT / f"labels_{split}.parquet")[ID_AND_LABELS]
    return feats.merge(graph, on="account_id").merge(labels, on="account_id")


def feature_columns(train_df):
    """All model inputs, minus constant columns (e.g. currency mismatch is 0 for every account)."""
    cols = [c for c in train_df.columns if c not in ID_AND_LABELS]
    return [c for c in cols if train_df[c].nunique() > 1]


def training_rows(df):
    """Grey accounts (touched laundering money but not pass-through) are left out of TRAINING only.

    Their label 0 is unreliable, so they would blur the mule-vs-clean contrast. Validation and
    test keep them, because a bank scoring live accounts would not know who is grey.
    """
    return df[df["is_grey"] == 0]


def lr_inputs(df, cols):
    X = df[cols].copy()
    logs = [c for c in LOG_COLS if c in cols]
    X[logs] = np.log1p(X[logs])
    return X


def lr_columns(cols, collinear=False):
    return list(cols) if collinear else [c for c in cols if c != LR_BASELINE_SHARE]


def fit_lr(df, cols, C):
    # class_weight="balanced": each mule counts as much as ~1,500 normal accounts, no fake data needed
    model = make_pipeline(StandardScaler(),
                          LogisticRegression(C=C, class_weight="balanced", max_iter=5000))
    return model.fit(lr_inputs(df, cols), df["is_mule"])


def fit_hgb(df, cols, learning_rate, max_depth, max_iter):
    # early_stopping=False: its default carves a RANDOM validation slice out of training,
    # which would reintroduce the leakage the time-based split exists to avoid
    model = HistGradientBoostingClassifier(learning_rate=learning_rate, max_depth=max_depth,
                                           max_iter=max_iter, class_weight="balanced",
                                           early_stopping=False, random_state=0)
    return model.fit(df[cols], df["is_mule"])


def predict(model, kind, df, cols):
    X = df[cols] if kind == "gradient_boosting" else lr_inputs(df, cols)
    return model.predict_proba(X)[:, 1]


def rule_score(df, cfg):
    """Weighted rules-only score from Phase 3 (no learning involved)."""
    weights = pd.Series(cfg["rules"]["weights"])[RULES]
    return fire_rules(df, cfg["rules"]).astype(int).values @ weights.values


def cost_optimal_alerts(y, score, cost_review, cost_miss):
    """Number of top-ranked accounts to alert that minimises reviews x cost + missed mules x cost."""
    order = np.argsort(-score, kind="stable")
    caught = np.concatenate([[0], np.cumsum(y.values[order])])     # mules caught if we alert the top k
    k = np.arange(len(caught))
    total = k * cost_review + (y.sum() - caught) * cost_miss
    best = int(np.argmin(total))
    return best, float(total[best])


def tune(cfg):
    """Fit each candidate on inner_train, score validation. Returns one row per candidate."""
    inner, val = load_split("inner_train"), load_split("validation")
    cols = feature_columns(inner)
    fit_rows = training_rows(inner)
    val_days = window_days(*window_bounds(cfg, "validation"))
    rows = []

    def record(kind, params, score):
        k, _ = cost_optimal_alerts(val["is_mule"], score, cfg["cost_per_review_inr"], cfg["cost_per_missed_mule_inr"])
        rows.append({"model": kind, "params": json.dumps(params),
                     "val_pr_auc": average_precision_score(val["is_mule"], score),
                     "val_roc_auc": roc_auc_score(val["is_mule"], score),
                     "val_cost_optimal_alerts_per_day": k / val_days})

    record("rules", {}, rule_score(val, cfg))
    for kind, collinear in [("logistic_regression", False), ("logistic_regression_v1", True)]:
        lr_cols = lr_columns(cols, collinear)
        for p in LR_GRID:
            record(kind, p, predict(fit_lr(fit_rows, lr_cols, **p), kind, val, lr_cols))
    for p in HGB_GRID:
        record("gradient_boosting", p, predict(fit_hgb(fit_rows, cols, **p), "gradient_boosting", val, cols))
        print("tuned", p, f"val PR-AUC {rows[-1]['val_pr_auc']:.4f}")
    return pd.DataFrame(rows), {"inner_mules": int(inner.is_mule.sum()), "inner_rows_used": len(fit_rows),
                                "inner_grey_dropped": int(inner.is_grey.sum()),
                                "validation_mules": int(val.is_mule.sum()), "validation_accounts": len(val)}


def best_per_model(tuning):
    best = tuning.loc[tuning.groupby("model")["val_pr_auc"].idxmax()].set_index("model")
    return best


def final_fit_and_score(cfg, best):
    """Refit the chosen settings on the full training window and score the test window ONCE."""
    train, test = load_split("train"), load_split("test")
    cols = feature_columns(train)
    fit_rows = training_rows(train)
    lr = fit_lr(fit_rows, lr_columns(cols), **json.loads(best.loc["logistic_regression", "params"]))
    lr_v1 = fit_lr(fit_rows, lr_columns(cols, collinear=True), **json.loads(best.loc["logistic_regression_v1", "params"]))
    hgb = fit_hgb(fit_rows, cols, **json.loads(best.loc["gradient_boosting", "params"]))

    scores = test[ID_AND_LABELS].copy()
    scores["rules"] = rule_score(test, cfg)
    scores["logistic_regression"] = predict(lr, "logistic_regression", test, lr_columns(cols))
    scores["logistic_regression_v1"] = predict(lr_v1, "logistic_regression_v1", test, lr_columns(cols, collinear=True))
    scores["gradient_boosting"] = predict(hgb, "gradient_boosting", test, cols)
    joblib.dump({"logistic_regression": lr, "gradient_boosting": hgb, "columns": cols,
                 "lr_columns": lr_columns(cols)}, OUT / "models.joblib")
    info = {"train_rows_used": len(fit_rows), "train_grey_dropped": int(train.is_grey.sum()),
            "train_mules": int(train.is_mule.sum()), "n_features": len(cols), "features": cols,
            "dropped_constant": [c for c in train.columns if c not in ID_AND_LABELS and c not in cols]}
    return scores, info


def lr_coefficients(models):
    """Standardised LR weights: comparable across features, the basis of LR reason codes."""
    lr = models["logistic_regression"]
    coef = lr.named_steps["logisticregression"].coef_[0]
    return pd.Series(coef, index=models["lr_columns"]).sort_values(key=abs, ascending=False)


if __name__ == "__main__":
    cfg = load_config()
    tuning, tune_info = tune(cfg)
    best = best_per_model(tuning)
    # The primary model is chosen on VALIDATION, before the test set is scored
    # (the superseded collinear LR is kept only for the record, never a candidate)
    primary = best.loc[["logistic_regression", "gradient_boosting"], "val_pr_auc"].idxmax()
    scores, fit_info = final_fit_and_score(cfg, best)
    scores.to_parquet(OUT / "test_scores.parquet", index=False)
    tuning.to_csv(OUT / "tuning.csv", index=False)
    (OUT / "model_info.json").write_text(json.dumps({
        "primary_model": primary, **tune_info, **fit_info,
        "best_params": {m: json.loads(best.loc[m, "params"]) for m in best.index},
        "val_cost_optimal_alerts_per_day": best["val_cost_optimal_alerts_per_day"].to_dict(),
        "val_pr_auc": best["val_pr_auc"].to_dict()}, indent=2))
    print(tuning.round(4).to_string())
    print("primary model (chosen on validation):", primary)
