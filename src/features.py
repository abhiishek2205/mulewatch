"""Build one row of behavioural features per account, per window, using the SQL in sql/.

Run:  python -m src.features
"""
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from src.load import load_config, window_bounds

SQL_DIR = Path("sql")


def window_days(start, end):
    return (pd.Timestamp(end) - pd.Timestamp(start)).days


def compute_fx_rates(parquet, start, end):
    """USD rate per currency, implied by transfers in [start, end). Always call with the TRAIN window."""
    sql = (SQL_DIR / "fx_rates.sql").read_text()
    return duckdb.connect().execute(sql, {"parquet": parquet, "start": start, "end": end}).df()


def check_fx_coverage(parquet, fx):
    """Fail loudly if any currency in the data has no rate: those transfers would silently vanish in the JOIN."""
    used = duckdb.sql(f"""
        SELECT payment_currency AS c FROM read_parquet('{parquet}')
        UNION SELECT receiving_currency FROM read_parquet('{parquet}')""").df()["c"]
    missing = set(used) - set(fx["currency"])
    if missing:
        raise ValueError(f"No exchange rate for: {sorted(missing)}")


def compute_features(parquet, start, end, fx, settings):
    """Run sql/features.sql for one window and return a DataFrame with one row per account."""
    con = duckdb.connect()
    con.execute("SET enable_progress_bar = false")
    con.register("fx_df", fx)
    con.execute("CREATE TABLE fx AS SELECT currency, usd_per_unit FROM fx_df")
    params = {
        "parquet": parquet,
        "start": start,
        "end": end,
        "window_days": window_days(start, end),
        "dwell_fill_hours": settings["dwell_fill_hours"],
        "round_multiple": settings["round_amount_multiple"],
        "ratio_cap": settings["pass_through_ratio_cap"],
    }
    return con.execute((SQL_DIR / "features.sql").read_text(), params).df()


def sanity_checks(features, labels, settings):
    """Structural checks; raise on the first failure so a broken table never reaches the models."""
    assert features["account_id"].is_unique, "duplicate accounts"
    assert set(features["account_id"]) == set(labels["account_id"]), "accounts differ from labels"
    assert not features.isna().any().any(), "missing values"
    share_cols = [c for c in features if c.endswith("_share") or c == "first_seen_frac"]
    assert features[share_cols].apply(lambda s: s.between(0, 1).all()).all(), "share outside [0, 1]"
    assert (features.drop(columns="account_id") >= 0).all().all(), "negative value"
    assert features["pass_through_ratio"].max() <= settings["pass_through_ratio_cap"], "ratio above cap"
    assert features["dwell_median_hours"].max() <= settings["dwell_fill_hours"], "dwell above fill value"
    fmt_cols = [c for c in features if c.startswith("fmt_")]
    active = features[["in_count_per_day", "out_count_per_day"]].sum(axis=1) > 0
    assert np.allclose(features.loc[active, fmt_cols].sum(axis=1), 1), "format shares don't sum to 1"


def spot_check(parquet, start, end, features, fx, n=5, seed=0):
    """Recompute simple features for a few random active accounts in pandas, independently of the SQL."""
    days = window_days(start, end)
    rate = dict(zip(fx["currency"], fx["usd_per_unit"]))
    sample = features[features["in_count_per_day"] > 0].sample(n, random_state=seed)["account_id"].tolist()
    txn = duckdb.sql(f"""
        SELECT * FROM read_parquet('{parquet}')
        WHERE ts >= '{start}' AND ts < '{end}' AND src <> dst
          AND (src IN {tuple(sample)} OR dst IN {tuple(sample)})""").df()
    for acct in sample:
        inc, out = txn[txn["dst"] == acct], txn[txn["src"] == acct]
        expected = {
            "in_count_per_day": len(inc) / days,
            "out_count_per_day": len(out) / days,
            "fan_in_per_day": inc["src"].nunique() / days,
            "fan_out_per_day": out["dst"].nunique() / days,
            "in_usd_per_day": (inc["amount_received"] * inc["receiving_currency"].map(rate)).sum() / days,
            "out_usd_per_day": (out["amount_paid"] * out["payment_currency"].map(rate)).sum() / days,
        }
        got = features.set_index("account_id").loc[acct]
        for col, value in expected.items():
            assert np.isclose(got[col], value), f"{acct} {col}: sql={got[col]} pandas={value}"
    return sample


def summarise_by_group(features, labels):
    """Median of each feature for mules, grey accounts, clean active accounts and self-transfer-only accounts.

    Run on TRAIN only: it guides rule thresholds in Phase 3, so it must never look at the test set.
    """
    df = features.merge(labels[["account_id", "is_mule", "is_grey"]], on="account_id")
    active = (df["in_count_per_day"] + df["out_count_per_day"]) > 0
    group = np.select(
        [df["is_mule"] == 1, df["is_grey"] == 1, active],
        ["mule", "grey", "clean_active"], default="self_transfer_only")
    feature_cols = [c for c in features.columns if c != "account_id"]
    summary = df[feature_cols].groupby(group).median().T
    summary.loc["n_accounts"] = pd.Series(group).value_counts()
    return summary[["mule", "grey", "clean_active", "self_transfer_only"]]


def to_markdown(df):
    """Plain markdown table, so we don't need an extra package just for formatting."""
    cols = [str(c) for c in df.columns]
    lines = ["| | " + " | ".join(cols) + " |", "|---" * (len(cols) + 1) + "|"]
    for idx, row in df.iterrows():
        lines.append(f"| {idx} | " + " | ".join(f"{v:,.3f}".rstrip("0").rstrip(".") for v in row) + " |")
    return "\n".join(lines)


def build_all(cfg):
    parquet = cfg["data"]["transactions_parquet"]
    settings = cfg["features"]
    train_start, train_end = window_bounds(cfg, "train")
    fx = compute_fx_rates(parquet, train_start, train_end)
    check_fx_coverage(parquet, fx)
    fx.to_csv("data/processed/fx_rates.csv", index=False)

    out = {}
    for split in ["train", "test"]:
        start, end = window_bounds(cfg, split)
        features = compute_features(parquet, start, end, fx, settings)
        labels = pd.read_parquet(f"data/processed/labels_{split}.parquet")
        sanity_checks(features, labels, settings)
        checked = spot_check(parquet, start, end, features, fx)
        features.to_parquet(f"data/processed/features_{split}.parquet", index=False)
        print(f"{split}: {len(features):,} accounts x {features.shape[1] - 1} features | "
              f"checks passed | spot-checked {len(checked)} accounts in pandas")
        out[split] = features
        if split == "train":
            summary = summarise_by_group(features, labels)
            Path("reports").mkdir(exist_ok=True)
            Path("reports/feature_summary_train.md").write_text(
                "# Feature medians by group (TRAIN window only)\n\n"
                "Generated by `python -m src.features`. Used to propose rule thresholds in Phase 3.\n\n"
                + to_markdown(summary) + "\n")
    return fx, out


if __name__ == "__main__":
    build_all(load_config())
