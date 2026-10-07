"""Interpretable rules layer: each rule is one reason code an analyst can read.

Run:  python -m src.rules   (writes reports/rules_train.md, training data only)
"""
from pathlib import Path

import pandas as pd

from src.load import load_config, window_bounds
from src.features import to_markdown, window_days

RULES = ["R1_RAPID_PASS_THROUGH", "R2_FAN_IN", "R3_FAN_OUT",
         "R4_NEW_AND_BURSTY", "R5_HIGH_RISK_CHANNEL", "R6_CROSS_BANK_HOPS"]


def fire_rules(features, rules_cfg):
    """Return one True/False column per rule, aligned with `features`."""
    f, c = features, rules_cfg
    r1, r2, r3 = c["R1_RAPID_PASS_THROUGH"], c["R2_FAN_IN"], c["R3_FAN_OUT"]
    r4, r5, r6 = c["R4_NEW_AND_BURSTY"], c["R5_HIGH_RISK_CHANNEL"], c["R6_CROSS_BANK_HOPS"]
    txn_per_day = f["in_count_per_day"] + f["out_count_per_day"]
    return pd.DataFrame({
        # has_dwell: the 168h fill value means "never forwarded", which must never count as rapid
        "R1_RAPID_PASS_THROUGH": (f["has_dwell"] == 1)
                                 & (f["dwell_median_hours"] < r1["max_dwell_hours"])
                                 & f["pass_through_ratio"].between(r1["min_pass_through_ratio"],
                                                                   r1["max_pass_through_ratio"]),
        "R2_FAN_IN": f["fan_in_per_day"] >= r2["min_fan_in_per_day"],
        "R3_FAN_OUT": f["fan_out_per_day"] >= r3["min_fan_out_per_day"],
        "R4_NEW_AND_BURSTY": (f["first_seen_frac"] >= r4["min_first_seen_frac"])
                             & (f["burst_share"] >= r4["min_burst_share"])
                             & (txn_per_day >= r4["min_txn_per_day"]),
        "R5_HIGH_RISK_CHANNEL": (f["fmt_cash_share"] + f["fmt_bitcoin_share"]) >= r5["min_cash_bitcoin_share"],
        "R6_CROSS_BANK_HOPS": (f["cross_bank_out_share"] >= r6["min_cross_bank_out_share"])
                              & (f["out_count_per_day"] >= r6["min_out_per_day"]),
    }, index=features.index)


def reason_text(row, days):
    """Human-readable explanation for each rule, in whole-window counts an analyst understands."""
    return {
        "R1_RAPID_PASS_THROUGH": f"RAPID_PASS_THROUGH: forwarded {row.pass_through_ratio:.0%} of incoming money, "
                                 f"median wait {row.dwell_median_hours:.1f}h",
        "R2_FAN_IN": f"FAN_IN: {round(row.fan_in_per_day * days)} distinct senders in {days} days",
        "R3_FAN_OUT": f"FAN_OUT: {round(row.fan_out_per_day * days)} distinct receivers in {days} days",
        "R4_NEW_AND_BURSTY": f"NEW_AND_BURSTY: first seen {row.first_seen_frac:.0%} into the window, "
                             f"{row.burst_share:.0%} of activity in one 24h stretch",
        "R5_HIGH_RISK_CHANNEL": f"HIGH_RISK_CHANNEL: {row.fmt_cash_share + row.fmt_bitcoin_share:.0%} of transfers "
                                f"in cash or bitcoin",
        "R6_CROSS_BANK_HOPS": f"CROSS_BANK_HOPS: all {round(row.out_count_per_day * days)} outgoing transfers "
                              f"went to other banks",
    }


def score_accounts(features, rules_cfg, days):
    """Fire rules, compute the weighted rules-only score, and attach reason codes."""
    fired = fire_rules(features, rules_cfg)
    weights = pd.Series(rules_cfg["weights"])[RULES]
    out = pd.concat([features[["account_id"]], fired], axis=1)
    out["n_rules_fired"] = fired.sum(axis=1)
    out["rule_score"] = fired.astype(int) @ weights
    # Reason codes only for accounts where something fired: building text for 500k rows is wasted work
    any_fired = out["n_rules_fired"] > 0
    out["reason_codes"] = ""
    out.loc[any_fired, "reason_codes"] = [
        " | ".join(text for rule, text in reason_text(row, days).items() if fired.at[idx, rule])
        for idx, row in features[any_fired].iterrows()
    ]
    return out


def rule_report(scored, labels):
    """Precision, recall and lift per rule. Pass TRAINING data only: thresholds are chosen from it."""
    df = scored.merge(labels[["account_id", "is_mule", "is_grey"]], on="account_id")
    n_mules, base_rate = df["is_mule"].sum(), df["is_mule"].mean()
    rows = {}
    for rule in RULES:
        hit = df[df[rule]]
        precision = hit["is_mule"].mean() if len(hit) else 0.0
        rows[rule] = {"accounts_flagged": len(hit), "mules_caught": hit["is_mule"].sum(),
                      "grey_among_flagged": hit["is_grey"].sum(),
                      "precision_pct": 100 * precision, "recall_pct": 100 * hit["is_mule"].sum() / n_mules,
                      "lift": precision / base_rate}
    return pd.DataFrame(rows).T


def score_report(scored, labels):
    """How the weighted score performs at each distinct cut-off (training data only)."""
    df = scored.merge(labels[["account_id", "is_mule"]], on="account_id")
    rows = {}
    for cut in sorted(df.loc[df["rule_score"] > 0, "rule_score"].unique()):
        hit = df[df["rule_score"] >= cut]
        rows[f"score >= {cut}"] = {"accounts_flagged": len(hit), "mules_caught": hit["is_mule"].sum(),
                                   "precision_pct": 100 * hit["is_mule"].mean(),
                                   "recall_pct": 100 * hit["is_mule"].sum() / df["is_mule"].sum()}
    return pd.DataFrame(rows).T


if __name__ == "__main__":
    cfg = load_config()
    start, end = window_bounds(cfg, "train")
    features = pd.read_parquet("data/processed/features_train.parquet")
    labels = pd.read_parquet("data/processed/labels_train.parquet")
    scored = score_accounts(features, cfg["rules"], window_days(start, end))
    per_rule, by_score = rule_report(scored, labels), score_report(scored, labels)
    base_rate = 100 * labels["is_mule"].mean()
    Path("reports/rules_train.md").write_text(
        "# Rules layer on the TRAINING window\n\n"
        "Generated by `python -m src.rules`. Test data is not used here.\n\n"
        f"Accounts: {len(labels):,} | mules: {labels['is_mule'].sum()} | base rate: {base_rate:.3f}%\n\n"
        "## Per rule\n\n" + to_markdown(per_rule) + "\n\n"
        "## Weighted rules-only score\n\n" + to_markdown(by_score) + "\n")
    print(per_rule.round(2).to_string(), "\n")
    print(by_score.round(2).to_string())
