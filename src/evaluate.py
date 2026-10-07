"""Evaluate the three scorers on the TEST window and write reports/results.md.

Everything a bank cares about, never accuracy: ranking quality (PR-AUC, ROC-AUC), the alert
budget, a cost-based threshold, recall per laundering typology, and example alerts.
Run after src.model:  python -m src.evaluate
"""
import json
from pathlib import Path

import duckdb
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score

from src.features import to_markdown, window_days
from src.load import load_config, parse_patterns, window_bounds
from src.model import OUT, lr_coefficients, lr_inputs
from src.rules import RULES, fire_rules, reason_text

MODELS = ["rules", "logistic_regression", "gradient_boosting"]
# The first LR had collinear payment-format shares; kept in the ranking and budget tables for the record
SUPERSEDED = ["logistic_regression_v1"]
NAMES = {"rules": "Rules only", "logistic_regression": "Logistic regression",
         "gradient_boosting": "Gradient boosting",
         "logistic_regression_v1": "LR v1 (collinear shares, superseded)"}
COLOURS = {"rules": "#2a78d6", "logistic_regression": "#eb6834", "gradient_boosting": "#1baf7a"}
FIG = Path("reports/figures")


def ranked(scores, model):
    # Ties (common for the rules score) are broken by account_id: deterministic, carries no information
    return scores.sort_values([model, "account_id"], ascending=[False, True])


def ranking_metrics(scores, val_pr_auc):
    base = scores["is_mule"].mean()
    rows = {}
    for m in MODELS + SUPERSEDED:
        pr = average_precision_score(scores["is_mule"], scores[m])
        rows[NAMES[m]] = {"PR-AUC": pr, "PR-AUC / base rate": pr / base,
                          "ROC-AUC": roc_auc_score(scores["is_mule"], scores[m]),
                          "validation PR-AUC": val_pr_auc[m]}
    return pd.DataFrame(rows).T


def at_budget(scores, model, n_alerts):
    top = ranked(scores, model).head(n_alerts)
    caught = int(top["is_mule"].sum())
    return {"alerts": n_alerts, "mules caught": caught,
            "precision %": 100 * caught / n_alerts, "recall %": 100 * caught / scores["is_mule"].sum(),
            "grey among alerts": int(top["is_grey"].sum()),
            "clean false alarms": int(((top["is_mule"] == 0) & (top["is_grey"] == 0)).sum())}


def budget_table(scores, per_day, days):
    rows = {}
    for apd in per_day:
        for m in MODELS + SUPERSEDED:
            rows[f"{apd}/day | {NAMES[m]}"] = at_budget(scores, m, apd * days)
    return pd.DataFrame(rows).T


def total_cost(scores, model, n_alerts, cfg):
    caught = ranked(scores, model).head(n_alerts)["is_mule"].sum()
    return n_alerts * cfg["cost_per_review_inr"] + (scores["is_mule"].sum() - caught) * cfg["cost_per_missed_mule_inr"]


def cost_curve(scores, model, cfg):
    y = ranked(scores, model)["is_mule"].values
    caught = np.concatenate([[0], np.cumsum(y)])
    k = np.arange(len(caught))
    return k, k * cfg["cost_per_review_inr"] + (y.sum() - caught) * cfg["cost_per_missed_mule_inr"]


def cost_table(scores, info, cfg, days):
    """Cost at three cut-offs. The cost-optimal rate is CHOSEN on validation; hindsight is shown only for reference."""
    rows = {}
    for m in MODELS:
        val_per_day = info["val_cost_optimal_alerts_per_day"][m]
        n_val = int(round(val_per_day * days))
        k, cost = cost_curve(scores, m, cfg)
        rows[NAMES[m]] = {
            "alert nobody (Rs)": total_cost(scores, m, 0, cfg),
            f"budget {cfg['alerts_per_day']}/day (Rs)": total_cost(scores, m, cfg["alerts_per_day"] * days, cfg),
            "cost-optimal alerts/day (chosen on validation)": val_per_day,
            "cost at that rate on test (Rs)": total_cost(scores, m, n_val, cfg),
            "hindsight best alerts/day on test (not used)": k[np.argmin(cost)] / days,
            "hindsight best cost (Rs)": cost.min(),
        }
    return pd.DataFrame(rows).T


def mule_typologies(cfg, split):
    """Map each account to the laundering typologies it took part in, inside the split's window."""
    start, end = window_bounds(cfg, split)
    p = parse_patterns(cfg["data"]["patterns_txt"])
    p = p[(p["ts"] >= start) & (p["ts"] < end)]
    return pd.concat([p[["src", "typology"]].rename(columns={"src": "account_id"}),
                      p[["dst", "typology"]].rename(columns={"dst": "account_id"})]).drop_duplicates()


def typology_recall(scores, typ, n_alerts):
    mules = scores[scores["is_mule"] == 1][["account_id"]]
    tagged = mules.merge(typ, on="account_id", how="left").fillna({"typology": "UNTAGGED"})
    out = tagged.groupby("typology").size().rename("mules").to_frame()
    for m in MODELS:
        alerted = set(ranked(scores, m).head(n_alerts)["account_id"])
        out[f"{NAMES[m]} recall %"] = tagged.assign(hit=tagged["account_id"].isin(alerted)) \
                                            .groupby("typology")["hit"].mean() * 100
    return out.sort_values("mules", ascending=False)


def caught_vs_missed(scores, test_df, model, n_alerts, days):
    """Profile of test mules inside vs outside the alert budget: what kind of mule do we miss?"""
    alerted = set(ranked(scores, model).head(n_alerts)["account_id"])
    m = scores.loc[scores["is_mule"] == 1, ["account_id"]].merge(test_df, on="account_id")
    m["transfers in window"] = (m["in_count_per_day"] + m["out_count_per_day"]) * days
    m["group"] = np.where(m["account_id"].isin(alerted), "caught", "missed")
    cols = ["transfers in window", "fmt_ach_share", "pass_through_ratio", "dwell_median_hours",
            "fan_in_per_day", "fan_out_per_day"]
    out = m.groupby("group")[cols].median()
    out.insert(0, "mules", m.groupby("group").size())
    out["share with > 20 transfers"] = m.groupby("group")["transfers in window"].apply(lambda s: (s > 20).mean())
    return out


def lr_drivers(models, feats_row_df, top=3):
    """Top features pushing an account's LR score UP: weight x standardised value."""
    lr, cols = models["logistic_regression"], models["lr_columns"]
    z = lr.named_steps["standardscaler"].transform(lr_inputs(feats_row_df, cols))
    contrib = pd.DataFrame(z * lr.named_steps["logisticregression"].coef_[0], columns=cols, index=feats_row_df.index)
    return contrib.apply(lambda r: ", ".join(f"{c} (+{v:.1f})" for c, v in r.nlargest(top).items() if v > 0), axis=1)


def examples(scores, model, n_alerts, test_df, models, days, cfg):
    """Top-ranked true positives and false positives inside the alert budget, with reasons."""
    top = ranked(scores, model).head(n_alerts).reset_index(drop=True)
    top["rank"] = top.index + 1
    picks = pd.concat([top[top["is_mule"] == 1].head(5), top[top["is_mule"] == 0].head(3)])
    rows = test_df.set_index("account_id").loc[picks["account_id"]]
    fired = fire_rules(rows, cfg["rules"])
    picks = picks.set_index("account_id")
    picks["rule reason codes"] = [
        # "; " not " | ": the text goes inside markdown tables
        "; ".join(t for r, t in reason_text(row, days).items() if fired.at[idx, r]) or "(no rule fired)"
        for idx, row in rows.iterrows()]
    picks["LR top drivers"] = lr_drivers(models, rows)
    keep = ["pass_through_ratio", "dwell_median_hours", "fan_in_per_day", "fan_out_per_day",
            "senders_pass_through_share", "receivers_pass_through_share", "short_cycles_per_day"]
    return picks.join(rows[keep].round(2)).reset_index()


def plot_pr(scores):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for m in MODELS:
        p, r, _ = precision_recall_curve(scores["is_mule"], scores[m])
        ax.plot(r, p, color=COLOURS[m], linewidth=2, label=f"{NAMES[m]} (PR-AUC {average_precision_score(scores['is_mule'], scores[m]):.3f})")
    ax.axhline(scores["is_mule"].mean(), color="#52514e", linestyle="--", linewidth=1, label="Random (base rate)")
    ax.set_xlabel("Recall (share of test mules caught)")
    ax.set_ylabel("Precision (share of alerts that are mules)")
    ax.set_title("Test window: precision vs recall", loc="left", fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(color="#e6e5e1", linewidth=0.6)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "05_pr_curves.png", dpi=150)
    plt.close(fig)


def plot_cost(scores, model, info, cfg, days):
    k, cost = cost_curve(scores, model, cfg)
    n_budget = cfg["alerts_per_day"] * days
    n_val = int(round(info["val_cost_optimal_alerts_per_day"][model] * days))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(k[1:], cost[1:] / 1e6, color=COLOURS[model], linewidth=2)
    ax.set_xscale("log")
    # Labels on opposite sides of their lines so they never collide
    for n, label, ls, side in [(n_budget, f"Alert budget ({cfg['alerts_per_day']}/day)", "--", "right"),
                               (n_val, "Cost-optimal rate\nchosen on validation", ":", "left")]:
        ax.axvline(n, color="#52514e", linestyle=ls, linewidth=1)
        ax.annotate(f"{label}\n{n:,} alerts\nRs {cost[n] / 1e6:.2f}M", (n, cost[0] / 1e6 * 1.45),
                    textcoords="offset points", xytext=(-6 if side == "right" else 6, 0),
                    ha=side, va="top", fontsize=9, color="#52514e")
    # Beyond ~1.6x the "alert nobody" cost the curve only shoots up; capping keeps the U-shape readable
    ax.set_ylim(0, cost[0] / 1e6 * 1.6)
    ax.set_xlabel("Number of accounts alerted in the 4-day test window (log scale)")
    ax.set_ylabel("Expected total cost (Rs million)")
    ax.set_title(f"{NAMES[model]}: reviews x Rs {cfg['cost_per_review_inr']} + missed mules x "
                 f"Rs {cfg['cost_per_missed_mule_inr']:,} (assumed costs)", loc="left", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(color="#e6e5e1", linewidth=0.6)
    fig.tight_layout()
    fig.savefig(FIG / "05_cost_curve.png", dpi=150)
    plt.close(fig)


def plot_typology(typ_table):
    t = typ_table.drop(index="UNTAGGED", errors="ignore").sort_values("mules")
    fig, ax = plt.subplots(figsize=(8, 4.8))
    y = np.arange(len(t))
    h = 0.26
    for i, m in enumerate(MODELS):
        ax.barh(y + (i - 1) * h, t[f"{NAMES[m]} recall %"], height=h - 0.03, color=COLOURS[m], label=NAMES[m])
    ax.set_yticks(y, [f"{idx} ({n} mules)" for idx, n in t["mules"].items()])
    ax.set_xlabel("Recall within the alert budget (%)")
    ax.set_title("Test window: share of each typology's mules alerted", loc="left", fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", color="#e6e5e1", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIG / "05_typology_recall.png", dpi=150)
    plt.close(fig)


def fmt_table(df, decimals=3):
    return to_markdown(df.astype(float).round(decimals))


def data_volumes(cfg):
    """How much data the pipeline actually processed, counted from the data, not typed in."""
    s = cfg["split"]
    row = duckdb.sql(f"""
        WITH t AS (SELECT * FROM read_parquet('{cfg["data"]["transactions_parquet"]}')),
             used AS (SELECT * FROM t WHERE ts >= '{s["data_start"]}' AND ts < '{s["data_end"]}')
        SELECT (SELECT count(*) FROM t)                                  AS rows_in_file,
               (SELECT count(*) FROM used)                               AS transactions_used,
               (SELECT sum(is_laundering) FROM used)::INT                AS laundering_transactions_used,
               (SELECT count(*) FROM (SELECT src FROM used UNION SELECT dst FROM used)) AS accounts_used
    """).df().iloc[0]
    return {k: int(v) for k, v in row.items()}


def readme_summary(vol, metrics, budget, costs, typ, missed, cfg, primary, n_acc, n_mules, n_budget, days):
    """Compact results for the README, regenerated on every run so the README can never drift from the data."""
    apd = cfg["alerts_per_day"]
    main_rows = budget.loc[[f"{apd}/day | {NAMES[m]}" for m in MODELS]]
    models_tbl = pd.DataFrame({
        "PR-AUC": metrics.loc[[NAMES[m] for m in MODELS], "PR-AUC"].values,
        "x random": metrics.loc[[NAMES[m] for m in MODELS], "PR-AUC / base rate"].values,
        "ROC-AUC": metrics.loc[[NAMES[m] for m in MODELS], "ROC-AUC"].values,
        f"mules in top {n_budget}": main_rows["mules caught"].values,
        "precision %": main_rows["precision %"].values,
        "recall %": main_rows["recall %"].values,
    }, index=[NAMES[m] for m in MODELS])
    sens = budget.loc[[f"{a}/day | {NAMES[primary]}" for a in (apd // 2, apd, apd * 2)],
                      ["alerts", "mules caught", "precision %", "recall %"]]
    sens.index = [f"{a} alerts/day" for a in (apd // 2, apd, apd * 2)]
    c = costs.loc[NAMES[primary]]
    head = main_rows.loc[f"{apd}/day | {NAMES[primary]}"]
    typ_tbl = typ[["mules", f"{NAMES[primary]} recall %"]].drop(index="UNTAGGED", errors="ignore")
    return f"""*Auto-generated by `python run_pipeline.py`. Full detail: [reports/results.md](reports/results.md).*

**Data processed:** {vol['transactions_used']:,} transactions (Sep 1-10; {vol['rows_in_file']:,} rows in the file) across
{vol['accounts_used']:,} accounts. **Test window (Sep 7-10):** {n_acc:,} accounts scored, {n_mules} label-B mules
(base rate {100 * n_mules / n_acc:.3f}%).

**Headline:** at {apd} alerts/day ({n_budget} alerts over {days} days), {NAMES[primary]} caught
**{int(head['mules caught'])} of {n_mules} test mules ({head['recall %']:.1f}% recall) at {head['precision %']:.1f}% precision**.

#### Model comparison (test, {apd} alerts/day)
{fmt_table(models_tbl, 3)}

#### If analyst capacity changes ({NAMES[primary]})
{fmt_table(sens, 1)}

#### Cost (assumed: Rs {cfg['cost_per_review_inr']} per review, Rs {cfg['cost_per_missed_mule_inr']:,} per missed mule)
Alert nobody: Rs {c['alert nobody (Rs)'] / 1e6:.2f}M. At {apd}/day: Rs {c[f'budget {apd}/day (Rs)'] / 1e6:.2f}M.
Cost-optimal rate chosen on validation, {c['cost-optimal alerts/day (chosen on validation)']:.0f}/day: Rs {c['cost at that rate on test (Rs)'] / 1e6:.2f}M.

#### Recall by laundering typology ({NAMES[primary]}, {apd} alerts/day)
{fmt_table(typ_tbl, 1)}

#### Caught vs missed test mules (medians)
{fmt_table(missed, 2)}
"""


def main():
    cfg = load_config()
    info = json.loads((OUT / "model_info.json").read_text())
    scores = pd.read_parquet(OUT / "test_scores.parquet")
    models = joblib.load(OUT / "models.joblib")
    test_df = pd.read_parquet(OUT / "features_test.parquet").merge(
        pd.read_parquet(OUT / "graph_features_test.parquet"), on="account_id")
    days = window_days(*window_bounds(cfg, "test"))
    n_budget = cfg["alerts_per_day"] * days
    primary = info["primary_model"]
    FIG.mkdir(parents=True, exist_ok=True)

    metrics = ranking_metrics(scores, info["val_pr_auc"])
    budget = budget_table(scores, [cfg["alerts_per_day"] // 2, cfg["alerts_per_day"], cfg["alerts_per_day"] * 2], days)
    costs = cost_table(scores, info, cfg, days)
    typ = typology_recall(scores, mule_typologies(cfg, "test"), n_budget)
    ex = examples(scores, primary, n_budget, test_df, models, days, cfg)
    missed = caught_vs_missed(scores, test_df, primary, n_budget, days)
    coefs = lr_coefficients(models)
    plot_pr(scores)
    plot_cost(scores, primary, info, cfg, days)
    plot_typology(typ)

    n_mules, n_acc = int(scores["is_mule"].sum()), len(scores)
    main_row = budget.loc[f"{cfg['alerts_per_day']}/day | {NAMES[primary]}"]
    vol = data_volumes(cfg)
    ex_cols = ["rank", "account_id", primary, "is_grey", "rule reason codes", "LR top drivers"] + \
              ["pass_through_ratio", "dwell_median_hours", "fan_in_per_day", "fan_out_per_day",
               "senders_pass_through_share", "receivers_pass_through_share", "short_cycles_per_day"]

    def ex_md(df):
        lines = ["| " + " | ".join(ex_cols) + " |", "|" + "---|" * len(ex_cols)]
        for _, r in df[ex_cols].iterrows():
            lines.append("| " + " | ".join(f"{v:.4f}" if isinstance(v, float) else str(v) for v in r) + " |")
        return "\n".join(lines)

    md = f"""# MuleWatch results

Generated by `python -m src.evaluate` from an actual pipeline run. **Do not edit by hand.**
Data is IBM's synthetic AML dataset (HI-Small), not real or Indian bank data.

## Setup
- Data processed: {vol['transactions_used']:,} transactions from Sep 1-10 ({vol['laundering_transactions_used']:,} marked laundering)
  across {vol['accounts_used']:,} accounts; the file has {vol['rows_in_file']:,} rows (Sep 11-18 dropped as a simulator artifact).
- Label B (pass-through mule): received AND sent at least 1 laundering transfer in the window.
- Train: Sep 1-6 ({info['train_mules']} mules; {info['train_grey_dropped']:,} grey accounts left out of training).
  Tuning: fit on Sep 1-4 ({info['inner_mules']} mules), judged on Sep 5-6 ({info['validation_mules']} mules).
- **Test: Sep 7-10, {n_acc:,} accounts, {n_mules} mules, base rate {100 * n_mules / n_acc:.3f}%.** Scored once.
- {info['n_features']} model features. Dropped as constant: {', '.join(info['dropped_constant']) or 'none'}.
- Chosen settings (on validation): {json.dumps(info['best_params'])}.
- **Primary model, chosen on validation PR-AUC before testing: {NAMES[primary]}.**
- Alert budget: {cfg['alerts_per_day']} alerts/day x {days} days = {n_budget} alerts. Best possible recall at this budget: {100 * min(1, n_budget / n_mules):.1f}%.
- Costs are ASSUMPTIONS: Rs {cfg['cost_per_review_inr']} per review, Rs {cfg['cost_per_missed_mule_inr']:,} per missed mule.
- **Post-hoc fix, disclosed:** the first test run showed that logistic regression's six payment-format shares always
  sum to 1 (collinear), which made its weights unstable and its reason codes meaningless. `fmt_cheque_share` was then
  dropped as the baseline category (no other change), re-tuned on validation, and the test set scored again.
  "LR v1" in the tables is the original run. The primary model did not change; gradient boosting was chosen on validation both times.

## Headline
At {cfg['alerts_per_day']} alerts/day, {NAMES[primary]} caught **{int(main_row['mules caught'])} of {n_mules} test mules
({main_row['recall %']:.1f}% recall) at {main_row['precision %']:.1f}% precision**. A further
{int(main_row['grey among alerts'])} of the {n_budget} alerts were grey accounts (touched laundering money without being pass-through).

## 1. Ranking quality (test)
PR-AUC of a random ranking equals the base rate ({scores['is_mule'].mean():.5f}).

{fmt_table(metrics, 4)}

## 2. Alert budget and sensitivity (test)
"{cfg['alerts_per_day'] // 2}/day" and "{cfg['alerts_per_day'] * 2}/day" show what happens if analyst capacity is halved or doubled.

{fmt_table(budget, 2)}

## 3. Cost-based threshold (test, assumed costs)
The cost-optimal alert rate is chosen on validation and then applied to test. The hindsight optimum on test is shown only for reference; it was not used to choose anything.

{fmt_table(costs, 1)}

![cost curve](figures/05_cost_curve.png)

## 4. Recall per laundering typology (test, within the alert budget)
A mule counts toward every typology it took part in. UNTAGGED = mules in no Patterns.txt scheme (Patterns covers about 62% of laundering transfers).

{fmt_table(typ, 1)}

![typology recall](figures/05_typology_recall.png)

### What kind of mule is missed? ({NAMES[primary]}, medians of test mules inside vs outside the alert budget)
{fmt_table(missed, 2)}

## 5. Example alerts ({NAMES[primary]}, within the alert budget)
### True positives (top-ranked mules)
{ex_md(ex[ex['is_mule'] == 1])}

### False positives (top-ranked non-mules)
{ex_md(ex[ex['is_mule'] == 0])}

## 6. Logistic regression weights (standardised; positive = pushes towards mule)
{fmt_table(coefs.to_frame('weight'), 3)}

![PR curves](figures/05_pr_curves.png)
"""
    Path("reports/results.md").write_text(md, encoding="utf-8")
    Path("reports/results_summary.md").write_text(
        readme_summary(vol, metrics, budget, costs, typ, missed, cfg, primary, n_acc, n_mules, n_budget, days),
        encoding="utf-8")
    print(metrics.round(4).to_string(), "\n")
    print(budget.round(2).to_string(), "\n")
    print(costs.round(1).to_string(), "\n")
    print(typ.round(1).to_string())


if __name__ == "__main__":
    main()
