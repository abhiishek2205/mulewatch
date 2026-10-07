# MuleWatch: account-level money-mule risk scoring

Scores every bank account for money-mule risk, explains each alert with readable reason codes, and picks
how many alerts to raise based on **analyst capacity and cost**, not on model accuracy.

Python · DuckDB SQL · NetworkX · scikit-learn

> **Synthetic data.** Built on IBM's synthetic *Transactions for Anti Money Laundering* dataset.
> No real customers, and it is **not Indian bank data**. The Indian UPI framing below is an analogy for the problem,
> not a claim about any real bank's data.

---

## The problem

A **money mule** is an account used to receive and pass on the proceeds of fraud, so the money trail is hard to follow.
Banks need to catch mule accounts early, but fraud analysts can only review a limited number of alerts per day.
MuleWatch ranks accounts by risk, attaches reasons, and evaluates the ranking at a fixed daily alert budget.

**The analogy: a typical Indian UPI mule chain**
```
 Victims of a UPI scam (fake job, KYC, loan offers)
        │  pay via UPI
        ▼
 Mule account 1        receives from many victims          (fan-in)
        │  within hours, splits the money
        ▼
 Mule accounts 2..n    often at different banks             (layering)
        │
        ▼
 Cash withdrawal / crypto / transfer abroad                 (cash-out)
```

**Why accounts, not transactions:** banks act on accounts (freeze, enhanced due diligence), and mule behaviour is a
pattern across many transfers, not a property of one payment.

---

## Data

[IBM Transactions for Anti Money Laundering (AML)](https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml),
**HI-Small** version, on Kaggle: `HI-Small_Trans.csv` and `HI-Small_Patterns.txt`. Not included in this repo; see *How to run*.

Checked on load rather than assumed:
- The CSV has **two columns named `Account`** (sender, then receiver); the loader renames them explicitly.
- Bank codes have leading zeros, so IDs are read as text. **Account ID = bank + account number**, because the same
  account number exists at more than one bank.
- `Is Laundering` is used **only as the label**, never as a feature.
- `Patterns.txt` (which laundering scheme each transfer belongs to) is used **only for evaluation**.

---

## Key decisions and why

| Decision | Choice | Why |
|---|---|---|
| Unit scored | Account, per time window | Banks act on accounts; mule behaviour shows up across transfers |
| Label | **Pass-through mule**: the account both received and sent at least one laundering transfer in the window (self-transfers ignored) | That is what a mule does. "Touched any laundering money" mixes sources, mules and end accounts, and most such accounts touched it only once |
| Train / test split | **By time**: train Sep 1-6, test Sep 7-10 | A bank scores the future using the past. A random split leaks: both halves of the same mule chain land on both sides |
| Sep 11-18 | **Dropped** | Normal customers stop there in the simulator and only laundering continues, so "active late" would be a fake perfect signal |
| Tuning | Fit on Sep 1-4, judge on Sep 5-6, then refit on Sep 1-6 | Settings, the primary model and the cost threshold were all chosen **before** the test set was scored |
| Grey accounts | Left out of **training only** | Accounts that touched laundering money without being pass-through have an unreliable label 0. Test keeps them and reports them separately |
| Window length | Counts and totals **per day** | Train is 6 days and test 4; the same behaviour must give the same number |
| Currencies | Converted to USD with rates **implied by the data itself** (median, training window only) | 15 currencies; no outside rates for a synthetic world |
| Rare class | `class_weight="balanced"`, **no SMOTE** | Makes each mule count more without inventing fake mule accounts |
| Metrics | PR-AUC, alert budget, cost; **never accuracy** | With so few mules, "nobody is a mule" is almost 100% accurate and useless |

---

## Approach

```
 HI-Small_Trans.csv ──► src/load.py ──► clean parquet + labels per window
                                              │
             ┌────────────────────────────────┼─────────────────────────────────┐
             ▼                                ▼                                 ▼
  sql/features.sql (DuckDB)        src/rules.py                      src/graph_features.py (NetworkX)
  21 behavioural features          6 rules = 6 reason codes          PageRank, neighbours' behaviour,
  per account per window           (thresholds in config.yaml)       short money loops (2-4 hops)
             └────────────────────────────────┼─────────────────────────────────┘
                                              ▼
                   src/model.py: rules-only score vs logistic regression vs gradient boosting
                                              ▼
       src/evaluate.py: PR-AUC, 50 alerts/day budget, cost curve, recall per typology, example alerts
                                              ▼
                              reports/results.md  (every number below comes from here)
```

**Rules** (each one is a reason code an analyst can read; thresholds were set from training-data percentiles):
`RAPID_PASS_THROUGH` · `FAN_IN` · `FAN_OUT` · `NEW_AND_BURSTY` · `HIGH_RISK_CHANNEL` · `CROSS_BANK_HOPS`

**Graph features:** in/out-degree were checked to equal the SQL fan-in/fan-out, so they were not duplicated. The graph
adds what single-account features cannot: position in the money network (PageRank), whether an account's senders and
receivers themselves look like pass-through accounts (their behaviour, never their labels), and short money loops.

---

## Results

<!-- RESULTS:START -->
*Auto-generated by `python run_pipeline.py`. Full detail: [reports/results.md](reports/results.md).*

**Data processed:** 5,077,237 transactions (Sep 1-10; 5,078,345 rows in the file) across
515,078 accounts. **Test window (Sep 7-10):** 367,656 accounts scored, 322 label-B mules
(base rate 0.088%).

**Headline:** at 50 alerts/day (200 alerts over 4 days), Gradient boosting caught
**84 of 322 test mules (26.1% recall) at 42.0% precision**.

#### Model comparison (test, 50 alerts/day)
| | PR-AUC | x random | ROC-AUC | mules in top 200 | precision % | recall % |
|---|---|---|---|---|---|---|
| Rules only | 0.009 | 10.329 | 0.71 | 14 | 7 | 4.348 |
| Logistic regression | 0.038 | 43.869 | 0.979 | 6 | 3 | 1.863 |
| Gradient boosting | 0.271 | 309.876 | 0.986 | 84 | 42 | 26.087 |

#### If analyst capacity changes (Gradient boosting)
| | alerts | mules caught | precision % | recall % |
|---|---|---|---|---|
| 25 alerts/day | 100 | 63 | 63 | 19.6 |
| 50 alerts/day | 200 | 84 | 42 | 26.1 |
| 100 alerts/day | 400 | 102 | 25.5 | 31.7 |

#### Cost (assumed: Rs 500 per review, Rs 50,000 per missed mule)
Alert nobody: Rs 16.10M. At 50/day: Rs 12.00M.
Cost-optimal rate chosen on validation, 236/day: Rs 9.97M.

#### Recall by laundering typology (Gradient boosting, 50 alerts/day)
| | mules | Gradient boosting recall % |
|---|---|---|
| STACK | 102 | 23.5 |
| CYCLE | 95 | 18.9 |
| SCATTER-GATHER | 76 | 43.4 |
| RANDOM | 65 | 20 |
| FAN-OUT | 26 | 7.7 |
| GATHER-SCATTER | 25 | 4 |
| BIPARTITE | 23 | 13 |
| FAN-IN | 11 | 18.2 |

#### Caught vs missed test mules (medians)
| | mules | transfers in window | fmt_ach_share | pass_through_ratio | dwell_median_hours | fan_in_per_day | fan_out_per_day | share with > 20 transfers |
|---|---|---|---|---|---|---|---|---|
| caught | 84 | 3 | 1 | 1 | 28.03 | 0.25 | 0.25 | 0.01 |
| missed | 238 | 16.5 | 0.25 | 0.94 | 8.52 | 0.5 | 0.5 | 0.39 |
<!-- RESULTS:END -->

Figures: [precision-recall](reports/figures/05_pr_curves.png) ·
[cost curve](reports/figures/05_cost_curve.png) · [typology recall](reports/figures/05_typology_recall.png)

---

## What the system misses, and why

- **Mules hidden inside busy accounts.** The model is best at "pure relay" mules: few transfers, all ACH, forwarding
  almost everything. Mules whose few laundering transfers sit among many ordinary cheque and card payments get
  diluted, because account-level averages wash them out (see *Caught vs missed* above). This is also why typologies
  with busy hub accounts are caught less than ones built from pure relays.
- **Long or slow cycles.** The loop search stops at 4 hops inside one window; laundering cycles in this data can be longer.
- **Mules whose chain crosses a window boundary.** Money received before the cut and sent after it makes the account
  look like "only received" on one side and "only sent" on the other, so it is labelled 0 in both.
- **Look-alikes.** Some false positives behave exactly like a one-hop mule (receive, forward almost all of it within
  hours). Transfers alone cannot separate them; real banks would use account age, KYC and device data.
- **Logistic regression and the "band" problem.** The strongest signal is forwarding *about* 100% of what arrives.
  A linear model can only say "higher is riskier" or "lower is riskier", so it cannot use this signal; trees can.

## Rules and models: why keep both

The gradient-boosting model ranks the alert queue because it ranks far better. Every alert still carries the rule
reason codes and key feature values, so an analyst or auditor sees *why*, not just a score. The rules also act as a
transparent baseline and challenger: if the model's alerts drift away from anything the rules recognise, that is a
signal to investigate.

## Limitations

- **Synthetic data.** Patterns are generated by a simulator. Some findings are artefacts of it: cash and bitcoin are
  used mostly by normal accounts here, and almost every transfer is cross-bank, so two rules carry weight 0.
- **No KYC, device, IP, login, account-age or beneficiary data**, which real bank systems rely on heavily.
- **Short history:** ten usable days. Scoring is per window (top 50/day × 4 days), not a true daily rolling system.
- **Static thresholds** in `config.yaml`; real mule gangs adapt, so thresholds and models need drift monitoring.
- **Cost figures are assumptions** (Rs 500 per review, Rs 50,000 per missed mule), kept in config and clearly labelled.
- **Patterns.txt covers only part of the laundering transfers**, so typology recall is measured on the tagged part.
- **Disclosed post-hoc fix:** after the first test run, logistic regression's payment-format shares turned out to be
  collinear (they always sum to 1). One share was dropped as the baseline, the model was re-tuned on validation and
  re-scored. Both runs are reported in `reports/results.md`; the primary model was unchanged.

## Future work

- A feature for distance from a pass-through ratio of 1, so linear models can see the band.
- "Same counterparty on both sides" and time-sliced features, so a laundering burst inside a busy account is not averaged away.
- Longer, time-respecting cycle search; daily rolling scoring; drift monitoring.

---

## How to run

1. Download `HI-Small_Trans.csv` and `HI-Small_Patterns.txt` from the Kaggle link above into `data/raw/`.
2. Install dependencies (developed on Python 3.14):
   ```bash
   pip install -r requirements.txt
   ```
3. Run everything (tests, then every step; about 10-15 minutes):
   ```bash
   python run_pipeline.py
   ```
   This writes `reports/results.md`, the figures, and refreshes the Results section above.
   Add `--notebooks` to also re-execute the notebooks.

**Reproducible:** two complete runs from raw data produce a byte-identical `reports/results.md`. The DuckDB feature and
graph queries run single-threaded with a fixed row order, because parallel floating-point sums come out in a different
order each run. An earlier multi-threaded run differed in the last decimal places, which was enough to flip a close
gradient-boosting tuning choice. The numbers above are from the deterministic version.

Run only the tests (small hand-made transfers where the right answer was worked out by hand):
```bash
python -m pytest tests
```

## Repository layout

```
config.yaml            all thresholds, the split dates, alert budget and cost assumptions
run_pipeline.py        one command, end to end
sql/                   labels.sql, fx_rates.sql, features.sql (DuckDB)
src/                   load, features, rules, graph_features, model, evaluate
tests/                 feature, rule, graph and evaluation tests on hand-made data
notebooks/             01 explore · 02 rules · 03 graph · 04 models and evaluation
reports/               results.md (generated), rules and feature summaries, figures/
data/raw, processed/   local only, gitignored
```

## License

MIT; see [LICENSE](LICENSE). The dataset is not included and is subject to its own terms on Kaggle.
