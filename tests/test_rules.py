"""Rules tested end to end on tiny hand-made transfers: transfers -> feature SQL -> rules.

Each toy account is built so we know in advance exactly which rules must fire and which must not.
Thresholds come from config.yaml, so these tests check the rules as actually configured.
"""
import pandas as pd
import pytest

from src.features import compute_features
from src.load import load_config
from src.rules import RULES, score_accounts
from toy import txn, write_parquet

CFG = load_config()
SETTINGS = CFG["features"]
USD_ONLY = pd.DataFrame({"currency": ["US Dollar"], "usd_per_unit": [1.0]})


def gather_scatter(acct, day="2022-09-02"):
    """6 senders pay 1,000 each in the morning; 8 receivers at other banks get 740 each that afternoon."""
    rows = [txn(f"{day} 0{8 + i // 2}:{30 * (i % 2):02d}", f"1{i}0_S{i}", acct, 1000.0) for i in range(6)]
    rows += [txn(f"{day} {14 + i // 2}:{30 * (i % 2):02d}", acct, f"2{i}0_R{i}", 740.0) for i in range(8)]
    return rows


TOY = (
    gather_scatter("009_MULE")
    # forwards 95%, but only after 72 hours: not rapid
    + [txn("2022-09-01 10:00", "301_A", "009_SLOW", 1000.0), txn("2022-09-04 10:00", "009_SLOW", "302_B", 950.0)]
    # forwards quickly, but keeps 70%: not pass-through
    + [txn("2022-09-02 10:00", "303_A", "009_KEEPER", 1000.0), txn("2022-09-02 11:00", "009_KEEPER", "304_B", 300.0)]
    # 5 distinct senders: one short of the fan-in threshold
    + [txn(f"2022-09-03 1{i}:00", f"4{i}0_S", "009_FIVE", 100.0) for i in range(5)]
    # first appears on Sep 5 (67% into the window), 3 transfers within 2 hours, forwards only 25%
    + [txn("2022-09-05 10:00", "501_A", "009_NEW", 1000.0), txn("2022-09-05 11:00", "502_B", "009_NEW", 1000.0),
       txn("2022-09-05 12:00", "009_NEW", "503_C", 500.0)]
    # same, but only 2 transfers: too little activity for burst to mean anything
    + [txn("2022-09-05 10:00", "601_A", "009_NEW2", 1000.0), txn("2022-09-05 12:00", "009_NEW2", "602_C", 200.0)]
    # 2 of 3 transfers in cash
    + [txn("2022-09-02 10:00", "701_A", "009_CASH", 900.0, fmt="Cash"),
       txn("2022-09-03 10:00", "702_B", "009_CASH", 800.0, fmt="Cash"),
       txn("2022-09-04 10:00", "703_C", "009_CASH", 700.0, fmt="Cheque")]
    # 3 outgoing transfers, all to other banks
    + [txn(f"2022-09-0{d} 10:00", "009_HOPS", f"80{d}_X", 100.0) for d in (2, 3, 4)]
    # 3 outgoing transfers, one to its own bank
    + [txn("2022-09-02 10:00", "009_SAMEBANK", "811_X", 100.0), txn("2022-09-03 10:00", "009_SAMEBANK", "812_Y", 100.0),
       txn("2022-09-04 10:00", "009_SAMEBANK", "009_Z", 100.0)]
    # only pays itself
    + [txn("2022-09-03 10:00", "009_SELF", "009_SELF", 5000.0, fmt="Reinvestment")]
)

EXPECTED = {
    "009_MULE": {"R1_RAPID_PASS_THROUGH", "R2_FAN_IN", "R3_FAN_OUT", "R6_CROSS_BANK_HOPS"},
    "009_SLOW": set(),
    "009_KEEPER": set(),
    "009_FIVE": set(),
    "009_NEW": {"R4_NEW_AND_BURSTY"},
    "009_NEW2": set(),
    "009_CASH": {"R5_HIGH_RISK_CHANNEL"},
    "009_HOPS": {"R6_CROSS_BANK_HOPS"},
    "009_SAMEBANK": set(),
    "009_SELF": set(),
}


@pytest.fixture(scope="module")
def scored(tmp_path_factory):
    path = write_parquet(TOY, tmp_path_factory.mktemp("toy"))
    feats = compute_features(path, "2022-09-01", "2022-09-07", USD_ONLY, SETTINGS)
    return score_accounts(feats, CFG["rules"], days=6).set_index("account_id")


@pytest.mark.parametrize("account", EXPECTED)
def test_exactly_the_expected_rules_fire(scored, account):
    fired = {rule for rule in RULES if scored.at[account, rule]}
    assert fired == EXPECTED[account]


def test_score_is_sum_of_weights_of_fired_rules(scored):
    w = CFG["rules"]["weights"]
    expected = w["R1_RAPID_PASS_THROUGH"] + w["R2_FAN_IN"] + w["R3_FAN_OUT"] + w["R6_CROSS_BANK_HOPS"]
    assert scored.at["009_MULE", "rule_score"] == expected
    assert scored.at["009_CASH", "rule_score"] == w["R5_HIGH_RISK_CHANNEL"]   # 0: a reason code, not a score


def test_reason_codes_are_readable(scored):
    text = scored.at["009_MULE", "reason_codes"]
    assert "FAN_IN: 6 distinct senders in 6 days" in text
    assert "FAN_OUT: 8 distinct receivers in 6 days" in text
    assert "RAPID_PASS_THROUGH: forwarded 99% of incoming money" in text   # 5,920 / 6,000
    assert scored.at["009_SELF", "reason_codes"] == ""


def test_same_behaviour_fires_in_a_shorter_window(tmp_path):
    # 4 senders in a 4-day test window is the same rate (1 per day) as 6 senders in 6 days
    rows = [txn(f"2022-09-08 1{i}:00", f"9{i}0_S", "009_T", 100.0) for i in range(4)]
    feats = compute_features(write_parquet(rows, tmp_path), "2022-09-07", "2022-09-11", USD_ONLY, SETTINGS)
    scored = score_accounts(feats, CFG["rules"], days=4).set_index("account_id")
    assert scored.at["009_T", "R2_FAN_IN"]
    assert "FAN_IN: 4 distinct senders in 4 days" in scored.at["009_T", "reason_codes"]
