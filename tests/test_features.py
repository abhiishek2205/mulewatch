"""Feature SQL tested on tiny hand-made transfers where every answer was worked out by hand.

Account M is the worked example from the Phase 2 explanation: three quick deposits from
different senders, forwarded within hours, plus one self-transfer that must be ignored.
"""
import pandas as pd
import pytest

from src.features import compute_features, compute_fx_rates

START, END = "2022-09-01", "2022-09-07"   # 6-day window, like the training split
SETTINGS = {"dwell_fill_hours": 168, "round_amount_multiple": 100, "pass_through_ratio_cap": 10}
USD_ONLY = pd.DataFrame({"currency": ["US Dollar"], "usd_per_unit": [1.0]})


def txn(ts, src, dst, amount, fmt="ACH", paid_ccy="US Dollar", recv_ccy="US Dollar", recv_amount=None):
    return {
        "ts": pd.Timestamp(ts), "src": src, "dst": dst,
        "from_bank": src.split("_")[0], "to_bank": dst.split("_")[0],
        "amount_received": amount if recv_amount is None else recv_amount, "receiving_currency": recv_ccy,
        "amount_paid": amount, "payment_currency": paid_ccy,
        "payment_format": fmt, "is_laundering": 0,
    }


def write_parquet(rows, tmp_path):
    path = tmp_path / "toy.parquet"
    pd.DataFrame(rows).to_parquet(path, index=False)
    return str(path).replace("\\", "/")


ACCOUNT_M = [
    txn("2022-09-02 10:00", "001_P", "009_M", 2000.00),
    txn("2022-09-02 10:30", "002_Q", "009_M", 1487.20),
    txn("2022-09-02 11:00", "003_R", "009_M", 3000.00),
    txn("2022-09-02 12:00", "009_M", "004_X", 3200.00),
    txn("2022-09-02 13:00", "009_M", "005_Y", 3105.40),
    txn("2022-09-04 09:00", "001_P", "009_M", 500.00, fmt="Cheque"),
    txn("2022-09-04 15:00", "009_M", "009_Z", 100.00, fmt="Cash"),
    txn("2022-09-05 08:00", "009_M", "009_M", 1000.00, fmt="Reinvestment"),   # self-transfer: ignored
]


@pytest.fixture
def m_features(tmp_path):
    feats = compute_features(write_parquet(ACCOUNT_M, tmp_path), START, END, USD_ONLY, SETTINGS)
    return feats.set_index("account_id").loc["009_M"]


def test_counts_and_totals_are_per_day(m_features):
    assert m_features.in_count_per_day == pytest.approx(4 / 6)
    assert m_features.out_count_per_day == pytest.approx(3 / 6)        # self-transfer not counted
    assert m_features.in_usd_per_day == pytest.approx(6987.20 / 6)
    assert m_features.out_usd_per_day == pytest.approx(6405.40 / 6)


def test_fan_in_counts_distinct_senders(m_features):
    assert m_features.fan_in_per_day == pytest.approx(3 / 6)           # P sent twice, counted once
    assert m_features.fan_out_per_day == pytest.approx(3 / 6)


def test_pass_through_ratio(m_features):
    assert m_features.pass_through_ratio == pytest.approx(6405.40 / 6987.20)


def test_dwell_is_median_hours_to_next_outgoing(m_features):
    # waits: 2.0, 1.5, 1.0 (to the 12:00 transfer) and 6.0 (Sep 4) -> median 1.75
    assert m_features.dwell_median_hours == pytest.approx(1.75)
    assert m_features.has_dwell == 1


def test_burst_uses_rolling_24_hours(m_features):
    assert m_features.burst_share == pytest.approx(5 / 7)              # 5 of 7 transfers on Sep 2


def test_first_seen_is_fraction_of_window(m_features):
    assert m_features.first_seen_frac == pytest.approx(34 / 144)       # Sep 2 10:00 is 34h into 144h


def test_shares(m_features):
    assert m_features.cross_bank_out_share == pytest.approx(2 / 3)     # Z is at M's own bank
    assert m_features.fmt_ach_share == pytest.approx(5 / 7)
    assert m_features.fmt_cheque_share == pytest.approx(1 / 7)
    assert m_features.fmt_cash_share == pytest.approx(1 / 7)
    assert m_features.round_amount_share == pytest.approx(5 / 7)       # 1487.20 and 3105.40 are not round
    assert m_features.ccy_mismatch_share == 0


def test_burst_is_not_split_at_midnight(tmp_path):
    # 4 transfers between 22:00 and 01:00: one burst, even though it crosses midnight
    rows = [txn(f"2022-09-02 {h}:00", f"00{i}_S{i}", "009_B", 50.0) for i, h in enumerate(["22", "23"])]
    rows += [txn(f"2022-09-03 0{h}:00", f"01{i}_S{i}", "009_B", 50.0) for i, h in enumerate(["0", "1"])]
    rows += [txn("2022-09-05 12:00", "020_S9", "009_B", 50.0)]
    feats = compute_features(write_parquet(rows, tmp_path), START, END, USD_ONLY, SETTINGS)
    assert feats.set_index("account_id").loc["009_B"].burst_share == pytest.approx(4 / 5)


def test_self_transfer_only_account_gets_no_activity_values(tmp_path):
    rows = [txn("2022-09-03 10:00", "010_S", "010_S", 500.0, fmt="Reinvestment")]
    s = compute_features(write_parquet(rows, tmp_path), START, END, USD_ONLY, SETTINGS).set_index("account_id").loc["010_S"]
    assert s.in_count_per_day == 0 and s.out_count_per_day == 0
    assert s.dwell_median_hours == 168 and s.has_dwell == 0
    assert s.first_seen_frac == 1.0 and s.burst_share == 0


def test_receive_only_account_gets_dwell_fill(tmp_path):
    rows = [txn("2022-09-03 10:00", "001_P", "009_K", 500.0)]
    k = compute_features(write_parquet(rows, tmp_path), START, END, USD_ONLY, SETTINGS).set_index("account_id").loc["009_K"]
    assert k.dwell_median_hours == 168 and k.has_dwell == 0
    assert k.pass_through_ratio == 0


def test_amounts_converted_to_usd_by_side(tmp_path):
    # paid 1000 Euro, received 1200 USD: sender's out is 1000 EUR x 1.2, receiver's in is 1200 USD
    rows = [txn("2022-09-03 10:00", "001_E", "002_U", 1000.0, paid_ccy="Euro", recv_amount=1200.0)]
    fx = pd.DataFrame({"currency": ["US Dollar", "Euro"], "usd_per_unit": [1.0, 1.2]})
    feats = compute_features(write_parquet(rows, tmp_path), START, END, fx, SETTINGS).set_index("account_id")
    assert feats.loc["001_E"].out_usd_per_day == pytest.approx(1200 / 6)
    assert feats.loc["002_U"].in_usd_per_day == pytest.approx(1200 / 6)
    assert feats.loc["001_E"].ccy_mismatch_share == 1


def test_fx_rate_is_median_of_implied_rates(tmp_path):
    rows = [txn("2022-09-03 10:00", f"00{i}_A", "009_B", 100.0, paid_ccy="Euro", recv_amount=r)
            for i, r in enumerate([117.0, 118.0, 500.0])]                # 500 is an odd row; median ignores it
    fx = compute_fx_rates(write_parquet(rows, tmp_path), START, END).set_index("currency")
    assert fx.loc["Euro", "usd_per_unit"] == pytest.approx(1.18)
    assert fx.loc["US Dollar", "usd_per_unit"] == 1.0
