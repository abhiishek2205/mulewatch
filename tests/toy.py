"""Helpers to build tiny hand-made transfer tables for tests."""
import pandas as pd


def txn(ts, src, dst, amount, fmt="ACH", paid_ccy="US Dollar", recv_ccy="US Dollar", recv_amount=None):
    """One transfer row in the same shape as data/processed/transactions.parquet. Bank = text before '_'."""
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
