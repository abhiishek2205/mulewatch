"""Load the raw IBM AML transactions once, check the file is what we expect, and save a clean parquet.

Every later phase reads the parquet, not the 475 MB CSV, so loading cost is paid once.
"""
import csv
import re
from pathlib import Path

import duckdb
import pandas as pd
import yaml

# The CSV header has two columns both called "Account" (sender first, receiver second).
# We check the header matches exactly, then give every column an unambiguous name.
EXPECTED_HEADER = [
    "Timestamp", "From Bank", "Account", "To Bank", "Account",
    "Amount Received", "Receiving Currency", "Amount Paid", "Payment Currency",
    "Payment Format", "Is Laundering",
]

# Bank and account IDs are read as text: bank IDs have leading zeros ("010"),
# and reading them as numbers would turn "010" into 10 and could merge different banks.
RAW_COLUMNS = {
    "ts": "VARCHAR",
    "from_bank": "VARCHAR",
    "from_acct": "VARCHAR",
    "to_bank": "VARCHAR",
    "to_acct": "VARCHAR",
    "amount_received": "DOUBLE",
    "receiving_currency": "VARCHAR",
    "amount_paid": "DOUBLE",
    "payment_currency": "VARCHAR",
    "payment_format": "VARCHAR",
    "is_laundering": "INTEGER",
}


def load_config(path="config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


def check_header(csv_path):
    with open(csv_path, newline="") as f:
        header = next(csv.reader(f))
    if header != EXPECTED_HEADER:
        raise ValueError(f"Unexpected CSV header: {header}")
    return header


def build_parquet(csv_path, parquet_path):
    """Convert the CSV to parquet with clean names, parsed timestamps and account IDs.

    Account ID = bank + account number, because the same account number
    can exist at more than one bank.
    """
    check_header(csv_path)
    Path(parquet_path).parent.mkdir(parents=True, exist_ok=True)
    columns_sql = ", ".join(f"'{name}': '{dtype}'" for name, dtype in RAW_COLUMNS.items())
    duckdb.sql(f"""
        COPY (
            SELECT
                row_number() OVER () AS txn_id,          -- stable ID for joins and debugging
                strptime(ts, '%Y/%m/%d %H:%M') AS ts,
                from_bank || '_' || from_acct AS src,    -- sender account ID
                to_bank   || '_' || to_acct   AS dst,    -- receiver account ID
                from_bank, to_bank,
                amount_received, receiving_currency,
                amount_paid, payment_currency,
                payment_format, is_laundering
            FROM read_csv('{csv_path}', header = true, columns = {{{columns_sql}}})
        ) TO '{parquet_path}' (FORMAT parquet)
    """)


def parse_patterns(patterns_path):
    """Parse Patterns.txt into one row per laundering transaction, tagged with its typology.

    Used ONLY for evaluation (recall per typology), never as a model input.
    """
    rows, attempt_id, typology = [], 0, None
    begin = re.compile(r"BEGIN LAUNDERING ATTEMPT - ([A-Z\-]+)")
    with open(patterns_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            match = begin.match(line)
            if match:
                attempt_id += 1
                typology = match.group(1).rstrip("-")
            elif line.startswith("END LAUNDERING ATTEMPT"):
                typology = None
            elif typology:
                p = line.split(",")
                rows.append({
                    "attempt_id": attempt_id, "typology": typology,
                    "ts": pd.to_datetime(p[0], format="%Y/%m/%d %H:%M"),
                    "src": f"{p[1]}_{p[2]}", "dst": f"{p[3]}_{p[4]}",
                    "amount_paid": float(p[7]),
                })
    return pd.DataFrame(rows)


def window_bounds(cfg, split):
    """Return (start, end) for 'train' or 'test'; start inclusive, end exclusive."""
    s = cfg["split"]
    if split == "train":
        return s["data_start"], s["cut_date"]
    if split == "test":
        return s["cut_date"], s["data_end"]
    raise ValueError(f"Unknown split: {split}")


def build_labels(cfg, split):
    """Label every account active in the split's window, using sql/labels.sql."""
    start, end = window_bounds(cfg, split)
    sql = Path("sql/labels.sql").read_text()
    params = {"parquet": cfg["data"]["transactions_parquet"], "start": start, "end": end}
    labels = duckdb.connect().execute(sql, params).df()
    out = Path(f"data/processed/labels_{split}.parquet")
    labels.to_parquet(out, index=False)
    return labels


if __name__ == "__main__":
    cfg = load_config()
    build_parquet(cfg["data"]["raw_csv"], cfg["data"]["transactions_parquet"])
    print("Wrote", cfg["data"]["transactions_parquet"])
    for split in ["train", "test"]:
        labels = build_labels(cfg, split)
        print(f"{split}: {len(labels)} accounts, {labels.is_mule.sum()} mules, {labels.is_grey.sum()} grey")
