"""Run MuleWatch end to end: tests, then every pipeline step in order, then refresh the README results.

    python run_pipeline.py               # tests + pipeline (about 10-15 minutes)
    python run_pipeline.py --notebooks   # also re-execute the notebooks afterwards

Needs data/raw/HI-Small_Trans.csv and data/raw/HI-Small_Patterns.txt (see README).
Each step is its own `python -m` process, so it runs exactly as it would by hand,
and the run stops at the first failure instead of producing half-updated results.
"""
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

STEPS = [
    ("Tests (hand-made toy data)",                          ["-m", "pytest", "-q", "tests"]),
    ("Load CSV, build labels for every window",             ["-m", "src.load"]),
    ("Account features (DuckDB SQL)",                       ["-m", "src.features"]),
    ("Rules report (training window)",                      ["-m", "src.rules"]),
    ("Graph features (NetworkX)",                           ["-m", "src.graph_features"]),
    ("Models: tune on validation, score test once",         ["-m", "src.model"]),
    ("Evaluation: reports/results.md and figures",          ["-m", "src.evaluate"]),
]
NOTEBOOKS = ["01_explore", "02_rules", "03_graph", "04_model_and_eval"]
README_START, README_END = "<!-- RESULTS:START -->", "<!-- RESULTS:END -->"


def run(label, args):
    print(f"\n=== {label}", flush=True)
    started = time.time()
    # UTF-8 output so DuckDB/pandas tables print on Windows consoles too
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    result = subprocess.run([sys.executable, *args], env=env)
    if result.returncode != 0:
        sys.exit(f"FAILED: {label} (exit code {result.returncode})")
    print(f"--- done in {time.time() - started:.0f}s", flush=True)


def check_raw_data():
    missing = [p for p in ["data/raw/HI-Small_Trans.csv", "data/raw/HI-Small_Patterns.txt"] if not Path(p).exists()]
    if missing:
        sys.exit(f"Missing raw data: {missing}. Download them from Kaggle first (see README).")


def refresh_readme():
    """Replace the README block between the markers with the summary the evaluation just wrote."""
    readme = Path("README.md")
    text = readme.read_text(encoding="utf-8")
    if README_START not in text or README_END not in text:
        sys.exit("README.md is missing the RESULTS markers")
    summary = Path("reports/results_summary.md").read_text(encoding="utf-8").strip()
    before, rest = text.split(README_START, 1)
    after = rest.split(README_END, 1)[1]
    readme.write_text(f"{before}{README_START}\n{summary}\n{README_END}{after}", encoding="utf-8")
    print("\n=== README results block refreshed from reports/results_summary.md")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--notebooks", action="store_true", help="also re-execute the notebooks")
    args = parser.parse_args()

    check_raw_data()
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    started = time.time()
    for label, step in STEPS:
        run(label, step)
    refresh_readme()
    if args.notebooks:
        for nb in NOTEBOOKS:
            run(f"Notebook {nb}", ["-m", "nbconvert", "--to", "notebook", "--execute", "--inplace",
                                   f"notebooks/{nb}.ipynb", "--ExecutePreprocessor.timeout=1800"])
    print(f"\nAll steps finished in {(time.time() - started) / 60:.1f} minutes. Results: reports/results.md")


if __name__ == "__main__":
    main()
