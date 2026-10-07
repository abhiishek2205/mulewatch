"""Cost cut-off and alert-budget logic on toy score lists where the answer can be worked out by hand."""
import numpy as np
import pandas as pd

from src.evaluate import at_budget
from src.model import cost_optimal_alerts


def test_cost_optimal_alerts_balances_reviews_and_misses():
    # ranked by score: mule, normal, mule, normal x 7. Review = 1, miss = 10.
    # alert 0 -> 2 misses = 20; alert 1 -> 1 + 10 = 11; alert 3 -> 3 + 0 = 3 (best); alert 10 -> 10
    y = pd.Series([1, 0, 1] + [0] * 7)
    score = np.linspace(1, 0, 10)
    k, cost = cost_optimal_alerts(y, score, cost_review=1, cost_miss=10)
    assert (k, cost) == (3, 3.0)


def test_cost_optimal_alerts_is_zero_when_reviews_are_too_expensive():
    y = pd.Series([1, 0, 0, 0])
    k, _ = cost_optimal_alerts(y, np.array([0.9, 0.8, 0.7, 0.6]), cost_review=100, cost_miss=10)
    assert k == 0


def test_at_budget_counts_mules_grey_and_breaks_ties_by_account_id():
    scores = pd.DataFrame({"account_id": ["a", "b", "c", "d"], "is_mule": [0, 1, 0, 1],
                           "is_grey": [1, 0, 0, 0], "rules": [5, 5, 5, 0]})
    r = at_budget(scores, "rules", 2)       # tie at 5: a and b come first alphabetically
    assert r["mules caught"] == 1 and r["grey among alerts"] == 1 and r["clean false alarms"] == 0
    assert r["recall %"] == 50.0
