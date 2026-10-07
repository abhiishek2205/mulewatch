"""Graph features on tiny hand-drawn graphs where the right answer is obvious."""
import itertools

import pandas as pd
import pytest

from src.graph_features import build_graph, neighbour_pass_through, pagerank_scaled, short_cycle_counts


def edges(pairs):
    return pd.DataFrame([{"src": s, "dst": d, "n_transfers": 1, "total_usd": 100.0} for s, d in pairs])


def test_neighbour_shares_use_neighbours_behaviour():
    e = edges([("A", "M"), ("B", "M"), ("M", "C")])
    r1 = pd.Series({"A": True, "B": False, "C": True, "M": False})
    shares = neighbour_pass_through(e, r1)
    assert shares.loc["M", "senders_pass_through_share"] == 0.5      # A fires R1, B doesn't
    assert shares.loc["M", "receivers_pass_through_share"] == 1.0    # C fires R1


def test_pagerank_scaled_averages_one_and_rewards_collectors():
    # four accounts pay S; S pays nobody
    pr = pagerank_scaled(build_graph(edges([("A", "S"), ("B", "S"), ("C", "S"), ("D", "S")])), alpha=0.85)
    assert pr.sum() == pytest.approx(5)          # 5 accounts, average 1
    assert pr["S"] == pr.max()


CYCLES = edges([
    ("A", "B"), ("B", "C"), ("C", "A"),                      # 3-hop loop
    ("D", "E"), ("E", "D"),                                  # 2-hop loop (mutual payments)
    ("F", "G"), ("G", "H"), ("H", "I"), ("I", "J"), ("J", "F"),   # 5-hop loop: too long
    ("X", "Y"), ("Y", "Z"), ("Z", "X"),                      # 3-hop loop far from any seed
])


def test_counts_loops_up_to_max_length_near_seeds():
    counts, info = short_cycle_counts(CYCLES, seeds={"A", "D", "F"}, max_length=4, hub_max_degree=72, time_limit_s=60)
    assert info["status"] == "complete"
    assert all(counts.get(a, 0) == 1 for a in "ABCDE")
    assert all(counts.get(a, 0) == 0 for a in "FGHIJ")        # 5 hops > max_length
    assert all(counts.get(a, 0) == 0 for a in "XYZ")          # not near a seed: not searched


def test_hubs_are_not_followed():
    # A -> HUB -> B -> A is a loop, but HUB has 10 counterparties, above the cap of 5
    pairs = [("A", "HUB"), ("HUB", "B"), ("B", "A")] + [(f"P{i}", "HUB") for i in range(8)]
    counts, _ = short_cycle_counts(edges(pairs), seeds={"A"}, max_length=4, hub_max_degree=5, time_limit_s=60)
    assert counts.get("A", 0) == 0


def test_time_limit_returns_nothing_rather_than_a_partial_count():
    dense = edges([(a, b) for a, b in itertools.permutations(range(12), 2)])   # thousands of short loops
    counts, info = short_cycle_counts(dense, seeds=set(range(12)), max_length=4, hub_max_degree=72, time_limit_s=-1)
    assert counts is None and info["status"] == "timed out"
