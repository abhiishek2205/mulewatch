"""Graph features: where an account sits in the money-flow network, and what its neighbours look like.

One directed graph per window (train / test), built only from that window's transfers.
Run:  python -m src.graph_features
"""
import time
from collections import Counter

import duckdb
import networkx as nx
import pandas as pd

from src.features import window_days
from src.load import load_config, window_bounds
from src.rules import fire_rules

CORE_RULES = ["R1_RAPID_PASS_THROUGH", "R2_FAN_IN", "R3_FAN_OUT", "R4_NEW_AND_BURSTY"]


def load_edges(parquet, start, end, fx):
    """Collapse repeated transfers between the same pair into one edge with a count and a USD total."""
    con = duckdb.connect()
    con.execute("SET enable_progress_bar = false")
    con.register("fx", fx)
    return con.execute("""
        SELECT t.src, t.dst,
               count(*)                            AS n_transfers,
               sum(t.amount_paid * fx.usd_per_unit) AS total_usd
        FROM read_parquet($parquet) t
        JOIN fx ON fx.currency = t.payment_currency
        WHERE t.ts >= $start AND t.ts < $end
          AND t.src <> t.dst                        -- self-transfers are not money moving between accounts
        GROUP BY t.src, t.dst
    """, {"parquet": parquet, "start": start, "end": end}).df()


def build_graph(edges):
    graph = nx.DiGraph()
    graph.add_edges_from(
        (s, d, {"n_transfers": n, "total_usd": u})
        for s, d, n, u in edges[["src", "dst", "n_transfers", "total_usd"]].itertuples(index=False))
    return graph


def pagerank_scaled(graph, alpha):
    """Unweighted PageRank, multiplied by the number of accounts in the graph.

    Raw PageRank sums to 1, so its values shrink as the graph grows: the test graph (fewer accounts)
    would get systematically bigger numbers than train. Scaling by N makes the average 1 in both.
    """
    pr = nx.pagerank(graph, alpha=alpha, weight=None)
    return pd.Series(pr, name="pagerank_scaled") * graph.number_of_nodes()


def neighbour_pass_through(edges, r1_fires):
    """Share of an account's senders, and of its receivers, that fire R1 (rapid pass-through).

    Uses neighbours' BEHAVIOUR only, never their labels: mules sit in chains with other mule-like
    accounts, and a bank could compute exactly this on live data.
    """
    e = edges[["src", "dst"]].copy()
    e["src_r1"] = e["src"].map(r1_fires).fillna(False).astype(float)
    e["dst_r1"] = e["dst"].map(r1_fires).fillna(False).astype(float)
    senders = e.groupby("dst")["src_r1"].mean().rename("senders_pass_through_share")
    receivers = e.groupby("src")["dst_r1"].mean().rename("receivers_pass_through_share")
    return pd.concat([senders, receivers], axis=1)


def short_cycle_counts(edges, seeds, max_length, hub_max_degree, time_limit_s):
    """Count loops of up to `max_length` hops each account belongs to, searched only near `seeds`.

    Search area = seed accounts + their 1-hop neighbours, minus giant hubs. Returns (counts, info);
    counts is None if the time limit was hit, so a half-finished count is never used as a feature.
    """
    degree = pd.concat([edges["src"], edges["dst"]]).value_counts()   # distinct counterparties per account
    allowed = set(degree.index[degree <= hub_max_degree])
    seeds = set(seeds) & allowed
    near = edges[edges["src"].isin(seeds) | edges["dst"].isin(seeds)]
    nodes = (set(near["src"]) | set(near["dst"])) & allowed
    sub = edges[edges["src"].isin(nodes) & edges["dst"].isin(nodes)]
    graph = nx.DiGraph(list(zip(sub["src"], sub["dst"])))

    counts, n_cycles, started = Counter(), 0, time.time()
    for cycle in nx.simple_cycles(graph, length_bound=max_length):
        n_cycles += 1
        counts.update(cycle)
        if n_cycles % 1000 == 0 and time.time() - started > time_limit_s:
            info = dict(status="timed out", search_nodes=graph.number_of_nodes(),
                        search_edges=graph.number_of_edges(), cycles_found=n_cycles,
                        hubs_excluded=int((degree > hub_max_degree).sum()), seconds=round(time.time() - started))
            return None, info
    info = dict(status="complete", search_nodes=graph.number_of_nodes(), search_edges=graph.number_of_edges(),
                cycles_found=n_cycles, hubs_excluded=int((degree > hub_max_degree).sum()),
                seconds=round(time.time() - started))
    return pd.Series(counts, name="short_cycles", dtype=float), info


def build_graph_features(cfg, split, fx):
    parquet, g = cfg["data"]["transactions_parquet"], cfg["graph"]
    start, end = window_bounds(cfg, split)
    days = window_days(start, end)
    features = pd.read_parquet(f"data/processed/features_{split}.parquet")
    fired = fire_rules(features, cfg["rules"]).set_index(features["account_id"])

    edges = load_edges(parquet, start, end, fx)
    graph = build_graph(edges)
    out = pd.DataFrame(index=features["account_id"])
    out = out.join(pagerank_scaled(graph, g["pagerank_alpha"]))
    out = out.join(neighbour_pass_through(edges, fired["R1_RAPID_PASS_THROUGH"]))

    seeds = fired.index[fired[CORE_RULES].any(axis=1)]
    cycles, info = short_cycle_counts(edges, seeds, g["cycle_max_length"], g["cycle_hub_max_degree"],
                                      g["cycle_time_limit_minutes"] * 60)
    if cycles is not None:
        # per day, like every other count, so 6-day and 4-day windows are comparable
        out = out.join((cycles / days).rename("short_cycles_per_day"))

    # Accounts with only self-transfers are not in the graph: no position, no neighbours, no loops
    out = out.fillna(0.0).reset_index()
    info.update(split=split, graph_nodes=graph.number_of_nodes(), graph_edges=graph.number_of_edges(),
                transfers_collapsed=int(edges["n_transfers"].sum()))
    return out, info


def sanity_checks(graph_feats, features, graph_nodes):
    assert graph_feats["account_id"].is_unique, "duplicate accounts"
    assert set(graph_feats["account_id"]) == set(features["account_id"]), "accounts differ from features"
    assert not graph_feats.isna().any().any(), "missing values"
    for col in ["senders_pass_through_share", "receivers_pass_through_share"]:
        assert graph_feats[col].between(0, 1).all(), f"{col} outside [0, 1]"
    # scaled PageRank over the graph's accounts sums to the number of accounts (average 1)
    assert abs(graph_feats["pagerank_scaled"].sum() - graph_nodes) / graph_nodes < 1e-6, "PageRank does not sum to N"


if __name__ == "__main__":
    cfg = load_config()
    fx = pd.read_csv("data/processed/fx_rates.csv")
    for split in ["train", "test"]:
        feats, info = build_graph_features(cfg, split, fx)
        sanity_checks(feats, pd.read_parquet(f"data/processed/features_{split}.parquet"), info["graph_nodes"])
        feats.to_parquet(f"data/processed/graph_features_{split}.parquet", index=False)
        print(info)
