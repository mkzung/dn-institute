"""Why the screen does not use a luck replay.

A luck replay redraws outcomes at the traded prices and asks how often a wallet does as well as it
really did. To keep a ladder of dates ("by March 31", "by April 30") consistent it has to draw one
number U per event and resolve a trade YES when U is below the trade's YES price. Prices move,
so the real outcome can be one that no U produces: it needs U above every YES price paid in the
markets that resolved NO and below every YES price paid in the markets that resolved YES. When
the first is higher than the second, the replay gives the real history probability zero, and a
wallet that bought the right side of both gets a profit no replay can reach.

For every event that is not negRisk and has markets resolving both ways, this writes the two
prices and whether the real outcome is reachable (replay_check.csv).
"""
import importlib.util
import os
import sys

import numpy as np
import pandas as pd

_spec = importlib.util.spec_from_file_location("features", os.path.join(os.path.dirname(__file__), "06_features.py"))
features = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(features)


def main(datadir, out):
    scope = features.read_scope(datadir)
    t = features.read_tape(datadir).merge(scope[["m", "event_id", "negRisk", "winner_index"]], on="m")
    t = t[~t["negRisk"].astype(str).str.lower().eq("true")]
    t["y"] = np.where(t["o"] == 0, t["price"], 1 - t["price"])
    t["usd"] = t["size"] * t["price"]
    t["resolved_yes"] = t["winner_index"] == 0
    g = t.groupby(["event_id", "resolved_yes"])["y"]
    tab = pd.DataFrame({"max_yes_price_in_no_markets": g.max().xs(False, level="resolved_yes"),
                        "min_yes_price_in_yes_markets": g.min().xs(True, level="resolved_yes")}).dropna()
    tab["real_outcome_reachable"] = tab["max_yes_price_in_no_markets"] < tab["min_yes_price_in_yes_markets"]
    tab["markets"] = t.groupby("event_id")["m"].nunique().reindex(tab.index)
    tab["usd"] = t.groupby("event_id")["usd"].sum().reindex(tab.index)
    tab["event_slug"] = scope.drop_duplicates("event_id").set_index("event_id")["event_slug"].reindex(tab.index)
    tab = tab.reset_index().sort_values("usd", ascending=False)
    cols = ["event_slug", "markets", "max_yes_price_in_no_markets", "min_yes_price_in_yes_markets", "real_outcome_reachable", "usd"]
    tab[cols].round(4).to_csv(out, index=False)
    bad = ~tab["real_outcome_reachable"]
    print(len(tab), "events;", int(bad.sum()), "with the real outcome out of reach;",
          round(tab.loc[bad, "usd"].sum() / tab["usd"].sum(), 4), "of their dollars")


if __name__ == "__main__":
    main(*sys.argv[1:3])
