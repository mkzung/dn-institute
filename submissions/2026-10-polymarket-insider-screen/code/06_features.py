"""Turn the committed tape into one row per wallet and market.

Every trade in the tape was placed before the market's t_known, so every row here is a
prediction. Positions are netted: shares held at t_known are buys minus sells per token. When a
wallet sells more of a token than it bought, the shortfall came from minting a YES and NO pair
for $1 (a split), so the pair is added at that cost. Profit is cash out minus cash in plus $1 per
held share of the winning token.

early_longshot_usd is the money put on the eventual winner at LONGSHOT or less in the LEAD_S
before t_known, counted only for shares the wallet still held at t_known: a wallet that bought
cheap and sold again within minutes did not bet on the outcome.
"""
import glob
import os
import sys

import numpy as np
import pandas as pd

LONGSHOT = 0.30
LEAD_S = 7 * 24 * 3600


def read_tape(datadir):
    parts = sorted(glob.glob(os.path.join(datadir, "tape_part*.csv.xz")))
    return pd.concat([pd.read_csv(p) for p in parts], ignore_index=True)


def read_scope(datadir):
    scope = pd.read_csv(os.path.join(datadir, "markets_scope.csv"), dtype={"id": str, "event_id": str})
    scope["event_id"] = scope["event_id"].fillna("m" + scope["m"].astype(str))
    return scope


def wallet_market(tape, scope):
    """One row per wallet and market from a tape and the market table, counting only trades
    before each market's t_known."""
    t = tape.merge(scope[["m", "winner_index", "t_known"]], on="m", how="left")
    t = t[t["ts"] < t["t_known"]].copy()
    t["usd"] = t["size"] * t["price"]
    t["buy"] = t["side"] == "B"
    t["win_tok"] = t["o"] == t["winner_index"]
    t["early"] = t["buy"] & t["win_tok"] & (t["price"] <= LONGSHOT) & (t["t_known"] - t["ts"] <= LEAD_S)
    t["lead_h_x_usd"] = np.where(t["buy"] & t["win_tok"], (t["t_known"] - t["ts"]) / 3600 * t["usd"], 0.0)
    keys = ["w", "m"]
    g = t.groupby(keys)
    out = pd.DataFrame({"n_trades": g.size(), "first_ts": g["ts"].min(), "last_ts": g["ts"].max()})
    for o in (0, 1):
        for side, flag in (("buy", True), ("sell", False)):
            s = t[(t["o"] == o) & (t["buy"] == flag)].groupby(keys)
            out[f"{side}_sh_{o}"] = s["size"].sum()
            out[f"{side}_usd_{o}"] = s["usd"].sum()
    out = out.fillna(0.0)
    eg = t[t["early"]].groupby(keys)
    out["early_usd"] = eg["usd"].sum().reindex(out.index).fillna(0.0)
    out["early_sh"] = eg["size"].sum().reindex(out.index).fillna(0.0)
    out["lead_h_x_usd"] = g["lead_h_x_usd"].sum()
    out = out.reset_index().merge(scope[["m", "winner_index", "event_id", "negRisk"]], on="m")

    held0 = out["buy_sh_0"] - out["sell_sh_0"]
    held1 = out["buy_sh_1"] - out["sell_sh_1"]
    minted = np.maximum(np.maximum(-held0, -held1), 0.0)
    out["minted_sh"] = minted
    out["held_0"] = held0 + minted
    out["held_1"] = held1 + minted
    out["cash"] = out["sell_usd_0"] + out["sell_usd_1"] - out["buy_usd_0"] - out["buy_usd_1"] - minted
    out["cost"] = out["buy_usd_0"] + out["buy_usd_1"] + minted
    win_held = np.where(out["winner_index"] == 0, out["held_0"], out["held_1"])
    esh = out["early_sh"].to_numpy()
    kept = np.minimum(esh, np.maximum(win_held, 0.0))
    out["early_longshot_usd"] = np.where(esh > 0, out["early_usd"].to_numpy() * kept / np.where(esh > 0, esh, 1), 0.0)
    out["pnl"] = out["cash"] + win_held
    win_buy = np.where(out["winner_index"] == 0, out["buy_usd_0"], out["buy_usd_1"])
    out["win_buy_usd"] = win_buy
    out["lose_buy_usd"] = out["buy_usd_0"] + out["buy_usd_1"] - win_buy
    out["win_lead_h_usdw"] = np.where(win_buy > 0, out["lead_h_x_usd"] / np.where(win_buy > 0, win_buy, 1), np.nan)
    out = out.drop(columns=["early_usd", "early_sh", "lead_h_x_usd"])
    return out.sort_values(keys).reset_index(drop=True)


def main(datadir, out):
    wm = wallet_market(read_tape(datadir), read_scope(datadir))
    wm.round(6).to_csv(out, index=False)
    print(len(wm), "wallet-market rows,", wm["w"].nunique(), "wallets,", wm["m"].nunique(), "markets")


if __name__ == "__main__":
    main(*sys.argv[1:3])
