"""
LBMA Gold Fix vs Live XAU/USD - does price "respect" the fix level?

Every run:
  1. Pulls 5-minute gold bars (COMEX GC=F via Yahoo) and merges them into data/prices_5m.csv
  2. Takes a live XAU/USD spot snapshot (gold-api.com) -> data/spot_snapshots.csv
     (spot - futures gap = "basis", used to convert between futures and spot terms)
  3. Builds daily fix levels:
       AM = price at 10:30 London, PM = price at 15:00 London   (fix-time proxy)
       If data/lbma_fixes.csv has the official fix for that day, that is used instead.
  4. Runs two tests and compares them against placebo levels:
       Respect test : when price comes back to the level, does it bounce more than it
                      would at an ordinary level?
       Magnet test  : when price drifts away from the level, does it tend to come back?
  5. Writes docs/data/summary.json for the dashboard.
"""
from __future__ import annotations

import json
import math
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
OUT = os.path.join(ROOT, "docs", "data")

PRICES_CSV = os.path.join(DATA, "prices_5m.csv")
SPOT_CSV = os.path.join(DATA, "spot_snapshots.csv")
FIXES_CSV = os.path.join(DATA, "lbma_fixes.csv")

LONDON = "Europe/London"
FIX_TIMES = {"AM": "10:30", "PM": "15:00"}
# Placebo "fix" times: same method, ordinary times of day
PLACEBO_TIMES = {"P1": "12:45", "P2": "19:00", "P3": "08:15"}
PLACEBO_OFFSETS = [-0.010, -0.0075, -0.005, -0.0025, 0.0025, 0.005, 0.0075, 0.010]

BAND = 0.001          # +/-0.10% of level counts as "at the level" (~$4 at $4,000)
WINDOW_BARS = 12 * 24  # follow each level for 24 hours of 5-min bars
LOOKAHEAD = 12        # 1 hour to decide bounce vs break
MAGNET_H = 24         # 2-hour forward return for the magnet test
CHART_START = "2026-10-09"   # chart begins here and keeps growing
UA = {"User-Agent": "Mozilla/5.0 (lbma-gold-fix-monitor; research project)"}


# ----------------------------------------------------------------------------- data
def fetch_yahoo(symbol: str = "GC=F", rng: str = "60d") -> pd.DataFrame:
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
    r = requests.get(url, params={"interval": "5m", "range": rng}, headers=UA, timeout=30)
    r.raise_for_status()
    res = r.json()["chart"]["result"][0]
    q = res["indicators"]["quote"][0]
    df = pd.DataFrame(
        {
            "ts": pd.to_datetime(res["timestamp"], unit="s", utc=True),
            "open": q["open"], "high": q["high"], "low": q["low"], "close": q["close"],
        }
    ).dropna()
    return df


def update_prices() -> pd.DataFrame:
    old = None
    if os.path.exists(PRICES_CSV):
        old = pd.read_csv(PRICES_CSV, parse_dates=["ts"])
        old["ts"] = pd.to_datetime(old["ts"], utc=True)
    try:
        new = fetch_yahoo(rng="5d" if old is not None and len(old) > 1000 else "60d")
    except Exception as e:  # keep going with what we have
        print("price fetch failed:", e)
        new = None
    frames = [f for f in (old, new) if f is not None]
    if not frames:
        raise SystemExit("no price data available")
    df = pd.concat(frames).drop_duplicates("ts", keep="last").sort_values("ts")
    df = df.reset_index(drop=True)
    df.to_csv(PRICES_CSV, index=False, float_format="%.2f")
    return df


def snapshot_spot(last_fut: float) -> pd.DataFrame:
    rows = pd.read_csv(SPOT_CSV) if os.path.exists(SPOT_CSV) else pd.DataFrame(
        columns=["ts", "spot", "futures"])
    try:
        r = requests.get("https://api.gold-api.com/price/XAU", headers=UA, timeout=20)
        r.raise_for_status()
        spot = float(r.json()["price"])
        now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        rows = pd.concat([rows, pd.DataFrame([{"ts": now, "spot": spot, "futures": last_fut}])])
        rows.to_csv(SPOT_CSV, index=False, float_format="%.2f")
    except Exception as e:
        print("spot snapshot failed:", e)
    return rows


def current_basis(snaps: pd.DataFrame) -> float:
    """futures - spot, median of recent snapshots (robust to stale ticks)."""
    if snaps is None or snaps.empty:
        return 0.0
    gap = (snaps["futures"].astype(float) - snaps["spot"].astype(float)).tail(48)
    gap = gap[gap.abs() < 150]  # ignore roll-day garbage
    return float(gap.median()) if len(gap) else 0.0


# ----------------------------------------------------------------------- fix levels
def levels_at(df: pd.DataFrame, hhmm: str) -> pd.DataFrame:
    """Close of the 5-min bar starting at hh:mm London, one per day."""
    loc = df["ts"].dt.tz_convert(LONDON)
    mask = loc.dt.strftime("%H:%M") == hhmm
    out = df.loc[mask, ["ts", "close"]].copy()
    out["date"] = loc[mask].dt.date.astype(str)
    out = out[pd.to_datetime(out["date"]).dt.dayofweek < 5]
    return out.rename(columns={"close": "level"})


def build_fix_levels(df: pd.DataFrame, basis: float) -> pd.DataFrame:
    official = None
    if os.path.exists(FIXES_CSV):
        official = pd.read_csv(FIXES_CSV, comment="#")
        official["date"] = pd.to_datetime(official["date"]).dt.date.astype(str)
        official = official.set_index("date")
    rows = []
    for kind, hhmm in FIX_TIMES.items():
        lv = levels_at(df, hhmm)
        for _, r in lv.iterrows():
            src, level_fut = "proxy", float(r["level"])
            col = kind.lower()
            if official is not None and r["date"] in official.index and col in official.columns:
                v = official.loc[r["date"], col]
                if pd.notna(v):
                    src, level_fut = "official", float(v) + basis  # spot fix -> futures terms
            rows.append({"kind": kind, "date": r["date"], "ts": r["ts"],
                         "level_fut": level_fut, "level_spot": level_fut - basis, "source": src})
    return pd.DataFrame(rows).sort_values("ts").reset_index(drop=True)


# ------------------------------------------------------------------------- tests
def touch_outcomes(px: pd.DataFrame, start: int, level: float) -> list[str]:
    """Walk forward from bar `start`; every time price comes back to the level after
    being away, classify what happens next as bounce / break / chop."""
    b = level * BAND
    end = min(start + WINDOW_BARS, len(px) - LOOKAHEAD)
    hi, lo, cl = px["high"].values, px["low"].values, px["close"].values
    outcomes, away_side, i = [], 0, start
    while i < end:
        c = cl[i]
        if abs(c - level) > 2 * b:
            away_side = 1 if c > level else -1
            i += 1
            continue
        touched = lo[i] <= level + b and hi[i] >= level - b
        if away_side != 0 and touched:
            s = away_side
            res = "chop"
            for j in range(i + 1, i + 1 + LOOKAHEAD):
                if s * (cl[j] - level) < -2 * b:
                    res = "break"
                    break
                if s * ((hi[j] if s > 0 else lo[j]) - level) >= 3 * b:
                    res = "bounce"
                    break
            outcomes.append(res)
            away_side = 0
            i += LOOKAHEAD  # one touch per hour, avoid double counting
            continue
        i += 1
    return outcomes


def two_prop_z(k1, n1, k2, n2):
    if min(n1, n2) == 0:
        return None, None
    p = (k1 + k2) / (n1 + n2)
    se = math.sqrt(p * (1 - p) * (1 / n1 + 1 / n2)) if 0 < p < 1 else None
    if not se:
        return None, None
    z = (k1 / n1 - k2 / n2) / se
    pval = math.erfc(abs(z) / math.sqrt(2))
    return z, pval


def respect_test(px: pd.DataFrame, levels: pd.DataFrame, placebo_levels: pd.DataFrame):
    idx = pd.Index(px["ts"])

    def run(lv: pd.DataFrame, offsets=(0.0,)):
        res = []
        for _, r in lv.iterrows():
            pos = idx.get_indexer([r["ts"]])[0]
            if pos < 0:
                continue
            for off in offsets:
                res += touch_outcomes(px, pos + 3, r["level_fut"] * (1 + off))  # skip 15 min
        return res

    fix = run(levels)
    plc = run(placebo_levels) + run(levels, PLACEBO_OFFSETS)

    def pack(o):
        n = len(o)
        k = sum(1 for x in o if x == "bounce")
        br = sum(1 for x in o if x == "break")
        return {"touches": n, "bounces": k, "breaks": br,
                "bounce_rate": round(k / n, 4) if n else None}

    f, p = pack(fix), pack(plc)
    z, pval = two_prop_z(f["bounces"], f["touches"], p["bounces"], p["touches"])
    return {"fix": f, "placebo": p, "z": None if z is None else round(z, 2),
            "p_value": None if pval is None else round(pval, 4)}


def magnet_beta(px: pd.DataFrame, lv: pd.DataFrame):
    """OLS of 2h forward return on distance from the level (both in % of level).
    beta < 0 => price drifts back toward the level."""
    idx = pd.Index(px["ts"])
    cl = px["close"].values
    xs, ys = [], []
    for _, r in lv.iterrows():
        pos = idx.get_indexer([r["ts"]])[0]
        if pos < 0:
            continue
        L = r["level_fut"]
        for t in range(pos + 3, min(pos + WINDOW_BARS, len(cl) - MAGNET_H), MAGNET_H):
            xs.append((cl[t] - L) / L)
            ys.append((cl[t + MAGNET_H] - cl[t]) / L)
    n = len(xs)
    if n < 10:
        return {"n": n, "beta": None, "t": None}
    x, y = np.array(xs), np.array(ys)
    X = np.column_stack([np.ones(n), x])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ coef
    s2 = resid @ resid / (n - 2)
    se = math.sqrt(s2 * np.linalg.inv(X.T @ X)[1, 1])
    return {"n": n, "beta": round(float(coef[1]), 4), "t": round(float(coef[1] / se), 2)}


def placebo_levels(df: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for name, hhmm in PLACEBO_TIMES.items():
        lv = levels_at(df, hhmm).rename(columns={"level": "level_fut"})
        lv["kind"] = name
        parts.append(lv)
    return pd.concat(parts).sort_values("ts").reset_index(drop=True)


def verdict(resp, mag_fix, mag_plc) -> dict:
    lines = []
    f, p = resp["fix"], resp["placebo"]
    respect = False
    if f["touches"] and p["touches"] and resp["p_value"] is not None:
        diff = (f["bounce_rate"] - p["bounce_rate"]) * 100
        respect = diff > 0 and resp["p_value"] < 0.05
        lines.append(
            f"At the fix level price bounced {f['bounce_rate']*100:.0f}% of the time "
            f"({f['touches']} touches) vs {p['bounce_rate']*100:.0f}% at placebo levels "
            f"({diff:+.0f} pts, p={resp['p_value']}).")
    magnet = False
    if mag_fix["beta"] is not None and mag_plc["beta"] is not None:
        magnet = mag_fix["t"] < -2 and mag_fix["beta"] < mag_plc["beta"]
        lines.append(
            f"Magnet beta: fix {mag_fix['beta']} (t={mag_fix['t']}) vs placebo "
            f"{mag_plc['beta']} (t={mag_plc['t']}). Negative = price pulled back toward the level.")
    if respect and magnet:
        head = "YES - price respects the LBMA fix level and gets pulled back to it."
    elif respect:
        head = "PARTLY - price bounces off the fix more than off random levels, but no magnet effect."
    elif magnet:
        head = "PARTLY - price drifts back toward the fix, but doesn't bounce off it more than random levels."
    else:
        head = "NOT YET - no statistically clear edge; the fix behaves like an ordinary price level so far."
    return {"headline": head, "details": lines, "respect": respect, "magnet": magnet}


# ---------------------------------------------------------------------------- main
def main():
    os.makedirs(OUT, exist_ok=True)
    df = update_prices()
    snaps = snapshot_spot(float(df["close"].iloc[-1]))
    basis = current_basis(snaps)

    fixes = build_fix_levels(df, basis)
    plc = placebo_levels(df)

    resp = respect_test(df, fixes, plc)
    resp_am = respect_test(df, fixes[fixes.kind == "AM"], plc)
    resp_pm = respect_test(df, fixes[fixes.kind == "PM"], plc)
    mag_fix = magnet_beta(df, fixes)
    mag_plc = magnet_beta(df, plc)

    # chart data (spot terms): fixed start date, grows every day from there.
    # last 7 days at 5-min, older history thinned to 30-min to keep the file small
    cutoff = pd.Timestamp(CHART_START, tz="UTC")
    hist = df[df["ts"] >= cutoff]
    fine_from = df["ts"].iloc[-1] - pd.Timedelta(days=7)
    old_part = hist[hist["ts"] < fine_from]
    old_part = old_part[old_part["ts"].dt.minute % 30 == 0]
    recent = pd.concat([old_part, hist[hist["ts"] >= fine_from]])
    rec_fix = fixes[fixes["ts"] >= cutoff - pd.Timedelta(days=3)]

    latest = {k: None for k in FIX_TIMES}
    for kind in FIX_TIMES:
        sub = fixes[fixes.kind == kind]
        if len(sub):
            r = sub.iloc[-1]
            latest[kind] = {"date": r["date"], "level_spot": round(r["level_spot"], 2),
                            "source": r["source"]}

    last_spot = None
    if snaps is not None and len(snaps):
        last_spot = {"ts": str(snaps.iloc[-1]["ts"]), "spot": float(snaps.iloc[-1]["spot"])}

    summary = {
        "updated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "data_from": df["ts"].iloc[0].strftime("%Y-%m-%d"),
        "data_to": df["ts"].iloc[-1].strftime("%Y-%m-%dT%H:%MZ"),
        "bars": int(len(df)),
        "fix_days": int(fixes["date"].nunique()),
        "official_fixes_used": int((fixes["source"] == "official").sum()),
        "basis_fut_minus_spot": round(basis, 2),
        "last_spot_snapshot": last_spot,
        "latest_fix": latest,
        "respect": {"all": resp, "AM": resp_am, "PM": resp_pm},
        "magnet": {"fix": mag_fix, "placebo": mag_plc},
        "verdict": verdict(resp, mag_fix, mag_plc),
        "settings": {"band_pct": BAND * 100, "window_hours": WINDOW_BARS / 12,
                     "lookahead_min": LOOKAHEAD * 5, "magnet_horizon_min": MAGNET_H * 5},
        "fix_history": [
            {"date": d,
             "am": (lambda s: round(float(s.iloc[0]), 2) if len(s) else None)(
                 g.loc[g.kind == "AM", "level_spot"]),
             "pm": (lambda s: round(float(s.iloc[0]), 2) if len(s) else None)(
                 g.loc[g.kind == "PM", "level_spot"]),
             "official": bool((g["source"] == "official").any())}
            for d, g in fixes.groupby("date", sort=True)
        ],
        "chart": {
            "t": [int(t.timestamp()) for t in recent["ts"]],
            "spot": [round(float(c) - basis, 2) for c in recent["close"]],
            "fixes": [{"kind": r["kind"], "t": int(r["ts"].timestamp()),
                       "level": round(r["level_spot"], 2), "source": r["source"]}
                      for _, r in rec_fix.iterrows()],
        },
    }
    with open(os.path.join(OUT, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    print(json.dumps(summary["verdict"], indent=1))


if __name__ == "__main__":
    main()
