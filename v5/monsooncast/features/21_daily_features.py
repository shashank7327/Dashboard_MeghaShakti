r"""
v5/monsooncast/features/21_daily_features.py  —  district features at DAILY
resolution, so the dashboard can be scrubbed day by day instead of only month
by month.

WHY ROLLING WINDOWS AND NOT DAILY VALUES
  A single day's rainfall departure is close to meaningless: most district-days
  are dry, so the ratio is either -100% or an enormous positive number, and the
  map would flicker between the two.  What a trader or an agronomist actually
  watches is the RUNNING TOTAL against normal, so the layers here are rolling
  and cumulative windows ending on the selected day:

      dep_7d       last 7 days vs the normal for those same 7 calendar days
      dep_30d      last 30 days, likewise
      dep_season   1 June to the selected day (the monsoon season to date)

  Every window is day-matched by construction -- the normal covers exactly the
  same calendar days as the actual -- so the partial-month problem that
  corrupted the monthly panel cannot arise here.  A window ending today is
  complete on both sides by definition.  This is the same reasoning that fixed
  `normal_mm` in step 01, applied from the start rather than as a repair.

NORMALS
  Day-of-year climatology over 1971-2020, IMD's own LPA window, smoothed with a
  centred 15-day window.  Smoothing matters: a raw single-calendar-day mean over
  50 years is noisy enough that the departure would inherit the noise of the
  denominator, and 29 February would be built from ~12 years rather than 50.

SOURCES AND THEIR EDGES
  IMD rain / tmax / tmin   daily, to the last published day
  ERA5-Land AET/PET/SM     daily, but lags IMD by about a week
  Each layer is written only as far as its own source runs, and the export
  records that edge per layer so the dashboard can say what is available when.

OUTPUT -> v5/data_lgd/
  daily_features_lgd.csv        district x day, recent years (analysis format)
  daily_layers_lgd.json         compact payload for the dashboard

Run:  py -3.13 -X utf8 "v5/monsooncast/features/21_daily_features.py"
"""
import json
import pathlib
import sys

import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
V5 = HERE.parents[1]
ROOT = HERE.parents[2]
IMD = ROOT / "IMD_Data"
OUTD = V5 / "data_lgd"
sys.path.insert(0, str(V5))
from common_v5 import log  # noqa

NORM_LO, NORM_HI = 1971, 2020      # IMD's own LPA window
SMOOTH = 15                        # centred day-of-year smoothing, days
CSV_FROM = 2015                    # how far back the CSV goes
DASH_DAYS = 120                    # days carried into the dashboard payload
MONSOON_START = (6, 1)             # 1 June, for season-to-date
MIN_NORM = 5.0                     # mm; below this a departure is undefined


def doy_climatology(w):
    """Smoothed day-of-year normal, shape (366, n_districts)."""
    b = w[(w.index.year >= NORM_LO) & (w.index.year <= NORM_HI)]
    g = b.groupby(b.index.dayofyear).mean()
    g = g.reindex(range(1, 367))
    # wrap-around smoothing so 1 January borrows from late December
    trip = pd.concat([g, g, g], ignore_index=True)
    sm = trip.rolling(SMOOTH, center=True, min_periods=1).mean()
    out = sm.iloc[len(g):2 * len(g)].reset_index(drop=True)
    out.index = range(1, 367)
    return out


def expand(clim, idx):
    """Day-of-year climatology broadcast onto a real date index."""
    return pd.DataFrame(clim.reindex(idx.dayofyear).to_numpy(),
                        index=idx, columns=clim.columns)


def roll_dep(act, nrm, win):
    """Rolling-window departure, %, day-matched on both sides."""
    a = act.rolling(win, min_periods=max(3, win // 2)).sum()
    n = nrm.rolling(win, min_periods=max(3, win // 2)).sum()
    return (100.0 * (a - n) / n.where(n >= MIN_NORM)).replace(
        [np.inf, -np.inf], np.nan)


def season_dep(act, nrm):
    """1 June -> day departure, %, reset each year."""
    y = act.index.year
    mark = ((act.index.month > MONSOON_START[0])
            | ((act.index.month == MONSOON_START[0])
               & (act.index.day >= MONSOON_START[1])))
    grp = np.where(mark, y, y - 1)          # a season belongs to its June
    a = act.groupby(grp).cumsum()
    n = nrm.groupby(grp).cumsum()
    dep = 100.0 * (a - n) / n.where(n >= MIN_NORM)
    dep[~mark] = np.nan                     # only meaningful inside the season
    return dep.replace([np.inf, -np.inf], np.nan)


def load_era5_daily(folder, cols):
    reg = pd.read_csv(IMD / "registry_lgd791.csv")
    sys.path.insert(0, str(IMD))
    from build_crosswalk import norm_state, norm_key  # noqa
    lut = {(s, d): i for s, d, i in zip(norm_state(reg["state"]),
                                        norm_key(reg["district"]),
                                        reg["district_id"])}
    fr = []
    for f in sorted((ROOT / folder).glob("*.csv")):
        y = "".join(c for c in f.stem if c.isdigit())[-4:]
        if y and int(y) < CSV_FROM:
            continue
        d = pd.read_csv(f, parse_dates=["date"])
        d["did"] = [lut.get((a, b), -1) for a, b in
                    zip(norm_state(d["state"]), norm_key(d["district"]))]
        fr.append(d[d["did"] >= 0][["date", "did"] + cols])
    if not fr:
        return None
    return pd.concat(fr, ignore_index=True)


def main():
    log("=" * 74)
    log("DAILY FEATURES — rolling, day-matched district layers")
    log("=" * 74)

    rain = pd.read_pickle(IMD / "_district_daily_rain_lgd.pkl")
    tmax = pd.read_pickle(IMD / "_district_daily_tmax_lgd.pkl")
    tmin = pd.read_pickle(IMD / "_district_daily_tmin_lgd.pkl")
    for d in (rain, tmax, tmin):
        d.columns = d.columns.astype(int)
    log(f"  IMD daily: rain to {rain.index.max():%Y-%m-%d}, "
        f"tmax to {tmax.index.max():%Y-%m-%d}, {rain.shape[1]} districts")

    rn = expand(doy_climatology(rain), rain.index)
    xn = expand(doy_climatology(tmax), tmax.index)
    nn = expand(doy_climatology(tmin), tmin.index)
    log(f"  day-of-year normals: {NORM_LO}-{NORM_HI}, {SMOOTH}-day smoothing")

    lay = {
        "rain_mm": rain,
        "dep_7d": roll_dep(rain, rn, 7),
        "dep_30d": roll_dep(rain, rn, 30),
        "dep_season": season_dep(rain, rn),
        "tmax_anom_d": tmax - xn,
        "tmin_anom_d": tmin - nn,
        "sdd_7d": (tmax - 34).clip(lower=0).rolling(7, min_periods=3).sum(),
    }

    # ERA5 daily layers, which stop earlier than IMD
    er = load_era5_daily("SM_LGD", ["swvl2"])
    if er is not None:
        w = er.pivot_table(index="date", columns="did", values="swvl2",
                           aggfunc="mean")
        lay["swvl2_d"] = w.reindex(columns=rain.columns)
        log(f"  ERA5 soil moisture to {w.index.max():%Y-%m-%d} "
            f"(lags IMD by {(rain.index.max()-w.index.max()).days} days)")

    # ---- CSV: the analysis format ----------------------------------------
    keep = rain.index[rain.index.year >= CSV_FROM]
    frames = []
    for name, df in lay.items():
        s = df.reindex(keep).stack(future_stack=True).rename(name)
        frames.append(s)
    D = pd.concat(frames, axis=1).reset_index()
    D.columns = ["date", "district_id"] + list(lay)
    reg = pd.read_csv(IMD / "registry_lgd791.csv")[
        ["district_id", "state", "district"]]
    D = D.merge(reg, on="district_id", how="left")
    # `year` lets the master exporter window the workbook by year the same way
    # it does the monthly tables; without it the XLSX falls back to a blind
    # tail-truncation that keeps an arbitrary slice of districts.
    D["year"] = pd.to_datetime(D["date"]).dt.year
    D = D[["date", "year", "district_id", "state", "district"] + list(lay)]
    for c in lay:
        D[c] = D[c].astype(float).round(2)
    D.to_csv(OUTD / "daily_features_lgd.csv", index=False)
    log(f"  daily_features_lgd.csv: {len(D):,} rows x {D.shape[1]} cols "
        f"({CSV_FROM}-{keep.max().year})")

    # ---- compact payload for the dashboard --------------------------------
    # Positional arrays aligned to one district-id list, not {id: value} maps:
    # 120 days x 791 districts x 8 layers is ~760k numbers, and the dict form
    # would roughly triple the byte count for no added information.
    days = rain.index[-DASH_DAYS:]
    ids = [int(c) for c in rain.columns]
    payload = {"ids": ids, "dates": [d.strftime("%Y-%m-%d") for d in days],
               "layers": {}, "edge": {}, "agg": {}, "method": {}}

    # All-India and per-state values, computed here rather than in the browser
    # for the same reason as the monthly panel: a rainfall departure is a RATIO,
    # so it has to be rebuilt from its own numerator and denominator.  Averaging
    # the district percentages answers a different question and gives a
    # materially different answer.
    reg = pd.read_csv(IMD / "registry_lgd791.csv")
    area = reg.set_index("district_id")["area_km2"].astype(float).reindex(
        rain.columns).to_numpy()
    stt = reg.set_index("district_id")["state"].reindex(rain.columns).to_numpy()
    states = sorted(set(str(s) for s in stt if pd.notna(s)))
    smask = {s: (stt == s) for s in states}
    # numerator/denominator pairs for the ratio layers
    win = {"dep_7d": 7, "dep_30d": 30}
    num_den = {}
    for k, w in win.items():
        num_den[k] = (rain.rolling(w, min_periods=max(3, w // 2)).sum(),
                      rn.rolling(w, min_periods=max(3, w // 2)).sum())
    ymark = ((rain.index.month > MONSOON_START[0])
             | ((rain.index.month == MONSOON_START[0])
                & (rain.index.day >= MONSOON_START[1])))
    sgrp = np.where(ymark, rain.index.year, rain.index.year - 1)
    num_den["dep_season"] = (rain.groupby(sgrp).cumsum(),
                             rn.groupby(sgrp).cumsum())

    def ratio_over(nu, de, m):
        d = np.nansum(de * area * m)
        return (float(100.0 * (np.nansum(nu * area * m) - d) / d)
                if d > 0 else None)

    def wmean_over(v, m):
        ok = np.isfinite(v) & m & np.isfinite(area)
        return (float((v[ok] * area[ok]).sum() / area[ok].sum())
                if ok.any() else None)

    for name, df in lay.items():
        sub = df.reindex(index=days, columns=rain.columns)
        payload["layers"][name] = [
            [None if not np.isfinite(v) else round(float(v), 1) for v in row]
            for row in sub.to_numpy()]
        last = df.dropna(how="all").index.max()
        payload["edge"][name] = (None if pd.isna(last)
                                 else last.strftime("%Y-%m-%d"))
        isr = name in num_den
        payload["method"][name] = ("area-weighted ratio of sums" if isr
                                   else "area-weighted mean")
        nat, byst = [], []
        for d in days:
            if isr:
                nu = num_den[name][0].loc[d].to_numpy(float)
                de = num_den[name][1].loc[d].to_numpy(float)
                nat.append(ratio_over(nu, de, np.ones(len(area), bool)))
                byst.append({s: ratio_over(nu, de, smask[s]) for s in states})
            else:
                v = sub.loc[d].to_numpy(float)
                nat.append(wmean_over(v, np.ones(len(area), bool)))
                byst.append({s: wmean_over(v, smask[s]) for s in states})
        payload["agg"][name] = {
            "national": [None if v is None or not np.isfinite(v)
                         else round(v, 2) for v in nat],
            "state": [{k: round(x, 2) for k, x in s.items()
                       if x is not None and np.isfinite(x)} for s in byst]}
    p = OUTD / "daily_layers_lgd.json"
    p.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    log(f"  daily_layers_lgd.json: {len(days)} days x {len(ids)} districts "
        f"x {len(lay)} layers ({p.stat().st_size/1e6:.1f} MB)")
    log("  per-layer last day: " + ", ".join(
        f"{k} {v}" for k, v in payload["edge"].items() if v))

    # ---- sanity: the season-to-date must agree with the monthly panel ------
    last = days[-1]
    sd = lay["dep_season"].loc[last]
    ar = pd.read_csv(IMD / "registry_lgd791.csv").set_index(
        "district_id")["area_km2"]
    m = sd.notna()
    aw = float((sd[m] * ar.reindex(sd.index)[m]).sum()
               / ar.reindex(sd.index)[m].sum())
    log(f"\n  season-to-date (1 Jun -> {last:%d %b}) area-weighted mean of the "
        f"district departures: {aw:+.1f}%")
    log(f"  wrote daily_features_lgd.csv and daily_layers_lgd.json")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
