r"""v5/monsooncast/validation/28_audit_training_data.py  —  interrogate the
sample matrix the models are actually fitted on.

WHY THIS EXISTS
  Every score in this project is computed on a frame nobody looks at. The
  bake-off prints skill and the dashboard prints a forecast, and both would
  print something plausible if a column had silently become constant, if the
  target had inverted, if a join had filled a feature with the wrong month, or
  if a unit change upstream had shifted a whole block by 273.15. This project
  has met every one of those in some form.

  The failures that survive are the ones that produce PLAUSIBLE NUMBERS. So
  this file checks the things a plausible number cannot hide.

WHAT IT CHECKS
  1  SHAPE        rows, columns, period, districts, train/test split sizes
  2  TARGET       distribution against the physical bound (-100% floor), and
                  that train and test are drawn from the same kind of thing
  3  DEGENERACY   constant columns, all-NaN columns, single-value columns --
                  a feature with no variance contributes nothing and usually
                  means a join failed
  4  MISSINGNESS  per-column NaN rate, and whether it differs between train
                  and test (a feature present in training and absent at test
                  time is a silent skill killer)
  5  LEAKAGE      correlation of every feature with the FORWARD target
                  compared with its correlation to the BACKWARD equivalent.
                  A feature that knows the future better than the past is the
                  signature of the look-ahead this project already had once.
  6  MJO          that the daily join actually landed: coverage, amplitude
                  range, phase spread, and that the phase composite has the
                  physical sign (Indian Ocean phases dry, W Pacific wet)
  7  RANGES       every feature against a physically admissible interval,
                  so a unit slip or a double conversion is caught by name

Run:  py -3.13 -X utf8 "v5/monsooncast/validation/28_audit_training_data.py"
"""
import pathlib
import sys

import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
V5 = HERE.parents[1]
OUTD = V5 / "data_lgd"
sys.path.insert(0, str(V5))
from common_v5 import log  # noqa

CACHE = sorted(OUTD.glob("_bakeoff_samples_*.pkl"))
TRAIN_END, TEST_START = 2016, 2020

#   Physically admissible intervals. A value outside these is not a bad
#   forecast, it is a broken pipeline.
BOUNDS = {
    "dep_7": (-100, 5000), "dep_14": (-100, 5000),
    "ant_dep_7": (-100, 20000), "ant_dep_14": (-100, 20000),
    "ant_dep_30": (-100, 20000), "ant_dep_60": (-100, 20000),
    "ant_dep_90": (-100, 20000),
    "ant_rain_7": (0, 5000), "ant_rain_90": (0, 20000),
    "wet30": (0, 30), "extreme30": (0, 30), "dsw": (0, 400),
    "doy_sin": (-1, 1), "doy_cos": (-1, 1),
    "spei_era5_1": (-5, 5), "spei_era5_4": (-5, 5), "spei_era5_12": (-5, 5),
    "spei_harg_4": (-5, 5),
    "mai": (0, 1.5), "sdd": (0, 1000), "gdd_kharif": (0, 1500),
    "tmax_anom": (-20, 20), "tmin_anom": (-20, 20),
    "swvl1": (0, 1), "swvl2": (0, 1), "swvl3": (0, 1), "swvl4": (0, 1),
    "aet_mm": (0, 1000), "pet_mm": (0, 1000),
    "oni": (-4, 4), "nino34_anom": (-5, 5), "iod_dmi": (-3, 3),
    "mjo_amp": (0, 6), "mjo_sin": (-1, 1), "mjo_cos": (-1, 1),
    "romi1": (-5, 5), "romi2": (-5, 5),
}


def main():
    log("=" * 78)
    log("TRAINING-DATA AUDIT — is the sample matrix real, or plausible junk?")
    log("=" * 78)
    if not CACHE:
        log("  no _bakeoff_samples_*.pkl — run modelling/08_model_bakeoff.py")
        return
    p = CACHE[-1]
    d = pd.read_pickle(p)
    fails = []

    # ---- 1 shape --------------------------------------------------------
    log(f"\n[1] SHAPE   {p.name}")
    log(f"    {len(d):,} rows x {d.shape[1]} cols")
    log(f"    {d['date'].min():%Y-%m-%d} .. {d['date'].max():%Y-%m-%d}, "
        f"{d['district_id'].nunique()} districts")
    tr, te = d[d.year <= TRAIN_END], d[d.year >= TEST_START]
    gap = d[(d.year > TRAIN_END) & (d.year < TEST_START)]
    log(f"    train <={TRAIN_END}: {len(tr):,}   "
        f"validation {TRAIN_END+1}-{TEST_START-1}: {len(gap):,}   "
        f"test >={TEST_START}: {len(te):,}")
    if not len(tr) or not len(te):
        fails.append("train or test split is empty")

    #   Every district must appear on both sides, or the model is being asked
    #   at test time about places it never saw.
    miss = set(te["district_id"]) - set(tr["district_id"])
    log(f"    districts in test but never in train: {len(miss)}"
        + (f"  {sorted(miss)[:6]}" if miss else ""))
    if miss:
        fails.append(f"{len(miss)} districts appear only at test time")

    # ---- 2 target -------------------------------------------------------
    log("\n[2] TARGET")
    for H in (7, 14):
        c = f"dep_{H}"
        if c not in d:
            continue
        s = d[c].dropna()
        below = (s < -100.001).sum()
        log(f"    {c}: n={len(s):,}  mean {s.mean():+.2f}  sd {s.std():.2f}  "
            f"min {s.min():+.1f}  p99 {s.quantile(.99):+.1f}  "
            f"max {s.max():+.1f}")
        log(f"        below the -100% physical floor: {below}")
        if below:
            fails.append(f"{c} has {below} values below -100%, "
                         f"which is less rain than no rain")
        a, b = tr[c].dropna(), te[c].dropna()
        log(f"        train mean {a.mean():+.2f} sd {a.std():.1f}  |  "
            f"test mean {b.mean():+.2f} sd {b.std():.1f}")
        #   A large shift between train and test is not automatically wrong --
        #   climate drifts -- but a factor-of-two change in spread means the
        #   two halves are not the same measurement.
        if a.std() > 0 and not (0.5 < b.std() / a.std() < 2.0):
            fails.append(f"{c}: test spread is {b.std()/a.std():.2f}x train "
                         f"— the halves may not be the same quantity")

    # ---- 3 degeneracy ---------------------------------------------------
    log("\n[3] DEGENERACY")
    num = d.select_dtypes(include=[np.number])
    allnan = [c for c in num.columns if num[c].isna().all()]
    const = [c for c in num.columns
             if not num[c].isna().all() and num[c].nunique(dropna=True) <= 1]
    log(f"    all-NaN columns: {len(allnan)} {allnan[:8]}")
    log(f"    constant columns: {len(const)} {const[:8]}")
    if allnan:
        fails.append(f"all-NaN columns: {allnan[:8]}")
    if const:
        fails.append(f"constant columns carry no information: {const[:8]}")

    # ---- 4 missingness --------------------------------------------------
    log("\n[4] MISSINGNESS  (train vs test, worst 12)")
    rows = []
    for c in num.columns:
        if c in ("year", "month", "fyear", "fmonth", "district_id"):
            continue
        rows.append((c, tr[c].isna().mean() * 100, te[c].isna().mean() * 100))
    rows.sort(key=lambda r: -max(r[1], r[2]))
    log(f"    {'column':<26}{'train %':>9}{'test %':>9}{'delta':>9}")
    for c, a, b in rows[:12]:
        flag = "  <-- differs" if abs(a - b) > 20 else ""
        log(f"    {c:<26}{a:9.1f}{b:9.1f}{b - a:+9.1f}{flag}")
        if abs(a - b) > 20:
            fails.append(f"{c}: {a:.0f}% missing in train vs {b:.0f}% in test")

    # ---- 5 leakage ------------------------------------------------------
    log("\n[5] LEAKAGE  — does any feature know the future better than the past?")
    log("    A predictor should correlate with what came BEFORE at least as")
    log("    strongly as with what comes AFTER. The reverse is the signature")
    log("    of the look-ahead this pipeline already carried once.")
    s = d.sample(min(200_000, len(d)), random_state=0)
    hits = []
    for c in num.columns:
        if c.startswith("dep_") or c.startswith("fwd") or c in (
                "year", "month", "fyear", "fmonth", "district_id"):
            continue
        v = s[c]
        if v.isna().all() or v.nunique(dropna=True) <= 1:
            continue
        fwd = v.corr(s["dep_7"])
        bwd = v.corr(s["ant_dep_7"]) if "ant_dep_7" in s else np.nan
        if np.isfinite(fwd) and np.isfinite(bwd) and abs(fwd) > 0.30 \
                and abs(fwd) > abs(bwd) * 1.5:
            hits.append((c, fwd, bwd))
    if hits:
        for c, f_, b_ in sorted(hits, key=lambda x: -abs(x[1]))[:8]:
            log(f"    ! {c:<24} corr(forward) {f_:+.3f}  "
                f"corr(backward) {b_:+.3f}")
        fails.append(f"{len(hits)} feature(s) correlate more with the future "
                     f"than the past — possible look-ahead")
    else:
        log("    none — no feature is suspiciously future-aware")

    # ---- 6 MJO ----------------------------------------------------------
    log("\n[6] MJO JOIN")
    if "mjo_amp" not in d.columns:
        log("    absent — the model has no intraseasonal information")
    else:
        cov = d["mjo_amp"].notna().mean() * 100
        log(f"    coverage {cov:.1f}%  "
            f"amp {d['mjo_amp'].min():.2f}..{d['mjo_amp'].max():.2f}")
        u = np.hypot(d["mjo_sin"], d["mjo_cos"])
        log(f"    unit-circle norm: {u.min():.3f}..{u.max():.3f} "
            f"(must be 1.000 where defined)")
        if np.nanmax(np.abs(u - 1)) > 0.01:
            fails.append("mjo_sin/mjo_cos are not on the unit circle")
        #   The physical test: with convection over the Indian Ocean the
        #   monsoon breaks; over the W Pacific it is active. If the join is
        #   misaligned this composite flattens.
        act = d[(d["mjo_amp"] > 1) & (d["month"].isin([6, 7, 8, 9]))]
        if len(act):
            ang = np.degrees(np.arctan2(act["romi2"], act["romi1"]))
            ph = (((ang + 180) % 360) // 45).astype(int) + 1
            g = act.groupby(ph)["dep_7"].mean()
            io = g.reindex([2, 3]).mean()
            wp = g.reindex([7, 8]).mean()
            log(f"    JJAS, amp>1: phases 2-3 (Indian Ocean) "
                f"{io:+.1f}%   phases 7-8 (W Pacific) {wp:+.1f}%")
            log(f"    spread across phases: {g.max() - g.min():.1f} pp")
            if not (wp > io):
                fails.append("MJO phase composite has the wrong sign — "
                             "the daily join may be misaligned")

    # ---- 7 ranges -------------------------------------------------------
    log("\n[7] PHYSICAL RANGES")
    bad = []
    for c, (lo, hi) in BOUNDS.items():
        if c not in d.columns:
            continue
        v = d[c].dropna()
        if v.empty:
            continue
        if v.min() < lo - 1e-6 or v.max() > hi + 1e-6:
            bad.append((c, float(v.min()), float(v.max()), lo, hi))
    if bad:
        for c, mn, mx, lo, hi in bad:
            log(f"    ! {c:<20} {mn:+12.3f}..{mx:+12.3f}   "
                f"expected [{lo}, {hi}]")
        fails.append(f"{len(bad)} column(s) outside physical range")
    else:
        log(f"    all {sum(1 for c in BOUNDS if c in d.columns)} checked "
            f"columns within physical bounds")

    # ---- verdict --------------------------------------------------------
    log("\n" + "=" * 78)
    if fails:
        log(f"{len(fails)} PROBLEM(S):")
        for f_ in fails:
            log(f"  - {f_}")
        sys.exit(1)
    log("PASS — the sample matrix is internally consistent, physically "
        "bounded,")
    log("       balanced across the split, and shows no look-ahead.")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
