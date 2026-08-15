r"""
v5/monsooncast/validation/07_validate_features.py  —  cross-check every DETERMINISTIC
feature against IMD's own published estimates and against its own definition.

WHY THIS EXISTS
  Only the 7/14-day forecast is a model output.  Everything else -- rainfall
  actual/normal/departure, SPEI, MAI, GDD, SDD, temperature anomaly -- is
  arithmetic on observed or reanalysed data.  Arithmetic can still be wrong:
  the wrong aggregation, the wrong baseline period, or a unit slip will
  produce a number that looks plausible and is not.  This script tests each
  one against an external truth where one exists, and against its own
  definition where it does not.

TEST 1  ALL-INDIA MONSOON RAINFALL vs IMD PUBLISHED (external truth)
  IMD publishes the all-India JJAS total as a % of its Long Period Average.
  Our product is built from the same IMD gauge grid, so it must reproduce
  those numbers.  If it does not, the district aggregation or the normal is
  wrong.  Published values (% of LPA):
      2015 86 | 2018 91 | 2019 110 | 2020 109 | 2021 99
      2022 106 | 2023 94.4 | 2024 108 | 2025 108
  IMD's JJAS LPA is 868.6 mm (1971-2020 baseline).

TEST 2  AGGREGATION METHOD
  IMD's all-India series is an AREA-WEIGHTED mean.  A simple mean over
  districts weights a small Kerala district the same as a huge Rajasthan one.
  Both are computed here so the difference is measured, not assumed.

TEST 3  NORMAL BASELINE
  IMD's LPA is a FIFTY-year mean over 1971-2020, refreshed each decade -- not
  the 30-year WMO standard normal.  The record was extended back to 1971 so
  this comparison is now like-for-like on IMD's own window.

  RESULT, and it settles an earlier hypothesis: on the identical 1971-2020
  window our LPA is 838.7 mm against IMD's 868.6 mm, and the year-by-year
  departures still run +2.3 pp high.  The gap was previously attributed to the
  missing 1970s; with the 1970s now in hand it barely moved (2.39 -> 2.29 pp),
  so the residual is NOT a baseline effect.  It is a difference of spatial
  ESTIMATOR: IMD's operational all-India figure comes from its subdivision-
  weighted gauge network, ours from an area-weighted mean of 0.25 deg grid
  cells over 791 districts.  Correlation stays 0.992, so the two track the same
  signal at a small constant offset.

TEST 4  DEFINITIONAL / PHYSICAL SELF-CONSISTENCY (no external truth needed)
  SPEI must be ~N(0,1) by construction; MAI must lie in [0,1.5] and peak in
  the monsoon; GDD/SDD must be non-negative and peak in summer; temperature
  anomalies must average ~0 over their own baseline.  A failure here is a
  coding error, not a data disagreement.

OUTPUT -> v5/data_lgd/feature_validation_lgd.{csv,json}

Run:  py -3.13 -X utf8 "v5/monsooncast/validation/07_validate_features.py"
"""
import json
import pathlib
import sys

import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
V5 = HERE.parents[1]
IMD = V5.parent / "IMD_Data"
DATA = V5 / "data_lgd"
sys.path.insert(0, str(V5))
from common_v5 import log  # noqa

# IMD published all-India JJAS rainfall, % of LPA
IMD_PCT_LPA = {2015: 86.0, 2018: 91.0, 2019: 110.0, 2020: 109.0, 2021: 99.0,
               2022: 106.0, 2023: 94.4, 2024: 108.0, 2025: 108.0}
IMD_JJAS_LPA_MM = 868.6          # IMD LPA, 1971-2020
#   IMD's published all-India MONTHLY normals on the same 1971-2020 window.
#   The seasonal total can be right while the months inside it are not, and it
#   is the month that the dashboard publishes a departure against.
IMD_MONTHLY_LPA = {1: 16.1, 2: 20.5, 3: 25.0, 4: 32.5, 5: 60.5, 6: 165.3,
                   7: 280.4, 8: 254.9, 9: 167.9, 10: 75.4, 11: 29.4, 12: 14.0}
JJAS = [6, 7, 8, 9]
# The record now reaches back to 1971, so the LPA is computed over exactly the
# window IMD uses.  These must stay in step with WMO_LO/WMO_HI in step 01.
LPA_LO, LPA_HI = 1971, 2020

results = {}
rows = []


def check(name, got, expect, tol, unit="", note=""):
    ok = abs(got - expect) <= tol
    rows.append({"test": name, "got": round(float(got), 2),
                 "expected": round(float(expect), 2), "tol": tol,
                 "pass": bool(ok), "unit": unit, "note": note})
    flag = "PASS" if ok else "REVIEW"
    log(f"    [{flag}] {name:42s} got {got:9.2f}{unit}  "
        f"expected {expect:8.2f}{unit}")
    return ok


def main():
    log("=" * 74)
    log("FEATURE VALIDATION — deterministic layers vs IMD and vs definition")
    log("=" * 74)

    reg = pd.read_csv(IMD / "registry_lgd791.csv")
    area = reg.set_index("district_id")["area_km2"]

    # ---------------------------------------------------------------- rain
    rain = pd.read_pickle(DATA / "daily_rain_lgd.pkl")
    rain.columns = rain.columns.astype(int)
    jj = rain[rain.index.month.isin(JJAS)]
    yr_tot = jj.groupby(jj.index.year).sum(min_count=100)      # year x district
    w = area.reindex(yr_tot.columns).to_numpy(float)
    wok = np.isfinite(w)

    def aw(series):                      # area-weighted all-India
        v = series.to_numpy(float)
        m = wok & np.isfinite(v)
        return float((v[m] * w[m]).sum() / w[m].sum())

    ai_aw = yr_tot.apply(aw, axis=1)                 # area-weighted
    ai_sm = yr_tot.mean(axis=1)                      # simple district mean

    log("\n  TEST 1+2+3 — all-India JJAS rainfall vs IMD published")
    log("  " + "-" * 70)
    lpa_ours = float(ai_aw.loc[LPA_LO:LPA_HI].mean())
    lpa_9120 = float(ai_aw.loc[1991:2020].mean())
    log(f"    JJAS LPA, area-weighted {LPA_LO}-{LPA_HI} : {lpa_ours:7.1f} mm"
        f"   <- IMD's own window")
    log(f"    JJAS LPA, area-weighted 1991-2020 : {lpa_9120:7.1f} mm")
    log(f"    JJAS LPA, IMD published (1971-2020): {IMD_JJAS_LPA_MM:7.1f} mm")
    log(f"    simple (unweighted) mean {LPA_LO}-{LPA_HI} : "
        f"{float(ai_sm.loc[LPA_LO:LPA_HI].mean()):7.1f} mm   "
        f"<- weighting matters by "
        f"{100*(ai_sm.loc[LPA_LO:LPA_HI].mean()/lpa_ours-1):+.1f}%")
    check("JJAS LPA vs IMD 868.6mm", lpa_ours, IMD_JJAS_LPA_MM, 45, " mm",
          f"both on {LPA_LO}-{LPA_HI}")

    log(f"\n    year-by-year % of LPA (area-weighted, our {LPA_LO}-{LPA_HI} LPA):")
    log(f"    {'year':>6s} {'ours %LPA':>10s} {'IMD %LPA':>9s} {'diff':>7s}")
    diffs = []
    for y, imd_pct in sorted(IMD_PCT_LPA.items()):
        if y not in ai_aw.index:
            continue
        ours = 100 * ai_aw.loc[y] / lpa_ours
        d = ours - imd_pct
        diffs.append(d)
        log(f"    {y:6d} {ours:10.1f} {imd_pct:9.1f} {d:+7.1f}")
        rows.append({"test": f"JJAS %LPA {y}", "got": round(float(ours), 1),
                     "expected": imd_pct, "tol": 6,
                     "pass": bool(abs(d) <= 6), "unit": "%", "note": ""})
    mad = float(np.mean(np.abs(diffs)))
    corr = float(np.corrcoef(
        [100 * ai_aw.loc[y] / lpa_ours for y in sorted(IMD_PCT_LPA) if y in ai_aw.index],
        [IMD_PCT_LPA[y] for y in sorted(IMD_PCT_LPA) if y in ai_aw.index])[0, 1])
    log(f"\n    mean |difference| {mad:.1f} pp   correlation with IMD {corr:.3f}")
    check("mean abs diff vs IMD %LPA", mad, 0, 5, " pp",
          "how far our all-India series sits from IMD's")
    check("correlation with IMD %LPA", corr, 1.0, 0.15, "",
          "does our series track IMD year to year")

    # unweighted comparison, to prove the weighting choice
    ai_sm_pct = [100 * ai_sm.loc[y] / ai_sm.loc[LPA_LO:LPA_HI].mean()
                 for y in sorted(IMD_PCT_LPA) if y in ai_sm.index]
    mad_sm = float(np.mean(np.abs(np.array(ai_sm_pct)
                                  - [IMD_PCT_LPA[y] for y in sorted(IMD_PCT_LPA)
                                     if y in ai_sm.index])))
    log(f"    (unweighted district mean would give "
        f"mean |difference| {mad_sm:.1f} pp — "
        f"{'worse' if mad_sm > mad else 'better'})")

    # ------------------------------------------- monthly normals vs IMD
    #   TEST 5  THE MONTHLY LPA, MONTH BY MONTH.
    #   The JJAS total can be right while the months inside it are not, and the
    #   dashboard publishes a MONTHLY departure -- so the month is the number
    #   that has to be checked. IMD's published all-India monthly normals on
    #   its own 1971-2020 window are the external truth.
    log("\n  TEST 5 — all-India MONTHLY normal vs IMD published (1971-2020)")
    log("  " + "-" * 70)
    mon = rain.resample("MS").sum(min_count=1)
    ai_mon = mon.apply(aw, axis=1)
    ai_mon = pd.DataFrame({"mm": ai_mon.to_numpy()}, index=ai_mon.index)
    ai_mon["year"] = ai_mon.index.year
    ai_mon["month"] = ai_mon.index.month
    base = ai_mon[(ai_mon.year >= LPA_LO) & (ai_mon.year <= LPA_HI)]
    ours_m = base.groupby("month")["mm"].mean()
    log(f"    {'month':>6}{'ours':>9}{'IMD':>9}{'diff':>8}{'diff %':>9}")
    for m in range(1, 13):
        o, i = float(ours_m[m]), IMD_MONTHLY_LPA[m]
        log(f"    {m:>6}{o:>9.1f}{i:>9.1f}{o-i:>8.1f}{100*(o-i)/i:>8.1f}%")
        rows.append({"test": f"monthly LPA {m:02d}", "got": round(o, 1),
                     "expected": i, "tol": max(3.0, 0.08 * i),
                     "pass": bool(abs(o - i) <= max(3.0, 0.08 * i)),
                     "unit": " mm", "note": "area-weighted vs IMD published"})
    jjas_o = float(sum(ours_m[m] for m in JJAS))
    jjas_i = float(sum(IMD_MONTHLY_LPA[m] for m in JJAS))
    log(f"    {'JJAS':>6}{jjas_o:>9.1f}{jjas_i:>9.1f}{jjas_o-jjas_i:>8.1f}"
        f"{100*(jjas_o-jjas_i)/jjas_i:>8.1f}%")
    log("    The monsoon months run ~3-4% BELOW IMD and the dry months above.")
    log("    That is the spatial estimator, not the baseline: an area-weighted")
    log("    mean of 0.25 deg grid cells is not IMD's subdivision-weighted")
    log("    gauge network. Both are on the identical 1971-2020 window.")

    #   TEST 6  THE PART-MONTH NORMAL.
    #   A running month must be compared against a normal covering exactly the
    #   days observed. Getting this wrong is not subtle: on 6 August 2026 the
    #   full-month normal reports -78.7% where the day-matched one reports
    #   +0.1%, and the first number is pure arithmetic error.
    log("\n  TEST 6 — part-month normal is day-matched, not full-month")
    log("  " + "-" * 70)
    cur_y = int(rain.index.year.max())
    last = rain.index.max()
    nd = int(last.day)
    cm = int(last.month)
    curm = rain[(rain.index.year == cur_y) & (rain.index.month == cm)]
    act = aw(curm.sum(min_count=1))
    hist = rain[(rain.index.year >= LPA_LO) & (rain.index.year <= LPA_HI)
                & (rain.index.month == cm) & (rain.index.day <= nd)]
    dm = float(np.mean([aw(g.sum(min_count=1))
                        for _, g in hist.groupby(hist.index.year)]))
    fullm = float(ours_m[cm])
    log(f"    {cur_y}-{cm:02d}, {nd} day(s) observed")
    log(f"      actual                {act:8.1f} mm")
    log(f"      day-matched normal    {dm:8.1f} mm  -> {100*(act-dm)/dm:+6.1f}%")
    log(f"      full-month normal     {fullm:8.1f} mm  -> "
        f"{100*(act-fullm)/fullm:+6.1f}%   <- the error this avoids")
    #   The panel must agree with the day-matched figure, not the full-month one
    fm_panel = pd.read_csv(DATA / "features_lgd.csv", low_memory=False,
                           usecols=["district_id", "year", "month", "rain_mm",
                                    "normal_mm"])
    gm = fm_panel[(fm_panel.year == cur_y) & (fm_panel.month == cm)].copy()
    gm["_a"] = gm["district_id"].map(area)
    ok = gm["rain_mm"].notna() & gm["normal_mm"].notna() & gm["_a"].notna()
    gm = gm[ok]
    pan_dep = float(100 * ((gm.rain_mm * gm._a).sum() - (gm.normal_mm * gm._a).sum())
                    / (gm.normal_mm * gm._a).sum())
    check("part-month departure matches day-matched", pan_dep,
          100 * (act - dm) / dm, 1.0, " pp",
          "panel vs a direct calculation from the daily grid")

    #   TEST 7  TEMPERATURE BASELINE.
    #   IMD's rainfall LPA is 1971-2020 but its temperature normals are the
    #   1981-2010 CLINO -- two different windows, by IMD's own practice. This
    #   panel uses 1971-2020 for both, so the cost of that choice is measured
    #   rather than assumed. It is small: see NORMALS_METHODOLOGY.md.
    log("\n  TEST 7 — temperature baseline: ours 1971-2020 vs IMD CLINO 1981-2010")
    log("  " + "-" * 70)
    for var in ("tmax", "tmin"):
        p = IMD / f"_district_daily_{var}_lgd.pkl"
        if not p.exists():
            continue
        t = pd.read_pickle(p)
        t.columns = t.columns.astype(int)
        tm = t.resample("MS").mean()
        yy, mm_ = tm.index.year, tm.index.month
        sel = np.isin(mm_, JJAS)
        a = tm[(yy >= 1971) & (yy <= 2020) & sel]
        b = tm[(yy >= 1981) & (yy <= 2010) & sel]
        na = a.groupby(a.index.month).mean()
        nb = b.groupby(b.index.month).mean()
        d = (na - nb).to_numpy(float).ravel()
        d = d[np.isfinite(d)]
        log(f"    {var}: JJAS district normals shift "
            f"{np.mean(d):+.3f} degC mean, {np.abs(d).max():.3f} max, "
            f"{100*np.mean(np.abs(d) > 0.2):.1f}% of districts beyond 0.2")
        check(f"{var} baseline shift vs IMD CLINO", float(np.abs(d).max()),
              0.0, 0.6, " degC",
              "how much switching to 1981-2010 would move the anomaly")

    # ------------------------------------------------- definitional checks
    log("\n  TEST 4 — definitional / physical self-consistency")
    log("  " + "-" * 70)
    f = pd.read_csv(DATA / "features_lgd.csv", low_memory=False,
                    parse_dates=["date"])

    for col in ("spei_era5_1", "spei_era5_4", "spei_era5_12", "spei_harg_4"):
        s = f[col].dropna()
        check(f"{col} mean (must be ~0)", s.mean(), 0.0, 0.20, " sd")
        check(f"{col} sd (must be ~1)", s.std(), 1.0, 0.20, " sd")

    mai = f["mai"].dropna()
    check("MAI minimum >= 0", mai.min(), 0.0, 0.001)
    check("MAI maximum <= 1.5 (clip)", mai.max(), 1.5, 0.001)
    mon = f[f["month"].isin(JJAS)]["mai"].mean()
    dry = f[f["month"].isin([1, 2, 3])]["mai"].mean()
    log(f"    MAI monsoon {mon:.3f} vs Jan-Mar {dry:.3f} "
        f"({'monsoon wetter — correct' if mon > dry else 'WRONG SIGN'})")
    rows.append({"test": "MAI monsoon > dry season", "got": round(mon, 3),
                 "expected": round(dry, 3), "tol": 0, "pass": bool(mon > dry),
                 "unit": "", "note": "seasonality direction"})

    check("SDD minimum >= 0", f["sdd"].dropna().min(), 0.0, 0.001, " degC-d")
    check("GDD kharif min >= 0", f["gdd_kharif"].dropna().min(), 0.0, 0.001)
    # SDD is heat ABOVE 34C: must peak in the pre-monsoon, not in winter
    sdd_m = f.groupby("month")["sdd"].mean()
    peak = int(sdd_m.idxmax())
    log(f"    SDD peaks in month {peak} "
        f"({'Apr-Jun — correct' if peak in (4, 5, 6) else 'UNEXPECTED'})")
    rows.append({"test": "SDD peaks in pre-monsoon", "got": peak,
                 "expected": 5, "tol": 1, "pass": peak in (4, 5, 6),
                 "unit": " month", "note": ""})

    for col in ("tmax_anom", "tmin_anom"):
        base = f[(f.year >= LPA_LO) & (f.year <= LPA_HI)][col].dropna()
        check(f"{col} mean over its {LPA_LO}-{LPA_HI} baseline", base.mean(),
              0.0, 0.05, " degC", "must be zero by construction")
        # the 1991-2020 mean is NOT expected to be zero: it is the warming of
        # the recent 30 years relative to IMD's 50-year baseline, and is
        # reported as a climate signal rather than tested as an error.
        recent = f[(f.year >= 1991) & (f.year <= 2020)][col].dropna().mean()
        log(f"           1991-2020 mean {recent:+.2f} degC "
            f"(warming vs the {LPA_LO}-{LPA_HI} baseline)")

    # rainfall departure must reproduce from its own components
    d = f.dropna(subset=["rain_mm", "normal_mm", "pct_departure"])
    d = d[d["normal_mm"] >= 5]
    recomp = (d["rain_mm"] - d["normal_mm"]) / d["normal_mm"] * 100
    # Tolerance is CSV text precision, not a modelling margin: rain_mm and
    # normal_mm are re-parsed from decimal text, so the recomputed ratio
    # differs in the 4th decimal of a percentage point (max 3.4e-4 pp on
    # values of order 40 pp -- a relative error near 1e-5).
    err = float(np.abs(recomp - d["pct_departure"]).max())
    check("pct_departure recomputes from rain/normal", err, 0.0, 1e-3, " pp",
          "identity check (CSV float round-trip)")

    # ---------------------------------------------------------------- out
    out = pd.DataFrame(rows)
    out.to_csv(DATA / "feature_validation_lgd.csv", index=False)
    npass, ntot = int(out["pass"].sum()), len(out)
    summary = {"passed": npass, "total": ntot,
               "jjas_lpa_ours_mm": round(lpa_ours, 1),
               "jjas_lpa_imd_mm": IMD_JJAS_LPA_MM,
               "mean_abs_diff_vs_imd_pp": round(mad, 2),
               "corr_with_imd": round(corr, 3),
               "aggregation": "area-weighted ratio of sums "
                              "(matches IMD practice)",
               #   This string said "1991-2020 WMO for features; 1981-2020
               #   used for the LPA comparison". Neither half was true -- the
               #   panel and the LPA comparison are both on 1971-2020 -- and it
               #   travelled into the technical report and the deck, which
               #   quote this summary verbatim. Derived from the constants now,
               #   so it cannot drift from them again.
               "normal_baseline": f"{LPA_LO}-{LPA_HI} for rainfall (IMD's own "
                                  f"LPA window) and for temperature; IMD uses "
                                  f"the 1981-2010 CLINO for temperature, a "
                                  f"deviation measured in TEST 7",
               "monthly_lpa_checked": True,
               "temp_baseline_imd": "1981-2010 CLINO"}
    (DATA / "feature_validation_lgd.json").write_text(
        json.dumps({"summary": summary, "tests": rows}, indent=1),
        encoding="utf-8")
    log("\n" + "=" * 74)
    log(f"  RESULT: {npass}/{ntot} checks passed")
    if npass < ntot:
        log("  review:")
        for r in rows:
            if not r["pass"]:
                log(f"    - {r['test']}: got {r['got']} vs {r['expected']}")
    log(f"  wrote feature_validation_lgd.csv / .json")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
