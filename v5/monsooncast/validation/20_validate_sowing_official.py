r"""
v5/monsooncast/validation/20_validate_sowing_official.py  —  check our UPAg-derived
sowing figures against the Ministry of Agriculture & Farmers' Welfare (DA&FW)
weekly all-India release.

WHY THIS CHECK IS WORTH HAVING
  Our sowing numbers are built from the UPAg state-level weekly releases and go
  through several non-trivial steps before they become a national figure: unit
  repair, removal of the "India"/"All India" aggregate rows, segmentation of the
  cumulative counter at season boundaries, and a roll-up over states.  Any of
  those could be wrong in a way that is invisible from the inside.

  DA&FW publishes its own consolidated all-India crop-wise area every week, from
  the same underlying state reporting.  It is the closest thing to a reference
  for this quantity, and it is genuinely INDEPENDENT of our processing chain --
  so agreement is evidence the chain is right, and disagreement localises the
  problem to a crop.

  Unlike the earlier rainfall work, there is no ambiguity about what is being
  measured here: both sides are "area sown to date, all-India, this crop, this
  week".  A gap is therefore a real gap, not a difference of estimator.

THE BENCHMARK
  Ministry of Agriculture & Farmers' Welfare, weekly kharif sowing progress,
  as of 17 July 2026, as reported 21 July 2026.  Figures are lakh hectares
  (1 lakh ha = 0.1 M ha = 100,000 ha) exactly as published.

  Recorded here as a literal table rather than scraped, because the release is
  a press statement rather than a machine-readable feed, and a hard-coded
  benchmark with a stated date is auditable.  It must be updated by hand when a
  newer release is used -- BENCHMARK_DATE is what makes a stale check obvious.

WHAT THE FIRST RUN FOUND
  Maize, sugarcane and tur reproduce the official figure to within 0.03%, which
  validates the whole chain.  Rice sits 6.8% low and soybean 14.6% low; the
  soybean gap is mostly staleness -- our latest soybean report is a week older
  than the official as-of date, during the steepest part of its sowing curve.
  That is why this script reports each crop's OWN as-of week rather than
  presenting everything under one headline date.

OUTPUT -> v5/data_lgd/sowing_validation_official.csv

Run:  py -3.13 -X utf8 "v5/monsooncast/validation/20_validate_sowing_official.py"
"""
import pathlib
import sys

import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
V5 = HERE.parents[1]
DATA = V5 / "data_lgd"
sys.path.insert(0, str(V5))
from common_v5 import log  # noqa

BENCHMARK_DATE = "2026-07-17"
BENCHMARK_SRC = ("Ministry of Agriculture & Farmers' Welfare, weekly kharif "
                 "sowing progress as of 17 Jul 2026")

# crop -> (area sown 2026, same period 2025), lakh hectares, as published
OFFICIAL = {
    "Rice":       (166.41, 167.83),
    "Soybean":    (106.02, 111.05),
    "Cotton":     (92.53, 98.40),
    "Maize":      (67.89, 71.48),
    "Sugarcane":  (57.58, 56.72),
    "Bajra":      (39.98, 48.98),
    "Groundnut":  (34.52, 37.69),
    "Tur":        (24.80, 30.17),
}
# published groupings, carried for context only -- we do not score against them
# because our crop list does not span the whole of any one category
CONTEXT = {
    "All kharif crops": (658.19, 700.47),
    "Pulses (total)":   (69.23, 81.52),
    "Oilseeds (total)": (147.09, 155.72),
    "Coarse cereals":   (119.03, 134.09),
}
LAKH_HA = 1e5
CLOSE, LOOSE = 2.0, 10.0        # percent, the bands used for the verdict


def verdict(d):
    if not np.isfinite(d):
        return "not reported by us"
    a = abs(d)
    return "matches" if a <= CLOSE else ("close" if a <= LOOSE else "DIVERGES")


def main():
    log("=" * 78)
    log("SOWING VALIDATION — our UPAg roll-up vs the DA&FW all-India release")
    log("=" * 78)
    log(f"  benchmark: {BENCHMARK_SRC}")

    p = DATA / "sowing_status_current.csv"
    if not p.exists():
        log("  ! sowing_status_current.csv absent — run 19_sowing_dynamics first")
        return
    c = pd.read_csv(p, parse_dates=["week_end"])
    k = c[c["season"] == "Kharif"]
    # The PUBLISHED year-on-year, which withholds a ratio whose matched basis is
    # too thin.  Recomputing it here from summed columns instead would report
    # maize at +187% while sowing_status_national.csv correctly shows nothing --
    # two of our own outputs disagreeing about the same number.
    pubp = DATA / "sowing_status_national.csv"
    pub = (pd.read_csv(pubp).query("season == 'Kharif'")
           .set_index("crop")[["yoy_pct", "yoy_basis_pct"]]
           if pubp.exists() else pd.DataFrame())
    ours = k.groupby("crop").agg(
        ours_ha=("area_to_date_ha", "sum"),
        ours_ly_ha=("ly_same_week_ha", "sum"),
        states=("state_raw", "nunique"),
        as_of=("week_end", "max"),
        wk=("wk", "max"))

    bd = pd.Timestamp(BENCHMARK_DATE)
    rows = []
    for crop, (a26, a25) in OFFICIAL.items():
        o = ours.loc[crop] if crop in ours.index else None
        our_ha = float(o["ours_ha"]) if o is not None else np.nan
        our_ly = float(o["ours_ly_ha"]) if o is not None else np.nan
        off_ha, off_ly = a26 * LAKH_HA, a25 * LAKH_HA
        d = 100 * (our_ha / off_ha - 1) if np.isfinite(our_ha) else np.nan
        dly = 100 * (our_ly / off_ly - 1) if np.isfinite(our_ly) and our_ly > 0 else np.nan
        stale = ((bd - o["as_of"]).days // 7) if o is not None else np.nan
        rows.append({
            "crop": crop,
            "our_mha": our_ha / 1e6, "official_mha": off_ha / 1e6,
            "diff_pct": d, "verdict": verdict(d),
            "our_ly_mha": our_ly / 1e6, "official_ly_mha": off_ly / 1e6,
            "ly_diff_pct": dly,
            "our_yoy_pct": (float(pub.loc[crop, "yoy_pct"])
                            if crop in pub.index else np.nan),
            "our_yoy_basis_pct": (float(pub.loc[crop, "yoy_basis_pct"])
                                  if crop in pub.index else np.nan),
            "naive_yoy_pct": (100 * (our_ha / our_ly - 1)
                              if np.isfinite(our_ly) and our_ly > 0 else np.nan),
            "official_yoy_pct": 100 * (a26 / a25 - 1),
            "states_reporting": (int(o["states"]) if o is not None else 0),
            "our_as_of": (o["as_of"].date().isoformat() if o is not None else ""),
            "weeks_behind_benchmark": stale,
            "benchmark_date": BENCHMARK_DATE,
            "benchmark_source": BENCHMARK_SRC,
        })
    V = pd.DataFrame(rows).sort_values("official_mha", ascending=False)

    log(f"\n  LEVEL — area sown to date, all-India (M ha):")
    log(f"    {'crop':<11}{'ours':>8}{'official':>10}{'diff':>9}"
        f"{'states':>8}{'our as-of':>13}{'lag':>6}  verdict")
    for r in V.itertuples():
        ov = f"{r.our_mha:.2f}" if np.isfinite(r.our_mha) else "--"
        df = f"{r.diff_pct:+.1f}%" if np.isfinite(r.diff_pct) else "--"
        lg = (f"{int(r.weeks_behind_benchmark)}w"
              if np.isfinite(r.weeks_behind_benchmark) else "--")
        log(f"    {r.crop:<11}{ov:>8}{r.official_mha:>10.2f}{df:>9}"
            f"{r.states_reporting:>8}{r.our_as_of:>13}{lg:>6}  {r.verdict}")

    ok = V[V["verdict"] == "matches"]
    log(f"\n    {len(ok)} of {int(V['our_mha'].notna().sum())} reported crops "
        f"within {CLOSE:.0f}% of the official figure"
        + (f" ({', '.join(ok['crop'])})" if len(ok) else ""))

    log(f"\n  YEAR-ON-YEAR — the comparison a buyer actually trades on:")
    log(f"    {'crop':<11}{'our YoY':>10}{'official':>10}{'gap':>9}  "
        f"note")
    for r in V.itertuples():
        if not np.isfinite(r.naive_yoy_pct):
            continue
        if np.isfinite(r.our_yoy_pct):
            ours = f"{r.our_yoy_pct:+.1f}%"
            gap = f"{r.our_yoy_pct - r.official_yoy_pct:+.1f}pp"
        else:
            ours, gap = "withheld", "--"
        note = ("last-year base low by "
                f"{abs(r.ly_diff_pct):.0f}%" if np.isfinite(r.ly_diff_pct)
                and abs(r.ly_diff_pct) > CLOSE else "consistent")
        if not np.isfinite(r.our_yoy_pct):
            note += f" (naive {r.naive_yoy_pct:+.0f}% on "
            note += (f"{r.our_yoy_basis_pct:.0f}% basis)"
                     if np.isfinite(r.our_yoy_basis_pct) else "a thin basis)")
        log(f"    {r.crop:<11}{ours:>10}{r.official_yoy_pct:>+9.1f}%"
            f"{gap:>9}  {note}")

    log(f"\n  CONTEXT — published groupings, not scored (our crop list does not "
        f"span them):")
    for name, (a26, a25) in CONTEXT.items():
        log(f"    {name:<20}{a26:>8.2f}{a25:>9.2f} lakh ha   "
            f"{100*(a26/a25-1):+.1f}% YoY")

    V.round(3).to_csv(DATA / "sowing_validation_official.csv", index=False)
    log(f"\n  wrote sowing_validation_official.csv")
    bad = V[V["verdict"] == "DIVERGES"]
    if len(bad):
        log(f"  ! {len(bad)} crop(s) diverge by more than {LOOSE:.0f}%: "
            f"{', '.join(bad['crop'])} — check state coverage and report lag "
            f"before quoting these")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
