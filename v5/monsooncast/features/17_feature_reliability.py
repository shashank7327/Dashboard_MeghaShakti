r"""
v5/monsooncast/features/17_feature_reliability.py  —  how much should each district's
observed features be trusted, and how uncertain is the observation itself?

WHY THIS REPLACES THE "AGREE WITH IMD" FRAMING
  Earlier validation treated IMD's district bulletins as ground truth and
  reported our disagreement as error.  That framing is wrong.  Those bulletins
  are averages of the rain-gauge stations that happen to lie INSIDE a district.
  Where a district holds one station the bulletin is that station; where it
  holds none the value is borrowed.  It is an observation with its own
  sampling error, not a reference standard -- and it does not cover every LGD
  district we carry.

  An area-weighted mean of an interpolated analysis and a one-station average
  are two different estimators of the same quantity.  Neither is the truth.
  The honest thing is to compute our estimate by a defensible method, and then
  say how uncertain it is -- which is what this does.

WHAT IS MEASURED

  1. SUPPORT — how many independent grid cells actually cover the district,
     and how concentrated the areal weights are.  The effective number of
     cells (inverse Simpson index of the weights, 1/sum(w^2)) is the honest
     count: a district drawing 95% of its weight from one cell has an
     effective support near 1 however many cells it touches.  Small districts
     on the 0.25 deg rainfall grid, and most districts on the 1 deg
     temperature grid, are resolution-limited and are flagged as such.

  2. OBSERVATIONAL SPREAD — the same quantity computed from an INDEPENDENT
     dataset.  CHIRPS (Funk et al. 2015, Sci. Data 2:150066) is satellite-plus-
     station and completely independent of IMD's gauge analysis, so the spread
     between the two is a lower bound on how well district rainfall is known
     at all.  This is a measurement of uncertainty, NOT a correction: neither
     product is adjusted toward the other.

  3. Both are combined into a per-district reliability grade, so a user can
     see where a district value is well-supported and where it is a single
     coarse cell.

OUTPUT -> v5/data_lgd/feature_reliability_lgd.csv

Run:  py -3.13 -X utf8 "v5/monsooncast/features/17_feature_reliability.py"
"""
import pathlib
import sys

import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
V5 = HERE.parents[1]
ROOT = V5.parent
IMD = ROOT / "IMD_Data"
DATA = V5 / "data_lgd"
CHIRPS = ROOT / "Chirps"
sys.path.insert(0, str(V5))
sys.path.insert(0, str(IMD))
from common_v5 import log  # noqa
from build_crosswalk import norm_state, norm_key  # noqa

CMP_YEARS = range(2005, 2021)      # overlap window for the independent check
JJAS = [6, 7, 8, 9]


def support():
    """Effective grid-cell support per district, from the areal weights."""
    out = {}
    for grid, label in (("rain", "rain"), ("temp", "temp")):
        cw = pd.read_csv(IMD / f"crosswalk_{grid}_areal.csv")
        g = cw.groupby("district_id")["weight"]
        n_cells = g.size().rename(f"{label}_cells")
        # inverse Simpson: 1/sum(w^2) -- the number of cells that actually
        # carry the district, discounting near-zero slivers
        eff = g.apply(lambda w: 1.0 / np.sum(np.square(w / w.sum()))
                      ).rename(f"{label}_eff_cells")
        out[label] = pd.concat([n_cells, eff], axis=1)
    return out["rain"].join(out["temp"], how="outer")


def chirps_source():
    r"""Which CHIRPS export to read, and on which column.

    There are two vintages on disk and they are NOT interchangeable:

      Chirps/       the original GAUL-2024 export — 701 units, no Jammu &
                    Kashmir or Ladakh, value column `rain_mm`
      Chirps_LGD/   the current export on the 791 LGD units, written by
                    GEE_scripts/04_rainfall_crosscheck/chirps_daily_LGD.js,
                    value column `chirps_mm`

    The LGD one is preferred where it exists, because it lands on exactly
    the geometry everything else in this system uses — the older files have
    to be matched to the registry BY NAME across two boundary vintages, and
    whatever fails to match silently drops out of the spread statistic.

    Returns (folder, value_column, label) so the caller can say which it used.
    Reporting the source matters here: the published observational spread
    (median 22.1%, 43.6% at the 90th percentile) is the yardstick every
    disagreement with a bulletin is judged against, and it is not the same
    number if it was computed on a different district set.
    """
    lgd = ROOT / "Chirps_LGD"
    if lgd.exists() and any(lgd.glob("CHIRPS_district_daily_*.csv")):
        return lgd, "chirps_mm", "Chirps_LGD (791 LGD units)"
    return CHIRPS, "rain_mm", "Chirps (701 GAUL units, matched by name)"


def chirps_jjas(reg):
    """All-India-independent check: CHIRPS district JJAS totals by name."""
    folder, vcol, label = chirps_source()
    log(f"  CHIRPS source: {label}")
    lut = {(s, d): i for s, d, i in
           zip(norm_state(reg["state"]), norm_key(reg["district"]),
               reg["district_id"])}
    uniq = {}
    for k, v in lut.items():
        uniq.setdefault(k[1], []).append(v)
    rows = []
    for y in CMP_YEARS:
        f = folder / f"CHIRPS_district_daily_{y}.csv"
        if not f.exists():
            continue
        d = pd.read_csv(f, parse_dates=["date"],
                        usecols=["date", "state", "district", vcol])
        d = d[d["date"].dt.month.isin(JJAS)]
        d["ks"], d["kd"] = norm_state(d["state"]), norm_key(d["district"])
        d["did"] = [lut.get((a, b), np.nan) for a, b in zip(d["ks"], d["kd"])]
        d = d.dropna(subset=["did"])
        d[vcol] = pd.to_numeric(d[vcol], errors="coerce")
        g = d.groupby("did", as_index=False)[vcol].sum().rename(
            columns={vcol: "rain_mm"})
        g["year"] = y
        rows.append(g)
    if not rows:
        return None
    c = pd.concat(rows, ignore_index=True)
    c["did"] = c["did"].astype(int)
    return c.rename(columns={"rain_mm": "chirps_jjas"})


def main():
    log("=" * 74)
    log("FEATURE RELIABILITY — support and observational spread per district")
    log("=" * 74)
    reg = pd.read_csv(IMD / "registry_lgd791.csv")

    sup = support()
    log(f"  areal support:")
    for c, lab in (("rain_eff_cells", "rainfall 0.25deg"),
                   ("temp_eff_cells", "temperature 1deg")):
        s = sup[c].dropna()
        log(f"    {lab:20s} effective cells per district: "
            f"median {s.median():.1f}, "
            f"{int((s < 1.5).sum())} districts below 1.5 "
            f"({100*(s < 1.5).mean():.0f}%)")

    # ---- independent observational spread, IMD grid vs CHIRPS ------------
    ours = pd.read_pickle(DATA / "daily_rain_lgd.pkl")
    ours.columns = ours.columns.astype(int)
    jj = ours[ours.index.month.isin(JJAS)]
    o = (jj.groupby(jj.index.year).sum(min_count=100)
         .loc[[y for y in CMP_YEARS if y in set(jj.index.year)]])
    o.index.name = "year"
    o = o.reset_index().melt(id_vars="year", var_name="did",
                             value_name="imd_jjas")
    o["did"] = o["did"].astype(int)

    c = chirps_jjas(reg)
    if c is None:
        log("  ! CHIRPS not available — spread not computed")
        rel = sup.reset_index()
    else:
        m = o.merge(c, on=["did", "year"], how="inner").dropna()
        log(f"\n  independent check vs CHIRPS: {m['did'].nunique()} districts, "
            f"{m['year'].nunique()} JJAS seasons, {len(m):,} district-years")
        nat_i = m.groupby("year")["imd_jjas"].mean()
        nat_c = m.groupby("year")["chirps_jjas"].mean()
        log(f"    all-India JJAS mean: IMD grid {nat_i.mean():.0f} mm, "
            f"CHIRPS {nat_c.mean():.0f} mm "
            f"(CHIRPS {100*(nat_c.mean()/nat_i.mean()-1):+.1f}%)")
        log(f"    year-to-year correlation of the two national series: "
            f"{np.corrcoef(nat_i, nat_c)[0,1]:.3f}")

        def agg(g):
            x, y2 = g["imd_jjas"].to_numpy(), g["chirps_jjas"].to_numpy()
            ok = np.isfinite(x) & np.isfinite(y2) & (x + y2 > 0)
            if ok.sum() < 5:
                return pd.Series({"spread_pct": np.nan, "corr_chirps": np.nan})
            x, y2 = x[ok], y2[ok]
            # symmetric relative spread: |a-b| / mean(a,b)
            sp = float(np.mean(np.abs(x - y2) / ((x + y2) / 2)) * 100)
            cr = float(np.corrcoef(x, y2)[0, 1]) if len(x) > 2 else np.nan
            return pd.Series({"spread_pct": sp, "corr_chirps": cr})
        sp = m.groupby("did").apply(agg, include_groups=False)
        log(f"    district JJAS spread between the two products: "
            f"median {sp['spread_pct'].median():.1f}%, "
            f"90th pct {sp['spread_pct'].quantile(0.9):.1f}%")
        log(f"    -> that is the observational uncertainty in district "
            f"rainfall, before any model is involved")
        rel = sup.join(sp, how="outer").reset_index()

    rel = rel.rename(columns={"index": "district_id"})
    rel = reg[["district_id", "state", "district", "area_km2"]].merge(
        rel, on="district_id", how="left")

    # ---- grade -----------------------------------------------------------
    # A district is well-supported when several grid cells genuinely cover it
    # AND the two independent products agree on it.
    eff = rel["rain_eff_cells"].fillna(0)
    spd = rel["spread_pct"]
    score = (np.clip(eff / 4.0, 0, 1) * 0.5
             + np.clip(1 - spd.fillna(spd.median()) / 60.0, 0, 1) * 0.5)
    rel["reliability"] = score.round(3)
    rel["reliability_grade"] = pd.cut(
        score, [-0.01, 0.35, 0.55, 0.75, 1.01],
        labels=["Low", "Moderate", "Good", "High"]).astype(str)
    rel["resolution_limited_temp"] = (rel["temp_eff_cells"].fillna(0) < 1.5).astype(int)
    rel.to_csv(DATA / "feature_reliability_lgd.csv", index=False)

    log(f"\n  reliability grades: " + ", ".join(
        f"{k} {v}" for k, v in rel["reliability_grade"].value_counts().items()))
    log(f"  temperature resolution-limited (eff cells < 1.5): "
        f"{int(rel['resolution_limited_temp'].sum())} of {len(rel)} districts")
    worst = rel.nsmallest(6, "reliability")[["state", "district",
                                             "rain_eff_cells", "spread_pct"]]
    log("  least-supported districts:")
    for r in worst.itertuples():
        log(f"    {r.district:26s} {r.state:22s} "
            f"eff cells {r.rain_eff_cells:.1f}  spread {r.spread_pct:.0f}%")
    log(f"\n  wrote feature_reliability_lgd.csv")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
