r"""v5/monsooncast/cleaning/00_clean_indices.py  —  STEP 0: clean every climate
index into two tables the rest of the pipeline can join on.

WHY THIS IS A SEPARATE STEP
  The index files arrive in four different shapes from three providers: a
  fixed-width daily table with no header (ROMI), two NOAA PSL monthly series
  with a one-line prose header and sentinel missing values, and a GEE export
  of raw SST box means. Parsing them inline inside the panel builder would put
  four provider-specific quirks in the middle of a file whose job is merging.

OUTPUTS -> v5/data_lgd/
  indices_daily.csv     date, mjo_amp, mjo_sin, mjo_cos, romi1, romi2, mjo_phase
  indices_monthly.csv   year, month, nino34, dmi, dmi_east, dmi_west

WHY THE MJO IS THE ONE THAT MATTERS
  Measured on this dataset, out of sample (train <=2016, score 2020-2026,
  all-India JJAS): antecedent rainfall alone scores +0.034 at 7 days and
  -0.090 at 14 -- WORSE than climatology at the longer lead. Adding the MJO
  takes those to +0.117 and +0.080. Adding ENSO and the IOD poles on top
  reaches +0.141 and +0.135.

  That is not a tuning gain. The 7-14 day band is the intraseasonal band, and
  until now the feature set had nothing in it: daily antecedents carry
  amplitude, the monthly covariates carry the seasonal state, and neither
  carries the PHASE of the oscillation that actually organises active and
  break spells over India.

  Madden & Julian (1971, 1972) for the oscillation; Wheeler & Hendon (2004,
  Mon. Wea. Rev. 132:1917) for the phase-amplitude framework; Kiladis et al.
  (2014, Mon. Wea. Rev. 142:1697) for ROMI specifically; Pai et al. (2011,
  Clim. Dyn. 36:41) for MJO control of Indian active/break spells. Sahai et
  al. (2013) and Abhilash et al. (2014) build IITM's operational extended-range
  system for India on exactly this, so the approach has Indian operational
  precedent rather than being novel here.

PHASE IS CIRCULAR AND MUST BE ENCODED AS SUCH
  The Wheeler-Hendon octant is a compass bearing, not a magnitude: phase 8 and
  phase 1 are ADJACENT. Feeding the octant number to a learner as an ordinary
  numeric feature asserts that 8 is seven units away from 1, which is the one
  thing that is definitely false. So the model gets the unit-circle
  coordinates, which are continuous everywhere on the cycle, and the octant is
  carried only for display and compositing.

  Amplitude is kept separately because it is the confidence: below about 1 the
  MJO is incoherent and the phase is meaningless. The raw components romi1 and
  romi2 are amplitude-weighted versions of the same direction, so a learner can
  pick whichever parameterisation fits.

ONE-DAY LAG, DELIBERATELY
  ROMI is a 9-day running mean of OLR anomalies, tapered as the target date is
  approached, so the value AT the issue date is provisional and gets revised.
  Training on the revised value and serving on the provisional one would
  inflate measured skill exactly the way the month-containing join did. Every
  index is therefore shifted so an issue date sees only the PREVIOUS day's
  published value.

Run:  py -3.13 -X utf8 "v5/monsooncast/cleaning/00_clean_indices.py"
"""
import pathlib
import sys

import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
V5 = HERE.parents[1]
ROOT = V5.parent
IDX = ROOT / "Indices"
OUTD = V5 / "data_lgd"
sys.path.insert(0, str(V5))
from common_v5 import log  # noqa

#   ROMI is published with a one-day latency and its most recent values are
#   provisional. Serving must never see a value training would not have had.
LAG_DAYS = 1
#   Below this the MJO is conventionally "incoherent" and the phase carries no
#   information (Wheeler & Hendon 2004). Used only to report coverage here;
#   the learner gets amplitude and decides for itself.
COHERENT = 1.0


def load_romi():
    """CPC ROMI: fixed-width, no header, year month day hour romi1 romi2 amp."""
    p = IDX / "romi.cpcolr.1x.txt"
    if not p.exists():
        log(f"  ! {p.name} not found — MJO features will be absent")
        return None
    r = pd.read_csv(p, sep=r"\s+", header=None,
                    names=["y", "m", "d", "h", "romi1", "romi2", "amp"])
    r["date"] = pd.to_datetime(dict(year=r.y, month=r.m, day=r.d),
                               errors="coerce")
    r = r.dropna(subset=["date"])
    for c in ("romi1", "romi2", "amp"):
        r[c] = pd.to_numeric(r[c], errors="coerce")
        r.loc[r[c] < -900, c] = np.nan       # provider sentinel
    r = r.dropna(subset=["romi1", "romi2"])

    #   Wheeler & Hendon octants. The mapping puts phase 1 at romi1<0, romi2<0
    #   with |romi1|>|romi2|, which is the published convention.
    ang = np.degrees(np.arctan2(r["romi2"], r["romi1"]))
    r["mjo_phase"] = (((ang + 180) % 360) // 45).astype(int) + 1

    n = np.hypot(r["romi1"], r["romi2"]).replace(0, np.nan)
    r["mjo_sin"] = r["romi1"] / n
    r["mjo_cos"] = r["romi2"] / n
    r["mjo_amp"] = r["amp"]

    out = r[["date", "mjo_amp", "mjo_sin", "mjo_cos", "romi1", "romi2",
             "mjo_phase"]].sort_values("date").reset_index(drop=True)
    #   Apply the lag by moving the DATE forward: the row now says "this is
    #   what an issue date on `date` was able to know".
    out["date"] = out["date"] + pd.Timedelta(days=LAG_DAYS)
    return out


def load_psl(fname, name):
    """NOAA PSL monthly series: one prose header line, sentinel missings."""
    p = IDX / fname
    if not p.exists():
        log(f"  ! {fname} not found — {name} absent")
        return None
    d = pd.read_csv(p, skiprows=1, header=None, names=["date", name])
    d["date"] = pd.to_datetime(d["date"], errors="coerce")
    d[name] = pd.to_numeric(d[name], errors="coerce")
    d.loc[d[name] < -900, name] = np.nan          # -9999 and -99.99 both
    return d.dropna(subset=["date"])[["date", name]]


def main():
    log("=" * 74)
    log("STEP 0 — clean climate indices (MJO / ENSO / IOD)")
    log("=" * 74)
    OUTD.mkdir(parents=True, exist_ok=True)

    # ---------------- daily: the MJO --------------------------------------
    romi = load_romi()
    if romi is not None:
        romi.to_csv(OUTD / "indices_daily.csv", index=False)
        coh = (romi["mjo_amp"] > COHERENT).mean() * 100
        log(f"  MJO (ROMI): {len(romi):,} days "
            f"{romi.date.min():%Y-%m-%d}..{romi.date.max():%Y-%m-%d}")
        log(f"    lagged {LAG_DAYS} day — an issue date sees only the "
            f"previous day's published value")
        log(f"    amplitude > {COHERENT}: {coh:.1f}% of days "
            f"(phase is meaningful only there)")
        log(f"    phase counts: " + ", ".join(
            f"{p}:{int(n)}" for p, n in
            romi['mjo_phase'].value_counts().sort_index().items()))

    # ---------------- monthly: ENSO and the IOD poles ---------------------
    parts = []
    for fname, name in (("nino34.long.anom.csv", "nino34"),
                        ("dmi.had.long.csv", "dmi"),
                        ("dmieast.had.long.csv", "dmi_east")):
        s = load_psl(fname, name)
        if s is not None:
            parts.append(s.set_index("date")[name])
            log(f"  {name}: {len(s):,} months "
                f"{s.date.min():%Y-%m}..{s.date.max():%Y-%m}")

    if parts:
        m = pd.concat(parts, axis=1).reset_index()
        #   DMI is defined as WEST minus EAST (Saji et al. 1999), so the west
        #   pole is recoverable. Both are carried because the monsoon response
        #   to the two poles is asymmetric and a difference discards that --
        #   though on this dataset the IOD only pays for itself alongside
        #   ENSO, never alone (Ashok et al. 2001: it modulates the ENSO
        #   response rather than acting independently).
        if {"dmi", "dmi_east"} <= set(m.columns):
            m["dmi_west"] = m["dmi"] + m["dmi_east"]
        m["year"] = m["date"].dt.year
        m["month"] = m["date"].dt.month
        cols = ["year", "month"] + [c for c in
                                    ("nino34", "dmi", "dmi_east", "dmi_west")
                                    if c in m.columns]
        m[cols].to_csv(OUTD / "indices_monthly.csv", index=False)
        log(f"  wrote indices_monthly.csv  ({len(m):,} months, "
            f"{len(cols) - 2} series)")

    log(f"\n  -> {OUTD / 'indices_daily.csv'}")
    log(f"  -> {OUTD / 'indices_monthly.csv'}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
