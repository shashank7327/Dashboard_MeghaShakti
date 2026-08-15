r"""
v5/monsooncast/lib/upag_common.py  —  shared loader for the UPAg weekly sowing
releases, with the two corrections that both 14_build_crop_mask.py and
19_sowing_dynamics.py need.

CORRECTION 1 — THE FILE CONTAINS NATIONAL TOTAL ROWS
  StateName is not purely a state column.  It carries two aggregate labels:
  "All India" and, separately, "India".  Filtering only "All India" -- which the
  first version of the mask did -- leaves the "India" rows in, and they are the
  national total, so every summed-over-states figure comes out at almost exactly
  twice the truth:

      Wheat 2025-26, summing every StateName   66.83 M ha
      the "India" row alone                    33.42 M ha   <- the real total
      the 35 genuine state rows                33.41 M ha
      published all-India wheat area          ~33    M ha

  Both labels are dropped from the state panel here and returned separately as
  `national`, where they are useful as a published cross-check on our own sum.

CORRECTION 2 — ONE season_year HOLDS SEVERAL SEASONS, AND THE COUNTER RESETS
  UPAg reports a CUMULATIVE sown area that restarts at each new season, but the
  season is not labelled (the `Season` column is almost entirely empty).  Rice
  under SowingYear 2025 runs up to 44,158 (thousand ha) by week 39 as Kharif
  finishes, then RESTARTS at 618 in week 45 as Rabi rice begins.

  Differencing that series after sorting by week number -- again, what the first
  version did -- treats the restart as a negative step, clips it to zero, and
  then adds the whole second ramp on top of the first season's final total.  The
  fix is to detect the resets and cut the series into separate seasons.

  Seasons are labelled from the calendar month in which each segment starts:
      Jun-Sep -> Kharif      Oct-Jan -> Rabi      Feb-May -> Summer
  which follows the standard Indian cropping calendar.

WEEK NUMBERING
  SowingWeekNumber is the ISO calendar week and SowingYear is the year the
  season STARTED, so week numbers align across years and a same-week
  year-on-year comparison is well defined.  Rabi segments cross the new year
  (week 45 -> week 52 -> week 1), which is why segments are ordered by
  week-ending DATE and never by week number.
"""
import pathlib

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[3]
UPAG = ROOT / "UPAJ"

# StateName values that are national totals, not states
NATIONAL = {"india", "all india"}

# UPAg commodity name -> the crop name used by the stress model
ALIAS = {
    "Paddy": "Rice", "Rice": "Rice", "Maize": "Maize", "Soyabean": "Soybean",
    "Soybean": "Soybean", "Arhar/Tur": "Tur", "Tur": "Tur", "Arhar": "Tur",
    "Bajra": "Bajra", "Groundnut": "Groundnut", "Castor": "Castor",
    "Castorseed": "Castor", "Sugarcane": "Sugarcane", "Cotton": "Cotton",
    "Sesamum": "Sesame", "Sesame": "Sesame", "Wheat": "Wheat",
    "Rapeseed & Mustard": "Mustard", "Mustard": "Mustard", "Gram": "Chana",
    "Chana": "Chana", "Barley": "Barley", "Ragi": "Ragi",
    "Guar Seed": "Guar Seed", "Guarseed": "Guar Seed",
    "Soybeans": "Soybean", "Bengal Gram": "Chana", "Rapeseed": "Mustard",
}

# a drop to below this fraction of the running maximum is read as a new season
RESET_FRAC = 0.60
# A silence longer than this ends the season regardless of the values.  Chosen
# from the observed reporting cadence rather than picked: of 8,100 consecutive
# report gaps, 7,535 are exactly 7 days and 100 are 14 (a skipped week), then
# the distribution falls away sharply -- 29 at 21 days, 30 at 28, 328 beyond 56.
# Cutting above 21 keeps a skipped week inside its season and treats the long
# tail as the season boundary it is.  At 56 days Karnataka's Kharif maize was
# swallowed by the Summer segment across a 28-day break, halving the national
# maize area.
GAP_DAYS = 21


def parse_area(d):
    r"""Area in hectares from the Value/Unit pair, repairing two corruptions
    present in the published files.

    SHIFTED ROWS (50 rows, all Sugarcane 2026 weeks 23-24).  Value is the
    literal 1000 and Unit holds the real figure:

        CommodityName  StateName  Value      Unit
        Sugarcane      Bihar     1000.0    189.16     <- 189.16 thousand ha

    Reading Value here gives every affected state a flat 1.0 M ha of sugarcane.
    Taking the per-season maximum then locks that spurious value in, which is
    why the mask reported 26.1 M ha of sugarcane against a published 5.7 M ha.
    The live snapshot escaped it only because the latest week happened to be an
    intact row -- the two scripts disagreeing is what exposed the corruption.

    GARBAGE UNIT STRINGS (1540 rows).  Unit reads "1002 Ha", "1003 Ha", ... in
    an incrementing series that is plainly a serial number rather than a unit.
    Value itself is sound in these rows, so only the unit is discarded.

    Everything else is "1000 Ha" / "Thousand Ha" / "1000Ha" and a handful of
    plain "Ha".  Unrecognised units default to thousand-hectares, the
    overwhelming majority convention in the file.
    """
    val = pd.to_numeric(d["Value"], errors="coerce")
    unit = d["Unit"].astype(str).str.strip()
    unit_num = pd.to_numeric(unit, errors="coerce")

    shifted = unit_num.notna() & val.eq(1000)
    val = val.where(~shifted, unit_num)

    # plain hectares, the only unit that is not already in thousands
    scale = np.where(unit.str.fullmatch(r"(?i)ha"), 1.0, 1000.0)
    return val * scale


def season_of(month):
    if 6 <= month <= 9:
        return "Kharif"
    if month >= 10 or month == 1:
        return "Rabi"
    return "Summer"


def load_upag(alias=None):
    """Return (state_panel, national_panel).  Both are deduplicated, parsed and
    segmented into seasons; areas are in hectares."""
    alias = ALIAS if alias is None else alias
    frames = []
    for f in sorted(UPAG.glob("Sowing*.csv")):
        d = pd.read_csv(f)
        d["_src"] = f.name
        frames.append(d)
    if not frames:
        return None, None
    d = pd.concat(frames, ignore_index=True)

    d["crop"] = d["CommodityName"].astype(str).str.strip().map(alias)
    d["state_raw"] = d["StateName"].astype(str).str.strip()
    d["area_ha"] = parse_area(d)
    d["wk"] = pd.to_numeric(d["SowingWeekNumber"], errors="coerce")
    d["season_year"] = pd.to_numeric(d["SowingYear"], errors="coerce")
    d["week_end"] = pd.to_datetime(d["WeekEndingDate"], errors="coerce",
                                   dayfirst=True)
    d = d.dropna(subset=["crop", "area_ha", "season_year", "week_end"])

    is_nat = d["state_raw"].str.lower().isin(NATIONAL)
    state = segment(d[~is_nat].copy(), "state_raw")
    # the two national labels are alternative names for the same series; keep
    # the more complete one per crop-week rather than adding them together
    nat = segment(d[is_nat].assign(state_raw="India").copy(), "state_raw")
    return state, nat


def segment(d, scope_col):
    r"""Deduplicate to one row per scope x crop x reporting week, cut the
    cumulative series wherever the counter resets, and label each segment with
    the season and year IT ACTUALLY STARTED IN.

    WHY season_year IS DERIVED AND NOT READ FROM SowingYear
      UPAg's SowingYear does not mean the same thing for every commodity.  For
      wheat it is the year sowing began (SowingYear 2025, weeks ending Nov 2025
      to Feb 2026).  For groundnut the same column carries the MARKET year:
      every row of the June-October 2025 kharif crop is stamped SowingYear 2026.

      Keying on it therefore files one crop's season under the wrong year, and
      the year-on-year join -- which matches season_year to season_year - 1 --
      silently compares the wrong pair or finds nothing at all.

      Segmenting on (scope, crop) alone and taking the year from each segment's
      FIRST reporting week removes the dependency entirely: whatever UPAg calls
      it, a series that starts in June 2026 is Kharif 2026.  The published
      SowingYear is kept as `upag_sowing_year` so the disagreement stays
      inspectable rather than being quietly overwritten.
    """
    if d.empty:
        return d
    key = [scope_col, "crop"]
    # dedup on the reporting DATE: week numbers repeat every year, so they are
    # not a unique key once season_year is no longer part of the grouping
    d = (d.sort_values(key + ["week_end", "area_ha"])
         .groupby(key + ["week_end"], as_index=False)
         .agg(area_ha=("area_ha", "max"), wk=("wk", "max"),
              upag_sowing_year=("season_year", "max")))
    d = d.sort_values(key + ["week_end"]).reset_index(drop=True)

    def cut(g):
        r"""Two independent season breaks, because one is not enough.

        VALUE RESET -- the counter drops sharply, e.g. rice falling from 44.2
        M ha as Kharif closes to 0.6 M ha as Rabi opens.

        TIME GAP -- reporting stops for months and resumes.  Without this rule,
        a season whose first report EXCEEDS the previous season's final total
        never triggers a value reset and the two silently merge.  Sugarcane in
        Uttar Pradesh does exactly that: 2,705,000 ha at the close of Kharif
        2025, then 2,797,450 ha when Kharif 2026 opens eight months later.  The
        merged segment took the 2025 label, which collapsed maize from an exact
        match with the ministry's figure to -64% and pushed sugarcane's
        year-on-year to +838%.

        A season's weekly series is contiguous by construction, so a silence of
        more than GAP_DAYS is a season boundary whatever the values do.
        """
        v = g["area_ha"].to_numpy()
        t = g["week_end"].to_numpy()
        seg, s, run_max = np.zeros(len(v), int), 0, -np.inf
        for i, x in enumerate(v):
            gap = i > 0 and ((t[i] - t[i - 1]) / np.timedelta64(1, "D")
                             > GAP_DAYS)
            if gap or (run_max > 0 and x < RESET_FRAC * run_max):
                s += 1
                run_max = x
            else:
                run_max = max(run_max, x)
            seg[i] = s
        g = g.copy()
        g["seg"] = seg
        return g

    d = d.groupby(key, group_keys=False)[d.columns.tolist()].apply(cut)
    start = d.groupby(key + ["seg"])["week_end"].transform("min")
    d["season"] = start.dt.month.map(season_of)
    # Rabi 2025 runs Oct 2025 -> Mar 2026.  A state whose first Rabi report
    # lands in January would otherwise be filed as Rabi 2026 and compared
    # against the wrong year, so January starts are rolled back one year.
    d["season_year"] = np.where(
        (d["season"] == "Rabi") & (start.dt.month <= 3),
        start.dt.year - 1, start.dt.year)
    return d.reset_index(drop=True)
