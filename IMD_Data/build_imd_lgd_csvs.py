r"""
IMD_Data/build_imd_lgd_csvs.py  —  aggregate the cached IMD grids to the 791
LGD units and write GEE-format per-year CSVs with actual, normal and % dep.

Reads the .grd cache already on disk (archive in raw/{var}/, real-time in raw/)
via the download_imd helpers, aggregates with the LGD crosswalks built by
build_lgd_system.py, forms the 1981-2025 day-of-year normal (+/-2 day window),
and writes the modelling-ready CSVs.

Every one of the 791 units gets a ROW.  Not every one gets a VALUE: IMD's
0.25 deg gauge rainfall grid has no valid cell anywhere over the Andaman,
Nicobar or Lakshadweep islands in any year 1971-2025, so those four units are
blank throughout, and the 1 deg temperature grid's land mask leaves a handful
of border and island units thin or blank.  That is a gap in the source, not a
gap in the crosswalk, and it is NOT filled from a neighbour: the nearest valid
rainfall cell to Nicobars is 1,600 km away.  The `coverage` column says so on
every row -- see below.

OUTPUT -> IMD_Data\lgd\{rain,tmax,tmin}\IMD_{var}_district_daily_YYYY.csv
          IMD_Data\_district_daily_{var}_lgd.pkl      (wide values)
          IMD_Data\_district_daily_{var}_cov_lgd.pkl  (wide coverage)

`coverage` is the fraction of the district's area that had data that day, so a
reader can tell the three reasons a cell looks empty apart:
  coverage = 0      no valid grid cell -> the value IS blank (islands, and one
                    IMD outage over coastal Saurashtra on 2023-10-12)
  0 < cov < 1       value came from part of the district only (Chennai draws on
                    28.6% of its area; treat as indicative)
  coverage = 1      full areal mean; a blank pct_departure here is the dry
                    season, not a hole (see the note on MIN_MM below)

Run:  py -3.13 -X utf8 "IMD_Data/build_imd_lgd_csvs.py"
"""
import pathlib
import sys

import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[0] / "v5"))
from common_v5 import log  # noqa
import download_imd as di  # noqa  (get_year, get_realtime_2026, aggregate...)

#   MIN_MM is why pct_departure is blank on 47% of all rainfall rows, and why
#   that share runs from 89% in January to 0.8% in July: a percentage against a
#   day-of-year normal below 1 mm is a division by roughly nothing, so it is
#   suppressed.  The blank tracks the monsoon, which is the point -- it is the
#   dry season showing through, not missing data.  Read rain_mm (and the
#   season-to-date sums built from it) outside the monsoon, never pct_departure.
SMOOTH, MIN_MM, MIN_C = 2, 1.0, 5.0

#   TWO DIFFERENT RANGES, AND THEY ARE NOT THE SAME RANGE.
#   ARCH_* is which years to read and aggregate: everything there is.
#   NORM_* is the climatology window the departures are measured against, and
#   it is 1971-2020 -- IMD's own Long Period Average window, adopted in April
#   2022 from 4,132 gauges.  The whole record was extended back to 1971
#   (extend_history_1971.py) precisely so this window could be IMD's, and
#   v5/monsooncast/cleaning/01_clean_merge_panel.py (WMO_LO, WMO_HI) uses it
#   for every normal in the panel, the features and the model target.
#
#   One constant used to serve both purposes, which quietly put these CSVs on a
#   1971-2025 climatology while the panel, the dashboard and the deck were all
#   on 1971-2020.  Same district, same day, two different departures depending
#   on which file you opened.  Splitting them fixes that; ARCH_HI still moves
#   forward every year, NORM_HI does not move until IMD moves it.
ARCH_LO, ARCH_HI = 1971, 2025
NORM_LO, NORM_HI = 1971, 2020
VARS = {"rain": ("rain_mm", "normal_mm", "rain"),
        "tmax": ("tmax_c", "normal_c", "temp"),
        "tmin": ("tmin_c", "normal_c", "temp")}


def load_cw(grid):
    """District x grid-cell weights.

    Prefers crosswalk_{grid}_areal.csv, where each cell is weighted by the
    fraction of the district's area it covers -- a true areal mean.  Falls back
    to the older cell-centre crosswalk (equal weights over the cells whose
    centre lies inside the polygon) only if the areal file is absent.

    The areal form matters most for TEMPERATURE: on the 1 deg grid the
    cell-centre method left 540 of 791 districts with no centre at all, so they
    were snapped to a single nearest cell.  Area weighting gives them a proper
    weighted mean of the cells that actually cover them (median 3).
    """
    import json
    m = json.loads((HERE / "lgd_system_meta.json").read_text())["grids"][grid]
    nlat, nlon = m["nlat"], m["nlon"]
    areal = HERE / f"crosswalk_{grid}_areal.csv"
    if areal.exists():
        cw = pd.read_csv(areal)
        wcol = cw["weight"].to_numpy(np.float32)
    else:
        cw = pd.read_csv(HERE / f"crosswalk_{grid}_lgd.csv").dropna(
            subset=["district_id"])
        wcol = np.ones(len(cw), dtype=np.float32)
    cw["district_id"] = cw["district_id"].astype(int)
    dids = np.sort(cw["district_id"].unique())
    pos = {d: i for i, d in enumerate(dids)}
    cell_flat = (cw["cell_row"].to_numpy() * nlon
                 + cw["cell_col"].to_numpy()).astype(np.int64)
    W = np.zeros((len(cw), len(dids)), dtype=np.float32)
    W[np.arange(len(cw)), cw["district_id"].map(pos).to_numpy()] = wcol
    return cell_flat, W, dids, nlat, nlon


def aggregate_variable(var):
    grid = "rain" if var == "rain" else "temp"
    cell_flat, onehot, dids, nlat, nlon = load_cw(grid)
    fill = di.FILL[var]
    frames, covs = [], []
    for y in range(ARCH_LO, ARCH_HI + 1):
        a = di.get_year(var, y)
        arr = a.values.astype(np.float32)
        if arr.shape[1:] != (nlat, nlon):
            continue
        dd, cc = di.aggregate(arr, fill, cell_flat, onehot, with_coverage=True)
        idx = pd.to_datetime(a.time.values)
        frames.append(pd.DataFrame(dd, index=idx, columns=dids))
        covs.append(pd.DataFrame(cc, index=idx, columns=dids))
    #   Hand the fetcher the days we already hold so it only asks IMD for what
    #   is new (plus its revision tail).  Without this the real-time loop walks
    #   1 January to today every single run.
    old_p = HERE / f"_district_daily_{var}_lgd.pkl"
    prev = None
    if old_p.exists():
        prev = pd.read_pickle(old_p)
        prev.columns = prev.columns.astype(int)
    have = (prev.index[prev.index.year == di.CUR] if prev is not None
            else None)
    rt, failed = di.get_realtime_2026(var, return_failures=True,
                                      have_days=have)
    if rt is not None:
        if grid == "temp":
            rt = di.subsample_temp_to_1deg(rt)
        arr = rt.values.astype(np.float32)
        if arr.shape[1:] == (nlat, nlon):
            dd, cc = di.aggregate(arr, fill, cell_flat, onehot,
                                  with_coverage=True)
            idx = pd.to_datetime(rt.time.values)
            frames.append(pd.DataFrame(dd, index=idx, columns=dids))
            covs.append(pd.DataFrame(cc, index=idx, columns=dids))
    wide = pd.concat(frames).sort_index()
    wide = wide[~wide.index.duplicated(keep="last")]
    wide.index.name = "date"
    cov = pd.concat(covs).sort_index()
    cov = cov[~cov.index.duplicated(keep="last")].reindex(wide.index)
    cov.index.name = "date"

    #   `wide` now holds the archive years plus ONLY the real-time days this
    #   run actually fetched.  Everything the incremental fetcher skipped is
    #   missing from it and has to come back from the previous product, or an
    #   incremental refresh would delete most of the current year.
    #   The freshly derived rows win on any date present in both.
    fresh = wide
    if prev is not None:
        keep = prev.index.difference(wide.index)
        if len(keep):
            wide = pd.concat([prev.loc[keep, wide.columns], wide]).sort_index()
            wide.index.name = "date"
            log(f"  {var}: carried {len(keep)} cached day(s) forward "
                f"({keep.min():%Y-%m-%d}..{keep.max():%Y-%m-%d})")
        cov = cov.reindex(wide.index)

    #   Before overwriting a product other scripts already consume, check that
    #   re-running the aggregation reproduces the days it produced last time.
    #   A silent change here would move every downstream departure, so it is
    #   reported rather than assumed.
    #   Compare against `fresh`, not `wide`: `wide` contains the previous
    #   product verbatim on every day the fetcher skipped, so checking it
    #   against `prev` would compare those rows with themselves and always
    #   pass.  `fresh` is the archive plus the days actually re-derived this
    #   run — the only rows where a genuine change could show up.
    if prev is not None:
        old = prev
        i = old.index.intersection(fresh.index)
        c = old.columns.intersection(fresh.columns)
        a1 = old.loc[i, c].to_numpy(np.float32)
        a2 = fresh.loc[i, c].to_numpy(np.float32)
        both = np.isfinite(a1) & np.isfinite(a2)
        dmax = float(np.abs(a1[both] - a2[both]).max()) if both.any() else 0.0
        flip = int((np.isfinite(a1) != np.isfinite(a2)).sum())
        log(f"  {var}: reproduced {len(i)} shared days — max |diff| {dmax:.6g}, "
            f"{flip} cells changed blank/not-blank")
        new_days = wide.index.difference(old.index)
        if len(new_days):
            log(f"  {var}: {len(new_days)} day(s) added: "
                f"{', '.join(f'{d:%Y-%m-%d}' for d in new_days[:10])}"
                + (" ..." if len(new_days) > 10 else ""))

    gaps = pd.date_range(wide.index.min(), wide.index.max(),
                         freq="D").difference(wide.index)
    if len(gaps):
        log(f"  {var}: WARNING {len(gaps)} calendar gap(s) remain: "
            f"{', '.join(f'{d:%Y-%m-%d}' for d in gaps[:10])}")
    if failed:
        log(f"  {var}: WARNING {len(failed)} real-time day(s) would not "
            f"download: {', '.join(failed[:10])}")

    #   Coverage has to be carried forward on the same days the values were,
    #   otherwise an incremental run leaves it NaN for most of the year and
    #   anything reading it sees a product that suddenly stopped in January.
    cov_p = HERE / f"_district_daily_{var}_cov_lgd.pkl"
    if cov_p.exists():
        pcov = pd.read_pickle(cov_p)
        pcov.columns = pcov.columns.astype(int)
        miss = cov.index[cov.isna().all(axis=1)]
        fill = pcov.index.intersection(miss)
        if len(fill):
            cov.loc[fill, :] = pcov.loc[fill, cov.columns].to_numpy()
            log(f"  {var}: carried {len(fill)} cached coverage row(s) forward")

    wide.to_pickle(old_p)
    cov.astype(np.float32).to_pickle(cov_p)
    log(f"  {var}: {wide.shape[0]} days x {wide.shape[1]} units "
        f"({wide.index.min():%Y-%m-%d}..{wide.index.max():%Y-%m-%d})")
    return wide, cov


def doy_normals(wide):
    """Day-of-year normal over NORM_LO..NORM_HI, pooled +/- SMOOTH days.

    Day-of-year is taken as pandas reports it, so in leap years everything
    after February is shifted by one against non-leap years and doy 60 pools
    29 Feb with 1 Mar.  The +/-2 day window is wider than that shift, so it
    washes out; doy 366 is the one to treat gently, since only the 13 leap
    years in the window feed it.
    """
    base = wide[(wide.index.year >= NORM_LO) & (wide.index.year <= NORM_HI)]
    per = base.groupby(base.index.dayofyear).mean().reindex(range(1, 367))
    arr = per.to_numpy()
    sm = np.nanmean(np.stack([np.roll(arr, k, axis=0)
                              for k in range(-SMOOTH, SMOOTH + 1)]), axis=0)
    return pd.DataFrame(sm, index=range(1, 367), columns=per.columns)


def write_csvs(var, wide, reg, cov=None):
    vc, nc, kind = VARS[var]
    norm = doy_normals(wide)
    dmap = reg.set_index("district_id")[["state", "district"]]
    outdir = HERE / "lgd" / var
    outdir.mkdir(parents=True, exist_ok=True)
    nm = norm.reset_index().melt(id_vars="index", var_name="district_id",
                                 value_name="normal").rename(
                                     columns={"index": "doy"})
    nm["district_id"] = nm["district_id"].astype(int)
    ny = 0
    for y, g in wide.groupby(wide.index.year):
        long = g.reset_index().melt(id_vars="date", var_name="district_id",
                                    value_name="value")
        long["district_id"] = long["district_id"].astype(int)
        long["doy"] = long["date"].dt.dayofyear
        long = long.merge(nm, on=["doy", "district_id"], how="left")
        long = long.merge(dmap, on="district_id", how="left")
        if cov is not None:
            cl = cov.loc[g.index].reset_index().melt(
                id_vars="date", var_name="district_id", value_name="coverage")
            cl["district_id"] = cl["district_id"].astype(int)
            long = long.merge(cl, on=["date", "district_id"], how="left")
            long["coverage"] = long["coverage"].astype(np.float32).round(4)
        long["date"] = long["date"].dt.strftime("%Y-%m-%d")
        with np.errstate(invalid="ignore", divide="ignore"):
            if kind == "rain":
                long["pct_departure"] = np.where(
                    long["normal"] >= MIN_MM,
                    (long["value"] - long["normal"]) / long["normal"] * 100,
                    np.nan)
                long = long.rename(columns={"value": vc, "normal": nc})
                cols = ["date", "state", "district", "district_id", vc, nc,
                        "pct_departure"]
                if cov is not None:
                    cols.append("coverage")
            else:
                long["anom_c"] = long["value"] - long["normal"]
                long["pct_departure"] = np.where(
                    long["normal"].abs() >= MIN_C,
                    (long["value"] - long["normal"]) / long["normal"] * 100,
                    np.nan)
                long = long.rename(columns={"value": vc, "normal": nc})
                cols = ["date", "state", "district", "district_id", vc, nc,
                        "anom_c", "pct_departure"]
                if cov is not None:
                    cols.append("coverage")
        long.sort_values(["date", "state", "district"])[cols].to_csv(
            outdir / f"IMD_{var}_district_daily_{y}.csv", index=False)
        ny += 1
    log(f"  {var}: wrote {ny} year CSVs to lgd/{var}/")


def main():
    log("=" * 68)
    log("IMD -> 791 LGD units: aggregate cached grids + write CSVs")
    log("=" * 68)
    reg = pd.read_csv(HERE / "registry_lgd791.csv")
    for var in ("rain", "tmax", "tmin"):
        wide, cov = aggregate_variable(var)
        write_csvs(var, wide, reg, cov)
    log("  done — LGD IMD pickles + CSVs written")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
