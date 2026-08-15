r"""
IMD_Data/download_imd.py  —  download IMD gridded rainfall and temperature
(1981-2026) and reduce every day to the 701-unit district registry, so the
result drops into the v5 pipeline exactly like the CHIRPS district files but
on the gauge-based IMD scale.

SOURCES (via imdlib)
  rain    0.25 deg daily, archive 1981-2025 (yearwise) + real-time 2026
  tmax    1.00 deg daily, archive 1981-2025 (yearwise) + real-time 2026
  tmin    1.00 deg daily, archive 1981-2025 (yearwise) + real-time 2026
  Sea/no-data fill is -999 for rain and 99.9 for temperature.  The real-time
  temperature grid is 0.5 deg on nodes that include every 1 deg archive node,
  so it is subsampled to the archive grid to keep the record on one grid.

AGGREGATION
  Grid -> district via IMD_Data\crosswalk_{rain,temp}.csv (built by
  build_crosswalk.py): each district's value on a day is the mean of the IMD
  cells assigned to it, computed as a masked matrix product so a cell that is
  the fill value on a given day is dropped from that day's average only.

OUTPUT -> IMD_Data\_district_daily_{var}.pkl
  a wide table, index = date, columns = district_id, values = the daily
  district mean (mm for rain, deg C for temperature).  The finaliser
  (finalize_imd.py) turns these into the GEE-format per-year CSVs.

Run:  py -3.13 -X utf8 "IMD_Data/download_imd.py"
"""
import pathlib
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "v5"))
from common_v5 import log  # noqa
import imdlib as imd
import socket
# imdlib issues requests with no timeout, so a stalled IMD server hangs the
# whole run forever (that is exactly what happened to tmin at year 2005).  A
# default socket timeout turns a stall into a retryable exception.
socket.setdefaulttimeout(90)

HERE = pathlib.Path(__file__).resolve().parent
RAW = HERE / "raw"
RAW.mkdir(parents=True, exist_ok=True)

ARCH_LO, ARCH_HI = 1981, 2025      # yearwise archive
CUR = 2026                         # real-time
FILL = {"rain": -999.0, "tmax": 99.9, "tmin": 99.9}
GRID = {"rain": "rain", "tmax": "temp", "tmin": "temp"}


def load_crosswalk(name):
    """Return (entry_cell_flat_index, district_group_index, district_ids,
    nlat, nlon) for a fast masked matrix aggregation."""
    import json
    meta = json.loads((HERE / "crosswalk_meta.json").read_text())["grids"][name]
    nlat, nlon = meta["nlat"], meta["nlon"]
    cw = pd.read_csv(HERE / f"crosswalk_{name}.csv").dropna(subset=["district_id"])
    cw["district_id"] = cw["district_id"].astype(int)
    dids = np.sort(cw["district_id"].unique())
    pos = {d: i for i, d in enumerate(dids)}
    cell_flat = (cw["cell_row"].to_numpy() * nlon
                 + cw["cell_col"].to_numpy()).astype(np.int64)
    grp = cw["district_id"].map(pos).to_numpy().astype(np.int64)
    # one-hot (n_entries x n_districts) as float32 for the matrix product
    onehot = np.zeros((len(cw), len(dids)), dtype=np.float32)
    onehot[np.arange(len(cw)), grp] = 1.0
    return cell_flat, onehot, dids, nlat, nlon


def aggregate(arr, fill, cell_flat, onehot, with_coverage=False):
    """arr: (T, nlat, nlon) -> (T, n_districts) masked cell-mean per district.

    With the areal crosswalk each district's weights sum to one, so `c` below
    is the FRACTION OF THE DISTRICT'S AREA that had data on that day, and the
    mean is renormalised over just that part.  That fraction is the single most
    useful piece of provenance this product can carry: it is 0 exactly where
    the output is blank, and it is small (not blank) for the narrow coastal
    units whose value comes from a sliver of land cells.  Pass
    with_coverage=True to get it back alongside the values.
    """
    T = arr.shape[0]
    flat = arr.reshape(T, -1)[:, cell_flat]          # (T, n_entries)
    valid = np.isfinite(flat) & (flat != fill)
    vals = np.where(valid, flat, 0.0).astype(np.float32)
    s = vals @ onehot                                 # (T, n_dist) sum
    c = valid.astype(np.float32) @ onehot             # (T, n_dist) count
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(c > 0, s / c, np.nan)
    return (out, c) if with_coverage else out


def get_year(var, y):
    """Open a cached .grd if present, else download; return the DataArray.

    Download is retried a few times because the IMD server occasionally
    times out on a single year (now caught by the socket timeout instead of
    hanging)."""
    try:
        ds = imd.open_data(var, y, y, "yearwise", str(RAW))
    except Exception:
        last = None
        for _ in range(3):
            try:
                ds = imd.get_data(var, y, y, fn_format="yearwise",
                                  file_dir=str(RAW))
                break
            except Exception as e:
                last = e
        else:
            raise last
    x = ds.get_xarray()
    return x[list(x.data_vars)[0]]


#   How many days back from the live edge are always re-fetched, even when
#   they are already aggregated.  IMD's real-time values are PRELIMINARY and
#   get revised for a few days after first publication, so trusting the cache
#   right up to today would freeze whatever the first release happened to say.
#   Ten days is comfortably past the revision window and costs ten requests.
REFETCH_TAIL = 10


def get_realtime_2026(var, return_failures=False, have_days=None):
    """Fetch the current year ONE DAY AT A TIME from the real-time endpoint,
    up to the most recent day that exists.

    INCREMENTAL BY DEFAULT.  `have_days` is the set of dates already present
    in the aggregated product; those are skipped unless they fall inside
    REFETCH_TAIL of the live edge.

      Without it this loop walks 1 January to today on EVERY run.  Measured on
      this machine that is about 50 seconds per day per variable once IMD's
      endpoint is slow -- roughly nine hours of requests to obtain the four
      days that were actually missing, and the run had stalled inside a retry
      long before finishing.  A refresh has to cost something proportional to
      what is new, or nobody runs it.

    The previous version requested a whole calendar month (e.g. 01--31 Jul);
    because the days after the last IMD release (24--31 Jul) do not exist yet,
    imdlib raised on the range and the ENTIRE month was dropped -- which is why
    July 2026 never reached the pickle even though 01--23 Jul were on disk.
    Fetching per day keeps every day that is actually published and simply
    stops at the live edge; cached days re-read instantly.

    EACH DAY IS RETRIED.  A single request that times out is not evidence that
    the day does not exist, but the first version of this loop treated it that
    way: one failed fetch of 07 Jun 2026 for tmin put a permanent one-day hole
    in the tmin record, which then sat in the published CSVs (the day was
    downloadable the whole time).  Missing days are retried and, if they still
    fail, returned to the caller so a hole is reported rather than silently
    shipped.  The end of the year is bounded by `<= today`, because IMD does
    publish the current day; the miss counter is what finds the live edge."""
    import datetime
    import xarray as xr
    parts, misses, failed = [], 0, []
    d = datetime.date(CUR, 1, 1)
    today = datetime.date.today()
    # `have_days or []` is ambiguous for a DatetimeIndex — pandas raises on
    # its truth value rather than treating a non-empty index as true.
    have = set() if have_days is None else {pd.Timestamp(x).date()
                                            for x in have_days}
    tail_from = today - datetime.timedelta(days=REFETCH_TAIL)
    skipped = 0
    while d <= today:
        s = d.isoformat()
        if d in have and d < tail_from:
            skipped += 1
            d += datetime.timedelta(days=1)
            continue
        got = None
        for _ in range(3):
            try:
                ds = imd.get_real_data(var, s, s, file_dir=str(RAW))
                got = ds.get_xarray()
                break
            except Exception:
                continue
        if got is not None:
            parts.append(got[list(got.data_vars)[0]])
            misses = 0
        else:
            misses += 1
            failed.append(s)
            # a run of misses at the end means we have reached the live edge
            if parts and misses >= 7:
                failed = failed[:-misses]     # the edge is not a hole
                break
        d += datetime.timedelta(days=1)
    if skipped:
        print(f"    {var}: {skipped} cached day(s) skipped, "
              f"{len(parts)} fetched (tail from {tail_from})")
    if not parts:
        return (None, failed) if return_failures else None
    out = xr.concat(parts, dim="time")
    return (out, failed) if return_failures else out


def subsample_temp_to_1deg(x):
    """Real-time temperature is 0.5 deg; take the nodes that coincide with the
    1 deg archive grid (7.5, 8.5, ...)."""
    lat = x.lat.values
    lon = x.lon.values
    ilat = np.where(np.isclose((lat - 7.5) % 1.0, 0))[0]
    ilon = np.where(np.isclose((lon - 67.5) % 1.0, 0))[0]
    return x.isel(lat=ilat, lon=ilon)


def build_variable(var):
    log("=" * 68)
    log(f"{var}: download + aggregate to districts")
    log("=" * 68)
    cell_flat, onehot, dids, nlat, nlon = load_crosswalk(GRID[var])
    fill = FILL[var]
    frames = []

    # ---- archive 1981-2025: reuse a completed pickle if present ----------
    # The archive never changes, so once a variable's pickle holds it there is
    # no reason to re-download 45 years just to refresh the current season.
    pkl = HERE / f"_district_daily_{var}.pkl"
    cached = None
    if pkl.exists():
        cached = pd.read_pickle(pkl)
        cached.columns = cached.columns.astype(int)
        arch = cached[cached.index.year <= ARCH_HI]
        if arch.index.year.min() <= ARCH_LO and arch.index.year.max() >= ARCH_HI:
            frames.append(arch)
            log(f"  archive reused from {pkl.name}: "
                f"{arch.index.min():%Y-%m-%d}..{arch.index.max():%Y-%m-%d}")
    if not frames:
        for y in range(ARCH_LO, ARCH_HI + 1):
            t0 = time.time()
            try:
                a = get_year(var, y)
            except Exception as e:
                log(f"  {y}: FAILED {type(e).__name__} {str(e)[:100]}")
                continue
            arr = a.values.astype(np.float32)
            if arr.shape[1:] != (nlat, nlon):
                log(f"  {y}: grid {arr.shape[1:]} != ({nlat},{nlon}) — skipped")
                continue
            dd = aggregate(arr, fill, cell_flat, onehot)
            idx = pd.to_datetime(a.time.values)
            frames.append(pd.DataFrame(dd, index=idx, columns=dids))
            if y % 5 == 0 or y == ARCH_HI:
                log(f"  {y}: {arr.shape[0]:3d} days -> districts "
                    f"({time.time()-t0:.1f}s)")

    # ---- current year via real-time -------------------------------------
    log(f"  {CUR}: real-time endpoint (day by day)")
    rt = get_realtime_2026(var)
    if rt is not None:
        if GRID[var] == "temp":
            rt = subsample_temp_to_1deg(rt)
        arr = rt.values.astype(np.float32)
        if arr.shape[1:] == (nlat, nlon):
            dd = aggregate(arr, fill, cell_flat, onehot)
            idx = pd.to_datetime(rt.time.values)
            frames.append(pd.DataFrame(dd, index=idx, columns=dids))
            log(f"  {CUR}: {arr.shape[0]} days -> "
                f"{idx.min():%Y-%m-%d}..{idx.max():%Y-%m-%d}")
        else:
            log(f"  {CUR}: grid {arr.shape[1:]} != ({nlat},{nlon}) — skipped")
    else:
        log(f"  {CUR}: no real-time data retrieved")

    wide = pd.concat(frames).sort_index()
    wide = wide[~wide.index.duplicated(keep="last")]
    wide.index.name = "date"
    out = HERE / f"_district_daily_{var}.pkl"
    wide.to_pickle(out)
    log(f"  wrote {out.name}: {wide.shape[0]} days x {wide.shape[1]} districts "
        f"({wide.index.min():%Y-%m-%d}..{wide.index.max():%Y-%m-%d})\n")


def main():
    for var in ("rain", "tmax", "tmin"):
        build_variable(var)
    log("done — district-daily pickles written for rain, tmax, tmin")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
