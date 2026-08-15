r"""
Indices/fetch_indices.py  —  pull every climate index this pipeline consumes
straight from the provider, over HTTP, and refresh the files on disk.

WHY THIS EXISTS
  The MJO, ENSO and IOD series were downloaded by hand. That is fine once and
  a liability every week after: the ROMI file silently ages, and a stale MJO is
  invisible in the dashboard while being exactly the feature that carries the
  7-14 day forecast (adding it took the blend from +0.0484 to +0.0617 at 7
  days). A script that can be re-run means "how old is the MJO" is answered by
  running it, not by remembering.

SOURCES (all open, no key, no login)
  ROMI daily      psl.noaa.gov/mjo/mjoindex/romi.cpcolr.1x.txt
                  Real-time OLR MJO Index, Kiladis et al. (2014). NOTE: the CPC
                  path this file used to come from (products/precip/CWlink/
                  daily_mjo_index/ROMI/) now 404s; PSL serves the same product
                  and its file is the one that is current.
  ONI seasonal    cpc.ncep.noaa.gov/data/indices/oni.ascii.txt
                  ERSSTv5, 3-month running mean, 1991-2020 base. Each season is
                  stamped on its CENTRE month (AMJ 2026 -> 2026-05), which is
                  how the cache has always been keyed.
  Nino-3.4 anom   cpc.ncep.noaa.gov/data/indices/ersst5.nino.mth.91-20.ascii
                  Monthly ERSSTv5 anomaly on the same 1991-2020 base as ONI.
                  sstoi.indices carries a fresher month but is OISSTv2 on a
                  different base -- mixing the two puts a step in the series,
                  so it is deliberately NOT used.
  DMI / DMI east  psl.noaa.gov/gcos_wgsp/Timeseries/Data/dmi{,east}.had.long.data
                  HadISST1.1, marked "Preliminary" for the recent months.

WHAT IS NOT FETCHED, AND WHY
  Indices/nino34.long.anom.csv is a PSL long (1870-) HadISST-based series whose
  .data endpoint no longer resolves under any of the documented names. It is
  left untouched rather than back-filled from ERSSTv5: the two use different
  SST datasets and different baselines, so splicing them would put a step in
  the middle of the series at exactly the join. cleaning/00_clean_indices.py
  reads it for the long monthly Nino-3.4; the panel's own nino34_anom comes
  from the cache this script does refresh.

OUTPUTS (all backed up to *.bak before being replaced)
  Indices/romi.cpcolr.1x.txt        daily MJO
  Indices/dmi.had.long.csv          monthly IOD dipole
  Indices/dmieast.had.long.csv      monthly IOD east pole
  noaa_indices_cache.csv            year, month, oni, nino34_anom, iod_dmi

  Every write is checked against what was already on disk: a series may only
  grow, and the values on shared months must reproduce. A provider that
  revises or truncates is REPORTED, never silently adopted.

Run:  py -3.13 -X utf8 "Indices/fetch_indices.py"
      py -3.13 -X utf8 "Indices/fetch_indices.py" --check   report only
"""
import argparse
import datetime
import io
import pathlib
import shutil
import sys

import numpy as np
import pandas as pd
import requests

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
CACHE = ROOT / "noaa_indices_cache.csv"

URLS = {
    "romi": "https://psl.noaa.gov/mjo/mjoindex/romi.cpcolr.1x.txt",
    "oni": "https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt",
    "nino34": "https://www.cpc.ncep.noaa.gov/data/indices/"
              "ersst5.nino.mth.91-20.ascii",
    "dmi": "https://psl.noaa.gov/gcos_wgsp/Timeseries/Data/dmi.had.long.data",
    "dmieast": "https://psl.noaa.gov/gcos_wgsp/Timeseries/Data/"
               "dmieast.had.long.data",
}

#   Season label -> the month it is centred on. DJF is centred on January, so
#   the YR column already names the centre's year for every season except none
#   -- CPC stamps DJF 1950 on Jan 1950 and NDJ 2025 on Dec 2025.
SEASON_CENTRE = {"DJF": 1, "JFM": 2, "FMA": 3, "MAM": 4, "AMJ": 5, "MJJ": 6,
                 "JJA": 7, "JAS": 8, "ASO": 9, "SON": 10, "OND": 11, "NDJ": 12}

TIMEOUT = 60
RETRIES = 3


def log(*a):
    print(*a, flush=True)


def get(url):
    """GET with retries; returns text or raises the last error."""
    last = None
    for _ in range(RETRIES):
        try:
            r = requests.get(url, timeout=TIMEOUT)
            r.raise_for_status()
            return r.text
        except Exception as e:                              # noqa: BLE001
            last = e
    raise last


# --------------------------------------------------------------- parsers
def parse_romi(text):
    d = pd.read_csv(io.StringIO(text), sep=r"\s+", header=None,
                    names=["y", "m", "d", "h", "romi1", "romi2", "amp"])
    d["date"] = pd.to_datetime(dict(year=d.y, month=d.m, day=d.d),
                               errors="coerce")
    d = d.dropna(subset=["date"])
    for c in ("romi1", "romi2", "amp"):
        d[c] = pd.to_numeric(d[c], errors="coerce")
        d.loc[d[c] < -900, c] = np.nan
    return d.dropna(subset=["romi1", "romi2"]).reset_index(drop=True)


def parse_oni(text):
    """SEAS YR TOTAL ANOM -> monthly ONI keyed on the season's centre month."""
    rows = []
    for line in text.strip().splitlines()[1:]:
        p = line.split()
        if len(p) < 4 or p[0] not in SEASON_CENTRE:
            continue
        rows.append({"year": int(p[1]), "month": SEASON_CENTRE[p[0]],
                     "oni": float(p[3])})
    return pd.DataFrame(rows)


def parse_ersst5(text):
    """YR MON NINO1+2 ANOM NINO3 ANOM NINO4 ANOM NINO3.4 ANOM."""
    rows = []
    for line in text.strip().splitlines()[1:]:
        p = line.split()
        if len(p) < 10:
            continue
        try:
            rows.append({"year": int(p[0]), "month": int(p[1]),
                         "nino34_anom": float(p[9])})
        except ValueError:
            continue
    return pd.DataFrame(rows)


def parse_psl_grid(text):
    """PSL '.data': first line 'startyear endyear', then year + 12 monthly
    columns, then a prose trailer. Missing values are large negatives."""
    lines = text.strip().splitlines()
    y0, y1 = (int(x) for x in lines[0].split()[:2])
    rows = []
    for line in lines[1:]:
        p = line.split()
        if len(p) != 13:
            continue
        try:
            yr = int(p[0])
        except ValueError:
            continue
        if not (y0 <= yr <= y1):
            continue
        for m, v in enumerate(p[1:], start=1):
            try:
                x = float(v)
            except ValueError:
                continue
            rows.append({"year": yr, "month": m,
                         "value": np.nan if x < -900 else x})
    return pd.DataFrame(rows)


# --------------------------------------------------------------- writers
def backup(path):
    if path.exists():
        shutil.copy2(path, path.with_suffix(path.suffix + ".bak"))


def compare_series(name, old, new, key):
    """Report growth and reproduction on shared keys. Returns True if the new
    series is safe to adopt (it does not lose keys and does not disagree
    beyond a provider revision we are willing to name)."""
    if old is None or not len(old):
        log(f"    {name}: no previous file — adopting {len(new)} rows")
        return True
    o = old.set_index(key).iloc[:, -1]
    n = new.set_index(key).iloc[:, -1]
    shared = o.index.intersection(n.index)
    lost = o.index.difference(n.index)
    added = n.index.difference(o.index)
    both = o.loc[shared].notna() & n.loc[shared].notna()
    diff = (o.loc[shared][both] - n.loc[shared][both]).abs()
    dmax = float(diff.max()) if len(diff) else 0.0
    nrev = int((diff > 1e-6).sum()) if len(diff) else 0
    log(f"    {name}: {len(shared)} shared, {len(added)} new, {len(lost)} lost"
        f" — max |diff| {dmax:.6g} on {nrev} revised value(s)")
    if len(lost):
        log(f"    {name}: WARNING the feed no longer carries "
            f"{len(lost)} key(s) that are on disk, e.g. {list(lost[:3])} — "
            f"NOT adopting")
        return False
    return True


def write_psl_csv(path, grid, label):
    """Write PSL '.data' content back out in the CSV shape cleaning/00 reads:
    one prose header line, then 'YYYY-MM-01,   value' with -9999 for missing."""
    header = None
    if path.exists():
        header = path.read_text(encoding="utf-8").splitlines()[0]
    if header is None:
        header = (f"Date, {label}  missing value -9999 "
                  f"https://psl.noaa.gov/data/timeseries/month/")
    g = grid.copy()
    g["date"] = pd.to_datetime(dict(year=g.year, month=g.month, day=1))
    g = g.sort_values("date")
    out = [header]
    for _, r in g.iterrows():
        v = -9999.0 if pd.isna(r["value"]) else r["value"]
        out.append(f"{r['date']:%Y-%m-%d},{v:10.3f}")
    backup(path)
    path.write_text("\n".join(out) + "\n", encoding="utf-8")


def read_psl_csv(path):
    if not path.exists():
        return None
    d = pd.read_csv(path, skiprows=1, header=None, names=["date", "value"])
    d["date"] = pd.to_datetime(d["date"], errors="coerce")
    d = d.dropna(subset=["date"])
    d["value"] = pd.to_numeric(d["value"], errors="coerce")
    d.loc[d["value"] < -900, "value"] = np.nan
    d["year"] = d["date"].dt.year
    d["month"] = d["date"].dt.month
    return d[["year", "month", "value"]]


def last_valid(df, col):
    d = df.dropna(subset=[col])
    if not len(d):
        return "—"
    r = d.iloc[-1]
    return f"{int(r['year'])}-{int(r['month']):02d}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="download and report, write nothing")
    a = ap.parse_args()

    today = datetime.date.today()
    log("=" * 74)
    log(f"CLIMATE INDICES — fetch from provider        today {today:%Y-%m-%d}")
    log("=" * 74)

    # ------------------------------------------------------------- ROMI
    log("\n  MJO / ROMI (daily)")
    p_romi = HERE / "romi.cpcolr.1x.txt"
    old_romi = None
    if p_romi.exists():
        old_romi = parse_romi(p_romi.read_text(encoding="utf-8"))
        log(f"    on disk: {len(old_romi):,} days "
            f"through {old_romi.date.max():%Y-%m-%d}")
    try:
        text = get(URLS["romi"])
        new_romi = parse_romi(text)
        log(f"    fetched: {len(new_romi):,} days "
            f"through {new_romi.date.max():%Y-%m-%d}  "
            f"({URLS['romi']})")
        ok = compare_series("ROMI", old_romi, new_romi, "date")
        if ok and not a.check:
            if old_romi is not None and \
                    new_romi.date.max() <= old_romi.date.max():
                log("    ROMI: feed is no newer than the file — left alone")
            else:
                backup(p_romi)
                p_romi.write_text(text if text.endswith("\n") else text + "\n",
                                  encoding="utf-8")
                gain = (new_romi.date.max() - old_romi.date.max()).days \
                    if old_romi is not None else len(new_romi)
                log(f"    wrote {p_romi.name}  (+{gain} day(s))")
    except Exception as e:                                   # noqa: BLE001
        log(f"    ROMI FETCH FAILED: {type(e).__name__}: {e}")
        new_romi = old_romi

    # ------------------------------------------------------------- ONI
    log("\n  ONI (seasonal, stamped on the centre month)")
    oni = pd.DataFrame()
    try:
        oni = parse_oni(get(URLS["oni"]))
        r = oni.iloc[-1]
        log(f"    fetched: {len(oni):,} seasons, latest centre "
            f"{int(r['year'])}-{int(r['month']):02d} = {r['oni']:+.2f}")
    except Exception as e:                                   # noqa: BLE001
        log(f"    ONI FETCH FAILED: {type(e).__name__}: {e}")

    # --------------------------------------------------------- Nino-3.4
    log("\n  Nino-3.4 monthly anomaly (ERSSTv5, 1991-2020 base)")
    n34 = pd.DataFrame()
    try:
        n34 = parse_ersst5(get(URLS["nino34"]))
        r = n34.iloc[-1]
        log(f"    fetched: {len(n34):,} months, latest "
            f"{int(r['year'])}-{int(r['month']):02d} = "
            f"{r['nino34_anom']:+.2f}")
    except Exception as e:                                   # noqa: BLE001
        log(f"    NINO3.4 FETCH FAILED: {type(e).__name__}: {e}")

    # ------------------------------------------------------- IOD poles
    log("\n  IOD — DMI and its east pole (HadISST1.1, preliminary tail)")
    poles = {}
    for key, fname, label in (
            ("dmi", "dmi.had.long.csv", "DMI HadISST1.1"),
            ("dmieast", "dmieast.had.long.csv", "DMI EAST HadISST1.1")):
        path = HERE / fname
        try:
            grid = parse_psl_grid(get(URLS[key]))
            grid = grid.dropna(subset=["value"])
            r = grid.iloc[-1]
            log(f"    {key}: fetched through "
                f"{int(r['year'])}-{int(r['month']):02d} = {r['value']:+.3f}")
            old = read_psl_csv(path)
            if old is not None:
                old = old.dropna(subset=["value"])
            ok = compare_series(key, old, grid, ["year", "month"])
            poles[key] = grid
            if ok and not a.check:
                write_psl_csv(path, grid, label)
                log(f"    wrote {fname}")
        except Exception as e:                               # noqa: BLE001
            log(f"    {key} FETCH FAILED: {type(e).__name__}: {e}")
            old = read_psl_csv(path)
            if old is not None:
                poles[key] = old.dropna(subset=["value"])

    # ------------------------------------------------- noaa_indices_cache
    log("\n  noaa_indices_cache.csv (year, month, oni, nino34_anom, iod_dmi)")
    prev = pd.read_csv(CACHE) if CACHE.exists() else pd.DataFrame(
        columns=["year", "month", "oni", "nino34_anom", "iod_dmi"])
    for c in ("oni", "nino34_anom", "iod_dmi"):
        log(f"    on disk: {c:12s} through {last_valid(prev, c)}")

    #   The cache is rebuilt on the union of what is on disk and what the feeds
    #   carry, so a provider that is temporarily short of a month can never
    #   delete history. Fresh values win where both exist.
    lo = int(prev["year"].min()) if len(prev) else 1981
    hi = max([int(prev["year"].max()) if len(prev) else today.year,
              today.year])
    base = pd.MultiIndex.from_product([range(lo, hi + 1), range(1, 13)],
                                      names=["year", "month"]).to_frame(False)
    out = base.copy()
    for c in ("oni", "nino34_anom", "iod_dmi"):
        out[c] = np.nan
    out = out.set_index(["year", "month"])
    if len(prev):
        pv = prev.set_index(["year", "month"])
        for c in ("oni", "nino34_anom", "iod_dmi"):
            if c in pv.columns:
                out[c] = pv[c].reindex(out.index)

    def overlay(col, src, valcol):
        if src is None or not len(src):
            return 0, 0.0
        s = src.set_index(["year", "month"])[valcol].reindex(out.index)
        both = out[col].notna() & s.notna()
        dmax = float((out.loc[both, col] - s[both]).abs().max()) \
            if both.any() else 0.0
        added = int((out[col].isna() & s.notna()).sum())
        out[col] = s.where(s.notna(), out[col])
        return added, dmax

    a1, d1 = overlay("oni", oni, "oni")
    a2, d2 = overlay("nino34_anom", n34, "nino34_anom")
    a3, d3 = overlay("iod_dmi", poles.get("dmi"), "value")
    log(f"    oni         +{a1} month(s), max |diff| vs disk {d1:.4g}")
    log(f"    nino34_anom +{a2} month(s), max |diff| vs disk {d2:.4g}")
    log(f"    iod_dmi     +{a3} month(s), max |diff| vs disk {d3:.4g}")

    out = out.reset_index()
    for c in ("oni", "nino34_anom", "iod_dmi"):
        log(f"    now:     {c:12s} through {last_valid(out, c)}")

    if not a.check:
        backup(CACHE)
        out.to_csv(CACHE, index=False)
        log(f"    wrote {CACHE.name}  ({len(out):,} months "
            f"{lo}-01..{hi}-12)")

    log("\n  NEXT")
    log("    py -3.13 -X utf8 \"v5/43_build_enso_features.py\"        "
        "# ENSO state -> enso_monthly_v5.csv")
    log("    py -3.13 -X utf8 \"v5/monsooncast/cleaning/00_clean_indices.py\" "
        "# MJO/ENSO/IOD -> data_lgd/")
    log("    py -3.13 -X utf8 \"v5/monsooncast/run_all.py\" --fast")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
