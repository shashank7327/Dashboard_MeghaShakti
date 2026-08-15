r"""
v5/monsooncast/forecast_input/25_ecmwf_opendata_lgd.py  —  ECMWF open-data
ensemble precipitation, aggregated to the 791 LGD districts.

IS THIS POSSIBLE?  YES FOR LIVE FORECASTS, NO FOR TRAINING.  Read this before
building anything on top of it, because the distinction decides what the data
can be used for.

  WHAT ECMWF OPEN DATA IS
    A real-time dissemination service, free and unlicensed for reuse, carrying
    the operational HRES and ENS forecasts as GRIB2 on a 0.25 deg grid.  The
    ensemble runs out to 360 hours (15 days) at 3-hourly steps to 144 h and
    6-hourly thereafter.  Data appears 7-9 hours after each run time.

  WHAT IT IS NOT
    An archive.  It is a rolling real-time feed: only the most recent runs are
    retained, and there is no reforecast or hindcast through this channel.

  WHY THAT MATTERS HERE
    Post-processing an NWP ensemble to district scale -- Model Output
    Statistics -- is the technique that would actually give this system skill
    at 7-14 days.  MOS requires a REFORECAST ARCHIVE: many years of past
    forecasts paired with what was observed, so the correction can be trained.
    Open data cannot supply that.  With open data alone you can DISPLAY today's
    ECMWF ensemble on the district map, bias-corrected only by a climatological
    adjustment, but you cannot train a correction and you cannot verify skill.

  WHERE THE TRAINING DATA WOULD COME FROM
    * ECMWF S2S reforecast database -- 20 years of reforecasts, research
      access, the standard source for this exact purpose
    * ECMWF MARS / Climate Data Store -- ERA5 reanalysis is open; operational
      reforecasts are licensed
    * AWS Open Data registry mirror of ECMWF real-time forecasts -- longer
      rolling retention than the FTP feed, but still not a full reforecast set
    * Simply archiving this script's output daily from now on: after two or
      three monsoon seasons there is enough paired data to fit a first MOS
      correction.  That is the cheapest route and it starts today.

DEPENDENCIES (none of these are installed on the machine this was written on)
    py -3.13 -m pip install ecmwf-opendata cfgrib eccodes xarray

  cfgrib needs the ECMWF eccodes binaries.  The `eccodes` wheel bundles them on
  Windows; if `import cfgrib` still fails, `pip install eccodes-python` or a
  conda-forge eccodes is the fallback.  The script checks and tells you rather
  than failing halfway through a download.

OUTPUT -> ECMWF_LGD/
    ecmwf_ens_district_YYYYMMDDHH.csv
      district_id, state, district, valid_date, step_h,
      tp_mm_mean, tp_mm_p10, tp_mm_p90, tp_spread_mm, members

Run:  py -3.13 -X utf8 "v5/monsooncast/forecast_input/25_ecmwf_opendata_lgd.py"
      py -3.13 ... --check          only verify dependencies and exit
      py -3.13 ... --steps 24,48,72 restrict the steps fetched
"""
import argparse
import pathlib
import sys

import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
V5 = HERE.parents[1]
ROOT = V5.parent
IMD = ROOT / "IMD_Data"
OUT = ROOT / "ECMWF_LGD"
sys.path.insert(0, str(V5))
from common_v5 import log  # noqa

# ENS is 3-hourly to 144 h then 6-hourly to 360 h; these are the daily
# accumulation boundaries that matter for a 7- and 14-day district product.
DEFAULT_STEPS = list(range(24, 145, 24)) + list(range(156, 361, 12))
GRID = 0.25                     # degrees, the only open-data resolution
INDIA = dict(north=38.0, south=6.0, west=67.0, east=98.0)


def check_deps(verbose=True):
    missing = []
    for mod, pipname in (("ecmwf.opendata", "ecmwf-opendata"),
                         ("cfgrib", "cfgrib"),
                         ("xarray", "xarray")):
        try:
            __import__(mod)
        except Exception:
            missing.append(pipname)
    if verbose:
        if missing:
            log("  MISSING dependencies: " + ", ".join(missing))
            log("    py -3.13 -m pip install " + " ".join(missing))
            log("    cfgrib also needs the eccodes binaries; the `eccodes` "
                "wheel bundles them on Windows.")
        else:
            log("  dependencies present: ecmwf-opendata, cfgrib, xarray")
    return missing


def build_crosswalk():
    r"""Grid cell -> district areal weights for the 0.25 deg ECMWF grid.

    Deliberately the SAME construction the IMD crosswalk uses: intersect the
    reprojected district polygons with the grid and weight by overlap area.
    Assigning each cell to the district containing its centre breaks down for
    districts smaller than a cell, and at 0.25 deg (~28 km) a great many LGD
    districts are smaller than a cell.
    """
    try:
        import shapefile
        from shapely.geometry import box, shape
        from shapely.strtree import STRtree
    except Exception as e:
        raise SystemExit(f"needs pyshp and shapely: {e}")

    shp = IMD / "lgd_shapefile" / "india_districts_lgd.shp"
    if not shp.exists():
        raise SystemExit(f"LGD shapefile not found at {shp}")
    reg = pd.read_csv(IMD / "registry_lgd791.csv")
    sys.path.insert(0, str(IMD))
    from build_crosswalk import norm_state, norm_key  # noqa
    lut = {(s, d): i for s, d, i in zip(norm_state(reg["state"]),
                                        norm_key(reg["district"]),
                                        reg["district_id"])}

    r = shapefile.Reader(str(shp))
    fld = [f[0] for f in r.fields[1:]]
    iD, iS = fld.index("DISTRICT"), fld.index("STATE_UT")
    geoms, dids = [], []
    for sh, rec in zip(r.iterShapes(), r.iterRecords()):
        did = lut.get((norm_state(pd.Series([rec[iS]]))[0],
                       norm_key(pd.Series([rec[iD]]))[0]))
        if did is None:
            continue
        geoms.append(shape(sh.__geo_interface__))
        dids.append(int(did))
    tree = STRtree(geoms)

    lats = np.arange(INDIA["south"], INDIA["north"] + 1e-9, GRID)
    lons = np.arange(INDIA["west"], INDIA["east"] + 1e-9, GRID)
    rows = []
    for la in lats:
        for lo in lons:
            cell = box(lo - GRID / 2, la - GRID / 2,
                       lo + GRID / 2, la + GRID / 2)
            # "intersects", not "contains": shapely 2.x applies the predicate
            # as input.predicate(tree_geom), so contains would ask whether the
            # cell contains the district and always be false
            for j in tree.query(cell, predicate="intersects"):
                inter = cell.intersection(geoms[j])
                if inter.is_empty:
                    continue
                rows.append({"lat": round(float(la), 4),
                             "lon": round(float(lo), 4),
                             "district_id": dids[j],
                             "w": float(inter.area)})
    cw = pd.DataFrame(rows)
    cw["w"] = cw["w"] / cw.groupby("district_id")["w"].transform("sum")
    log(f"  crosswalk: {len(cw):,} cell-district overlaps, "
        f"{cw['district_id'].nunique()} districts covered")
    return cw


def fetch(steps, run_time=0):
    from ecmwf.opendata import Client
    OUT.mkdir(parents=True, exist_ok=True)
    client = Client(source="ecmwf")
    tgt = OUT / "_ens_tp.grib2"
    log(f"  requesting ENS total precipitation, {len(steps)} steps "
        f"(max {max(steps)} h), run {run_time:02d}Z ...")
    # type="pf" is the perturbed ensemble: 50 members, which is what makes a
    # spread and percentiles meaningful. "em"/"es" would give only the mean and
    # standard deviation and throw away the distribution.
    client.retrieve(time=run_time, stream="enfo", type="pf",
                    param="tp", step=steps, target=str(tgt))
    log(f"  downloaded {tgt.stat().st_size/1e6:.1f} MB")
    return tgt


def to_districts(grib, cw, reg):
    import xarray as xr
    ds = xr.open_dataset(grib, engine="cfgrib",
                         backend_kwargs={"indexpath": ""})
    var = "tp" if "tp" in ds else list(ds.data_vars)[0]
    da = ds[var]
    log(f"  grib: {dict(da.sizes)}")

    lat_name = "latitude" if "latitude" in da.dims else "lat"
    lon_name = "longitude" if "longitude" in da.dims else "lon"
    frames = []
    steps = da["step"].values if "step" in da.dims else [None]
    for si, st in enumerate(steps):
        sl = da.isel(step=si) if st is not None else da
        # ECMWF tp is a cumulative metre-of-water accumulation from run time
        vals = sl.values * 1000.0                    # -> mm
        # members x lat x lon
        if vals.ndim == 2:
            vals = vals[None, ...]
        la = sl[lat_name].values
        lo = sl[lon_name].values
        li = {round(float(v), 4): k for k, v in enumerate(la)}
        oi = {round(float(v), 4): k for k, v in enumerate(lo)}
        cwv = cw[cw["lat"].isin(li) & cw["lon"].isin(oi)]
        if not len(cwv):
            continue
        ii = cwv["lat"].map(li).to_numpy()
        jj = cwv["lon"].map(oi).to_numpy()
        w = cwv["w"].to_numpy()
        did = cwv["district_id"].to_numpy()
        # members x overlaps -> area-weighted district mean per member
        cell = vals[:, ii, jj]
        df = pd.DataFrame(cell.T)
        df["district_id"] = did
        df["w"] = w
        g = df.groupby("district_id")
        wsum = g["w"].sum()
        mem = g.apply(lambda x: pd.Series(
            np.average(x.iloc[:, :cell.shape[0]].to_numpy(), axis=0,
                       weights=x["w"].to_numpy())), include_groups=False)
        hrs = int(pd.to_timedelta(st).total_seconds() // 3600) if st is not None else 0
        out = pd.DataFrame({
            "district_id": mem.index,
            "step_h": hrs,
            "tp_mm_mean": mem.mean(axis=1).to_numpy(),
            "tp_mm_p10": mem.quantile(0.10, axis=1).to_numpy(),
            "tp_mm_p90": mem.quantile(0.90, axis=1).to_numpy(),
            "tp_spread_mm": mem.std(axis=1).to_numpy(),
            "members": mem.shape[1],
        })
        frames.append(out)
    if not frames:
        raise SystemExit("no steps decoded — check the grib and the crosswalk")
    D = pd.concat(frames, ignore_index=True)
    base = pd.to_datetime(str(ds["time"].values)) if "time" in ds else None
    if base is not None:
        D["valid_date"] = (base + pd.to_timedelta(D["step_h"], unit="h")).dt.date
        D["run"] = base.strftime("%Y%m%d%H")
    D = D.merge(reg[["district_id", "state", "district"]], on="district_id",
                how="left")
    return D, (base.strftime("%Y%m%d%H") if base is not None else "unknown")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="verify dependencies and exit")
    ap.add_argument("--steps", default="",
                    help="comma-separated forecast hours, e.g. 24,48,72")
    ap.add_argument("--run", type=int, default=0, help="run hour: 0 or 12")
    a = ap.parse_args()

    log("=" * 74)
    log("ECMWF OPEN DATA -> 791 LGD DISTRICTS")
    log("=" * 74)
    log("  NOTE: this is a REAL-TIME feed, not an archive. It supports a live")
    log("  district forecast display. It CANNOT supply the reforecast history")
    log("  a trained MOS correction needs — see the module docstring.")

    missing = check_deps()
    if a.check:
        return
    if missing:
        log("\n  install the packages above, then re-run.")
        sys.exit(1)

    steps = ([int(x) for x in a.steps.split(",") if x.strip()]
             if a.steps else DEFAULT_STEPS)
    reg = pd.read_csv(IMD / "registry_lgd791.csv")
    cw = build_crosswalk()
    grib = fetch(steps, a.run)
    D, run = to_districts(grib, cw, reg)

    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / f"ecmwf_ens_district_{run}.csv"
    cols = ["district_id", "state", "district", "run", "valid_date", "step_h",
            "tp_mm_mean", "tp_mm_p10", "tp_mm_p90", "tp_spread_mm", "members"]
    D[[c for c in cols if c in D.columns]].round(3).to_csv(p, index=False)
    log(f"\n  wrote {p}")
    log(f"  {len(D):,} rows — {D['district_id'].nunique()} districts x "
        f"{D['step_h'].nunique()} steps, {int(D['members'].max())} members")
    log("  Archive this file every day. After two or three monsoon seasons "
        "the paired forecast/observation record is long enough to fit a first "
        "MOS correction, which is the cheapest route to real 7-14 day skill.")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
