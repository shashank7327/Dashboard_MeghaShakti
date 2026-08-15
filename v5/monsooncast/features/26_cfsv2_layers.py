r"""v5/monsooncast/features/26_cfsv2_layers.py  —  add CFSv2 as a SECOND
PRODUCT alongside IMD, never as a replacement for it.

WHAT THIS IS, AND WHAT IT IS NOT
  NOAA/CFSV2/FOR6H as Earth Engine serves it is indexed by VALID TIME, not by
  (initialisation, lead time). Every image is the model's state for the hour it
  is stamped with. That distinction decides what the data can honestly be
  called:

    it IS      a coupled model's analysis of the atmosphere -- precipitation,
               circulation, humidity transport, thermal state -- on the same
               791 districts, from a source completely independent of the IMD
               gauge network
    it is NOT  a 7 or 14-day forecast. There is no lead-time axis to select,
               so Model Output Statistics cannot be trained on it and no
               forecast skill can be inherited from it.

  Labelling these layers "CFSv2 forecast" would therefore be false, and the
  dashboard labels them "CFSv2 model" for that reason. What they genuinely
  provide is a SECOND OPINION on the same quantity from a different kind of
  instrument, which is the same role CHIRPS plays for rainfall: where the two
  disagree, the disagreement is information about how well the field is known.

WHY IT IS AN ADDITION AND NOT A SUBSTITUTION
  IMD gauge data remains the observational truth for every published figure
  and every feature. Nothing here enters the master panel, the crop model or
  the forecast features. These layers exist so a reader can put the model's
  view beside the gauge view and see the difference, which is a choice rather
  than a change of source.

DEPARTURE NEEDS CFSv2'S OWN CLIMATOLOGY
  A CFSv2 rainfall total must be compared with a CFSv2 normal, never with
  IMD's. Reanalysis precipitation carries its own biases -- over India CFSv2
  is generally wet -- so differencing a CFSv2 actual against an IMD normal
  reports the model's bias as a rainfall anomaly. The day-of-year climatology
  built here is computed from the CFSv2 archive itself over the same 1991-2020
  window, so the departure is internally consistent.

OUTPUT -> v5/data_lgd/cfsv2_layers.json
  { "dates": [...], "layers": {...}, "values": {layer: {date: {district_id: v}}} }
  shaped exactly like daily_layers_lgd.json so the dashboard consumes both
  through one code path.

Run:  py -3.13 -X utf8 "v5/monsooncast/features/26_cfsv2_layers.py"
"""
import json
import pathlib
import sys

import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
V5 = HERE.parents[1]
ROOT = V5.parent
IMD = ROOT / "IMD_Data"
OUTD = V5 / "data_lgd"
sys.path.insert(0, str(V5))
from common_v5 import log  # noqa

#   Folders searched, in order; the first that holds CSVs wins.
SRC = [ROOT / "CFSv2_LGD", ROOT / "CSFv2 updated"]
CLIM_LO, CLIM_HI = 1991, 2020      # matches the ROMI record and IMD's LPA end
KEEP_DAYS = 120                    # same window the dashboard's daily slider uses
SMOOTH = 2                         # +/- days on the day-of-year climatology

LAYERS = {
    "cfs_dep_7d":   "CFSv2 rain, last 7 days vs its own normal (%)",
    "cfs_dep_30d":  "CFSv2 rain, last 30 days vs its own normal (%)",
    "cfs_precip_mm": "CFSv2 rainfall that day (mm)",
    "cfs_tmax_c":   "CFSv2 daily maximum temperature (degC)",
    "cfs_q":        "CFSv2 specific humidity (kg/kg)",
    "cfs_wind_ms":  "CFSv2 wind speed (m/s)",
}


def load():
    for d in SRC:
        files = sorted(d.glob("CFS[Vv]2_district_daily_*.csv"))
        if files:
            log(f"  reading {len(files)} file(s) from {d.name}/")
            return files
    return []


def main():
    log("=" * 74)
    log("STEP 26 — CFSv2 as a second product (addition, not substitution)")
    log("=" * 74)
    files = load()
    if not files:
        log("  no CFSv2 district CSVs found — nothing to do")
        return

    reg = pd.read_csv(IMD / "registry_lgd791.csv")
    norm = lambda s: (s.astype(str).str.lower()
                      .str.replace(r"[^a-z0-9]", "", regex=True))
    lut = {(a, b): i for a, b, i in
           zip(norm(reg["state"]), norm(reg["district"]), reg["district_id"])}

    keep = ["date", "state", "district", "cfs_precip_mm", "cfs_tmax_c",
            "cfs_q", "cfs_wind_ms", "n_records"]
    frames = []
    for f in files:
        d = pd.read_csv(f, parse_dates=["date"])
        d = d[[c for c in keep if c in d.columns]]
        d["district_id"] = [lut.get((a, b), -1) for a, b in
                            zip(norm(d["state"]), norm(d["district"]))]
        d = d[d["district_id"] >= 0]
        #   The LGD asset holds two polygon records for Purba Medinipur, so
        #   every district export carries 792 rows per day. Collapse before
        #   anything is summed, exactly as the panel builder does.
        vals = [c for c in d.columns
                if c.startswith("cfs_") or c == "n_records"]
        d = d.groupby(["date", "district_id"], as_index=False)[vals].mean()
        frames.append(d)
    c = pd.concat(frames, ignore_index=True)

    #   A partial day extrapolates: precip = mean(rate) x 86400 over however
    #   many 6-hourly records exist. Four is a complete day.
    if "n_records" in c.columns:
        short = (c["n_records"] < 4).sum()
        if short:
            log(f"  dropping {short:,} district-days with fewer than 4 "
                f"6-hourly records (their daily total is an extrapolation)")
            c = c[c["n_records"] >= 4]

    log(f"  {len(c):,} district-days, "
        f"{c.date.min():%Y-%m-%d}..{c.date.max():%Y-%m-%d}")

    wide = c.pivot(index="date", columns="district_id",
                   values="cfs_precip_mm").sort_index()

    #   CFSv2's OWN day-of-year climatology. Using IMD's normal here would
    #   report the model's wet bias over India as a rainfall anomaly.
    base = wide[(wide.index.year >= CLIM_LO) & (wide.index.year <= CLIM_HI)]
    per = base.groupby(base.index.dayofyear).mean().reindex(range(1, 367))
    sm = np.nanmean(np.stack([np.roll(per.to_numpy(), k, axis=0)
                              for k in range(-SMOOTH, SMOOTH + 1)]), axis=0)
    clim = pd.DataFrame(sm, index=range(1, 367), columns=per.columns)
    nrm = pd.DataFrame(clim.reindex(wide.index.dayofyear).to_numpy(),
                       index=wide.index, columns=wide.columns)
    log(f"  climatology from CFSv2 itself, {CLIM_LO}-{CLIM_HI}, "
        f"+/-{SMOOTH} day smoothing")

    out_vals = {}
    for H, key in ((7, "cfs_dep_7d"), (30, "cfs_dep_30d")):
        a = wide.rolling(H, min_periods=H).sum()
        n = nrm.rolling(H, min_periods=H).sum()
        with np.errstate(invalid="ignore", divide="ignore"):
            out_vals[key] = pd.DataFrame(
                np.where(n >= 5.0, (a - n) / n * 100.0, np.nan),
                index=wide.index, columns=wide.columns)
    out_vals["cfs_precip_mm"] = wide
    for col in ("cfs_tmax_c", "cfs_q", "cfs_wind_ms"):
        if col in c.columns:
            out_vals[col] = c.pivot(index="date", columns="district_id",
                                    values=col).sort_index()

    #   EMIT THE SAME SHAPE THE DAILY PAYLOAD ALREADY USES, ON ITS AXES.
    #
    #   daily_layers_lgd.json stores each layer as a DENSE array of
    #   [values per district] per date, positionally aligned to its own `ids`
    #   and `dates` lists -- not as nested dictionaries keyed by id. Emitting a
    #   different shape here and merging it later means the merge has to
    #   re-align two coordinate systems, and a positional format punishes that
    #   silently: mismatched lengths do not raise, they just draw the wrong
    #   district.
    #
    #   So CFSv2 is reindexed onto the IMD axes at source. The IMD dates stay
    #   the master axis because the two products end on different days (CFSv2
    #   runs to 02 Aug, the gauge analysis to 04 Aug); CFSv2 is padded with
    #   nulls at the end rather than truncating the IMD layers to match.
    dp = OUTD / "daily_layers_lgd.json"
    if not dp.exists():
        log("  ! daily_layers_lgd.json absent — run features/21_daily_features"
            " first; CFSv2 has no axes to align to")
        return
    base = json.loads(dp.read_text(encoding="utf-8"))
    ids = [int(i) for i in base["ids"]]
    dates = list(base["dates"])
    didx = pd.to_datetime(dates)

    payload = {"layers": {}, "edge": {}, "method": {}, "labels": {},
               "source": "NOAA/CFSV2/FOR6H via Earth Engine — coupled model "
                         "analysis at valid time, NOT a lead-time forecast"}
    for key, label in LAYERS.items():
        if key not in out_vals:
            continue
        sub = out_vals[key].reindex(index=didx, columns=ids)
        arr = sub.to_numpy(dtype=float)
        payload["layers"][key] = [
            [None if not np.isfinite(v) else round(float(v), 2) for v in row]
            for row in arr]
        ok = didx[np.isfinite(arr).any(axis=1)]
        payload["edge"][key] = (f"{ok.max():%Y-%m-%d}" if len(ok) else None)
        payload["method"][key] = ("area-weighted ratio of sums"
                                  if key.startswith("cfs_dep")
                                  else "area-weighted mean")
        payload["labels"][key] = label

    p = OUTD / "cfsv2_layers.json"
    p.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    log(f"  wrote {p.name}: {len(payload['layers'])} layers x "
        f"{len(dates)} days x {len(ids)} districts, aligned to the IMD axes "
        f"({p.stat().st_size / 1e6:.1f} MB)")
    for k, e in payload["edge"].items():
        log(f"    {k:<16} last day with data: {e}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
