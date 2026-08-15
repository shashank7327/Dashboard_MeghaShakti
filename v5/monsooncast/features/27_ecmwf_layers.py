r"""v5/monsooncast/features/27_ecmwf_layers.py  —  expose the ECMWF ensemble
as a selectable forecast product alongside the statistical models.

THIS ONE REALLY IS A FORECAST — UNLIKE CFSv2
  The distinction matters and the labelling follows it exactly:

    CFSv2 (features/26)   Earth Engine serves NOAA/CFSV2/FOR6H indexed by
                          VALID TIME with no lead-time axis. It is the coupled
                          model's analysis of the atmosphere. Labelled
                          "CFSv2 model", never "forecast".

    ECMWF (this file)     ecmwf-opendata delivers the operational ENS with an
                          explicit STEP (lead time in hours), 50 perturbed
                          members, and therefore a real forecast distribution.
                          It is a forecast and is labelled as one.

  What it still is NOT is a VERIFIED forecast. ECMWF open data is a real-time
  feed with no reforecast archive, so no skill score can be computed for it
  here and none is claimed. The dashboard shows it with its ensemble spread
  and its run time, and says where it came from.

WHY THE SPREAD IS CARRIED, NOT JUST THE MEAN
  The ensemble mean of 50 members is smoother than any individual member and
  systematically under-represents heavy rain -- averaging washes out the
  convective tail. Publishing the mean alone would show a confident, damp
  forecast. p10 and p90 are carried so the width of the distribution is
  visible, which for a monsoon rainfall forecast is most of the information.

OUTPUT -> v5/data_lgd/ecmwf_forecast_latest.csv
  district_id, and per lead: ecmwf_{H}h_mean / _p10 / _p90 / _spread
  plus a sidecar ecmwf_meta.json carrying the run time and step list, so the
  dashboard can state the issue time rather than implying the forecast is
  current.

Run:  py -3.13 -X utf8 "v5/monsooncast/features/27_ecmwf_layers.py"
"""
import json
import pathlib
import sys

import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
V5 = HERE.parents[1]
ROOT = V5.parent
SRC = ROOT / "ECMWF_LGD"
OUTD = V5 / "data_lgd"
sys.path.insert(0, str(V5))
from common_v5 import log  # noqa

#   Leads offered to the dashboard, in hours. Anything present in the file is
#   used; anything absent is skipped rather than fabricated.
WANT = [24, 48, 72, 96, 120, 144, 168, 240, 336, 360]
#   A run older than this is reported as stale. ECMWF publishes four runs a
#   day, so anything beyond two days is a refresh that did not happen.
STALE_HOURS = 48


def main():
    log("=" * 74)
    log("STEP 27 — ECMWF ENS as a selectable forecast product")
    log("=" * 74)
    files = sorted(SRC.glob("ecmwf_ens_district_*.csv"))
    if not files:
        log(f"  no ECMWF district CSVs in {SRC.name}/ — run "
            f"forecast_input/25_ecmwf_opendata_lgd.py first")
        return

    #   Newest run wins. The filename carries the run stamp YYYYMMDDHH, so a
    #   lexical sort is a chronological sort.
    f = files[-1]
    d = pd.read_csv(f, parse_dates=["valid_date"])
    if len(files) > 1:
        log(f"  {len(files)} run files present; using the newest ({f.name})")

    run = str(d["run"].iloc[0])
    run_ts = pd.to_datetime(run, format="%Y%m%d%H")
    now = pd.Timestamp.now("UTC").tz_localize(None)
    age_h = (now - run_ts) / pd.Timedelta(hours=1)
    steps = sorted(d["step_h"].unique().tolist())
    members = int(d["members"].max())

    log(f"  run {run_ts:%Y-%m-%d %H}Z, {members} members, "
        f"steps {steps} h")
    log(f"  {d['district_id'].nunique()} districts, {len(d):,} rows")

    if age_h > STALE_HOURS:
        log(f"  ! this run is {age_h/24:.1f} DAYS old. ECMWF open data is a "
            f"real-time feed and keeps only recent runs — re-run "
            f"forecast_input/25_ecmwf_opendata_lgd.py for a current forecast.")
    missing = [s for s in WANT if s not in steps]
    if missing:
        log(f"  ! only {len(steps)} lead time(s) present; "
            f"{len(missing)} of the usual set are missing ({missing}).")
        log(f"    The fetch defaults to steps out to 360 h — this file covers "
            f"{max(steps)} h, so the display will stop "
            f"{max(steps)/24:.0f} day(s) ahead.")

    out = pd.DataFrame({"district_id":
                        sorted(d["district_id"].unique().astype(int))})
    layers = {}
    for s in steps:
        if s not in WANT:
            continue
        sub = d[d["step_h"] == s].set_index("district_id")
        day = s // 24
        for src, suf, lab in (("tp_mm_mean", "mean", "ensemble mean"),
                              ("tp_mm_p10", "p10", "10th percentile"),
                              ("tp_mm_p90", "p90", "90th percentile"),
                              ("tp_spread_mm", "spread", "ensemble spread")):
            if src not in sub.columns:
                continue
            col = f"ecmwf_{s}h_{suf}"
            out[col] = out["district_id"].map(sub[src])
            if suf == "mean":
                layers[col] = (f"ECMWF ENS rain to +{day}d ({lab}, mm)")

    OUTD.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTD / "ecmwf_forecast_latest.csv", index=False)

    meta = {"run": f"{run_ts:%Y-%m-%dT%H:00Z}",
            "age_hours": round(float(age_h), 1),
            "stale": bool(age_h > STALE_HOURS),
            "members": members,
            "steps_h": [int(s) for s in steps],
            "max_lead_days": round(max(steps) / 24, 1),
            "layers": layers,
            "source": "ECMWF open data, operational ENS, 0.25 deg, CC BY 4.0. "
                      "A real forecast with lead time and ensemble spread, but "
                      "NOT verified here: open data carries no reforecast "
                      "archive, so no skill score can be computed for it."}
    (OUTD / "ecmwf_meta.json").write_text(json.dumps(meta, indent=1),
                                          encoding="utf-8")

    m = out[[c for c in out.columns if c.endswith("_mean")]]
    log(f"\n  wrote ecmwf_forecast_latest.csv "
        f"({len(out)} districts x {len(layers)} lead(s))")
    for c in m.columns:
        log(f"    {c:24s} {m[c].min():6.2f} .. {m[c].max():7.2f} mm  "
            f"(mean {m[c].mean():5.2f})")
    log(f"  wrote ecmwf_meta.json")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
