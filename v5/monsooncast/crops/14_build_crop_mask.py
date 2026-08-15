r"""
v5/monsooncast/crops/14_build_crop_mask.py  —  crop-area mask on the 791 LGD
districts, from the UPAg weekly sowing releases.

WHY A MASK IS NECESSARY
  The climate inputs exist for every district, so a crop-stress index can be
  computed for all 791 x 18 district-crop pairs.  Most of those pairs are
  meaningless: rice stress in a district that sows no rice is a number with no
  referent, and it inflates every "districts under High/Severe stress" count.
  UPAg publishes weekly SOWN AREA by state, crop and season, which identifies
  where each crop is actually grown.

TWO CORRECTIONS OVER THE FIRST VERSION
  Both are implemented in upag_common.py and documented there in full:

    * StateName carries the national totals "India" AND "All India".  The first
      version filtered neither, so summing over StateName double-counted every
      crop -- wheat came out at 66.8 M ha against a true 33.4 M ha.

    * UPAg's cumulative counter RESTARTS at each new season within the same
      SowingYear, and the season is not labelled.  Rice under SowingYear 2025
      climbs to 44.2 M ha as Kharif ends, then restarts at 0.6 M ha for Rabi.
      The series is now cut at those resets and each segment is labelled Kharif,
      Rabi or Summer from the month it begins.

  Because the mask is now season-aware, Kharif rice and Rabi rice carry their
  own areas instead of sharing one figure.

AREA ALLOCATION, AND ITS LIMITATION
  UPAg reports at STATE level.  The state's sown area is divided evenly across
  the districts of that state, so district_area_ha is an ALLOCATION, not a
  measurement.  Consequences, stated rather than hidden:
    * ordering districts WITHIN a state by stress-weighted area is driven
      entirely by the stress index, since the area term is constant;
    * ordering BETWEEN states is meaningful, because the state totals are real.
  District-level sowing statistics would remove this limitation.

  For real-time coverage and the year-on-year comparison, see
  19_sowing_dynamics.py -- this script answers only "is it grown here".

OUTPUT -> v5/data_lgd/
  crop_area_mask_lgd.csv        district_id x crop x season -> grown, area
  sowing_progress_lgd.csv       state x crop x season x week, cumulative
  sowing_monthly_lgd.csv        area sown per month (cumulative differenced)
  sowing_monthly_national_lgd.csv

Run:  py -3.13 -X utf8 "v5/monsooncast/crops/14_build_crop_mask.py"
"""
import pathlib
import sys

import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
V5 = HERE.parents[1]
ROOT = V5.parent
IMD = ROOT / "IMD_Data"
OUTD = V5 / "data_lgd"
UPAG = ROOT / "UPAJ"
sys.path.insert(0, str(V5))
sys.path.insert(0, str(IMD))
sys.path.insert(0, str(HERE.parent / "lib"))
from common_v5 import log  # noqa
from build_crosswalk import norm_state  # noqa
from upag_common import ALIAS, load_upag  # noqa

# crops the stress model scores but UPAg does not publish: left unmasked and
# flagged, so a reader knows the difference between "not grown" and "unknown"
NO_UPAG = ["Ragi", "Guar Seed"]


def main():
    log("=" * 70)
    log("crop-area mask + sowing progress on the 791 LGD districts")
    log("=" * 70)
    reg = pd.read_csv(IMD / "registry_lgd791.csv")
    reg["ks"] = norm_state(reg["state"])
    n_units = len(reg)

    u, nat = load_upag()
    if u is None or u.empty:
        log(f"  ! no Sowing*.csv in {UPAG} — mask not built")
        return
    u["ks"] = norm_state(u["state_raw"])
    log(f"  UPAg state rows {len(u):,}; crops matched: "
        f"{u['crop'].nunique()} ({', '.join(sorted(u['crop'].unique()))})")
    log(f"  national total rows held out of the state panel: {len(nat):,}")
    raw = pd.concat([pd.read_csv(f) for f in sorted(UPAG.glob("Sowing*.csv"))],
                    ignore_index=True)
    unmatched = sorted(set(raw["CommodityName"].astype(str).str.strip())
                       - set(ALIAS))
    if unmatched:
        log(f"  UPAg commodities with no crop mapping (ignored): "
            f"{', '.join(unmatched[:12])}")

    # ---- weekly cumulative per state x crop x season x season-year ---------
    key = ["ks", "crop", "season", "season_year"]
    prog = u[key + ["state_raw", "wk", "week_end", "area_ha", "seg"]].copy()
    prog = prog.sort_values(key + ["week_end"])
    prog.to_csv(OUTD / "sowing_progress_lgd.csv", index=False)

    # the segment's largest cumulative value is that season's sown area
    peak = prog.groupby(key, as_index=False).agg(area_ha=("area_ha", "max"))
    log(f"  state x crop x season x year records: {len(peak):,}")
    nat_pk = (peak.groupby(["crop", "season", "season_year"], as_index=False)
              ["area_ha"].sum())
    last = nat_pk[nat_pk["season_year"] == nat_pk["season_year"].max() - 1]
    if len(last):
        log(f"  sanity check, last completed year "
            f"({int(last['season_year'].iloc[0])}), M ha: " + ", ".join(
                f"{r.crop} {r.season[:3]} {r.area_ha/1e6:.1f}"
                for r in last.nlargest(6, "area_ha").itertuples()))

    # ---- allocate the state area across that state's districts -------------
    dist_per_state = reg.groupby("ks")["district_id"].apply(list).to_dict()
    rows = []
    for r in peak.itertuples():
        ds = dist_per_state.get(r.ks)
        if not ds:
            continue
        n_d = max(len(ds), 1)
        for did in ds:
            rows.append({"district_id": did, "crop": r.crop,
                         "season": r.season,
                         "season_year": int(r.season_year),
                         "grown": 1,
                         "state_area_ha": float(r.area_ha),
                         "district_area_ha": float(r.area_ha) / n_d,
                         "n_districts_in_state": n_d,
                         "area_allocation": "state area / districts in state",
                         "source": "UPAg weekly sowing"})

    # crops with no UPAg coverage: flagged, not masked out
    ymax = int(peak["season_year"].max()) if len(peak) else 2026
    for crop in NO_UPAG:
        for did in reg["district_id"]:
            rows.append({"district_id": did, "crop": crop, "season": "Kharif",
                         "season_year": ymax, "grown": 1,
                         "state_area_ha": np.nan, "district_area_ha": np.nan,
                         "n_districts_in_state": np.nan,
                         "area_allocation": "no UPAg commodity",
                         "source": "unmasked (coverage unknown)"})
    MK = pd.DataFrame(rows)
    MK.to_csv(OUTD / "crop_area_mask_lgd.csv", index=False)
    masked = MK[MK["source"] == "UPAg weekly sowing"]
    log(f"  mask: {len(MK):,} district-crop-season rows "
        f"({masked['crop'].nunique()} masked crops, "
        f"{len(NO_UPAG)} left unmasked: {', '.join(NO_UPAG)})")
    log(f"  districts covered per crop (median): "
        f"{int(masked.groupby('crop')['district_id'].nunique().median())}"
        f" of {n_units}")

    # ---- monthly sown area: difference the cumulative WITHIN a segment ------
    # Differencing across a season boundary would read the counter restart as a
    # negative step; clipping that to zero then adds the whole next season on
    # top of this one.  `seg` keeps each season's ramp separate.
    p = prog.copy()
    p["ym"] = p["week_end"].dt.strftime("%Y-%m")
    seg_key = key + ["seg"]
    p = p.sort_values(seg_key + ["week_end"])
    # Within a season the cumulative sown area cannot fall, but the published
    # series does dip when a state revises an earlier week.  Differencing the
    # raw series and clipping the negative step to zero silently re-adds the
    # recovery, so 2026 summed to 70 M ha against 41 M ha of actual segment
    # peaks.  Taking the running maximum first makes the series monotone, and
    # the monthly increments then sum exactly to the season's final area.
    p["cum_ha"] = p.groupby(seg_key)["area_ha"].cummax()
    p["inc_ha"] = (p.groupby(seg_key)["cum_ha"].diff()
                   .fillna(p["cum_ha"]).clip(lower=0))
    me = (p.groupby(["ks", "crop", "season", "season_year", "ym"],
                    as_index=False)["inc_ha"].sum()
          .rename(columns={"inc_ha": "area_sown_ha"}))
    me.to_csv(OUTD / "sowing_monthly_lgd.csv", index=False)
    nt = (me.groupby(["crop", "season", "season_year", "ym"], as_index=False)
          ["area_sown_ha"].sum())
    nt.to_csv(OUTD / "sowing_monthly_national_lgd.csv", index=False)
    cur = nt[nt["season_year"] == nt["season_year"].max()]
    log(f"  sowing {int(nt['season_year'].max())}: "
        f"{cur['area_sown_ha'].sum()/1e6:.2f} M ha across "
        f"{cur['ym'].nunique()} months, {cur['crop'].nunique()} crops")
    log("  wrote crop_area_mask_lgd.csv, sowing_progress_lgd.csv, "
        "sowing_monthly_lgd.csv, sowing_monthly_national_lgd.csv")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
