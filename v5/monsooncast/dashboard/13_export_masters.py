r"""
v5/monsooncast/dashboard/13_export_masters.py  —  master data files for all 791
districts: one per observed variable, one for the full feature set, and one for
the forecast output.

FORMAT
  Every table is written twice:
    .csv   complete record, no row limit -- the analysis format
    .xlsx  same columns, with a frozen header row, an AUTOFILTER on every
           column and sensible widths, so it can be sorted and filtered in
           Excel directly

  Excel caps a worksheet at 1,048,576 rows.  The monthly panel is 527k rows and
  fits; the full FEATURE table is much wider, so its .xlsx is limited to the
  most recent XLSX_YEARS years while the .csv keeps everything.  Any truncation
  is stated in the log and in a "README" sheet inside the workbook, never
  silently.

OUTPUT -> v5/masters_lgd/
  MASTER_Rainfall.{csv,xlsx}           actual, normal, % departure
  MASTER_Temperature.{csv,xlsx}        Tmax/Tmin, normals, anomalies, GDD/SDD
  MASTER_Evapotranspiration.{csv,xlsx} AET, PET, moisture adequacy
  MASTER_SoilMoisture.{csv,xlsx}       ERA5-Land layers 1-4
  MASTER_Features_All.{csv,xlsx}       every modelling feature
  MASTER_Forecast.{csv,xlsx}           current 7/14-day forecast per district

Run:  py -3.13 -X utf8 "v5/monsooncast/dashboard/13_export_masters.py"
"""
import json
import pathlib
import sys

import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
V5 = HERE.parents[1]
IMD = V5.parent / "IMD_Data"
DATA = V5 / "data_lgd"
DASH = V5 / "dashboard_lgd"
OUT = V5 / "masters_lgd"
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(V5))
from common_v5 import log  # noqa

XLSX_MAX = 1_000_000          # keep clear of Excel's 1,048,576 hard limit
XLSX_YEARS = 15               # window used when a table is too wide/long
# Excel's real constraint is CELLS, not rows: the 63-column feature table at
# 527k rows is 33 million cells, which xlsxwriter will grind on for hours and
# Excel would struggle to open.  Cap on the cell count and window the years
# until it fits; the CSV always keeps the complete record.
XLSX_MAX_CELLS = 6_000_000

ID = ["date", "year", "month", "state", "district", "district_id"]

GROUPS = {
    "MASTER_Rainfall": (
        ["rain_mm", "normal_mm", "pct_departure"],
        "IMD gauge rainfall, monthly totals. normal_mm is the 1971-2020 "
        "climatological mean (IMD's own LPA window); pct_departure = "
        "100*(rain-normal)/normal, undefined where the normal is below 5 mm."),
    "MASTER_Temperature": (
        ["tmax_c", "tmax_normal_c", "tmax_anom", "tmin_c", "tmin_normal_c",
         "tmin_anom", "gdd_kharif", "gdd_rabi", "sdd"],
        "IMD gauge temperature, monthly means, with 1971-2020 normals and "
        "anomalies. GDD = growing degree-days (base 10 C Kharif / 5 C Rabi); "
        "SDD = stress degree-days above 34 C."),
    "MASTER_Evapotranspiration": (
        ["aet_mm", "pet_mm", "pet_hargreaves", "mai"],
        "ERA5-Land actual and potential evapotranspiration, monthly sums. "
        "mai = AET/PET, the moisture adequacy index, clipped to [0, 1.5]."),
    "MASTER_SoilMoisture": (
        ["swvl1", "swvl2", "swvl3", "swvl4"],
        "ERA5-Land volumetric soil water, monthly means, m3/m3. Layers are "
        "0-7 cm, 7-28 cm, 28-100 cm and 100-289 cm."),
}


def autofit(ws, df, wb, nrow):
    hdr = wb.add_format({"bold": True, "bg_color": "#1F365C",
                         "font_color": "white", "border": 1})
    for j, c in enumerate(df.columns):
        ws.write(0, j, str(c), hdr)
        w = max(len(str(c)) + 2, 11)
        if df[c].dtype == object:
            # `x or 10` does NOT guard against NaN here: float('nan') is truthy,
            # so an all-null text column passed NaN straight into int() and the
            # whole export died on MASTER_Forecast.  Check for the empty case
            # explicitly instead.
            ln = df[c].astype(str).str.len().head(2000).max()
            ln = 10 if ln is None or not np.isfinite(ln) else int(ln)
            w = min(max(w, ln + 2), 34)
        ws.set_column(j, j, w)
    ws.freeze_panes(1, 0)
    ws.autofilter(0, 0, nrow, len(df.columns) - 1)


LOCKED = []          # files a running program is holding open


def write_pair(df, name, note):
    r"""Write the CSV and XLSX for one table.

    A file the user happens to have open in Excel cannot be overwritten on
    Windows, and the whole pipeline used to die on that PermissionError at the
    very last step -- after the 28-minute retrain had already succeeded.  A
    spreadsheet left open is not a pipeline failure, so the file is skipped,
    recorded, and reported at the end with the fix.
    """
    csv = OUT / f"{name}.csv"
    try:
        df.to_csv(csv, index=False)
    except PermissionError:
        LOCKED.append(csv.name)
        log(f"    {name:28s} SKIPPED — {csv.name} is open in another program")
        return
    xl = df
    trunc = ""
    ncol = max(len(df.columns), 1)
    # Window on whichever year-like column the table actually has.  The crop
    # tables are keyed by season_year, not year; without this they fell through
    # to a blind tail-truncation that kept an arbitrary subset of crops.
    ycol = next((c for c in ("year", "season_year") if c in df.columns), None)
    if ycol:
        df = df.sort_values(ycol)
        yrs = XLSX_YEARS
        while yrs >= 1:
            lo = int(df[ycol].max()) - yrs + 1
            cand = df[df[ycol] >= lo]
            if len(cand) <= XLSX_MAX and len(cand) * ncol <= XLSX_MAX_CELLS:
                break
            yrs = yrs // 2 if yrs > 1 else 0
        if len(df) > XLSX_MAX or len(df) * ncol > XLSX_MAX_CELLS:
            xl = cand
            trunc = (f"XLSX limited to {ycol} {lo}-{int(df[ycol].max())} "
                     f"({len(xl):,} of {len(df):,} rows, {ncol} columns) so the "
                     f"workbook stays openable; the CSV holds the complete "
                     f"1971-2026 record.")
    if len(xl) > XLSX_MAX:
        xl = xl.tail(XLSX_MAX)
        trunc = f"XLSX truncated to the last {XLSX_MAX:,} rows; CSV is complete."
    path = OUT / f"{name}.xlsx"
    try:
        _write_xlsx(path, xl, df, name, note, trunc)
    except PermissionError:
        LOCKED.append(path.name)
        log(f"    {name:28s} csv ok, XLSX SKIPPED — {path.name} is open "
            f"in another program")
        return
    log(f"    {name:28s} csv {len(df):>9,} rows   xlsx {len(xl):>9,} rows"
        + ("  [" + trunc.split(";")[0] + "]" if trunc else ""))


def _write_xlsx(path, xl, df, name, note, trunc):
    with pd.ExcelWriter(path, engine="xlsxwriter") as w:
        xl.to_excel(w, sheet_name="data", index=False, startrow=1,
                    header=False)
        autofit(w.sheets["data"], xl, w.book, len(xl))
        rd = pd.DataFrame({"About": [name, note,
                                     f"rows in this sheet: {len(xl):,}",
                                     f"rows in the CSV: {len(df):,}",
                                     trunc or "no truncation",
                                     "Districts: 791 LGD units incl. J&K, "
                                     "Ladakh and the island UTs.",
                                     "Source: IMD gridded gauge data + "
                                     "ERA5-Land, aggregated to districts by "
                                     "area-weighted overlap."]})
        rd.to_excel(w, sheet_name="README", index=False)
        w.sheets["README"].set_column(0, 0, 110)


def main():
    log("=" * 72)
    log("MASTER EXPORTS — per-variable, feature and forecast tables")
    log("=" * 72)
    f = pd.read_csv(DATA / "features_lgd.csv", low_memory=False,
                    parse_dates=["date"])
    f = f.sort_values(["state", "district", "date"])
    log(f"  feature panel: {len(f):,} rows x {f.shape[1]} cols, "
        f"{f['district_id'].nunique()} districts, "
        f"{f['date'].min():%Y-%m}..{f['date'].max():%Y-%m}")

    log("\n  per-variable masters:")
    for name, (cols, note) in GROUPS.items():
        have = [c for c in cols if c in f.columns]
        miss = [c for c in cols if c not in f.columns]
        if miss:
            log(f"    ({name}: absent columns skipped -> {', '.join(miss)})")
        write_pair(f[ID + have].copy(), name, note)

    log("\n  full feature master:")
    write_pair(f.copy(), "MASTER_Features_All",
               "Every modelling feature per district-month: rainfall, "
               "temperature, evapotranspiration, soil moisture, SPEI-1/4/12, "
               "MAI, GDD/SDD, ENSO/IOD indices and the per-district ENSO-phase "
               "signatures. This is the table the forecast models consume.")

    log("\n  crop masters:")
    for name, src, note in (
        ("MASTER_CropStress_Current", "crop_stress_current_lgd.csv",
         "Crop-stress index for the season IN PROGRESS, per district and crop. "
         "csi_to_date rescales the partial FAO-33 product by the worst outcome "
         "attainable from the stages elapsed so far -- use it, not csi, while a "
         "season is running. csi_class buckets it Low/Moderate/High/Severe."),
        ("MASTER_CropStress_History", "crop_stress_history_lgd.csv",
         "Crop-stress index per district, crop and season-year, 1971-2026. "
         "csi is the FAO-33 multiplicative yield-loss index over the four "
         "phenological stages; complete seasons should be read on csi."),
        ("MASTER_CropStress_Stages", "crop_stage_stress_lgd.csv",
         "The same index broken out BY PHENOLOGICAL STAGE (Establishment, "
         "Vegetative, Reproductive, Maturity) with each stage's Ky yield-"
         "response factor and its own upper temperature threshold."),
        ("MASTER_CropAreaMask", "crop_area_mask_lgd.csv",
         "Where each crop is actually sown, from UPAg weekly sowing releases. "
         "district_area_ha is the STATE area divided evenly across that "
         "state's districts -- an allocation, not a measurement."),
        ("MASTER_DailyFeatures", "daily_features_lgd.csv",
         "DAILY district features, one row per district per day. dep_7d, "
         "dep_30d and dep_season are rolling or cumulative rainfall against the "
         "normal for those SAME calendar days, so each is day-matched by "
         "construction — unlike a monthly total, a window ending today is "
         "complete on both sides. tmax_anom_d / tmin_anom_d are against a "
         "smoothed 1971-2020 day-of-year normal. swvl2_d is ERA5-Land and lags "
         "IMD by about a week, so its last few days are blank by design."),
        ("MASTER_SowingStatus", "sowing_status_current.csv",
         "REAL-TIME SOWING, state x crop, from the UPAg weekly releases. "
         "area_to_date_ha is what is in the ground so far this season; "
         "coverage_pct expresses it as a share of the PRIOR season's final "
         "area; pace_pp is coverage_pct minus where last season stood in the "
         "same ISO week, so positive means ahead of last year. Ratios resting "
         "on under half the crop's area are left blank rather than published."),
        ("MASTER_SowingNational", "sowing_status_national.csv",
         "The same sowing snapshot summed to all-India. Totals cover every "
         "reporting state; the YoY and coverage RATIOS are computed only over "
         "states present in both years, and yoy_basis_pct / cov_basis_pct say "
         "what share of the crop that matched subset represents."),
        ("MASTER_SowingWeekly", "sowing_dynamics_state.csv",
         "The full weekly sowing panel behind the snapshot: state x crop x "
         "season x ISO week cumulative area, with the same week a year earlier "
         "joined on for the year-on-year comparison. Seasons are segmented at "
         "the points where UPAg's cumulative counter restarts."),
        ("MASTER_SowingDistrict", "sowing_status_district.csv",
         "The sowing snapshot allocated to the 791 LGD units. The state figure "
         "is divided evenly across that state's districts, so these values "
         "compare meaningfully BETWEEN states but are constant WITHIN one."),
        ("MASTER_CropParams", "crop_params_lgd.csv",
         "The crop table: stage calendar, Ky yield-response factors, upper "
         "temperature thresholds, base temperatures and water adequacy, with "
         "the source of each Ky."),
    ):
        p = DATA / src
        if not p.exists():
            log(f"    ({name}: {src} absent — skipped)")
            continue
        write_pair(pd.read_csv(p, low_memory=False), name, note)

    log("\n  forecast master:")
    dj = json.loads((DASH / "data.json").read_text(encoding="utf-8"))
    fc = pd.DataFrame(dj["districts"]).rename(columns={
        "id": "district_id", "d": "district", "s": "state",
        "jk": "is_jk_ladakh", "fc7": "forecast_7d_pct",
        "fc14": "forecast_14d_pct"})
    fc.insert(0, "issued", dj["issued"])
    sk = dj.get("skill", {})
    for h in (7, 14):
        k = f"dep_{h}"
        if k in sk:
            fc[f"skill_{h}d_vs_climatology"] = sk[k].get("mse_skill")
            fc[f"corr_{h}d"] = sk[k].get("corr")
    write_pair(fc, "MASTER_Forecast",
               f"Operational 7- and 14-day district rainfall-departure "
               f"forecast issued {dj['issued']}, from the "
               f"{sk.get('dep_14', {}).get('model', 'gradient-boosted blend')}. "
               "Positive = wetter than normal. The skill columns are the "
               "held-out 2020-2026 scores for the model that produced them.")

    log(f"\n  wrote {len(list(OUT.glob('*')))} files to {OUT}")
    if LOCKED:
        log(f"  ! {len(LOCKED)} file(s) could not be replaced because another "
            f"program has them open:")
        for n in LOCKED:
            log(f"      {n}")
        log("    Close them in Excel and re-run this step alone:")
        log("      py -3.13 -X utf8 \"v5/monsooncast/dashboard/"
            "13_export_masters.py\"")
        log("    Everything else was written; these files still hold the "
            "PREVIOUS run's numbers.")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
