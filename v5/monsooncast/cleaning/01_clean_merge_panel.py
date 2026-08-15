r"""
v5/monsooncast/cleaning/01_clean_merge_panel.py  —  STEP 1 of the IMD/LGD rebuild.

Cleans and merges every observed layer into ONE coherent monthly master panel
on the 791 LGD district units, ready for feature creation and modelling.  This
is the file that replaces master_panel_v5.csv, with CHIRPS and TerraClimate
fully removed and IMD in their place.

SOURCES (all already on the LGD boundary)
  IMD rainfall    IMD_Data/_district_daily_rain_lgd.pkl   (0.25 deg, gauge)
  IMD Tmax/Tmin   IMD_Data/_district_daily_{tmax,tmin}_lgd.pkl (1 deg, gauge)
  ERA5 AET/PET    Evapotranspiration_LGD/ERA5L_aet_pet_district_daily_*.csv
  ERA5 soil moist SM_LGD/ERA5L_soilmoisture_district_daily_*.csv
  ENSO indices    noaa_indices_cache.csv (ONI, Nino-3.4, IOD; IOD to 2026-07)

AGGREGATION (WMO / physical convention)
  extensive (rain, AET, PET) -> monthly SUM
  intensive (Tmax, Tmin, soil moisture) -> monthly MEAN
  NORMAL is the climatological monthly mean over 1981-2025 (45 years, the full
  record) -- chosen to follow IMD's own 50-year LPA practice rather than the
  30-year WMO standard normal; see the NORMAL BASELINE note below.
  pct_departure = 100 * (rain - normal) / normal, undefined where normal < 5 mm.

OUTPUT -> v5/data_lgd/
  master_panel_lgd.csv     monthly, one row per district-month, all layers
  daily_rain_lgd.pkl       daily IMD rain (wide) for antecedent/extreme features
  merge_report_lgd.json    coverage + provenance

Run:  py -3.13 -X utf8 "v5/monsooncast/cleaning/01_clean_merge_panel.py"
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
OUTD.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(V5))
from common_v5 import log  # noqa

# ------------------------------------------------------ NORMAL BASELINE
# IMD computes its rainfall normals over a FIFTY-year period, 1971-2020,
# refreshed once a decade (the current set was released in April 2022 from
# 4,132 gauges over 703 districts).  WMO-No. 1203's 30-year "standard normal"
# (1991-2020) is designed for climate-change MONITORING, not for operational
# rainfall departure, and IMD deliberately does not use it for the LPA.
#
# The reason matters here: district rainfall is far noisier than the national
# mean, so a short baseline carries real sampling error.  Measured on this
# panel, moving from a 30-year to a 40-year baseline shifts 72.6% of district-
# months by more than 2%, with a maximum of 33% -- invisible nationally (836.9
# vs 837.8 mm) but large where the product is actually used.
#
# The record has been extended back to 1971 (IMD_Data/extend_history_1971.py)
# precisely so this window can be IMD's own.  The normals below are therefore
# computed over 1971-2020 -- the identical 50-year period IMD uses for the LPA
# -- which makes our departures directly comparable to every figure IMD
# publishes, instead of approximately so.  The DAILY normal used for the model
# target and the antecedent features uses the same window, so the monthly
# feature and the model target are on one baseline.
WMO_LO, WMO_HI = 1971, 2020
MIN_NORMAL_MM = 5.0

# -------------------------------------------------- IMD LPA: DIAGNOSED, NOT
#                                                     "CORRECTED" (deliberate)
# 07_validate_features.py measures our area-weighted all-India JJAS mean at
# 837.8 mm against IMD's published Long Period Average of 868.6 mm, and our
# year-by-year % departures run about +2.4 pp above IMD's published figures
# (correlation 0.992 -- the shape is right, the level is offset).
#
# The tempting fix is to scale the normals up so the LPA matches IMD's.  That
# is WRONG and is deliberately not done, because % departure is a RATIO of two
# quantities from the SAME source:
#
#     our actual  vs IMD actual : -1.2%      <- these nearly agree
#     our LPA     vs IMD LPA    : -3.5%      <- the gap is in the BASELINE
#
# IMD's LPA is computed over 1971-2020; our record begins in 1981, so our
# normal cannot see the wetter 1970s and sits legitimately lower.  Scaling only
# the denominator to IMD's LPA -- while the numerator stays on our scale --
# manufactures a deficit that is not in the data: it drove the ENSO-neutral
# JJAS composite from +1.2% (correctly near zero) to -2.4%.
#
# So the normals stay on our own longest available baseline (1981-2025),
# internally consistent with the actuals they are divided into, and the residual
# offset against IMD's LPA-referenced figures is reported as a known,
# understood baseline difference rather than papered over.
IMD_JJAS_LPA_MM = 868.6          # IMD published all-India JJAS LPA, for report
JJAS = [6, 7, 8, 9]


def norm(s):
    s = pd.Series(s).astype(str).str.upper().str.replace(r"[^A-Z ]", " ",
                                                         regex=True)
    return s.str.replace(r"\s+", " ", regex=True).str.strip()


#   Minimum observed days before a monthly SUM is produced at all.
#
#   This was 20, which withheld the running month's rainfall entirely until
#   the 20th. The effect was invisible for most of the month and then obvious:
#   on 4 August the panel carried a normal for August and no actual, so every
#   monthly rainfall layer went blank and the export died formatting a null.
#
#   Twenty was the wrong guard for this quantity. A partial-month TOTAL is
#   indeed meaningless on its own, but nothing here compares it to a full
#   month: daymatched_normal() restricts the denominator to exactly the days
#   observed, so a 4-day sum is measured against a 4-day normal. That is the
#   same like-for-like construction the daily layers use, and the panel
#   already carries days_observed / month_complete so every consumer can see
#   what it is looking at.
#
#   Accumulating INDICES are a different matter and keep their own guard --
#   SPEI over 4 days against a distribution fitted on whole months really is
#   meaningless, which is what features/_partial_month.py exists to handle.
MIN_DAYS_FOR_SUM = 1


def monthly_from_pickle(name, how):
    w = pd.read_pickle(IMD / f"_district_daily_{name}_lgd.pkl")
    w.columns = w.columns.astype(int)
    m = (w.resample("MS").sum(min_count=MIN_DAYS_FOR_SUM) if how == "sum"
         else w.resample("MS").mean())
    return m


def daymatched_normal(w, month_normal):
    r"""Climatological normal covering exactly the days each month actually has.

    `w` is the daily district panel, `month_normal` the 12 x district full-month
    normal.  Returns (normal_by_date, days_observed).

    For a complete month the two agree to floating-point noise -- the sum of the
    per-calendar-day climatological means over a month IS that month's mean
    total.  The value of doing it this way is entirely in the running month,
    where summing only the elapsed days removes a denominator the actual has had
    no opportunity to fill.
    """
    base = w[(w.index.year >= WMO_LO) & (w.index.year <= WMO_HI)]
    clim = base.groupby([base.index.month, base.index.day]).mean()

    obs = w.notna().any(axis=1)
    idx = pd.MultiIndex.from_arrays([w.index.month, w.index.day])
    daily = clim.reindex(idx)
    daily.index = w.index
    daily = daily.where(obs, np.nan)          # days with no report add nothing
    normal_by_date = daily.resample("MS").sum(min_count=1)

    days = obs.astype(int).resample("MS").sum().rename("days_observed")
    days = days.reset_index()
    days["days_in_month"] = days["date"].dt.days_in_month
    days["month_complete"] = (days["days_observed"]
                              >= days["days_in_month"]).astype(int)
    n_part = int((days["month_complete"] == 0).sum())
    if n_part:
        p = days[days["month_complete"] == 0]
        log(f"  day-matched normals: {n_part} incomplete month(s) — " + ", ".join(
            f"{r.date:%Y-%m} ({r.days_observed}/{r.days_in_month} days)"
            for r in p.itertuples()))
    return normal_by_date, days


def normal_baseline(m):
    base = m[(m.index.year >= WMO_LO) & (m.index.year <= WMO_HI)]
    g = base.groupby(base.index.month).mean()             # 12 x district
    g.index.name = "nmn"
    return g


def long_imd(m, valcol):
    lo = m.reset_index().melt(id_vars="date", var_name="district_id",
                              value_name=valcol)
    lo["district_id"] = lo["district_id"].astype(int)
    return lo


def load_era5_folder(folder, valcols, how, days_col=None):
    """Read the per-year GEE CSVs, join to the LGD registry, aggregate monthly.

    `days_col` also returns how many distinct days each month actually contains.
    That matters because the sources do not end on the same date: IMD rainfall
    ran to 24 Jul 2026 while ERA5-Land ET stopped at 15 Jul.  Anything that
    SUBTRACTS one from the other -- the SPEI water balance is rain minus PET --
    then compares three-quarters of a month of rain against half a month of
    evaporative demand, and the running month comes out looking wetter than a
    normal July in a season running 13% below normal.
    """
    reg = pd.read_csv(IMD / "registry_lgd791.csv")
    reg["ks"], reg["kd"] = norm(reg["state"]), norm(reg["district"])
    lut = {(r.ks, r.kd): r.district_id for r in reg.itertuples()}
    frames, matched, unmatched, daysf = [], set(), set(), []
    dupes = set()
    for f in sorted((ROOT / folder).glob("*.csv")):
        d = pd.read_csv(f, parse_dates=["date"])
        if days_col:
            dd = d.assign(_m=d["date"].values.astype("datetime64[M]"))
            daysf.append(dd.groupby("_m")["date"].nunique()
                         .rename(days_col).reset_index()
                         .rename(columns={"_m": "date"}))
        d["did"] = [lut.get((a, b), -1) for a, b in
                    zip(norm(d["state"]), norm(d["district"]))]
        unmatched |= set(zip(d.loc[d["did"] < 0, "state"],
                             d.loc[d["did"] < 0, "district"]))
        d = d[d["did"] >= 0]
        matched |= set(d["did"])

        #   ONE ROW PER DISTRICT PER DAY, ENFORCED.
        #   The LGD asset holds TWO polygon records for Purba Medinipur, so
        #   every GEE district export carries 792 rows per day rather than
        #   791.  The monthly aggregation below SUMS extensive quantities,
        #   which silently doubled that district's water balance: its June
        #   2025 PET came out at 322 mm against a West Bengal median of 171,
        #   and its AET, moisture adequacy, SPEI and crop stress all
        #   inherited the error.
        #
        #   Collapsing by mean is an approximation — the two records are
        #   different polygons with different areas, and their true
        #   combination is area-weighted, which the export does not carry.
        #   It is much closer than doubling, and it is bounded.  The exact
        #   fix is upstream: merge the two records in the source asset into
        #   one multipolygon and re-export.
        n0 = len(d)
        d = d.groupby(["date", "did"], as_index=False)[valcols].mean()
        if len(d) < n0:
            dupes |= {n0 - len(d)}
        d["date"] = d["date"].values.astype("datetime64[M]")   # month start
        agg = {c: how for c in valcols}
        frames.append(d.groupby(["date", "did"], as_index=False).agg(agg))
    if dupes:
        log(f"  {folder}: collapsed {max(dupes)} duplicate district-day "
            f"row(s) per file before aggregating (Purba Medinipur has two "
            f"polygon records in the LGD asset)")
    out = pd.concat(frames, ignore_index=True).rename(columns={"did":
                                                               "district_id"})
    out = out.groupby(["date", "district_id"], as_index=False)[valcols].sum() \
        if how == "sum" else \
        out.groupby(["date", "district_id"], as_index=False)[valcols].mean()
    if days_col and daysf:
        dv = (pd.concat(daysf, ignore_index=True)
              .groupby("date", as_index=False)[days_col].max())
        # no leading underscore: itertuples() renames such columns positionally
        dv["dim"] = dv["date"].dt.days_in_month
        dv[days_col + "_complete"] = (dv[days_col] >= dv["dim"]).astype(int)
        part = dv[dv[days_col + "_complete"] == 0]
        if len(part):
            log(f"  {folder}: incomplete month(s) — " + ", ".join(
                f"{r.date:%Y-%m} ({getattr(r, days_col)}/{r.dim} days)"
                for r in part.itertuples()))
        return out, len(matched), sorted(unmatched)[:10], dv.drop(columns="dim")
    return out, len(matched), sorted(unmatched)[:10]


def main():
    log("=" * 70)
    log("STEP 1 — clean & merge IMD + ERA5 ET + soil moisture on 791 LGD units")
    log("=" * 70)
    reg = pd.read_csv(IMD / "registry_lgd791.csv")

    # ---------- IMD rainfall (extensive) --------------------------------
    rain_m = monthly_from_pickle("rain", "sum")
    rn = normal_baseline(rain_m)
    # kept for the day-matched normal: the monthly frame cannot say WHICH days
    # of an incomplete month were observed, only how many landed in the sum
    rain_raw_daily = pd.read_pickle(IMD / "_district_daily_rain_lgd.pkl")
    rain_raw_daily.columns = rain_raw_daily.columns.astype(int)

    # anchor the normals to IMD's published LPA (see IMD ANCHOR note above)
    area = reg.set_index("district_id")["area_km2"]
    jj = rain_m[rain_m.index.month.isin(JJAS)]
    yt = jj.groupby(jj.index.year).sum(min_count=3)
    w = area.reindex(yt.columns).to_numpy(float)
    ok = np.isfinite(w)

    def _aw(s):
        v = s.to_numpy(float)
        m = ok & np.isfinite(v)
        return (v[m] * w[m]).sum() / w[m].sum() if m.any() else np.nan
    lpa_ours = float(yt.loc[1981:2020].apply(_aw, axis=1).mean())
    anchor = 1.0                      # diagnosed only — see the note above
    gap = IMD_JJAS_LPA_MM / lpa_ours
    log(f"  LPA diagnostic: ours {lpa_ours:.1f} mm (1981-2020) vs IMD "
        f"{IMD_JJAS_LPA_MM} mm (1971-2020) = {100*(gap-1):+.1f}%; normals left "
        f"UNSCALED so departure stays a same-source ratio")

    rain = long_imd(rain_m, "rain_mm")
    # DAY-MATCHED NORMAL FOR AN INCOMPLETE MONTH
    #   The monthly total is a sum over whatever days exist, but the normal is a
    #   full-calendar-month mean, so the running month is compared against a
    #   denominator covering days it has not lived through yet.  In July 2026
    #   that meant 23 days of rain against a 31-day normal: the panel reported
    #   -22.0% when the day-matched departure was +0.8%.  IMD's own published
    #   figures agree with the day-matched number, not the naive one.
    #
    #   Only the trailing month is ever affected -- all 666 completed months in
    #   the record have a full complement of days -- so this changes nothing the
    #   models were trained on, and for a complete month the day-matched normal
    #   is identical to the full-month normal by construction.
    dn, days_obs = daymatched_normal(rain_raw_daily, rn)
    nlong = dn.reset_index().melt(id_vars="date", var_name="district_id",
                                  value_name="normal_mm")
    nlong["district_id"] = nlong["district_id"].astype(int)
    rain = rain.merge(nlong, on=["date", "district_id"], how="left")
    rain = rain.merge(days_obs, on="date", how="left")
    with np.errstate(invalid="ignore", divide="ignore"):
        rain["pct_departure"] = np.where(
            rain["normal_mm"] >= MIN_NORMAL_MM,
            (rain["rain_mm"] - rain["normal_mm"]) / rain["normal_mm"] * 100,
            np.nan)
    log(f"  IMD rain: {rain['date'].dt.year.min()}-{rain['date'].dt.year.max()}"
        f", {rain['district_id'].nunique()} units")

    panel = rain

    # ---------- IMD temperature (intensive) -----------------------------
    for v in ("tmax", "tmin"):
        m = monthly_from_pickle(v, "mean")
        nn = normal_baseline(m)
        lo = long_imd(m, f"{v}_c")
        lo["nmn"] = lo["date"].dt.month
        nl = nn.reset_index().melt(id_vars="nmn", var_name="district_id",
                                   value_name=f"{v}_normal_c")
        nl["district_id"] = nl["district_id"].astype(int)
        lo = lo.merge(nl, on=["nmn", "district_id"], how="left").drop(
            columns="nmn")
        lo[f"{v}_anom"] = lo[f"{v}_c"] - lo[f"{v}_normal_c"]
        panel = panel.merge(lo, on=["date", "district_id"], how="left")
    log(f"  IMD Tmax/Tmin merged")

    # ---------- ERA5 AET/PET (extensive) + soil moisture (intensive) ----
    et, met, un_et, etdays = load_era5_folder(
        "Evapotranspiration_LGD", ["aet_mm", "pet_mm"], "sum",
        days_col="et_days")
    panel = panel.merge(et, on=["date", "district_id"], how="left")
    panel = panel.merge(etdays, on="date", how="left")
    log(f"  ERA5 ET merged: {met} units matched; unmatched sample {un_et}")
    sm, msm, un_sm = load_era5_folder("SM_LGD",
                                      ["swvl1", "swvl2", "swvl3", "swvl4"],
                                      "mean")
    panel = panel.merge(sm, on=["date", "district_id"], how="left")
    log(f"  ERA5 soil moisture merged: {msm} units matched")

    # ---------- optional layers ------------------------------------------
    #   These are exported by GEE_scripts/03_temperature_era5 and /07_vegetation
    #   and are NOT required: the folder may legitimately be empty on a fresh
    #   clone or before the first export. Each is merged when present and
    #   skipped with a line in the log when not, so a missing optional layer
    #   never stops a refresh — but never disappears silently either.
    #
    #   Both are INTENSIVE (a ratio and a temperature), so they aggregate to
    #   months by MEAN. Getting that wrong would sum NDVI over a month and
    #   produce a number around 8 where the scale runs -1..+1.
    for folder, cols, label in (
            ("Vegetation_LGD", ["ndvi", "evi"], "MODIS vegetation"),
            ("TemperatureERA5_LGD", ["t2m_max_c", "t2m_min_c", "t2m_mean_c"],
             "ERA5-Land 2m temperature")):
        d = ROOT / folder
        if not d.exists() or not any(d.glob("*.csv")):
            log(f"  {label}: no export in {folder}/ — skipped (optional)")
            continue
        try:
            v, mv, un_v = load_era5_folder(folder, cols, "mean")
            panel = panel.merge(v, on=["date", "district_id"], how="left")
            log(f"  {label} merged: {mv} units matched, columns {cols}")
        except Exception as e:
            log(f"  ! {label} in {folder}/ failed to merge — "
                f"{type(e).__name__}: {e}. Continuing without it.")

    # ---------- ENSO indices by year-month ------------------------------
    enso = pd.read_csv(V5 / "data" / "enso_monthly_v5.csv")
    keep = ["year", "month", "oni", "nino34_anom", "iod_dmi", "enso_phase",
            "enso_warm", "enso_cold", "enso_developing", "enso_decaying",
            "oni_mam", "iod_z", "enso_iod_interact"]
    keep = [c for c in keep if c in enso.columns]
    panel["year"] = panel["date"].dt.year
    panel["month"] = panel["date"].dt.month
    panel = panel.merge(enso[keep], on=["year", "month"], how="left")

    #   IOD POLES, ADDED ALONGSIDE — NOT REPLACING
    #   enso_monthly_v5.csv carries oni / nino34_anom / iod_dmi and is current
    #   to the last complete month. The HadISST series in Indices/ lag two to
    #   three months because they come off an SST reanalysis, so adopting them
    #   wholesale would trade the pole separation for two months of currency.
    #   Only the EAST and WEST poles are taken from them; DMI itself stays on
    #   the fresher source.
    #
    #   Why the poles at all: DMI is defined as west minus east (Saji et al.
    #   1999), and the monsoon response to the two is asymmetric, which a
    #   difference discards. Measured on this dataset the IOD earns nothing on
    #   its own (it cost 0.007 of skill at 7 days) but pays alongside ENSO
    #   (+0.037), which is Ashok et al. (2001): it modulates the ENSO
    #   response rather than acting independently.
    ip = OUTD / "indices_monthly.csv"
    if ip.exists():
        ix = pd.read_csv(ip)
        pole = [c for c in ("dmi_east", "dmi_west") if c in ix.columns]
        if pole:
            panel = panel.merge(ix[["year", "month"] + pole],
                                on=["year", "month"], how="left")
            cov = panel[pole[0]].notna().mean() * 100
            log(f"  IOD poles merged: {', '.join(pole)} "
                f"({cov:.1f}% of district-months covered)")
    else:
        log("  indices_monthly.csv absent — run 00_clean_indices.py for the "
            "IOD poles (optional)")

    # ---------- district identity + MAI ---------------------------------
    panel = panel.merge(reg[["district_id", "state", "district", "lgd_code",
                             "is_jk_ladakh"]], on="district_id", how="left")
    with np.errstate(invalid="ignore", divide="ignore"):
        panel["mai"] = np.clip(panel["aet_mm"] / panel["pet_mm"], 0, 1.5)

    front = ["date", "year", "month", "state", "district", "district_id",
             "lgd_code", "is_jk_ladakh"]
    cols = front + [c for c in panel.columns if c not in front]
    panel = panel[cols].sort_values(["district_id", "date"])
    panel.to_csv(OUTD / "master_panel_lgd.csv", index=False)

    # daily rain (wide) for antecedent / extreme features later
    pd.read_pickle(IMD / "_district_daily_rain_lgd.pkl").to_pickle(
        OUTD / "daily_rain_lgd.pkl")

    rep = {
        "units": int(reg.shape[0]),
        "months": int(panel["date"].nunique()),
        "date_range": [str(panel["date"].min())[:10],
                       str(panel["date"].max())[:10]],
        "rows": int(len(panel)),
        "columns": list(panel.columns),
        "era5_et_units_matched": met,
        "era5_sm_units_matched": msm,
        "normal_period": f"{WMO_LO}-{WMO_HI} (WMO-No. 1203)",
        "imd_lpa_anchor": 1.0,
        "jjas_lpa_ours_mm": round(lpa_ours, 1),
        "jjas_lpa_imd_mm": IMD_JJAS_LPA_MM,
        "imd_lpa_note": (
            f"our JJAS LPA {lpa_ours:.1f} mm (1981-2020) vs IMD "
            f"{IMD_JJAS_LPA_MM} mm (1971-2020), {100*(gap-1):+.1f}%. Normals "
            "are NOT rescaled: departure is a same-source ratio and our actual "
            "agrees with IMD's to -1.2%, so the gap is a baseline-period "
            "difference (missing 1970s), reported not corrected."),
        "layers": {
            "rain_mm": "IMD 0.25deg gauge, monthly sum",
            "tmax_c/tmin_c": "IMD 1deg gauge, monthly mean",
            "aet_mm/pet_mm": "ERA5-Land, monthly sum",
            "swvl1-4": "ERA5-Land soil moisture, monthly mean",
            "oni/iod_dmi": "NOAA/BoM ENSO indices",
        },
    }
    (OUTD / "merge_report_lgd.json").write_text(json.dumps(rep, indent=1),
                                                encoding="utf-8")
    log(f"\n  master_panel_lgd.csv: {len(panel):,} rows x {len(cols)} cols, "
        f"{rep['date_range'][0]}..{rep['date_range'][1]}")
    log(f"  null rates (2026 rows):")
    r26 = panel[panel["year"] == 2026]
    for c in ["rain_mm", "pct_departure", "tmax_c", "aet_mm", "pet_mm",
              "swvl2", "mai", "oni", "iod_dmi"]:
        log(f"    {c:14s} {100*r26[c].isna().mean():5.1f}% null")
    log(f"  wrote master_panel_lgd.csv, daily_rain_lgd.pkl, merge_report_lgd.json")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
