"""
common_v5.py  —  shared configuration for the SELF-CONTAINED v5 pipeline.

v5 no longer depends on v4 for anything: the registry, the name harmonization,
the master panel and every derived index are built here in v5 from the raw
Google Earth Engine exports.

IMPORTANT — interpreter: a Python 3.14 with no packages owns the bare `py`
command on this machine.  Run every v5 script with **py -3.13**, e.g.
    py -3.13 -X utf8 "v5/41_build_master_panel.py"

Every path is anchored to this file's location, so scripts run from VS Code
regardless of the working directory.  If a path ever fails, the offending
absolute path is printed — send it and it is corrected here, in one place.

Grounding for the cleaning rules:
  * "Geospatial Data Harmonization and Panel Construction" (cleaning spec) —
    the Autonomous Entity Framework: the 8 GAUL "Disputed (...)" polygons are
    NOT dropped and NOT merged; they are preserved as standalone units with a
    synthetic LGD code block (9001+), and the panel is multi-keyed on
    [date, state, district_id].  Extensive variables aggregate by sum,
    intensive by mean, % departure by ratio-of-aggregates.
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------- paths
V5 = Path(__file__).resolve().parent                 # ...\El-nino forecast\v5
BASE = V5.parent                                     # ...\El-nino forecast
DATA = V5 / "data"
DATA.mkdir(exist_ok=True)
FIGS = V5 / "figures"
FIGS.mkdir(exist_ok=True)

CHIRPS_DIR = BASE / "Chirps"
ET_DIR     = BASE / "Evapotranspiration"
SM_DIR     = BASE / "moisture"
TEMP_DIR   = BASE / "Temperature"          # TerraClimate monthly
ERA5T_DIR  = BASE / "TemperatureERA5"      # ERA5-Land daily Tmax/Tmin (2026+)
FORE_DIR   = BASE / "Fore"
MAPS_DIR   = BASE / "maps-master" / "maps-master"

# ---- v5 outputs (all self-contained) --------------------------------------
REGISTRY_CSV   = DATA / "district_registry_v5.csv"
CFS_MONTHLY    = DATA / "cfsv2_district_monthly_v5.csv"
MASTER_CSV     = DATA / "master_panel_v5.csv"          # wide, district-month
MASTER_LONG    = DATA / "master_timeseries_v5.csv"     # LONG / time-linear
FEATURES_CSV   = DATA / "features_v5.csv"
NOAA_CACHE     = DATA / "noaa_indices_cache_v5.csv"

PANEL_START = "1981-01"

# ---------------------------------------------------------------- horizons
# 30- and 60-day horizons were REMOVED: at those leads the statistical signal
# collapses toward climatology, so publishing them implied a precision the
# method does not have.  Only the two defensible horizons are shipped.
HORIZONS = [7, 14]
ANTECEDENT = [7, 14, 30, 60, 90]     # look-back windows (features, not targets)
ISSUE_STRIDE = 5                     # sample an issue date every N days
MIN_NORMAL_MM = 5.0                  # ratio-of-aggregates guard

TRAIN_END = 2016                     # train issue-years <= 2016
VAL_END = 2019                       # val 2017-2019; test 2020-2026

# ---------------------------------------------------------------- cleaning
STATE_FIXES = {"Arunchal Pradesh": "Arunachal Pradesh"}
DISTRICT_FIXES = {
    ("Delhi", "Shahadra"):     "Shahdara",
    ("Assam", "Kamrup Metro"): "Kamrup Metropolitan",
    ("Assam", "Kamrup Rural"): "Kamrup",
}
DISPUTED_LGD_START = 9001

# CFSv2 PET flux -> water depth (cleaning spec sec 6.3):
# mm/day = W/m2 * 86400 s / (2.45e6 J/kg) ... = W/m2 * 0.035265
WM2_TO_MMDAY = 86400.0 / 2.45e6 * 1000.0 / 1000.0 * 1000.0 / 1000.0

# ---- feature lists shared by the modelling steps --------------------------
COVARIATES = ["swvl1", "swvl2", "swvl3", "swvl4",
              # cfs_soilm_5cm/70cm/150cm were REMOVED: CFSv2 ends 2025-12, so
              # they are present throughout training and entirely absent at
              # the 2026 issue date — a pure train/serve gap.  ERA5-Land
              # swvl1-4 measures the same physical quantity across the whole
              # 1981-2026 record, so nothing is lost by dropping them.
              "spei_era5_1", "spei_era5_4", "spei_era5_12", "mai", "sdd",
              "aet_mm", "pet_mm", "oni", "nino34_anom", "iod_dmi",
              # unified temperature: CFSv2 -> TerraClimate -> ERA5-Land.
              # Using the source-specific columns instead breaks training —
              # era5_* is 100% NaN before 2026 and cfs_* is 100% NaN in 2026.
              "tmax", "tmin", "spei_harg_4", "gdd_kharif",
              "wrsi_proxy", "extreme_days",
              # ---- ENSO / IOD state (built by 43_build_enso_features.py) ---
              # A raw monthly ONI is not what the monsoon responds to.  These
              # encode the aspects the literature identifies: the CPC
              # five-season phase, whether the event is DEVELOPING or decaying
              # (Kumar et al. 2006), the spring signal before the
              # predictability barrier (Webster & Yang 1992), and the IOD
              # compensation term (Ashok et al. 2001).  All strictly trailing.
              "enso_warm", "enso_cold", "enso_sign", "enso_developing",
              "enso_decaying", "enso_intensity", "oni_tend_3", "oni_tend_6",
              "oni_djf_prev", "oni_mam", "iod_z", "iod_pos", "iod_neg",
              "enso_iod_interact",
              # per-district ENSO signature, composited on TRAINING years only
              "dep_elnino", "dep_lanina", "dep_neutral", "enso_signature",
              "lanina_signature"]
ANT_FEATS = ["ant_rain_7", "ant_rain_14", "ant_rain_30", "ant_rain_60",
             "ant_rain_90", "ant_dep_7", "ant_dep_14", "ant_dep_30",
             "ant_dep_60", "ant_dep_90", "wet30", "extreme30", "dsw",
             "doy_sin", "doy_cos"]


def log(m=""):
    print(m, flush=True)


def safe_to_csv(df, path, **kw):
    """Write a CSV, retrying through transient OneDrive/Excel file locks."""
    for attempt in range(6):
        try:
            df.to_csv(path, **kw)
            return
        except PermissionError:
            if attempt == 5:
                alt = path.with_suffix(f".retry{path.suffix}")
                df.to_csv(alt, **kw)
                log(f"    ! {path.name} locked -> wrote {alt.name} instead")
                return
            time.sleep(2)


def clean_names(df, keep_disputed=True):
    """Trim/standardize names. Disputed units are KEPT by default (Autonomous
    Entity Framework, cleaning spec sec 5)."""
    for col in ("state", "district"):
        df[col] = (df[col].astype(str).str.strip()
                          .str.replace(r"\s+", " ", regex=True))
    df["state"] = df["state"].replace(STATE_FIXES)
    df["district"] = [DISTRICT_FIXES.get(k, k[1])
                      for k in zip(df["state"], df["district"])]
    if not keep_disputed:
        df = df[~df["state"].str.startswith("Disputed")].copy()
    return df


def build_registry():
    """Canonical registry of ALL units (official + disputed).  Official units
    get 0..N-1 (stable, sorted); disputed units get synthetic codes 9001+."""
    probe = pd.read_csv(CHIRPS_DIR / "CHIRPS_district_daily_1981.csv",
                        usecols=["state", "district"]).drop_duplicates()
    probe = clean_names(probe, keep_disputed=True)
    units = probe.drop_duplicates().sort_values(["state", "district"])
    units["is_disputed"] = units["state"].str.startswith("Disputed")

    official = units[~units["is_disputed"]].reset_index(drop=True)
    official["district_id"] = np.arange(len(official), dtype=int)
    disputed = units[units["is_disputed"]].reset_index(drop=True)
    disputed["district_id"] = DISPUTED_LGD_START + np.arange(len(disputed))
    return pd.concat([official, disputed], ignore_index=True)[
        ["state", "district", "district_id", "is_disputed"]]


def get_registry():
    if REGISTRY_CSV.exists():
        return pd.read_csv(REGISTRY_CSV)
    reg = build_registry()
    safe_to_csv(reg, REGISTRY_CSV, index=False)
    return reg


def latest_source_dates():
    """Auto-detect the true last observation date of every daily source, so the
    operational forecast follows the data instead of a hard-coded date."""
    out = {}
    for label, folder, pat in [
            ("chirps", CHIRPS_DIR, "CHIRPS_district_daily_*.csv"),
            ("et", ET_DIR, "ERA5L_aet_pet_district_daily_*.csv"),
            ("soil", SM_DIR, "ERA5L_soilmoisture_district_daily_*.csv"),
            ("temp", ERA5T_DIR, "ERA5L_tmax_tmin_district_daily_*.csv")]:
        files = sorted(folder.glob(pat))
        if not files:
            out[label] = None
            continue
        d = pd.read_csv(files[-1], usecols=["date"])
        out[label] = pd.to_datetime(d["date"]).max()
    return out


def load_daily_chirps(years=range(1981, 2031)):
    """Concatenate every CHIRPS daily district file, harmonize names, keep the
    disputed autonomous units, attach district_id.  No row is dropped."""
    reg = get_registry()
    key = reg.set_index(["state", "district"])["district_id"]
    frames, missing = [], []
    for y in years:
        f = CHIRPS_DIR / f"CHIRPS_district_daily_{y}.csv"
        if not f.exists():
            missing.append(y)
            continue
        d = pd.read_csv(f, parse_dates=["date"])
        d = clean_names(d, keep_disputed=True)
        d["district_id"] = d.set_index(["state", "district"]).index.map(key)
        frames.append(d[["date", "state", "district", "district_id",
                         "rain_mm", "normal_mm"]])
    out = pd.concat(frames, ignore_index=True)
    return out, reg


def build_antecedent_daily(daily):
    """Attach every antecedent / seasonality feature to a daily CHIRPS frame.

    Shared by 52_forecast_from_latest.py (one issue date) and
    56_forecast_backfill.py (an archive of issue dates) so the two cannot
    drift apart — a forecast issued for the dashboard's date picker must be
    built exactly the way the operational one is.

    The frame is modified and returned; it must already carry district_id,
    date, rain_mm and normal_mm.
    """
    import numpy as _np
    import pandas as _pd

    daily = daily.sort_values(["district_id", "date"]).reset_index(drop=True)
    for c in ("rain_mm", "normal_mm"):
        daily[c] = _pd.to_numeric(daily[c], errors="coerce")
    g = daily.groupby("district_id", sort=False, observed=True)

    for h in ANTECEDENT:
        ar = g["rain_mm"].rolling(h, min_periods=h).sum() \
            .reset_index(level=0, drop=True)
        an = g["normal_mm"].rolling(h, min_periods=h).sum() \
            .reset_index(level=0, drop=True)
        daily[f"ant_rain_{h}"] = ar
        with _np.errstate(invalid="ignore", divide="ignore"):
            daily[f"ant_dep_{h}"] = _np.where(
                an >= MIN_NORMAL_MM, (ar - an) / an * 100.0, _np.nan)

    wet = (daily["rain_mm"] > 1.0).astype(float)
    daily["wet30"] = wet.groupby(daily["district_id"], sort=False) \
        .rolling(30, min_periods=30).sum().reset_index(level=0, drop=True)
    wet_rain = daily["rain_mm"].where(daily["rain_mm"] > 1.0)
    p95 = wet_rain.groupby(daily["district_id"]).transform(
        lambda s: s.quantile(0.95))
    daily["extreme_day"] = (daily["rain_mm"] > p95).astype(float)
    daily["extreme30"] = daily["extreme_day"] \
        .groupby(daily["district_id"], sort=False) \
        .rolling(30, min_periods=30).sum().reset_index(level=0, drop=True)
    idx = _np.arange(len(daily), dtype=float)
    lastwet = _pd.Series(
        _np.where(daily["rain_mm"] > 1.0, idx, _np.nan), index=daily.index)
    lastwet = lastwet.groupby(daily["district_id"], sort=False).ffill()
    daily["dsw"] = idx - lastwet.to_numpy()
    doy = daily["date"].dt.dayofyear
    daily["doy_sin"] = _np.sin(2 * _np.pi * doy / 365.25)
    daily["doy_cos"] = _np.cos(2 * _np.pi * doy / 365.25)
    return daily
