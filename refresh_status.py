r"""refresh_status.py  —  what do I have to update today?

One command that reads every product on disk, works out how far behind it is,
and prints the exact action to bring it current. Run it before a refresh to
get the list, and again afterwards to confirm the list is empty.

    py -3.13 -X utf8 "refresh_status.py"

WHY A TOOL AND NOT A CHECKLIST IN A README
  A written checklist goes stale the moment a lag changes, and every product
  here lags real time by a DIFFERENT and NON-CONSTANT amount: IMD by 1-2 days,
  ERA5-Land by about a week, MODIS by up to 16, CHIRPS by two to three weeks.
  Without knowing the expected lag you cannot tell "this export failed" from
  "this data does not exist yet", and that distinction is the whole point of
  the exercise. So the expected lag is encoded per product and the report says
  CURRENT / DUE / STALE rather than just printing a date.

STATUS MEANINGS
  CURRENT   within its normal publication lag; nothing to do
  DUE       new data exists upstream that you do not have; run the action
  STALE     further behind than the source's own lag explains; something
            failed, or the export was never re-run
  MISSING   the product folder does not exist yet
"""
import datetime
import json
import pathlib
import sys

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent
IMD = ROOT / "IMD_Data"
TODAY = datetime.date.today()

#   (label, kind, path, expected lag in days, action)
#   `expected lag` is the source's own normal delay. Anything inside it is
#   CURRENT; DUE means the source has moved on and you have not.
PRODUCTS = [
    ("IMD rainfall",        "imd",  "_district_daily_rain_lgd.pkl", 2,
     'py -3.13 -X utf8 "IMD_Data/build_imd_lgd_csvs.py"'),
    ("IMD Tmax",            "imd",  "_district_daily_tmax_lgd.pkl", 2,
     'py -3.13 -X utf8 "IMD_Data/build_imd_lgd_csvs.py"'),
    ("IMD Tmin",            "imd",  "_district_daily_tmin_lgd.pkl", 2,
     'py -3.13 -X utf8 "IMD_Data/build_imd_lgd_csvs.py"'),
    ("ERA5-Land AET/PET",   "csv",  "Evapotranspiration_LGD",       8,
     "GEE  01_evapotranspiration/era5land_aet_pet_LGD.js   MODE='UPDATE'"),
    ("ERA5-Land soil water", "csv", "SM_LGD",                       8,
     "GEE  02_soil_moisture/era5land_soilmoisture_LGD.js   MODE='UPDATE'"),
    ("ERA5-Land 2m temp",   "csv",  "TemperatureERA5_LGD",          8,
     "GEE  03_temperature_era5/era5land_t2m_LGD.js         MODE='UPDATE'"),
    ("CHIRPS (validation)", "csv",  "Chirps_LGD",                  25,
     "GEE  04_rainfall_crosscheck/chirps_daily_LGD.js      MODE='UPDATE'"),
    #   CFSv2_LGD/ is where the export actually lands and what
    #   features/26_cfsv2_layers.py reads; "CSFv2 updated" was an earlier
    #   folder name (with the typo) and checking it reported a product that
    #   has 38 years on disk as MISSING.
    ("CFSv2 (roadmap)",     "csv",  "CFSv2_LGD",                    2,
     "GEE  05_forecast_cfsv2/cfsv2_daily_LGD.js            MODE='UPDATE'"),
    ("MODIS NDVI/EVI",      "csv",  "Vegetation_LGD",              20,
     "GEE  07_vegetation/modis_ndvi_evi_LGD.js             MODE='UPDATE'"),
    ("Ocean SST boxes",     "month", "Indices",                    40,
     "GEE  06_ocean_indices/oisst_nino34_iod.js            MODE='UPDATE'"),
    #   ONI IS DATED ON THE CENTRE OF A THREE-MONTH MEAN, so the newest value
    #   CPC has ever published is already about 95 days behind by this stamp:
    #   the AMJ season, centred on 1 May, appears in early August. A 40-day lag
    #   therefore reported STALE permanently, even on the day of publication --
    #   and a status that is always red is a status nobody reads. 100 days
    #   allows the current season plus a few days' publication slack; anything
    #   beyond that really is a missed monthly refresh.
    ("ONI / DMI cache",     "enso", "noaa_indices_cache.csv",     100,
     'py -3.13 -X utf8 "Indices/fetch_indices.py"'),
    ("UPAg sowing",         "upag", "UPAJ",                        10,
     "download the weekly release into UPAJ/"),
]

#   STATIC PRODUCTS ARE NOT JUDGED BY DATE
#   Irrigation and cropland have no newer vintage upstream (LGRIP30 is a
#   single 2015 epoch; the C3S land-cover series ends in 2022), so "how many
#   days behind" is meaningless for them. What CAN be wrong is the BOUNDARY
#   they were exported on, and that is invisible in a date check.
#
#   GlobalIrrigation/ is still the GAUL-2024 export at 701 units. The 90-odd
#   LGD districts with no GAUL equivalent -- every unit in Jammu & Kashmir and
#   Ladakh among them -- therefore have no irrigation value of their own and
#   fall back to a state median. Irrigation drives the Kharif buffer and the
#   whole Rabi water term, so those districts' crop stress rests on a stand-in.
STATIC = [
    ("Irrigated area", "GlobalIrrigation", 791,
     "GEE  08_static_layers/06_irrigation_timeseries_1992_2022.js"),
    ("Cropland extent", "LGRIP30", 791,
     "GEE  08_static_layers/05_lgrip30_cropland_2015.js"),
]


def static_units(folder):
    """Distinct (state, district) pairs in a static export, or None."""
    d = ROOT / folder
    if not d.exists():
        return None
    files = sorted(d.glob("*.csv"))
    if not files:
        return None
    try:
        dd = pd.read_csv(files[-1], usecols=["state", "district"])
        return len(dd.drop_duplicates())
    except Exception:
        return None


def last_date(kind, target):
    """Latest date a product carries, or None if it is not on disk."""
    try:
        if kind == "imd":
            p = IMD / target
            if not p.exists():
                return None
            return pd.read_pickle(p).index.max().date()

        if kind in ("csv", "month"):
            d = ROOT / target
            if not d.exists():
                return None
            files = sorted(d.glob("*.csv"))
            if not files:
                return None
            #   Only the newest file needs reading: these are per-year
            #   exports and the last one alphabetically is the current year.
            col = "month" if kind == "month" else "date"
            best = None
            for f in files[-2:]:
                dd = pd.read_csv(f, usecols=lambda c: c.strip() == col)
                if dd.empty:
                    continue
                v = pd.to_datetime(dd.iloc[:, 0], errors="coerce").max()
                if pd.notna(v):
                    best = max(best, v.date()) if best else v.date()
            return best

        if kind == "enso":
            hits = list(ROOT.glob(target)) or list(ROOT.glob("**/" + target))
            if not hits:
                return None
            d = pd.read_csv(hits[0])
            d = d.dropna(subset=["oni"])
            if d.empty:
                return None
            r = d.iloc[-1]
            return datetime.date(int(r["year"]), int(r["month"]), 1)

        if kind == "upag":
            d = ROOT / target
            if not d.exists():
                return None
            files = sorted(d.glob("*.csv"))
            if not files:
                return None
            #   UPAg dates are dd-mm-yyyy and the column name varies between
            #   releases, so find whichever column parses as dates.
            dd = pd.read_csv(files[-1], low_memory=False)
            best = None
            for c in dd.columns:
                if "date" not in c.lower():
                    continue
                v = pd.to_datetime(dd[c], errors="coerce", dayfirst=True).max()
                if pd.notna(v):
                    best = max(best, v.date()) if best else v.date()
            return best
    except Exception as e:
        print(f"    ! could not read {target}: {type(e).__name__}: {e}")
    return None


def main():
    print("=" * 78)
    print(f"MonsoonCast refresh status          today {TODAY:%Y-%m-%d}")
    print("=" * 78)
    print(f"{'PRODUCT':<22}{'LATEST':<13}{'BEHIND':>7}  {'STATUS':<9} ")
    print("-" * 78)

    todo = []
    for label, kind, target, lag, action in PRODUCTS:
        got = last_date(kind, target)
        if got is None:
            print(f"{label:<22}{'—':<13}{'—':>7}  {'MISSING':<9}")
            todo.append((label, action, "folder does not exist yet"))
            continue
        behind = (TODAY - got).days
        if behind <= lag:
            status = "CURRENT"
        elif behind <= lag * 2 + 3:
            status = "DUE"
        else:
            status = "STALE"
        print(f"{label:<22}{got:%Y-%m-%d}   {behind:>5} d  {status:<9}")
        if status != "CURRENT":
            todo.append((label, action, f"{behind} days behind "
                                        f"(source lag ~{lag})"))

    print("-" * 78)
    print("\nSTATIC LAYERS — judged on boundary vintage, not on date\n")
    for label, folder, want, action in STATIC:
        got = static_units(folder)
        if got is None:
            print(f"{label:<22}{'—':<13}{'—':>7}  {'MISSING':<9}")
            todo.append((label, action, f"{folder}/ has no export"))
            continue
        ok = got >= want
        print(f"{label:<22}{got} units{'':<4}{'':>7}  "
              f"{'CURRENT' if ok else 'WRONG BOUNDARY':<9}")
        if not ok:
            todo.append((label, action,
                         f"{got} units on disk, need {want} — the "
                         f"{want - got} LGD districts with no GAUL equivalent "
                         f"fall back to a state median"))

    print("-" * 78)
    if not todo:
        print("\nEverything is current. Rebuild when you want:")
        print('  py -3.13 -X utf8 "v5/monsooncast/run_all.py" --fast')
        return

    print(f"\n{len(todo)} PRODUCT(S) TO UPDATE\n")
    for i, (label, action, why) in enumerate(todo, 1):
        print(f"  {i}. {label}  —  {why}")
        print(f"     {action}\n")

    print("After the downloads land:")
    print('  py -3.13 -X utf8 "GEE_scripts/verify_exports.py"      '
          '# refuses overlapping files')
    print('  py -3.13 -X utf8 "v5/monsooncast/run_all.py" --fast   '
          '# rebuild panel -> dashboard')


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
