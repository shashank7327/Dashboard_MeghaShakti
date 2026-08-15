r"""
build_dashboard.py  —  the single entry point.

WHAT THIS IS FOR
  You have just cloned this repository and you want the dashboard. Run:

      py -3.13 -X utf8 build_dashboard.py

  It checks your environment, tells you exactly what is missing and how to get
  it, and then runs the pipeline end to end. If you only want to LOOK at the
  dashboard, you do not need this script at all -- run

      py -3.13 view_dashboard.py

  which serves the already-built dashboard and opens your browser at it.

WHAT IT DOES, IN ORDER
  1. checks the Python version and the installed packages
  2. checks which datasets are present and how old they are
  3. optionally fetches the data that can be fetched automatically
  4. runs the pipeline, which rebuilds the dashboard from scratch
  5. tells you where the output is

WHY A SCRIPT AND NOT A LIST OF COMMANDS IN THE README
  Because the failure that actually happens to a new user is not "I typed the
  command wrong", it is "I ran step 4 without noticing that step 2 had not
  produced anything", and the error that surfaces is a KeyError three files
  deep. Each stage here is checked before the next one starts.

Run:
  py -3.13 -X utf8 build_dashboard.py            check, then build
  py -3.13 -X utf8 build_dashboard.py --check    check only, build nothing
  py -3.13 -X utf8 build_dashboard.py --fetch    fetch fresh data first
  py -3.13 -X utf8 build_dashboard.py --fast     skip model retraining (default)
  py -3.13 -X utf8 build_dashboard.py --full     retrain the models too (~40 min)
"""
import argparse
import importlib
import pathlib
import shutil
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent
PY = [sys.executable, "-X", "utf8"]

REQUIRED = [
    ("pandas", "pandas"), ("numpy", "numpy"), ("sklearn", "scikit-learn"),
    ("xgboost", "xgboost"), ("lightgbm", "lightgbm"), ("shapefile", "pyshp"),
    ("shapely", "shapely"), ("requests", "requests"),
]
OPTIONAL = [
    ("imdlib", "imdlib", "fetching new IMD rainfall/temperature days"),
    ("ecmwf.opendata", "ecmwf-opendata", "fetching the ECMWF ensemble"),
    ("cfgrib", "cfgrib", "reading the ECMWF GRIB files"),
    ("openpyxl", "openpyxl", "writing the Excel master exports"),
]

#   Inputs the pipeline reads. "required" means the build cannot start.
INPUTS = [
    ("IMD_Data/_district_daily_rain_lgd.pkl", True,
     "IMD rainfall, aggregated to districts",
     'py -3.13 -X utf8 "IMD_Data/build_imd_lgd_csvs.py"'),
    ("IMD_Data/_district_daily_tmax_lgd.pkl", True, "IMD maximum temperature",
     'py -3.13 -X utf8 "IMD_Data/build_imd_lgd_csvs.py"'),
    ("IMD_Data/_district_daily_tmin_lgd.pkl", True, "IMD minimum temperature",
     'py -3.13 -X utf8 "IMD_Data/build_imd_lgd_csvs.py"'),
    ("IMD_Data/registry_lgd791.csv", True, "the 791-district registry",
     "shipped with this repository"),
    ("IMD_Data/lgd_shapefile/india_districts_lgd.shp", True,
     "district boundaries for the map", "shipped with this repository"),
    ("Evapotranspiration_LGD", False, "ERA5-Land AET/PET (Earth Engine)",
     "docs/02_DATA_EXTRACTION.md section 2"),
    ("SM_LGD", False, "ERA5-Land soil moisture (Earth Engine)",
     "docs/02_DATA_EXTRACTION.md section 2"),
    ("Vegetation_LGD", False, "MODIS NDVI/EVI (Earth Engine)",
     "docs/02_DATA_EXTRACTION.md section 2"),
    ("CFSv2_LGD", False, "CFSv2 model layers (Earth Engine)",
     "docs/02_DATA_EXTRACTION.md section 2"),
    ("ECMWF_LGD", False, "ECMWF ensemble forecast",
     'py -3.13 -X utf8 "v5/monsooncast/forecast_input/25_ecmwf_opendata_lgd.py"'),
    ("UPAJ", False, "UPAg / DA&FW sowing releases",
     "docs/02_DATA_EXTRACTION.md section 3 (manual download)"),
]


def hdr(t):
    print("\n" + "=" * 74)
    print(t)
    print("=" * 74, flush=True)


def flush():
    """Our own prints are buffered; a subprocess writes to the terminal
    directly. Without flushing first, the child's output appears ABOVE ours
    and the report reads in the wrong order."""
    sys.stdout.flush()


def has_content(rel):
    """A folder counts as present only if it holds CSVs, not merely exists."""
    p = HERE / rel
    if not p.exists():
        return False
    if p.is_dir():
        return any(p.glob("*.csv"))
    return p.stat().st_size > 0


def check_env():
    hdr("1. ENVIRONMENT")
    ok = True
    v = sys.version_info
    print(f"  Python {v.major}.{v.minor}.{v.micro}")
    if (v.major, v.minor) < (3, 11):
        print("    ! This project is built for Python 3.13. Older versions "
              "will fail on newer pandas syntax.")
        ok = False

    missing = []
    for mod, pip in REQUIRED:
        try:
            importlib.import_module(mod)
        except Exception:
            missing.append(pip)
    if missing:
        ok = False
        print(f"  MISSING required packages: {', '.join(missing)}")
        print(f"    fix:  {sys.executable} -m pip install -r requirements.txt")
    else:
        print(f"  all {len(REQUIRED)} required packages present")

    for mod, pip, why in OPTIONAL:
        try:
            importlib.import_module(mod)
        except Exception:
            print(f"  optional '{pip}' absent — needed only for {why}")
    return ok


def check_inputs():
    hdr("2. INPUT DATA")
    blocking = []
    for rel, required, what, how in INPUTS:
        present = has_content(rel)
        tag = "ok     " if present else ("MISSING" if required else "absent ")
        print(f"  {tag} {rel:<48} {what}")
        if not present:
            print(f"          -> {how}")
            if required:
                blocking.append(rel)
    if blocking:
        print(f"\n  {len(blocking)} required input(s) missing — cannot build.")
        print("  Everything else is optional: the pipeline skips a layer whose "
              "source is absent\n  rather than failing, so a partial build "
              "still produces a working dashboard.")
    return not blocking


def run(rel, label):
    print(f"\n>>> {label}\n    {rel}")
    r = subprocess.run(PY + [str(HERE / rel)])
    if r.returncode != 0:
        print(f"\n!!! FAILED: {rel} (exit {r.returncode})")
        return False
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="check only")
    ap.add_argument("--fetch", action="store_true",
                    help="download fresh IMD, indices and ECMWF first")
    ap.add_argument("--full", action="store_true",
                    help="retrain the models (~40 min) instead of reusing them")
    a = ap.parse_args()

    print("=" * 74)
    print("MonsoonCast — build the dashboard")
    print("=" * 74)
    print("  To simply VIEW the dashboard you do not need to build anything:")
    print("  run   py -3.13 view_dashboard.py")

    env_ok = check_env()
    data_ok = check_inputs()

    hdr("3. DATA CURRENCY")
    flush()
    subprocess.run(PY + [str(HERE / "refresh_status.py")])

    if a.check:
        print("\n--check given: nothing was built.")
        return
    if not env_ok:
        sys.exit("\nInstall the missing packages, then re-run.")
    if not data_ok:
        sys.exit("\nObtain the missing required inputs, then re-run. "
                 "See docs/02_DATA_EXTRACTION.md.")

    if a.fetch:
        hdr("4. FETCH")
        for rel, label in (
                ("Indices/fetch_indices.py", "climate indices (MJO, ENSO, IOD)"),
                ("IMD_Data/build_imd_lgd_csvs.py", "IMD days since the last run"),
                ("v5/monsooncast/forecast_input/25_ecmwf_opendata_lgd.py",
                 "ECMWF ensemble forecast")):
            if not run(rel, label):
                print("    (continuing — a failed fetch leaves the previous "
                      "data in place)")

    hdr("5. BUILD")
    cmd = PY + [str(HERE / "v5" / "monsooncast" / "run_all.py")]
    if not a.full:
        cmd.append("--fast")
        print("  --fast: reusing the trained models. Pass --full to retrain.")
    flush()
    if subprocess.run(cmd).returncode != 0:
        sys.exit("\nThe pipeline stopped. The failing step is named above; "
                 "re-run it alone to see the full error.")

    hdr("6. DONE")
    print(f"  Standalone dashboard : "
          f"{HERE / 'v5' / 'dashboard_lgd' / 'MonsoonCast_LGD.html'}")
    print(f"  Data payload         : "
          f"{HERE / 'v5' / 'dashboard_lgd' / 'data.json'}")
    print(f"  Spreadsheet exports  : {HERE / 'v5' / 'masters_lgd'}")
    print("\n  React version (optional, needs Node.js):")
    print("    npm --prefix v5/dashboard_react install")
    print("    npm --prefix v5/dashboard_react run dev")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
