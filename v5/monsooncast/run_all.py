r"""
v5/monsooncast/run_all.py  —  run the MonsoonCast pipeline end to end:
clean -> features -> crops -> model -> forecast -> dashboard -> masters.

PATHS ARE ABSOLUTE, ALWAYS
  Every script resolves its inputs and outputs from its own __file__, never from
  the current working directory, so this runs identically from VS Code's Run
  button, from an integrated terminal opened anywhere, or from a scheduled task.
  You do not need to cd into a particular folder first.

PREREQUISITES
  Evapotranspiration_LGD/, SM_LGD/    ERA5-Land GEE exports (manual, from GEE)
  UPAJ/Sowing*.csv                    UPAg weekly sowing releases (manual)
  See RUN_GUIDE_VSCODE.md for first-time setup.

  The IMD grids are NOT a prerequisite any more: `--update-imd` fetches every
  day published since the last run and re-aggregates to the 791 LGD units, so a
  refresh is one command rather than two remembered scripts.

USAGE
  py -3.13 -X utf8 "v5/monsooncast/run_all.py"               full run
  py -3.13 -X utf8 "v5/monsooncast/run_all.py" --update-imd  fetch new IMD days first
  py -3.13 -X utf8 "v5/monsooncast/run_all.py" --fast        skip model retraining
  py -3.13 -X utf8 "v5/monsooncast/run_all.py" --from 14     start at a given step
  py -3.13 -X utf8 "v5/monsooncast/run_all.py" --list        show the steps only

  The usual daily refresh is:
      run_all.py --update-imd --fast

WHY --fast EXISTS
  08_model_bakeoff retrains every model family and is by far the longest step.
  Nothing upstream of it changes when you only refresh observations, rebuild the
  crop layers or edit the dashboard, so --fast reuses the persisted model and
  turns a ~40 minute run into a few minutes.
"""
import pathlib
import subprocess
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
V5 = HERE.parent

# (step number, relative path, description, slow?)
STEPS = [
    ("01", "cleaning/01_clean_merge_panel.py",
     "clean & merge IMD + ERA5 + ENSO -> master panel", False),
    ("02", "cleaning/02_enso_phase_climatology.py",
     "ENSO-phase climatology features", False),
    ("03", "features/03_build_features.py",
     "SPEI-1/4/12, GDD, SDD, moisture adequacy", False),
    ("21", "features/21_daily_features.py",
     "daily rolling layers (7d / 30d / season-to-date)", False),
    ("17", "features/17_feature_reliability.py",
     "per-district support & observational spread", False),
    ("18", "features/18_irrigation_fraction.py",
     "irrigated fraction per district", False),
    ("07", "validation/07_validate_features.py",
     "validate features against IMD published figures", False),
    ("08", "modelling/08_model_bakeoff.py",
     "model bake-off, pick & persist the best", True),
    ("14", "crops/14_build_crop_mask.py",
     "crop-area mask from UPAg sowing", False),
    ("19", "crops/19_sowing_dynamics.py",
     "real-time sowing coverage & year-on-year pace", False),
    ("20", "validation/20_validate_sowing_official.py",
     "check sowing against the DA&FW release", False),
    ("15", "crops/15_crop_stress.py",
     "FAO-33 crop stress by phenological stage", True),
    ("05", "dashboard/05_forecast_export.py",
     "forecast latest date + export dashboard data", False),
    ("06", "dashboard/06_build_dashboard.py",
     "build standalone HTML + React project", False),
    ("13", "dashboard/13_export_masters.py",
     "master CSV/XLSX exports", True),
]
# Not in the default run:
#   04_build_samples_train.py   superseded by 08
#   09_calibration_test.py      diagnostic: calibration vs sharpness
#   10_model_v2_spatial.py      spatial/stacked experiment (did not beat 08)
#   11_train_final.py           called by 08
#   12_validate_district_vs_imd.py, 16_model_families_verify.py   diagnostics


# Fetching new IMD days lives outside the numbered steps: it reaches the
# network, it is the only stage that can fail for reasons nothing in the repo
# controls, and most runs do not need it.  Kept behind an explicit flag.
IMD_STEPS = [
    ("IMD_Data/download_imd.py", "fetch IMD days published since the last run"),
    ("IMD_Data/build_imd_lgd_csvs.py", "aggregate IMD grids to the 791 units"),
]


def run(path, label):
    t = time.time()
    r = subprocess.run(["py", "-3.13", "-X", "utf8", str(path)])
    if r.returncode != 0:
        print(f"\n!!! FAILED at {label} (exit {r.returncode}). Stopping.")
        return False
    print(f"    ok ({time.time() - t:.0f}s)")
    return True


def main():
    argv = sys.argv[1:]
    if "--list" in argv:
        for n, rel, desc, slow in STEPS:
            print(f"  {n}  {rel:<46} {desc}{'   [SLOW]' if slow else ''}")
        return
    fast = "--fast" in argv
    start = None
    if "--from" in argv:
        start = argv[argv.index("--from") + 1]

    steps = STEPS
    if start:
        idx = [i for i, s in enumerate(steps) if s[0] == start]
        if not idx:
            sys.exit(f"--from {start}: no such step (try --list)")
        steps = steps[idx[0]:]
    if fast:
        steps = [s for s in steps if s[0] != "08"]

    print("=" * 70)
    print("MonsoonCast pipeline — clean -> features -> crops -> dashboard")
    print(f"  {len(steps)} steps" + ("  (--fast: model retraining skipped)"
                                     if fast else ""))
    print("=" * 70)
    t0 = time.time()
    if "--update-imd" in argv:
        for rel, desc in IMD_STEPS:
            print(f"\n>>> [IMD] {rel}")
            print(f"    {desc}")
            if not run(V5.parent / rel, rel):
                sys.exit(1)
    for n, rel, desc, slow in steps:
        print(f"\n>>> [{n}] {rel}")
        print(f"    {desc}")
        if not run(HERE / rel, rel):
            print(f"    resume with:  --from {n}")
            sys.exit(1)
    print(f"\nALL DONE in {(time.time() - t0) / 60:.1f} min.")
    print(f"  standalone : {V5 / 'dashboard_lgd' / 'MonsoonCast_LGD.html'}")
    print(f"  React app  : cd {V5 / 'dashboard_react'} && npm run dev")
    print(f"  masters    : {V5 / 'masters_lgd'}")


if __name__ == "__main__":
    main()
