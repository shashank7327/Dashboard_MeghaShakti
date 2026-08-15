r"""v5/monsooncast/run_post_training.py  —  everything that must happen after
a retrain, in the order it has to happen, with a stop on the first failure.

WHY A RUNNER AND NOT A LIST IN A README
  These steps have real dependencies and one of them is easy to get wrong: the
  dashboard export must run AFTER the models are persisted, or it serves the
  previous run's models while reporting the new run's skill scores. That
  mismatch is invisible in the output -- the map renders, the numbers look
  plausible, and the forecast on screen came from a model the scores do not
  describe.

  Ordering it in a file also means the audit cannot be skipped when it is
  inconvenient, which is the moment it matters most.

ORDER
  28  audit the training data      is the sample matrix real, or plausible junk?
  07  validate features vs IMD     29 tests against IMD's published %LPA
  12  validate districts vs IMD    district by district, where a national
                                   mean can hide cancelling errors
  05  export dashboard data        on the NEWLY persisted models
  06  build the dashboard          standalone + React from one component
  13  export masters               CSV/XLSX per variable

Run:  py -3.13 -X utf8 "v5/monsooncast/run_post_training.py"
      py -3.13 -X utf8 "v5/monsooncast/run_post_training.py" --skip 12,13
"""
import pathlib
import subprocess
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent

STEPS = [
    ("28", "validation/28_audit_training_data.py",
     "audit the sample matrix the models were fitted on", True),
    ("07", "validation/07_validate_features.py",
     "features vs IMD published all-India figures", True),
    ("12", "validation/12_validate_district_vs_imd.py",
     "district-by-district vs IMD bulletins", False),
    ("05", "dashboard/05_forecast_export.py",
     "forecast + dashboard data on the new models", True),
    ("06", "dashboard/06_build_dashboard.py",
     "build standalone HTML + React project", True),
    ("13", "dashboard/13_export_masters.py",
     "master CSV/XLSX exports", False),
]


def main():
    argv = sys.argv[1:]
    skip = set()
    if "--skip" in argv:
        skip = {s.strip() for s in argv[argv.index("--skip") + 1].split(",")}

    print("=" * 72)
    print("POST-TRAINING — audit, validate, export, rebuild")
    print("=" * 72)
    t0 = time.time()
    results = []
    for n, rel, desc, critical in STEPS:
        if n in skip:
            print(f"\n>>> [{n}] skipped by request")
            continue
        print(f"\n>>> [{n}] {rel}")
        print(f"    {desc}")
        t = time.time()
        r = subprocess.run(["py", "-3.13", "-X", "utf8", str(HERE / rel)])
        dt = time.time() - t
        ok = r.returncode == 0
        results.append((n, rel, ok, dt))
        if ok:
            print(f"    ok ({dt:.0f}s)")
            continue
        #   A non-critical step failing is reported and stepped over: the
        #   district validation needs IMD bulletin files that are not always
        #   on disk, and the master export fails if a workbook is open in
        #   Excel. Neither should stop a dashboard rebuild.
        if critical:
            print(f"\n!!! FAILED at {n} (exit {r.returncode}). Stopping — "
                  f"later steps depend on this one.")
            sys.exit(1)
        print(f"    ! {n} failed (exit {r.returncode}) — non-critical, "
              f"continuing")

    print("\n" + "=" * 72)
    for n, rel, ok, dt in results:
        print(f"  {'ok ' if ok else 'FAIL'}  {n}  {rel:<44} {dt:6.0f}s")
    print(f"\nALL DONE in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
