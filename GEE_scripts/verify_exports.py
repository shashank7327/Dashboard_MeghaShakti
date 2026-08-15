r"""GEE_scripts/verify_exports.py  —  check a product folder before the
pipeline reads it.

WHY THIS EXISTS
  v5/monsooncast/cleaning/01_clean_merge_panel.py globs *.csv in each
  product folder, aggregates every file to monthly, and then COMBINES the
  files.  For an extensive quantity that combination is a SUM:

      out.groupby(["date", "district_id"])[valcols].sum()   if how == "sum"

  So two files that both cover July 2026 make July's evapotranspiration
  DOUBLE.  Nothing raises.  The month simply comes out twice as thirsty,
  PET doubles, the SPEI water balance (rain - PET) goes sharply negative,
  and the dashboard shows a drought that is not there.

  That is the exact shape of failure this project keeps meeting: a
  plausible number rather than an error.  The guard is cheap, so it runs
  before the pipeline rather than after somebody notices the map is wrong.

WHAT IT CHECKS, PER FOLDER
  1. OVERLAP     no two files may cover the same (year, month).  Fatal.
  2. GAPS        months missing inside the covered span.  Warning.
  3. COVERAGE    districts present per file against the 791-unit registry,
                 and the name-match rate against it.  Warning.
  4. FRESHNESS   the last date each product carries, and how far behind
                 the IMD rainfall analysis it is.  Informational, because
                 a lag is expected — ERA5-Land runs about a week behind
                 and CHIRPS two to three weeks.
  5. SCHEMA      the expected columns are present.  Fatal if not.

Run:  py -3.13 -X utf8 "GEE_scripts/verify_exports.py"
      py -3.13 -X utf8 "GEE_scripts/verify_exports.py" Evapotranspiration_LGD
"""
import pathlib
import sys

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
IMD = ROOT / "IMD_Data"

#   folder -> (required columns, whether the pipeline SUMS it across files)
#   The `sums` flag is what decides whether an overlap is fatal or merely
#   wrong: summed quantities double, averaged ones get silently reweighted.
PRODUCTS = {
    "Evapotranspiration_LGD": (["date", "state", "district",
                                "aet_mm", "pet_mm"], True),
    "SM_LGD": (["date", "state", "district",
                "swvl1", "swvl2", "swvl3", "swvl4"], False),
    "TemperatureERA5_LGD": (["date", "state", "district",
                             "t2m_max_c", "t2m_min_c", "t2m_mean_c"], False),
    "Chirps_LGD": (["date", "state", "district", "chirps_mm"], True),
    "CFSv2_LGD": (["date", "state", "district", "cfs_precip_mm"], True),
    "Vegetation_LGD": (["date", "state", "district", "ndvi", "evi"], False),
}


def norm(s):
    """Same normalisation the pipeline's registry join uses."""
    return (s.astype(str).str.lower().str.replace(r"[^a-z0-9]", "",
                                                  regex=True))


def imd_last():
    p = IMD / "_district_daily_rain_lgd.pkl"
    if not p.exists():
        return None
    return pd.read_pickle(p).index.max()


def check(folder, cols, sums, reg, ref):
    d = ROOT / folder
    print(f"\n{'=' * 72}\n{folder}\n{'=' * 72}")
    if not d.exists():
        print("  folder does not exist — skipped")
        return 0
    files = sorted(d.glob("*.csv"))
    if not files:
        print("  no CSV files — skipped")
        return 0

    fatal, months_seen, last_date = 0, {}, None
    for f in files:
        try:
            df = pd.read_csv(f, parse_dates=["date"])
        except Exception as e:
            print(f"  ! {f.name}: unreadable — {type(e).__name__}: {e}")
            fatal += 1
            continue

        missing = [c for c in cols if c not in df.columns]
        if missing:
            print(f"  ! {f.name}: MISSING COLUMNS {missing}")
            print(f"      has: {list(df.columns)}")
            fatal += 1
            continue

        ym = df["date"].dt.to_period("M")
        span = f"{ym.min()} .. {ym.max()}"
        last_date = max(last_date, df["date"].max()) if last_date \
            else df["date"].max()

        clash = sorted({str(m) for m in ym.unique()} & set(months_seen))
        if clash:
            others = sorted({months_seen[c] for c in clash})
            print(f"  ! {f.name}: OVERLAPS {', '.join(others)} on "
                  f"{len(clash)} month(s): {', '.join(clash[:6])}"
                  + (" ..." if len(clash) > 6 else ""))
            print(f"      -> the pipeline "
                  + ("SUMS this product across files, so those months will "
                     "be DOUBLE-COUNTED." if sums else
                     "averages this product across files, so those months "
                     "are silently reweighted."))
            print(f"      -> delete the older file; UPDATE mode is meant to "
                  f"REPLACE the year file, not sit beside it.")
            fatal += 1
        for m in ym.unique():
            months_seen[str(m)] = f.name

        #   Count (state, district) PAIRS, not district names.  786 district
        #   names cover all 791 units: Aurangabad exists in both Bihar and
        #   Maharashtra, and four other names repeat the same way.  Counting
        #   names reports a shortfall that is not there.
        matched = norm(df["state"]) + "|" + norm(df["district"])
        units = matched.nunique()
        rate = matched.isin(ref).mean() * 100 if ref is not None else float("nan")
        short = f" (-{len(reg) - units})" if reg is not None \
            and units < len(reg) else ""
        print(f"  {f.name:<48} {span}  rows {len(df):>8,}  "
              f"units {units:>4}{short:<5}  registry match {rate:5.1f}%")
        if reg is not None and units < len(reg):
            gone = sorted(set(ref) - set(matched.unique()))[:5]
            print(f"      ! {len(reg) - units} unit(s) absent from this file, "
                  f"e.g. {gone}")
        if ref is not None and rate < 98:
            bad = sorted(set(zip(df.loc[~matched.isin(ref), "state"],
                                 df.loc[~matched.isin(ref), "district"])))[:5]
            print(f"      ! only {rate:.1f}% of rows match the 791-unit "
                  f"registry. Unmatched e.g.: {bad}")

    # gaps inside the covered span
    if months_seen:
        ms = sorted(pd.Period(m) for m in months_seen)
        full = pd.period_range(ms[0], ms[-1], freq="M")
        gaps = [str(p) for p in full if str(p) not in months_seen]
        if gaps:
            print(f"  ! {len(gaps)} month(s) missing inside the span: "
                  f"{', '.join(gaps[:8])}" + (" ..." if len(gaps) > 8 else ""))

    if last_date is not None:
        msg = f"  last date: {last_date:%Y-%m-%d}"
        if ref is not None and imd_ref is not None:
            lag = (imd_ref - last_date).days
            msg += f"   ({lag} days behind IMD rainfall, {imd_ref:%Y-%m-%d})"
        print(msg)
    return fatal


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    reg = None
    ref = None
    rp = IMD / "registry_lgd791.csv"
    if rp.exists():
        reg = pd.read_csv(rp)
        ref = set(norm(reg["state"]) + "|" + norm(reg["district"]))
        print(f"registry: {len(reg)} LGD units")
    else:
        print("! registry_lgd791.csv not found — skipping the coverage check")

    imd_ref = imd_last()
    if imd_ref is not None:
        print(f"IMD rainfall analysis runs to {imd_ref:%Y-%m-%d}")

    want = sys.argv[1:] or list(PRODUCTS)
    total = 0
    for folder in want:
        if folder not in PRODUCTS:
            print(f"\n! unknown product folder: {folder}")
            print(f"  known: {', '.join(PRODUCTS)}")
            continue
        cols, sums = PRODUCTS[folder]
        total += check(folder, cols, sums, reg, ref)

    print(f"\n{'=' * 72}")
    if total:
        print(f"{total} FATAL problem(s). Fix these before running the "
              f"pipeline — it will not notice them itself.")
        sys.exit(1)
    print("no overlaps, no schema problems. Safe to run the pipeline.")
