r"""
IMD_Data/validate_imd.py  —  check the finalised IMD district product against
IMD's own published all-India figures, against CHIRPS, and against itself.

WHAT THIS USED TO DO, AND WHY IT STOPPED DOING IT
  It read IMD_Data/{var}/IMD_{var}_district_daily_YYYY.csv — the 701-unit
  layout written by finalize_imd.py.  That layout was retired when the product
  moved to the 791 LGD units (build_imd_lgd_csvs.py, output under lgd/), and
  those directories no longer exist.  load_year() therefore returned None for
  every year: check 1 skipped silently and checks 2-3 raised on a None.  The
  README's validation numbers were quoting a product that was no longer on
  disk.  This now reads lgd/.

  It also area-weighted with v5/data/district_area_v5.csv, which is keyed by
  the 701-unit district_id.  The LGD product's ids run 0..790 and mean
  different districts, so that join silently mismatched every weight; areas
  now come from registry_lgd791.csv, the registry the product is built on.

FOUR CHECKS
  1. NATIONAL.  Area-weighted all-India JJAS totals vs IMD's published numbers.
     If the crosswalk and the normals are right, these land on the published
     values.
  2. AGAINST CHIRPS.  Same districts, same month, to quantify the wet bias that
     motivated building this product.
  3. TEMPERATURE sanity, and the archive-vs-real-time mask difference that
     makes 2026 temperature inhomogeneous for a handful of units.
  4. FILE INTEGRITY + THE BLANK TAXONOMY.  Every year file: schema, row count
     against days x 791, duplicate keys, calendar gaps.  Then WHY cells are
     empty, which is the question this product gets asked most often —
     separating 'no grid cell here' from 'the dry season'.

Run:  py -3.13 -X utf8 "IMD_Data/validate_imd.py"
      (after build_imd_lgd_csvs.py)
"""
import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "v5"))
from common_v5 import CHIRPS_DIR, log  # noqa

HERE = pathlib.Path(__file__).resolve().parent
NUNITS = 791
VALCOL = {"rain": "rain_mm", "tmax": "tmax_c", "tmin": "tmin_c"}
NORMCOL = {"rain": "normal_mm", "tmax": "normal_c", "tmin": "normal_c"}

# IMD published references
PUB = {
    ("rain", 2023, "JJAS"): 816.0,
    ("rain", 2024, "JJAS"): 937.0,
}


def key(s):
    s = pd.Series(s).astype(str).str.upper().str.replace(r"[^A-Z ]", " ",
                                                         regex=True)
    return s.str.replace(r"\s+", " ", regex=True).str.strip()


def registry():
    return pd.read_csv(HERE / "registry_lgd791.csv")


def areas():
    return registry().set_index("district_id")["area_km2"]


def aw(df, val, w):
    d = df.dropna(subset=[val])
    ww = w.reindex(d["district_id"]).to_numpy()
    ok = np.isfinite(ww)
    return float((d[val].to_numpy()[ok] * ww[ok]).sum() / ww[ok].sum())


def year_path(var, y):
    return HERE / "lgd" / var / f"IMD_{var}_district_daily_{y}.csv"


def load_year(var, y):
    f = year_path(var, y)
    if not f.exists():
        return None
    return pd.read_csv(f, parse_dates=["date"])


def check_national(A):
    log("\n1. NATIONAL, area-weighted, vs IMD published")
    for y in (2023, 2024):
        d = load_year("rain", y)
        if d is None:
            log(f"   {y}: file missing — skipped")
            continue
        jjas = (d[d["date"].dt.month.isin([6, 7, 8, 9])]
                .groupby(["district_id"])["rain_mm"].sum().reset_index())
        val = aw(jjas, "rain_mm", A)
        pub = PUB[("rain", y, "JJAS")]
        log(f"   {y} JJAS  IMD-grid {val:6.1f} mm   published ~{pub:.0f}  "
            f"({100*(val/pub-1):+.1f}%)")

    d26 = load_year("rain", 2026)
    if d26 is not None:
        for label, mth in (("June", d26["date"].dt.month == 6),
                           ("1 Jun-6 Jul",
                            (d26["date"].dt.month == 6)
                            | ((d26["date"].dt.month == 7)
                               & (d26["date"].dt.day <= 6)))):
            g = d26[mth].groupby("district_id")["rain_mm"].sum().reset_index()
            log(f"   2026 {label:12s} IMD-grid {aw(g,'rain_mm',A):6.1f} mm")
        log("   (IMD published: June 2026 = 99.5 mm; 1 Jun-6 Jul = 170.7 mm; "
            "real-time grid runs preliminary/high)")
        log(f"   2026 observations run to {d26['date'].max():%Y-%m-%d}")


def check_chirps():
    log("\n2. IMD vs CHIRPS, June 2023, same districts")
    d = load_year("rain", 2023)
    ch_f = CHIRPS_DIR / "CHIRPS_district_daily_2023.csv"
    if d is None or not ch_f.exists():
        log("   IMD or CHIRPS 2023 file missing — skipped")
        return
    ij = d[d["date"].dt.month == 6].groupby("district_id")["rain_mm"].sum()
    ch = pd.read_csv(ch_f, parse_dates=["date"])
    #   Name-match CHIRPS onto the 791 LGD registry, NOT the 701 v5 registry:
    #   `ij` is indexed by LGD district_id, so a 701-keyed join would compare
    #   two different districts on every row.
    reg = registry()
    reg["ks"], reg["kd"] = key(reg["state"]), key(reg["district"])
    ch["ks"], ch["kd"] = key(ch["state"]), key(ch["district"])
    ch = ch.merge(reg[["ks", "kd", "district_id"]], on=["ks", "kd"],
                  how="inner")
    ch["rain_mm"] = pd.to_numeric(ch["rain_mm"], errors="coerce")
    cj = ch[ch["date"].dt.month == 6].groupby("district_id")["rain_mm"].sum()
    m = pd.concat([ij.rename("imd"), cj.rename("chirps")], axis=1).dropna()
    m = m[m["chirps"] > 0]
    if not len(m):
        log("   no districts matched by name — skipped")
        return
    r = m["chirps"].sum() / m["imd"].sum()
    log(f"   districts {len(m)}   CHIRPS/IMD = {r:.3f}  "
        f"(CHIRPS {100*(r-1):+.1f}% wet)   corr {m['imd'].corr(m['chirps']):.3f}")


def check_temperature():
    log("\n3. temperature sanity, June 2023")
    for var in ("tmax", "tmin"):
        d = load_year(var, 2023)
        if d is None:
            log(f"   {var}: file missing — skipped")
            continue
        vc = VALCOL[var]
        jun = d[d["date"].dt.month == 6].groupby("district_id")[vc].mean()
        log(f"   {var}: June land-district mean {jun.mean():.1f} C "
            f"(range {jun.min():.1f}..{jun.max():.1f})")

    #   IMD's real-time 0.5 deg temperature grid interpolates over slightly more
    #   sea than the 1 deg archive does, so a few units get 2026 values with no
    #   history behind them.  Their normal is NaN, so anom_c and pct_departure
    #   stay blank -- but the ACTUAL is not blank, and a reader comparing 2026
    #   to earlier years for these units is comparing to nothing.
    d26 = load_year("tmax", 2026)
    if d26 is None:
        return
    orphan = d26.dropna(subset=["tmax_c"])
    orphan = orphan[orphan["normal_c"].isna()]
    if len(orphan):
        u = orphan.groupby(["state", "district"]).size()
        log(f"   2026 tmax: {len(u)} unit(s) have values but NO normal "
            f"({len(orphan)} rows) — real-time mask is wider than the archive:")
        for (st, dt), n in u.items():
            log(f"      {st[:28]:28s} {dt[:24]:24s} {n} days")


def check_files_and_blanks():
    log("\n4. FILE INTEGRITY and the blank taxonomy")
    for var in ("rain", "tmax", "tmin"):
        files = sorted((HERE / "lgd" / var).glob(
            f"IMD_{var}_district_daily_*.csv"))
        if not files:
            log(f"   {var}: no files found — skipped")
            continue
        vc, nc = VALCOL[var], NORMCOL[var]
        bad, tot_rows, blank_val, blank_pct, blank_cov0 = [], 0, 0, 0, 0
        zero_val, mo = 0, []
        dmin, dmax = None, None
        for f in files:
            df = pd.read_csv(f, parse_dates=["date"])
            y = int(f.stem[-4:])
            nd = df["date"].nunique()
            gaps = len(pd.date_range(df["date"].min(), df["date"].max(),
                                     freq="D")) - nd
            dup = int(df.duplicated(["date", "district_id"]).sum())
            if len(df) != nd * NUNITS or dup or gaps:
                bad.append((y, len(df), nd * NUNITS, dup, gaps))
            tot_rows += len(df)
            blank_val += int(df[vc].isna().sum())
            blank_pct += int(df["pct_departure"].isna().sum())
            if "coverage" in df.columns:
                blank_cov0 += int((df["coverage"] == 0).sum())
            if var == "rain":
                zero_val += int((df[vc] == 0).sum())
                mo.append(df.assign(m=df["date"].dt.month).groupby("m").agg(
                    n=("district_id", "size"),
                    nullpct=("pct_departure", lambda s: s.isna().sum()),
                    zero=(vc, lambda s: (s == 0).sum())))
            dmin = df["date"].min() if dmin is None else min(dmin,
                                                             df["date"].min())
            dmax = df["date"].max() if dmax is None else max(dmax,
                                                             df["date"].max())
        log(f"\n   {var}: {len(files)} files, {tot_rows:,} rows, "
            f"{dmin:%Y-%m-%d}..{dmax:%Y-%m-%d}")
        if bad:
            log(f"      {len(bad)} file(s) with wrong row count / dups / gaps:")
            for y, n, e, dup, g in bad:
                log(f"        {y}: rows {n:,} expected {e:,} "
                    f"dups {dup} gaps {g}")
        else:
            log("      row counts, keys and calendars all complete")
        log(f"      blank {vc:9s} {blank_val:>10,} "
            f"({blank_val/tot_rows*100:5.2f}%)"
            + (f"  — all with coverage == 0" if blank_cov0 >= blank_val
               else ""))
        log(f"      coverage == 0   {blank_cov0:>10,} "
            f"({blank_cov0/tot_rows*100:5.2f}%)  no valid grid cell that day")
        log(f"      blank pct_dep   {blank_pct:>10,} "
            f"({blank_pct/tot_rows*100:5.2f}%)  normal below the "
            f"{'1 mm' if var == 'rain' else '5 C'} floor")
        if var == "rain":
            log(f"      exact zeros     {zero_val:>10,} "
                f"({zero_val/tot_rows*100:5.2f}%)  measured dry days")
            m = pd.concat(mo).groupby(level=0).sum()
            m["pct_null_dep"] = m["nullpct"] / m["n"] * 100
            m["pct_zero"] = m["zero"] / m["n"] * 100
            log("      by calendar month — the blanks ARE the monsoon:")
            log("        month  blank pct_dep %   exact-zero %")
            for mm, r in m.iterrows():
                log(f"        {mm:5d}  {r['pct_null_dep']:14.1f}  "
                    f"{r['pct_zero']:14.1f}")

    # units that never receive a value, per variable
    log("\n   units that are blank throughout (source has no cell for them):")
    reg = registry().set_index("district_id")
    for var in ("rain", "tmax", "tmin"):
        p = HERE / f"_district_daily_{var}_lgd.pkl"
        if not p.exists():
            continue
        w = pd.read_pickle(p)
        w.columns = w.columns.astype(int)
        dead = [c for c in w.columns if w[c].isna().all()]
        log(f"      {var}: {len(dead)}")
        for c in dead:
            log(f"         {reg.at[c,'state'][:28]:28s} "
                f"{reg.at[c,'district'][:24]:24s} "
                f"({reg.at[c,'area_km2']:,.0f} km2)")

    # thin units: a value, but drawn from a fraction of the district
    log("\n   thinnest units by mean coverage (value drawn from part of the "
        "district):")
    for var in ("rain", "tmax", "tmin"):
        p = HERE / f"_district_daily_{var}_cov_lgd.pkl"
        if not p.exists():
            log(f"      {var}: no coverage pickle — run build_imd_lgd_csvs.py")
            continue
        cv = pd.read_pickle(p)
        cv.columns = cv.columns.astype(int)
        mc = cv.mean(axis=0)
        thin = mc[(mc > 0) & (mc < 0.5)].sort_values()
        log(f"      {var}: {len(thin)} unit(s) below 50% coverage")
        for did, v in thin.head(8).items():
            log(f"         {reg.at[did,'state'][:26]:26s} "
                f"{reg.at[did,'district'][:22]:22s} {v*100:5.1f}%")


def main():
    log("=" * 70)
    log("IMD product validation — 791 LGD units")
    log("=" * 70)
    A = areas()
    check_national(A)
    check_chirps()
    check_temperature()
    check_files_and_blanks()
    log("\n  validation complete")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
