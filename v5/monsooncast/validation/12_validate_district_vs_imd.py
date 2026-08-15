r"""
v5/monsooncast/validation/12_validate_district_vs_imd.py  —  validate our features
DISTRICT BY DISTRICT against IMD's own published district bulletins.

WHY THIS IS THE TEST THAT MATTERS
  Step 07 compared the ALL-INDIA aggregate and looked good (correlation 0.992
  with IMD's published series).  That is not sufficient: a national mean can be
  right while individual districts are badly wrong, because errors of opposite
  sign cancel in the average.  The product is used district by district, so it
  has to be validated district by district.

GROUND TRUTH (uploaded IMD releases, district level)
  stage_pyexcel_factrates_rainfall_district_export.csv
        June 2026 cumulative: actual, normal and departure per district
  DISTRICT_RAINFALL_DISTRIBUTION_COUNTRY_INDIA_cd (45).pdf
        01 June - 23 July 2026, full country, same three fields

  Each gives IMD's own ACTUAL, its own NORMAL and its own DEPARTURE, so the
  three can be checked separately -- which localises any disagreement to the
  numerator, the denominator, or the arithmetic.

THE DISTINCTION THIS IS DESIGNED TO EXPOSE
  IMD publishes two different rainfall products and they are not the same
  number:
    - the 0.25 deg GRIDDED analysis (what this system is built from), an
      interpolated field;
    - the operational DISTRICT BULLETIN, an average of the rain-gauge stations
      that actually lie inside each district.
  Interpolation smooths: a gridded district value borrows from its neighbours,
  so it should be biased toward the regional mean and away from local extremes.
  If that is what is happening, the signature is specific -- a regression slope
  of our departure on IMD's below 1, not a random scatter -- and it is a
  property of the source, not a bug.

OUTPUT -> v5/data_lgd/district_vs_imd_lgd.csv  (per district, per window)

Run:  py -3.13 -X utf8 "v5/monsooncast/validation/12_validate_district_vs_imd.py"
"""
import pathlib
import re
import sys

import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
V5 = HERE.parents[1]
ROOT = V5.parent
IMD = ROOT / "IMD_Data"
DATA = V5 / "data_lgd"
DL = pathlib.Path(r"C:\Users\shash\Downloads")
sys.path.insert(0, str(V5))
sys.path.insert(0, str(IMD))
from common_v5 import log  # noqa
from build_crosswalk import norm_state, norm_key  # noqa

CENT = {"actual": 565, "normal": 644, "dep": 726}   # cd45 PERIOD columns


def matcher(reg):
    reg = reg.copy()
    reg["ks"], reg["kd"] = norm_state(reg["state"]), norm_key(reg["district"])
    exact = {(r.ks, r.kd): r.district_id for r in reg.itertuples()}
    by_state = {k: g for k, g in reg.groupby("ks")}
    cnt = reg["kd"].value_counts()
    uniq = dict(zip(reg[reg["kd"].isin(cnt[cnt == 1].index)]["kd"],
                    reg[reg["kd"].isin(cnt[cnt == 1].index)]["district_id"]))

    def m(kd, ks=None):
        if ks and (ks, kd) in exact:
            return exact[(ks, kd)]
        cand = by_state.get(ks) if ks else None
        if cand is not None:
            tk = set(kd.split())
            best, sc = None, 0.0
            for r in cand.itertuples():
                rt = set(r.kd.split())
                ov = len(tk & rt) / max(min(len(tk), len(rt)), 1)
                if ov > sc:
                    best, sc = r.district_id, ov
            if best is not None and sc >= 0.5:
                return best
        return uniq.get(kd, np.nan)
    return m


def parse_cd45():
    import fitz
    doc = fitz.open(str(DL / "DISTRICT_RAINFALL_DISTRIBUTION_COUNTRY_INDIA_cd (45).pdf"))
    rows, cur = [], None
    for pg in doc:
        buck = {}
        for w in pg.get_text("words"):
            buck.setdefault(round(w[1] / 3) * 3, []).append(w)
        for y in sorted(buck):
            ws = sorted(buck[y], key=lambda w: w[0])
            sno = any(w[0] < 50 and re.fullmatch(r"\d+", w[4]) for w in ws)
            name = " ".join(w[4] for w in ws if 55 < w[0] < 300).strip()
            if not name or re.search(r"ACTUAL|NORMAL|DEP|SUBDIV|DAY|PERIOD|DISTRICT RAIN", name):
                continue

            def col(cx):
                best, bd = None, 40
                for w in ws:
                    c = (w[0] + w[2]) / 2
                    if abs(c - cx) < bd and re.search(r"[-\d]", w[4]):
                        best, bd = w[4], abs(c - cx)
                try:
                    return float(best)
                except (TypeError, ValueError):
                    return np.nan
            a, n, dep = (col(CENT["actual"]), col(CENT["normal"]),
                         col(CENT["dep"]))
            if sno:
                rows.append((cur, name, a, n, dep))
            elif re.match(r"^[A-Z][A-Z &().\-]+$", name):
                cur = name
    return pd.DataFrame(rows, columns=["imd_state", "district", "imd_actual",
                                       "imd_normal", "imd_dep"])


def ours(rain, lo, hi, nrm_lo=1971, nrm_hi=2020):
    """Our actual + normal over a date window, per district."""
    sel = rain[(rain.index >= lo) & (rain.index <= hi)]
    act = sel.sum(min_count=1)
    doy = sel.index.dayofyear
    base = rain[(rain.index.year >= nrm_lo) & (rain.index.year <= nrm_hi)]
    per = base.groupby(base.index.dayofyear).mean().reindex(range(1, 367))
    arr = per.to_numpy()
    sm = np.nanmean(np.stack([np.roll(arr, k, axis=0) for k in range(-2, 3)]),
                    axis=0)
    per = pd.DataFrame(sm, index=range(1, 367), columns=rain.columns)
    nrm = per.reindex(doy).sum()
    return pd.DataFrame({"our_actual": act, "our_normal": nrm})


def report(df, label, out_rows):
    d = df.dropna(subset=["our_actual", "our_normal", "imd_actual",
                          "imd_normal", "imd_dep"])
    d = d[(d["imd_normal"] > 10) & (d["our_normal"] > 10)]
    d = d.assign(our_dep=100 * (d["our_actual"] - d["our_normal"])
                 / d["our_normal"])
    log(f"\n  {label}   ({len(d)} districts matched)")
    log("  " + "-" * 70)
    for fld, o, i in (("ACTUAL (mm)", "our_actual", "imd_actual"),
                      ("NORMAL (mm)", "our_normal", "imd_normal"),
                      ("DEPARTURE (%)", "our_dep", "imd_dep")):
        x, y = d[o].to_numpy(), d[i].to_numpy()
        r = float(np.corrcoef(x, y)[0, 1])
        bias = float(np.mean(x - y))
        mae = float(np.mean(np.abs(x - y)))
        rel = 100 * float(np.mean(x - y) / np.mean(y)) if fld != "DEPARTURE (%)" else np.nan
        slope = float(np.polyfit(y, x, 1)[0])
        log(f"    {fld:14s} r {r:.3f}   bias {bias:+8.2f}   MAE {mae:7.2f}"
            + (f"   ({rel:+.1f}% of IMD mean)" if np.isfinite(rel) else "")
            + f"   slope {slope:.3f}")
        out_rows.append({"window": label, "field": fld, "n": len(d),
                         "corr": round(r, 4), "bias": round(bias, 3),
                         "mae": round(mae, 3), "slope_ours_on_imd":
                         round(slope, 4)})
    # how often do the two land in the same IMD category?
    B = [-60, -20, 20, 60]
    same = np.mean(np.digitize(d["our_dep"], B) == np.digitize(d["imd_dep"], B))
    log(f"    same IMD category: {100*same:.1f}% of districts")
    out_rows.append({"window": label, "field": "IMD category agreement",
                     "n": len(d), "corr": round(float(same), 4), "bias": np.nan,
                     "mae": np.nan, "slope_ours_on_imd": np.nan})
    return d


def main():
    log("=" * 74)
    log("DISTRICT-LEVEL VALIDATION vs IMD's own district bulletins")
    log("=" * 74)
    reg = pd.read_csv(IMD / "registry_lgd791.csv")
    mt = matcher(reg)
    rain = pd.read_pickle(DATA / "daily_rain_lgd.pkl")
    rain.columns = rain.columns.astype(int)
    out_rows, keep = [], []

    # ---- June 2026, from the district CSV --------------------------------
    c = pd.read_csv(DL / "stage_pyexcel_factrates_rainfall_district_export.csv")
    c = c.rename(columns={"DistrictName": "district",
                          "ActualCumulative": "imd_actual",
                          "NormalCumulative": "imd_normal",
                          "DepreciationCumulative": "imd_dep"})
    c["imd_dep"] = c["imd_dep"] * 100.0
    c["district_id"] = [mt(k) for k in norm_key(c["district"])]
    c = c.dropna(subset=["district_id"])
    c["district_id"] = c["district_id"].astype(int)
    o = ours(rain, "2026-06-01", "2026-06-30")
    j = c.set_index("district_id").join(o, how="inner")
    keep.append(report(j, "June 2026", out_rows).assign(window="June 2026"))

    # ---- 01 Jun - 23 Jul 2026, from the cd45 PDF -------------------------
    try:
        p = parse_cd45()
        p["ks"] = norm_state(p["imd_state"])
        p["kd"] = norm_key(p["district"])
        p["district_id"] = [mt(k, s) for k, s in zip(p["kd"], p["ks"])]
        p = p.dropna(subset=["district_id"])
        p["district_id"] = p["district_id"].astype(int)
        p = p.drop_duplicates("district_id")
        o2 = ours(rain, "2026-06-01", "2026-07-23")
        j2 = p.set_index("district_id").join(o2, how="inner")
        keep.append(report(j2, "01 Jun - 23 Jul 2026",
                           out_rows).assign(window="01Jun-23Jul 2026"))
    except Exception as e:
        log(f"  cd45 PDF: {type(e).__name__} {str(e)[:80]}")

    pd.DataFrame(out_rows).to_csv(DATA / "district_vs_imd_summary_lgd.csv",
                                  index=False)
    pd.concat(keep).reset_index().to_csv(DATA / "district_vs_imd_lgd.csv",
                                         index=False)
    log(f"\n  wrote district_vs_imd_lgd.csv + _summary_lgd.csv")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
