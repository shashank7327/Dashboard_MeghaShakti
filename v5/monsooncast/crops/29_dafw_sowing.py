r"""v5/monsooncast/crops/29_dafw_sowing.py  —  parse the DA&FW all-India
cropwise progressive area-sown reports.

WHAT THESE FILES ARE, AND WHY THEY MATTER MORE THAN THEY LOOK
  Ministry of Agriculture & Farmers' Welfare (DA&FW), "Kharif All Crops: All
  India Cropwise Progressive Area Sown", one XLSX per reporting week. They
  carry four columns per crop: a NORMAL, this season's area, last season's
  area, and the difference.

  That NORMAL is the important part. Every sowing comparison this project
  publishes is against the PRIOR SEASON, not against a normal, and the reason
  is documented at length in 19_sowing_dynamics.py: UPAg's own NormalValue,
  PreviousValue and PreviousYearChange columns exist in the files and are
  100% EMPTY in every row. With only two prior seasons in the archive -- one
  of them thin -- calling their mean a climatological normal would have been
  dishonest, so the column names say "prior season" throughout.

  DA&FW publishes an actual normal. So for the national figures these files
  support a genuine coverage-against-normal, which the UPAg-derived product
  could not.

WHAT THEY ARE NOT
  National totals only, by crop. They cannot extend the district or state
  sowing layers, which need UPAg's state-level detail. They replace nothing;
  they add a national benchmark that is both fresher and better-founded.

  They also supersede the hard-coded benchmark in
  validation/20_validate_sowing_official.py, which pinned one week (17 Jul
  2026) into the source. A number typed into a script goes stale silently;
  these files are read at build time.

OUTPUT -> v5/data_lgd/
  dafw_sowing_national.csv   crop, normal_lakh_ha, sown_lakh_ha,
                             sown_ly_lakh_ha, diff_lakh_ha, pct_change,
                             pct_of_normal, as_of, source_file
  dafw_sowing_meta.json      as_of date of the newest report, crops covered

Run:  py -3.13 -X utf8 "v5/monsooncast/crops/29_dafw_sowing.py"
"""
import json
import pathlib
import re
import sys

import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
V5 = HERE.parents[1]
ROOT = V5.parent
UPAG = ROOT / "UPAJ"
OUTD = V5 / "data_lgd"
sys.path.insert(0, str(V5))
from common_v5 import log  # noqa

#   Rows that are group subtotals rather than crops. Kept in the output but
#   flagged, because summing them together with their members double-counts --
#   the same defect the UPAg loader exists to handle.
AGGREGATES = {"total pulses", "total coarse cereals / shri anna",
              "total oilseeds", "total kharif crops", "all crops",
              "total cereals", "total shri anna", "coarse cereals",
              "grand total", "total"}


def parse(path):
    """One DA&FW report -> tidy rows. Returns (DataFrame, as_of date)."""
    m = re.search(r"(\d{4}-\d{2}-\d{2})", path.name)
    as_of = pd.to_datetime(m.group(1)) if m else pd.NaT

    raw = pd.read_excel(path, sheet_name=0, header=None)
    #   The header block is prose: ministry, department, title, source, units.
    #   Find the row whose first cell is literally "Crop" rather than assuming
    #   a fixed offset -- the preamble length has changed between releases.
    hdr = None
    for i in range(min(15, len(raw))):
        if str(raw.iloc[i, 0]).strip().lower() == "crop":
            hdr = i
            break
    if hdr is None:
        log(f"  ! {path.name}: no 'Crop' header row found — skipped")
        return None, as_of

    d = raw.iloc[hdr + 1:, :5].copy()
    d.columns = ["crop", "normal", "sown", "sown_ly", "diff"]
    d["crop"] = d["crop"].astype(str).str.strip()
    d = d[d["crop"].notna() & (d["crop"] != "nan") & (d["crop"] != "")]
    for c in ("normal", "sown", "sown_ly", "diff"):
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.dropna(subset=["sown"])

    d["is_aggregate"] = d["crop"].str.lower().isin(AGGREGATES)
    #   Recomputed rather than read: the published % column is rounded to two
    #   decimals and occasionally blank, and a derived figure that disagrees
    #   with its own inputs is worse than no figure.
    with np.errstate(invalid="ignore", divide="ignore"):
        d["pct_change"] = np.where(d["sown_ly"] > 0,
                                   (d["sown"] - d["sown_ly"])
                                   / d["sown_ly"] * 100, np.nan)
        d["pct_of_normal"] = np.where(d["normal"] > 0,
                                      d["sown"] / d["normal"] * 100, np.nan)
    d["as_of"] = as_of
    d["source_file"] = path.name
    return d.reset_index(drop=True), as_of


def main():
    log("=" * 74)
    log("STEP 29 — DA&FW all-India cropwise sowing reports")
    log("=" * 74)
    files = sorted(UPAG.glob("Kharif-All-Crops*.xlsx"))
    if not files:
        log(f"  no DA&FW reports in {UPAG.name}/ "
            f"(expected Kharif-All-Crops*.xlsx)")
        return

    frames, dates = [], []
    for f in files:
        d, as_of = parse(f)
        if d is None:
            continue
        frames.append(d)
        dates.append(as_of)
        log(f"  {f.name[-20:]:<20} as on {as_of:%Y-%m-%d}  "
            f"{len(d)} rows ({int(d['is_aggregate'].sum())} aggregates)")
    if not frames:
        log("  nothing parsed")
        return

    out = pd.concat(frames, ignore_index=True).sort_values(["as_of", "crop"])
    out.to_csv(OUTD / "dafw_sowing_national.csv", index=False)

    latest = out[out["as_of"] == out["as_of"].max()]
    crops = latest[~latest["is_aggregate"]]
    log(f"\n  latest report: {out['as_of'].max():%Y-%m-%d}, "
        f"{len(crops)} crops + {int(latest['is_aggregate'].sum())} aggregates")
    log(f"  {'crop':<22}{'normal':>9}{'sown':>9}{'last yr':>9}"
        f"{'vs LY %':>9}{'% of normal':>12}")
    for r in crops.nlargest(8, "sown").itertuples():
        pn = f"{r.pct_of_normal:>11.1f}" if np.isfinite(r.pct_of_normal) else "          –"
        log(f"  {r.crop[:22]:<22}{r.normal:>9.1f}{r.sown:>9.1f}"
            f"{r.sown_ly:>9.1f}{r.pct_change:>+9.1f}{pn}")

    tot = latest[latest["crop"].str.lower().str.contains(
        "grand total|total kharif|all crops", na=False)]
    meta = {"as_of": f"{out['as_of'].max():%Y-%m-%d}",
            "reports": [f"{d:%Y-%m-%d}" for d in sorted(set(dates))],
            "n_crops": int(len(crops)),
            "units": "lakh hectares (1 lakh ha = 100,000 ha)",
            "source": "Ministry of Agriculture & Farmers' Welfare (DA&FW), "
                      "Kharif All Crops: All India Cropwise Progressive Area "
                      "Sown Report, source CWWG",
            "note": "National totals by crop. Carries a published NORMAL, "
                    "which the UPAg state releases do not — their normal "
                    "columns are empty in every row. Does not replace the "
                    "state/district sowing layers."}
    if len(tot):
        r = tot.iloc[0]
        meta["all_kharif"] = {"sown_lakh_ha": float(r["sown"]),
                              "normal_lakh_ha": float(r["normal"]),
                              "pct_of_normal": (float(r["pct_of_normal"])
                                                if np.isfinite(r["pct_of_normal"])
                                                else None),
                              "vs_last_year_pct": (float(r["pct_change"])
                                                   if np.isfinite(r["pct_change"])
                                                   else None)}
        log(f"\n  ALL KHARIF: {r['sown']:.1f} lakh ha sown against a normal of "
            f"{r['normal']:.1f} ({r['pct_of_normal']:.1f}% of normal), "
            f"{r['pct_change']:+.1f}% vs last year")
    (OUTD / "dafw_sowing_meta.json").write_text(json.dumps(meta, indent=1),
                                                encoding="utf-8")
    log(f"\n  wrote dafw_sowing_national.csv + dafw_sowing_meta.json")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
