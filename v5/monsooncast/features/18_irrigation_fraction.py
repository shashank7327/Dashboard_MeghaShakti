r"""
v5/monsooncast/features/18_irrigation_fraction.py  —  irrigated fraction per LGD
district, the variable that separates Kharif from Rabi risk.

WHY IT MATTERS, AND WHY IT DIFFERS BY SEASON
  Kharif (Jun-Oct) is the monsoon crop: rainfall IS the water supply, and
  irrigation is a BUFFER that decouples a district from monsoon failure.  A
  fully irrigated Punjab paddy district and a rainfed Vidarbha cotton district
  can see the same rainfall deficit and suffer completely different losses.

  Rabi (Oct-Apr) is the dry-season crop and is grown ON irrigation: winter
  rainfall is minor (western disturbances aside), so a rainfall-departure
  measure carries almost no information about Rabi water supply.  What matters
  is whether the district can irrigate at all, plus the soil moisture left
  behind by the monsoon at sowing.

  Treating both seasons with one rainfall-based water term -- which the
  previous version effectively did -- misstates Rabi risk in both directions.

SOURCE, AND WHICH BAND IS ACTUALLY USABLE
  GlobalIrrigation/GlobalIrrigation_districts_YYYY.csv (2001-2015) exports two
  bands from the global irrigation map: irrigated_any_ha (class >= 1) and
  irrigated_high_ha (class == 2).  Summed over India for 2015 they give:

      irrigated_any_ha    265.3 M ha
      irrigated_high_ha    74.2 M ha
      India net irrigated area, published        ~71 M ha
      India net SOWN area, published            ~140 M ha

  irrigated_any_ha exceeds the entire net sown area by almost 2x and covers
  ~80% of India's land surface, so the >= 1 class is not "irrigated" in any
  agronomic sense -- it captures nearly the whole mapped domain.  The eq(2)
  class lands within 5% of the published net irrigated area, so
  irrigated_high_ha is the band used here and irrigated_any_ha is carried only
  as a diagnostic.

  Fractions are taken against NET SOWN AREA, not district geographic area:
  irrigation is only meaningful over land that is actually cropped, and
  dividing by total area would make every forested or desert district look
  un-irrigated for the wrong reason.  Cropped area is approximated per district
  from the crop-area mask where available.

  Names are matched to the 791 LGD units; the source is on an older district
  vintage, so unmatched units fall back to their STATE median -- flagged in
  `irrig_source`.

OUTPUT -> v5/data_lgd/irrigation_fraction_lgd.csv
  district_id, irrig_frac (0-1), irrig_high_frac, irrig_source

Run:  py -3.13 -X utf8 "v5/monsooncast/features/18_irrigation_fraction.py"
"""
import pathlib
import sys

import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
V5 = HERE.parents[1]
ROOT = V5.parent
IMD = ROOT / "IMD_Data"
OUTD = V5 / "data_lgd"
SRC = ROOT / "GlobalIrrigation"
sys.path.insert(0, str(V5))
sys.path.insert(0, str(IMD))
from common_v5 import log  # noqa
from build_crosswalk import norm_state, norm_key  # noqa

USE_YEARS = 5          # average the most recent N annual maps


def main():
    log("=" * 72)
    log("IRRIGATED FRACTION per LGD district")
    log("=" * 72)
    reg = pd.read_csv(IMD / "registry_lgd791.csv")
    reg["ks"], reg["kd"] = norm_state(reg["state"]), norm_key(reg["district"])

    files = sorted(SRC.glob("GlobalIrrigation_districts_*.csv"))[-USE_YEARS:]
    if not files:
        log(f"  ! no irrigation files in {SRC}")
        return
    d = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    log(f"  averaging {len(files)} annual maps "
        f"({files[0].stem[-4:]}-{files[-1].stem[-4:]})")
    d["ks"], d["kd"] = norm_state(d["state"]), norm_key(d["district"])
    g = (d.groupby(["ks", "kd"], as_index=False)
         [["irrigated_any_ha", "irrigated_high_ha"]].mean())

    m = reg.merge(g, on=["ks", "kd"], how="left")
    matched = m["irrigated_high_ha"].notna()
    log(f"  matched {int(matched.sum())}/{len(m)} districts by name")
    log(f"  India totals: irrigated_high {g['irrigated_high_ha'].sum()/1e6:.1f} "
        f"M ha (published net irrigated ~71) | irrigated_any "
        f"{g['irrigated_any_ha'].sum()/1e6:.1f} M ha (implausible, diagnostic "
        f"only)")

    # Denominator.  Cropped area per district would be the agronomically
    # ideal denominator, but the only cropped-area figure available is the
    # crop mask's EVENLY ALLOCATED state area (uniform within a state), so
    # dividing a real district irrigated area by it produced a ratio that
    # saturated at 0 and 1 and carried no ordering.  Geographic area is used
    # instead: "share of the district under irrigation" is well defined, does
    # not clip, and nationally reproduces 70.4/328.7 = 21%.
    #
    # For the stress model what matters is the RELATIVE buffer between
    # districts, so a percentile index is also emitted and is what the crop
    # model consumes -- it is robust to the absolute level of the map.
    area_ha = m["area_km2"] * 100.0
    m["irrig_frac"] = (m["irrigated_high_ha"] / area_ha).clip(0, 1)
    m["irrig_any_frac_diag"] = (m["irrigated_any_ha"] / area_ha).clip(0, 1)
    m["irrig_source"] = np.where(matched, "district", "state median (unmatched)")

    # state-mean fallback for units the older vintage does not name
    for col in ("irrig_frac", "irrig_any_frac_diag"):
        sm = m.groupby("ks")[col].transform("median")
        m[col] = m[col].fillna(sm).fillna(m[col].median())

    # relative irrigation buffer, 0 (most monsoon-exposed) to 1 (most secure)
    m["irrig_index"] = m["irrig_frac"].rank(pct=True).round(4)
    out = m[["district_id", "state", "district", "irrig_frac", "irrig_index",
             "irrig_any_frac_diag", "irrig_source"]].copy()
    out[["irrig_frac", "irrig_any_frac_diag"]] = out[
        ["irrig_frac", "irrig_any_frac_diag"]].round(4)
    out.to_csv(OUTD / "irrigation_fraction_lgd.csv", index=False)

    log(f"  share of district area irrigated: median "
        f"{out['irrig_frac'].median():.3f}, IQR "
        f"{out['irrig_frac'].quantile(.25):.3f}-"
        f"{out['irrig_frac'].quantile(.75):.3f}  "
        f"(national {m['irrigated_high_ha'].sum()/area_ha.sum():.3f})")
    top = out.nlargest(6, "irrig_frac")[["state", "district", "irrig_frac"]]
    bot = out.nsmallest(6, "irrig_frac")[["state", "district", "irrig_frac"]]
    log("  most irrigated:")
    for r in top.itertuples():
        log(f"    {r.district:24s} {r.state:20s} {r.irrig_frac:.2f}")
    log("  least irrigated (most monsoon-exposed):")
    for r in bot.itertuples():
        log(f"    {r.district:24s} {r.state:20s} {r.irrig_frac:.2f}")
    log(f"  wrote irrigation_fraction_lgd.csv")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
