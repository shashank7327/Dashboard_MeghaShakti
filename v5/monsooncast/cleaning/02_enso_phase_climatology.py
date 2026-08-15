r"""
v5/monsooncast/cleaning/02_enso_phase_climatology.py  —  STEP 2: characterise the
district climatology by ENSO phase and turn it into modelling features.

WHAT THE USER ASKED, AND THE ACADEMIC CORRECTION APPLIED
  Goal: use the ocean indices to identify how each district's climate behaves
  under El Nino / neutral / La Nina, and carry that as a feature.  Two
  corrections from the literature are applied so the composite is honest:

  1. PHASE IS DEFINED CAUSALLY, NOT BY THE DJF PEAK LABEL.  An event is named
     for the winter it peaks in, but a forecast issued in the monsoon cannot
     know that peak.  Kumar et al. (2006, Science 314:115) show monsoon
     failure tracks events that are DEVELOPING through the summer, not events
     already decaying.  So the phase used here is the operational trailing
     ONI classification carried in the panel (enso_warm/enso_cold), which is
     knowable at issue time, and the developing flag is kept alongside.

  2. COMPOSITES ARE FIT ON TRAINING YEARS ONLY (<= 2016).  Compositing over
     the whole record and then predicting the held-out years would leak.

  3. THE RELATIONSHIP IS NON-STATIONARY AND MODULATED.  Krishna Kumar et al.
     (1999, Science 284:2156) document a weakening ENSO-monsoon correlation;
     Ashok et al. (2001, GRL 28:4499) show a positive IOD offsets the El Nino
     deficit.  So the composite is reported as a signal to be MODULATED by the
     IOD interaction term, not a deterministic rule.

METHOD
  For the monsoon quarter (JJAS) and for each district, composite the observed
  climate variables by phase over training years, and derive the per-district
  ENSO SIGNATURE = mean(JJAS metric | El Nino) - mean(JJAS metric | neutral).
  Variables: rainfall departure, soil-moisture adequacy (MAI), 4-month
  soil-moisture proxy (swvl2), temperature anomaly.

OUTPUT -> v5/data_lgd/
  enso_phase_composite_lgd.csv   national JJAS climate by phase (comparison)
  district_enso_signature_lgd.csv per-district signatures (features)
  panel_with_enso_features_lgd.csv  master panel + district ENSO features merged

Run:  py -3.13 -X utf8 "v5/monsooncast/cleaning/02_enso_phase_climatology.py"
"""
import pathlib
import sys

import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
V5 = HERE.parents[1]
OUTD = V5 / "data_lgd"
sys.path.insert(0, str(V5))
from common_v5 import log  # noqa

TRAIN_END = 2016
JJAS = [6, 7, 8, 9]
METRICS = {"pct_departure": "Rainfall departure (%)",
           "mai": "Soil-moisture adequacy (AET/PET)",
           "swvl2": "Root-zone soil moisture (m3/m3)",
           "tmax_anom": "Tmax anomaly (degC)"}


def phase_of(row):
    if row.get("enso_warm", 0) == 1:
        return "El Nino"
    if row.get("enso_cold", 0) == 1:
        return "La Nina"
    return "Neutral"


def main():
    log("=" * 70)
    log("STEP 2 — ENSO-phase climatology features (causal, training-only)")
    log("=" * 70)
    p = pd.read_csv(OUTD / "master_panel_lgd.csv", parse_dates=["date"])
    p["phase"] = p.apply(phase_of, axis=1)
    jjas = p[p["month"].isin(JJAS)].copy()
    train = jjas[jjas["year"] <= TRAIN_END]
    log(f"  JJAS training months <= {TRAIN_END}: {len(train):,} district-months")
    log(f"  phase share: " + ", ".join(
        f"{k} {v}" for k, v in train.drop_duplicates(['year', 'month'])
        ['phase'].value_counts().items()))

    # ---- national comparison across phases (the 'comparative dynamics') ----
    comp = (train.groupby("phase")[list(METRICS)].mean().round(3))
    comp = comp.reindex(["El Nino", "Neutral", "La Nina"])
    comp["n_district_months"] = train.groupby("phase").size()
    comp.to_csv(OUTD / "enso_phase_composite_lgd.csv")
    log("\n  NATIONAL JJAS CLIMATE BY PHASE (training years):")
    for ph, r in comp.iterrows():
        log(f"    {str(ph):8s}  rain {r['pct_departure']:+6.1f}%   "
            f"MAI {r['mai']:.3f}   swvl2 {r['swvl2']:.3f}   "
            f"Tmax anom {r['tmax_anom']:+.2f}")

    # ---- per-district ENSO signature = El Nino minus Neutral (features) ----
    sig_rows = []
    for did, g in train.groupby("district_id"):
        rec = {"district_id": did}
        for m in METRICS:
            en = g.loc[g["phase"] == "El Nino", m].mean()
            nu = g.loc[g["phase"] == "Neutral", m].mean()
            ln = g.loc[g["phase"] == "La Nina", m].mean()
            rec[f"{m}_elnino"] = en
            rec[f"{m}_neutral"] = nu
            rec[f"{m}_lanina"] = ln
            rec[f"{m}_signature"] = en - nu          # El Nino drying signal
        sig_rows.append(rec)
    sig = pd.DataFrame(sig_rows)
    reg = p[["district_id", "state", "district", "is_jk_ladakh"]].drop_duplicates()
    sig = sig.merge(reg, on="district_id", how="left")
    sig.to_csv(OUTD / "district_enso_signature_lgd.csv", index=False)

    drier = int((sig["pct_departure_signature"] < 0).sum())
    log(f"\n  PER-DISTRICT ENSO SIGNATURE (El Nino minus Neutral, JJAS rain):")
    log(f"    districts drier in El Nino: {drier}/{len(sig)} "
        f"({100*drier/len(sig):.0f}%)")
    log(f"    median signature: {sig['pct_departure_signature'].median():+.1f} pp"
        f"   worst decile: {sig['pct_departure_signature'].quantile(0.1):+.1f} pp")
    top = sig.nsmallest(8, "pct_departure_signature")
    log("    strongest El Nino drought signature:")
    for r in top.itertuples():
        log(f"      {r.district:24s} {r.state:20s} "
            f"{r.pct_departure_signature:+6.1f} pp")

    # ---- merge the district ENSO features back onto the panel --------------
    feat_cols = ["district_id"] + [c for c in sig.columns
                                   if c.endswith(("_elnino", "_neutral",
                                                  "_lanina", "_signature"))]
    out = p.merge(sig[feat_cols], on="district_id", how="left")
    out.to_csv(OUTD / "panel_with_enso_features_lgd.csv", index=False)
    log(f"\n  wrote enso_phase_composite_lgd.csv, "
        f"district_enso_signature_lgd.csv,")
    log(f"        panel_with_enso_features_lgd.csv "
        f"({len(out):,} rows x {out.shape[1]} cols)")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
