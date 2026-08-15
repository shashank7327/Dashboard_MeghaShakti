r"""
v5/43_build_enso_features.py  —  STEP 3b: ENSO / IOD state features and
per-district ENSO composites.

Runs AFTER 42_build_indices.py and augments features_v5.csv in place.

WHY THIS STEP EXISTS
  The panel already carried the raw ONI, Nino-3.4 and IOD DMI numbers, but a
  raw index is not the thing the monsoon responds to.  The literature is
  specific about WHICH aspect of an ENSO event matters for Indian rainfall,
  and none of those aspects is the bare monthly value:

  * Krishna Kumar, Rajagopalan & Cane (1999), Science 284:2156 — "On the
    weakening relationship between the Indian monsoon and ENSO".  The
    ENSO-monsoon correlation is non-stationary; the monsoon responds to the
    EVOLUTION of an event, not its instantaneous amplitude.
  * Kumar, Rajagopalan, Hoerling, Bates & Cane (2006), Science 314:115 —
    "Unraveling the mystery of Indian monsoon failure during El Nino".
    Monsoon failure tracks events that DEVELOP through the summer with warming
    in the central Pacific, not events already decaying by June.
  * Webster & Yang (1992), QJRMS 118:877 — the boreal-spring predictability
    barrier.  ONI through MAM is the last robust pre-monsoon signal, so it is
    carried as its own feature rather than being smeared into a running mean.
  * Saji, Goswami, Vinayachandran & Yamagata (1999), Nature 401:360 — the
    Indian Ocean Dipole; DMI is the west-minus-southeast tropical Indian Ocean
    SST anomaly gradient.
  * Ashok, Guan & Yamagata (2001), GRL 28:4499 — a positive IOD COMPENSATES
    the El Nino monsoon deficit.  The ENSO-monsoon relationship weakens
    precisely when the IOD is positive, which is why an interaction term is
    carried and not just the two indices side by side.
  * ENSO phase itself follows the NOAA CPC operational definition: an episode
    requires ONI at or beyond +/-0.5 C for five consecutive overlapping
    three-month seasons.  ONI is already a 3-month running mean, so
    consecutive months of the series ARE the overlapping seasons.

CAUSALITY — THE THING THAT IS EASY TO GET WRONG
  An ENSO event is conventionally labelled by the DJF peak it eventually
  reaches.  Using that label for a JJAS forecast issued in June leaks the
  future: in June the peak has not happened.  Every feature here is therefore
  built from a STRICTLY TRAILING window:

    * enso_phase uses the last five months (not a centred run)
    * oni_djf_prev is the PREVIOUS winter's peak, known since March
    * oni_mam is the current spring, known since May
    * oni_tend_3 is a backward difference

  A "developing" event is thus one that is rising NOW, which is knowable, not
  one that turns out to peak next winter, which is not.

  Per-district composites are fitted on TRAINING YEARS ONLY (<= TRAIN_END) so
  that the validation and test periods stay genuinely held out.

PROVENANCE
  Everything in this file is [OBS] or [DERIVED] — measured SST indices and
  deterministic arithmetic on them.  No model output enters here.  The feature
  manifest written alongside records that classification per column.

Outputs -> v5\data\:
  features_v5.csv              (augmented in place with the ENSO columns)
  enso_monthly_v5.csv          national monthly ENSO/IOD state
  district_enso_composite_v5.csv   per-district ENSO signature (train-only)
  feature_manifest_v5.csv      every feature: formula, provenance, source

Run:  py -3.13 -X utf8 "v5/43_build_enso_features.py"
"""

import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from common_v5 import (BASE, DATA, FEATURES_CSV, TRAIN_END, log,  # noqa
                       safe_to_csv)

NOAA = BASE / "noaa_indices_cache.csv"
EL_THRESH, LA_THRESH = 0.5, -0.5
RUN_LEN = 5                     # CPC: five consecutive overlapping seasons
IOD_THRESH = 0.4                # Saji et al. (1999) conventional cut
PERSIST_LIMIT = 3               # months an unpublished index may be carried


def trailing_run(flag, n):
    """True where `flag` has been continuously True for the last n months.

    Trailing, never centred: at month t this may only look at t-n+1 .. t, so
    the label is knowable at t.  A centred run would let a forecast issued in
    June know what the ocean does in September.
    """
    f = flag.astype(float).fillna(0.0)
    return f.rolling(n, min_periods=n).sum() >= n


def seasonal_mean(df, months, col):
    """Mean of `col` over the given calendar months, labelled by the year the
    season ENDS in, so it can be joined without leaking forward."""
    s = df[df["month"].isin(months)].groupby("year")[col].mean()
    return s


def main():
    log("=" * 70)
    log("v5 STEP 3b — ENSO / IOD state features and district composites")
    log("=" * 70)

    if not NOAA.exists():
        log(f"  ! {NOAA.name} not found — cannot build ENSO features")
        return

    ix = pd.read_csv(NOAA).sort_values(["year", "month"]).reset_index(drop=True)
    ix["date"] = pd.to_datetime(dict(year=ix["year"], month=ix["month"], day=1))

    for c in ("oni", "nino34_anom", "iod_dmi"):
        last = ix.loc[ix[c].notna(), "date"].max()
        log(f"    {c:12s} published through {last:%Y-%m}")

    # ---- 1. operational availability -------------------------------------
    # NOAA publishes ONI with a lag, and the IOD feed here stops in 2025-04.
    # Operational practice is to persist the last observed value rather than
    # drop the forecast; the panel records WHERE that happened so the
    # dashboard and the report can be honest about it.
    # PERSISTENCE IS BOUNDED.  An unlimited ffill runs to the end of the index
    # cache (2026-12), which is months past any observation — and it invented
    # a phantom El Nino event for 2026-08..2026-12 out of a value carried
    # forward seven months.  Operationally, persisting an ocean index for a
    # quarter is defensible; persisting it for half a year is fabrication.
    for c in ("oni", "nino34_anom", "iod_dmi"):
        ix[f"{c}_persisted"] = ix[c].isna()
        first_valid = ix[c].first_valid_index()
        if first_valid is not None:
            ix.loc[first_valid:, c] = ix.loc[first_valid:, c].ffill(
                limit=PERSIST_LIMIT)
        ix[f"{c}_persisted"] &= ix[c].notna()

    n_p = {c: int(ix[f"{c}_persisted"].sum())
           for c in ("oni", "nino34_anom", "iod_dmi")}
    log(f"    persisted months — oni {n_p['oni']}, nino34 {n_p['nino34_anom']}, "
        f"iod_dmi {n_p['iod_dmi']}")

    # ---- 2. ENSO phase, CPC five-season rule, trailing --------------------
    warm = trailing_run(ix["oni"] >= EL_THRESH, RUN_LEN)
    cold = trailing_run(ix["oni"] <= LA_THRESH, RUN_LEN)
    ix["enso_phase"] = np.where(warm, "El Nino",
                                np.where(cold, "La Nina", "Neutral"))
    ix["enso_warm"] = warm.astype(int)      # one-hot, model-friendly
    ix["enso_cold"] = cold.astype(int)
    ix["enso_sign"] = ix["enso_warm"] - ix["enso_cold"]

    # ---- 3. evolution: developing vs decaying ----------------------------
    # Kumar et al. (2006): a monsoon fails under an event that is STILL
    # WARMING through the summer.  Backward differences only.
    ix["oni_tend_3"] = ix["oni"] - ix["oni"].shift(3)
    ix["oni_tend_6"] = ix["oni"] - ix["oni"].shift(6)

    # previous winter's DJF mean, and the current spring — both strictly past
    djf = ix.copy()
    # DJF labelled by the January it contains: Dec(Y-1), Jan(Y), Feb(Y)
    djf["djf_year"] = np.where(djf["month"] == 12, djf["year"] + 1, djf["year"])
    djf_mean = (djf[djf["month"].isin([12, 1, 2])]
                .groupby("djf_year")["oni"].mean())
    mam_mean = seasonal_mean(ix, [3, 4, 5], "oni")

    ix["oni_djf_prev"] = ix["year"].map(djf_mean)
    # before March the current year's DJF is not yet complete -> use the
    # previous one, so nothing is known before it could be
    ix.loc[ix["month"] < 3, "oni_djf_prev"] = \
        ix.loc[ix["month"] < 3, "year"].sub(1).map(djf_mean)
    ix["oni_mam"] = ix["year"].map(mam_mean)
    ix.loc[ix["month"] < 6, "oni_mam"] = np.nan   # MAM not complete until June

    developing = (ix["oni_tend_3"] > 0.1) & (ix["oni"] >= 0.0)
    decaying = (ix["oni_djf_prev"] >= EL_THRESH) & (ix["oni_tend_3"] < -0.1)
    ix["enso_developing"] = developing.astype(int)
    ix["enso_decaying"] = decaying.astype(int)

    # ---- 4. intensity ------------------------------------------------------
    a = ix["oni"].abs()
    ix["enso_intensity"] = np.select(
        [a < 0.5, a < 1.0, a < 1.5, a < 2.0],
        [0, 1, 2, 3], 4).astype(float)        # none/weak/mod/strong/very

    # ---- 5. IOD and the ENSO x IOD interaction ----------------------------
    sd = ix["iod_dmi"].std()
    ix["iod_z"] = ix["iod_dmi"] / sd if sd and np.isfinite(sd) else np.nan
    ix["iod_pos"] = (ix["iod_dmi"] >= IOD_THRESH).astype(int)
    ix["iod_neg"] = (ix["iod_dmi"] <= -IOD_THRESH).astype(int)
    # Ashok et al. (2001): a positive IOD offsets El Nino.  The product is the
    # compensation term — large and negative when a warm Pacific meets a
    # positive dipole, which is exactly when the ENSO-monsoon link breaks.
    ix["enso_iod_interact"] = ix["oni"] * ix["iod_dmi"]

    # The RAW indices are carried too, in their persisted form.  Without this
    # the derived features were populated from the forward-filled series while
    # `oni` and `iod_dmi` themselves stayed NaN in features_v5.csv — so the
    # model saw an empty ONI at the June 2026 issue date while every ONI-
    # derived feature was present.  The *_persisted flags travel alongside, so
    # a filled value is never mistaken for an observed one.
    # Never emit ENSO state for months the panel does not cover.  The NOAA
    # cache is pre-allocated to 2026-12, and classifying those empty months
    # produced an "event" for dates with no data behind them at all.
    pan = pd.read_csv(FEATURES_CSV, usecols=["year", "month"])
    pan_end = pd.Period(
        f"{int(pan['year'].max())}-"
        f"{int(pan.loc[pan['year'] == pan['year'].max(), 'month'].max()):02d}",
        freq="M").to_timestamp()
    before = len(ix)
    ix = ix[ix["date"] <= pan_end]
    if len(ix) < before:
        log(f"    clipped {before - len(ix)} months beyond the panel end "
            f"({pan_end:%Y-%m}) — no data stands behind them")

    keep = ["year", "month", "oni", "nino34_anom", "iod_dmi",
            "enso_warm", "enso_cold", "enso_sign",
            "enso_developing", "enso_decaying", "enso_intensity",
            "oni_tend_3", "oni_tend_6", "oni_djf_prev", "oni_mam",
            "iod_z", "iod_pos", "iod_neg", "enso_iod_interact",
            "enso_phase", "oni_persisted", "nino34_anom_persisted",
            "iod_dmi_persisted"]
    enso = ix[keep].copy()
    safe_to_csv(enso, DATA / "enso_monthly_v5.csv", index=False)

    ph = ix.groupby("enso_phase").size()
    log(f"    phase months 1981-2026 — " +
        ", ".join(f"{k} {v}" for k, v in ph.items()))
    log(f"    developing months {int(ix['enso_developing'].sum())}, "
        f"decaying {int(ix['enso_decaying'].sum())}")

    # ---- 6. per-district ENSO composite, TRAIN YEARS ONLY -----------------
    log("  district ENSO composites (JJAS, training years only) ...")
    p = pd.read_csv(FEATURES_CSV)
    p = p.drop(columns=[c for c in enso.columns
                        if c not in ("year", "month") and c in p.columns])
    p = p.merge(enso, on=["year", "month"], how="left")

    jjas = p[p["month"].isin([6, 7, 8, 9]) & (p["year"] <= TRAIN_END)]
    if "pct_departure" in jjas.columns:
        comp = (jjas.groupby(["district_id", "enso_phase"])["pct_departure"]
                .mean().unstack())
        for c in ("El Nino", "La Nina", "Neutral"):
            if c not in comp.columns:
                comp[c] = np.nan
        comp = comp.rename(columns={"El Nino": "dep_elnino",
                                    "La Nina": "dep_lanina",
                                    "Neutral": "dep_neutral"})
        # the district's ENSO signature: how much drier an El Nino monsoon is
        # than a neutral one, in percentage points of departure
        comp["enso_signature"] = comp["dep_elnino"] - comp["dep_neutral"]
        comp["lanina_signature"] = comp["dep_lanina"] - comp["dep_neutral"]
        comp = comp.reset_index()[["district_id", "dep_elnino", "dep_lanina",
                                   "dep_neutral", "enso_signature",
                                   "lanina_signature"]]
        safe_to_csv(comp, DATA / "district_enso_composite_v5.csv", index=False)
        sig = comp["enso_signature"].dropna()
        log(f"    {len(comp)} districts; El Nino minus neutral JJAS departure "
            f"median {sig.median():+.1f} pp, "
            f"{int((sig < 0).sum())}/{len(sig)} districts drier")
        p = p.drop(columns=[c for c in comp.columns
                            if c != "district_id" and c in p.columns])
        p = p.merge(comp, on="district_id", how="left")

    safe_to_csv(p, FEATURES_CSV, index=False)
    log(f"  features_v5.csv augmented -> {p.shape[0]:,} rows x {p.shape[1]} cols")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
