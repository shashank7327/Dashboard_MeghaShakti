r"""
v5/monsooncast/features/03_build_features.py  —  STEP 3: deterministic monthly features
on the LGD panel (SPEI, Hargreaves-SPEI, GDD, SDD), completing the model-ready
feature table.  MAI, temperature anomalies, soil moisture and the ENSO features
are already on the panel from steps 1-2.

  SPEI-1/4/12   log-logistic, unbiased PWM (Hosking 1990; Vicente-Serrano 2010),
                per district x calendar month, from D = rain - PET(ERA5).
  spei_harg_4   same, with Hargreaves temperature-based PET (Hargreaves 1985);
                a PET-source sensitivity term, historically the #2 predictor.
  GDD/SDD       accumulated thermal time and heat load from DAILY IMD Tmax/Tmin
                (McMaster & Wilhelm 1997; Idso 1977) -> monthly sums.

OUTPUT -> v5/data_lgd/features_lgd.csv   (monthly, one row per district-month)

Run:  py -3.13 -X utf8 "v5/monsooncast/features/03_build_features.py"
"""
import pathlib
import sys

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import gamma

HERE = pathlib.Path(__file__).resolve().parent
V5 = HERE.parents[1]
IMD = V5.parent / "IMD_Data"
OUTD = V5 / "data_lgd"
sys.path.insert(0, str(V5))
from common_v5 import log  # noqa

GSC = 0.0820                        # solar constant MJ/m2/min


def loglogistic_spei(D):
    out = np.full(len(D), np.nan)
    idx = np.where(~np.isnan(D))[0]
    x = D[idx]
    if len(x) < 20:
        return out
    xs = np.sort(x)
    n = len(xs)
    j = np.arange(1, n + 1)
    w0 = xs.mean()
    w1 = np.sum((n - j) / (n * (n - 1)) * xs)
    w2 = np.sum((n - j) * (n - j - 1) / (n * (n - 1) * (n - 2)) * xs)
    try:
        beta = (2 * w1 - w0) / (6 * w1 - w0 - 6 * w2)
        g1, g2 = gamma(1 + 1 / beta), gamma(1 - 1 / beta)
        alpha = (w0 - 2 * w1) * beta / (g1 * g2)
        gam = w0 - alpha * g1 * g2
        if not np.isfinite([beta, alpha, gam]).all() or alpha <= 0:
            raise ValueError
        F = 1.0 / (1.0 + (alpha / (x - gam)) ** beta)
        out[idx] = stats.norm.ppf(np.clip(F, 1e-6, 1 - 1e-6))
    except Exception:
        F = (stats.rankdata(x) - 0.44) / (n + 0.12)
        out[idx] = stats.norm.ppf(np.clip(F, 1e-6, 1 - 1e-6))
    return out


def add_spei(panel, dcol, prefix, scales=(1, 4, 12)):
    panel = panel.sort_values(["district_id", "date"]).reset_index(drop=True)
    for s in scales:
        acc = (panel.groupby("district_id")[dcol]
                    .transform(lambda v: v.rolling(s, min_periods=s).sum()))
        col = f"{prefix}_{s}"
        panel[col] = np.nan
        for _, g in panel.groupby(["district_id", "month"]):
            panel.loc[g.index, col] = loglogistic_spei(acc.loc[g.index].values)
        log(f"    {col}: {int(panel[col].notna().sum()):,} values")
    return panel


def extraterrestrial_ra(lat_deg, month):
    """FAO-56 monthly Ra (mm/day equivalent) at a latitude, mid-month DOY."""
    doy = np.array([15, 46, 74, 105, 135, 166, 196, 227, 258, 288, 319, 349])
    J = doy[month - 1]
    phi = np.deg2rad(lat_deg)
    dr = 1 + 0.033 * np.cos(2 * np.pi * J / 365)
    dec = 0.409 * np.sin(2 * np.pi * J / 365 - 1.39)
    ws = np.arccos(np.clip(-np.tan(phi) * np.tan(dec), -1, 1))
    ra = (24 * 60 / np.pi) * GSC * dr * (
        ws * np.sin(phi) * np.sin(dec) + np.cos(phi) * np.cos(dec) * np.sin(ws))
    return ra * 0.408                       # MJ/m2/day -> mm/day equivalent


def district_latitudes():
    cw = pd.read_csv(IMD / "crosswalk_rain_lgd.csv")
    return cw.groupby("district_id")["lat"].mean()


def monthly_thermal():
    """GDD_kharif/rabi and SDD from daily IMD Tmax/Tmin -> monthly sums."""
    tx = pd.read_pickle(IMD / "_district_daily_tmax_lgd.pkl")
    tn = pd.read_pickle(IMD / "_district_daily_tmin_lgd.pkl")
    tx.columns = tx.columns.astype(int)
    tn.columns = tn.columns.astype(int)
    tmean = (tx + tn) / 2
    gdd_k = (tmean - 10).clip(lower=0).resample("MS").sum(min_count=20)
    gdd_r = (tmean - 5).clip(lower=0).resample("MS").sum(min_count=20)
    sdd = (tx - 34).clip(lower=0).resample("MS").sum(min_count=20)

    def melt(m, name):
        lo = m.reset_index().melt(id_vars="date", var_name="district_id",
                                  value_name=name)
        lo["district_id"] = lo["district_id"].astype(int)
        return lo
    out = melt(gdd_k, "gdd_kharif")
    out = out.merge(melt(gdd_r, "gdd_rabi"), on=["date", "district_id"])
    out = out.merge(melt(sdd, "sdd"), on=["date", "district_id"])

    #   NORMALS FOR THE DEGREE-DAY SUMS, AND WHY THEY ARE NEEDED.
    #
    #   Raw SDD answers "where is it hot", which is the same places every
    #   year: Vidarbha and western Rajasthan run 100-250 degree-days above the
    #   threshold every May whatever the season is doing. Displayed on its own
    #   it is a map of climate, not of stress, and it carries no information
    #   about whether THIS month is unusual -- which is the only thing a
    #   stress layer is for.
    #
    #   Worse, the raw sum is not comparable across the slider. Half the
    #   record is exactly zero (52.4%), the peak months sit near the top of
    #   any fixed scale, and a part-month sum is a fraction of a full one, so
    #   scrubbing from July to August looked like heat stress vanishing when
    #   it only meant four days of data.
    #
    #   So the anomaly is computed the same way tmax_anom already is: against
    #   a per-district, per-CALENDAR-MONTH mean over IMD's own 1971-2020
    #   window. A May value is then compared only with other Mays in the same
    #   district, and the layer reads as "hotter or cooler than usual here".
    lo, hi = 1971, 2020
    base = out[(out["date"].dt.year >= lo) & (out["date"].dt.year <= hi)].copy()
    base["cm"] = base["date"].dt.month
    out["cm"] = out["date"].dt.month
    for col in ("sdd", "gdd_kharif", "gdd_rabi"):
        nrm = (base.groupby(["district_id", "cm"])[col].mean()
               .rename(f"{col}_normal").reset_index())
        out = out.merge(nrm, on=["district_id", "cm"], how="left")
        out[f"{col}_anom"] = out[col] - out[f"{col}_normal"]
    out = out.drop(columns="cm")
    log(f"    degree-day normals on {lo}-{hi}: "
        f"sdd_normal {out['sdd_normal'].notna().sum():,} values, "
        f"median {out['sdd_normal'].median():.1f} degC-d")
    # how many days of temperature each month actually has, so a part-month
    # degree-day SUM is never compared against full-month history
    obs = tx.notna().any(axis=1)
    td = obs.astype(int).resample("MS").sum().rename("temp_days").reset_index()
    td["temp_days_complete"] = (td["temp_days"]
                                >= td["date"].dt.days_in_month).astype(int)
    return out, td


# Which completeness flags each accumulating feature depends on.  A monthly
# ACCUMULATION is only meaningful against a climatology of equally complete
# months, so a feature is withheld for any month its sources do not fully cover.
NEEDS_COMPLETE = {
    "spei_era5_1": ["month_complete", "et_days_complete"],
    "spei_era5_4": ["month_complete", "et_days_complete"],
    "spei_era5_12": ["month_complete", "et_days_complete"],
    "spei_harg_4": ["month_complete", "temp_days_complete"],
    "gdd_kharif": ["temp_days_complete"],
    "gdd_rabi": ["temp_days_complete"],
    "sdd": ["temp_days_complete"],
    "wb_era5": ["month_complete", "et_days_complete"],
    "wb_harg": ["month_complete", "temp_days_complete"],
    "pet_hargreaves": ["temp_days_complete"],
}


# The mask still runs: it removes the values that WOULD be wrong, and the
# partial-month pass immediately below replaces them with values that are
# right.  Set SKIP_MASK to keep the raw full-month arithmetic instead — useful
# only for reproducing the earlier behaviour when investigating a regression.
SKIP_MASK = False


def fill_partial_month(p):
    r"""Recompute the withheld indices for the trailing month on a
    month-to-date basis, so the running month shows a valid number rather than
    an empty map.

    Every value written here is compared against the SAME window in every other
    year of the record — see features/_partial_month.py for why that makes a
    25-day SPEI directly interpretable, and why it uses Hargreaves PET rather
    than ERA5 for this one case.

    Rows touched are flagged `is_month_to_date = 1` so the dashboard, the master
    exports and any downstream consumer can label them instead of silently
    mixing a to-date value with full-month history.
    """
    p["is_month_to_date"] = 0
    if "month_complete" not in p.columns:
        return p

    #   FILL EVERY MONTH THE MASK WILL WITHHOLD, NOT JUST THE LAST ONE.
    #
    #   These two functions used different definitions of "incomplete" and the
    #   difference left a hole. This one selected on month_complete, which is
    #   about RAINFALL; mask_incomplete_months withholds when month_complete
    #   OR et_days_complete is 0, and ERA5-Land lags the gauge analysis by
    #   about a week. So a month can have all 31 days of rain and only 27 days
    #   of ERA5 — complete by this function's test, incomplete by the mask's.
    #
    #   July 2026 was exactly that: masked, never refilled, and SPEI came out
    #   blank for July while June and August either side of it had values. A
    #   gap in the middle of a series reads as a broken product, and it is the
    #   opposite of what the month-to-date machinery exists to do.
    #
    #   The recomputation is on Hargreaves PET, which needs only temperature
    #   and therefore ends when the gauge data ends. For a month that is whole
    #   in rain and temperature, that yields a COMPLETE month on a different
    #   PET model rather than a partial one — which is why the flag below is
    #   set from days_observed, not from membership of this list.
    flags = [c for c in ("month_complete", "et_days_complete",
                         "temp_days_complete") if c in p.columns]
    ok = np.ones(len(p), bool)
    for c in flags:
        ok &= p[c].fillna(0).to_numpy() == 1
    inc = p[~ok]
    if not len(inc):
        log("  no incomplete month — nothing to recompute")
        return p

    #   Only the trailing run is recomputable: earlier gaps are missing source
    #   data, not a lagging edge, and inventing values there would be
    #   pro-rating by another name.
    allm = sorted(pd.Timestamp(x) for x in p["date"].unique())
    incm = {pd.Timestamp(x) for x in inc["date"].unique()}
    todo = []
    for m in reversed(allm):
        if m not in incm:
            break
        todo.append(m)
    todo = sorted(todo)
    log(f"  {len(todo)} incomplete month(s) to recompute: "
        + ", ".join(f"{m:%Y-%m}" for m in todo))

    import _partial_month as PM
    tx = pd.read_pickle(IMD / "_district_daily_tmax_lgd.pkl")
    tn = pd.read_pickle(IMD / "_district_daily_tmin_lgd.pkl")
    rn = pd.read_pickle(IMD / "_district_daily_rain_lgd.pkl")
    for fr in (tx, tn, rn):
        fr.columns = fr.columns.astype(int)
    lat = district_latitudes()

    for last in todo:
        sel = p["date"] == last
        ndays = int(p.loc[sel, "days_observed"].iloc[0])
        dim = int(p.loc[sel, "days_in_month"].iloc[0])
        #   The flag means "this value covers only part of its month", which
        #   is a statement about the RAIN/TEMPERATURE window, not about which
        #   PET model was used. A month whole in both is a complete month on
        #   Hargreaves PET and must not be labelled to-date.
        partial = ndays < dim
        p.loc[sel, "is_month_to_date"] = 1 if partial else 0
        log(f"    {last:%Y-%m}: {ndays} of {dim} days"
            + ("  (month-to-date)" if partial else
               "  (complete month, Hargreaves PET)"))

        sp = PM.partial_spei(rn, tx, tn, lat, last.month, last.year, ndays,
                             loglogistic_spei, scales=(1, 4, 12))
        for s, vals in sp.items():
            col = f"spei_era5_{s}"
            if col not in p.columns:
                continue
            p.loc[sel, col] = p.loc[sel, "district_id"].map(vals)
            log(f"      {col:16s} {len(vals):>4} districts")
        if 4 in sp and "spei_harg_4" in p.columns:
            # the partial index IS the Hargreaves formulation, so both columns
            # carry the same value rather than one staying blank
            p.loc[sel, "spei_harg_4"] = p.loc[sel, "district_id"].map(sp[4])

        th = PM.partial_thermal(tx, tn, last.month, last.year, ndays)
        for col, vals in th.items():
            if col not in p.columns:
                p[col] = np.nan
            # some panel columns arrive as float32; assigning float64 into
            # them raises LossySetitemError on current pandas
            if p[col].dtype != "float64":
                p[col] = p[col].astype("float64")
            p.loc[sel, col] = p.loc[sel, "district_id"].map(vals)
            log(f"      {col:16s} {len(vals):>4} districts")

        #   THE ANOMALY FOR A PART-MONTH USES THE DAY-MATCHED NORMAL.
        #   sdd_normal is a full-month climatology. Subtracting it from a
        #   four-day sum would report a large negative heat anomaly for the
        #   simple reason that four days is not thirty-one -- the same
        #   day-matching error the rainfall departure had, in a different
        #   variable. partial_thermal already returns a normal computed over
        #   the SAME first-D-days window in every year, so the anomaly is
        #   rebuilt from that instead.
        for col in ("sdd", "gdd_kharif", "gdd_rabi"):
            tdn, an = f"{col}_todate_normal", f"{col}_anom"
            if tdn in p.columns and an in p.columns:
                if p[an].dtype != "float64":
                    p[an] = p[an].astype("float64")
                have = sel & p[tdn].notna()
                p.loc[have, an] = (p.loc[have, col].to_numpy()
                                   - p.loc[have, tdn].to_numpy())
                log(f"      {an:16s} rebuilt on the day-matched normal "
                    f"({int(have.sum())} districts)")
    return p


def mask_incomplete_months(p):
    r"""Withhold accumulating features for months their sources do not cover.

    THE FAILURE THIS PREVENTS
      The sources end on different dates.  In July 2026 IMD rainfall ran to the
      24th (23 days) while ERA5-Land ET stopped on the 15th (15 days).  The SPEI
      water balance is `rain - PET`, so it subtracted half a month of
      evaporative demand from three-quarters of a month of rain:

          2026-07    rain 241.5 (23 d)  - PET 104.7 (15 d)  =  +137.2 mm
          1991-2025  rain 305.4 (31 d)  - PET 192.1 (31 d)  =  +113.9 mm

      The running month therefore scored WETTER than a normal July in a season
      running 13% below normal, and SPEI-12 displayed +0.40 while the rainfall
      layer showed a deficit -- two panels of the same dashboard contradicting
      each other.  Degree-day sums had the same defect in milder form.

    WHY WITHHOLD RATHER THAN PRO-RATE
      Scaling a part-month up to a full-month equivalent assumes the remaining
      days behave climatologically, which is exactly the assumption a drought
      index exists to test.  Operational SPI/SPEI products publish complete
      months only, and that is what is done here.

    NOT AFFECTED, and deliberately left live:
      pct_departure   day-matched against a normal covering the same days
      mai             AET/PET, a ratio of two sums over the SAME 15 days
      swvl*, tmax_c, tmin_c, *_anom   monthly means, not accumulations
    """
    flags = [c for c in ("month_complete", "et_days_complete",
                         "temp_days_complete") if c in p.columns]
    for c in flags:
        p[c] = p[c].fillna(0).astype(int)
    missing = [c for c in ("month_complete", "et_days_complete",
                           "temp_days_complete") if c not in p.columns]
    if missing:
        log(f"  ! completeness flags absent ({', '.join(missing)}) — "
            f"re-run step 01; accumulating features NOT masked")
        return p

    if SKIP_MASK:
        return p
    n, touched, per_col = 0, set(), {}
    for col, req in NEEDS_COMPLETE.items():
        if col not in p.columns:
            continue
        ok = np.ones(len(p), bool)
        for r in req:
            ok &= p[r].to_numpy() == 1
        hit = p[col].notna().to_numpy() & ~ok
        if hit.any():
            touched |= set(p.loc[hit, "date"].unique())
            per_col[col] = int(hit.sum())
            p.loc[hit, col] = np.nan
            n += int(hit.sum())
    if n:
        # Report only months where a value was ACTUALLY removed.  Counting every
        # month merely flagged incomplete listed 1971-1980 as well, which reads
        # as if a decade had been destroyed -- in fact ERA5 PET simply does not
        # exist before 1981, so SPEI was already empty there and nothing was
        # lost.  GDD, SDD and Hargreaves PET remain 100% present for those years.
        ms = sorted(pd.Timestamp(b).strftime("%Y-%m") for b in touched)
        log(f"  withheld {n:,} accumulating values in "
            f"{len(ms)} month(s): {', '.join(ms)}")
        log("    " + ", ".join(f"{k} {v:,}" for k, v in sorted(per_col.items())))
        log("    (pct_departure and MAI stay live — day-matched, and a ratio "
            "over a common window, respectively)")
    return p


def main():
    log("=" * 70)
    log("STEP 3 — deterministic features (SPEI, Hargreaves-SPEI, GDD, SDD)")
    log("=" * 70)
    p = pd.read_csv(OUTD / "panel_with_enso_features_lgd.csv",
                    parse_dates=["date"], low_memory=False)

    # ---- Hargreaves PET + water balances ----------------------------------
    lat = district_latitudes()
    p["lat"] = p["district_id"].map(lat)
    dim = p["date"].dt.days_in_month
    ra = np.array([extraterrestrial_ra(la, mo) if np.isfinite(la) else np.nan
                   for la, mo in zip(p["lat"], p["month"])])
    trange = (p["tmax_c"] - p["tmin_c"]).clip(lower=0)
    tmean = (p["tmax_c"] + p["tmin_c"]) / 2
    p["pet_hargreaves"] = (0.0023 * ra * np.sqrt(trange)
                           * (tmean + 17.8)).clip(lower=0) * dim
    p["wb_era5"] = p["rain_mm"] - p["pet_mm"]
    p["wb_harg"] = p["rain_mm"] - p["pet_hargreaves"]

    log("  SPEI from ERA5 water balance:")
    p = add_spei(p, "wb_era5", "spei_era5", scales=(1, 4, 12))
    log("  SPEI from Hargreaves water balance:")
    p = add_spei(p, "wb_harg", "spei_harg", scales=(4,))

    # ---- GDD / SDD from daily temperature ---------------------------------
    log("  thermal indices from daily Tmax/Tmin ...")
    th, tdays = monthly_thermal()
    p = p.merge(th, on=["date", "district_id"], how="left")
    p = p.merge(tdays, on="date", how="left")

    p = mask_incomplete_months(p)
    p = fill_partial_month(p)

    p = p.sort_values(["district_id", "date"])
    p.to_csv(OUTD / "features_lgd.csv", index=False)
    log(f"\n  features_lgd.csv: {len(p):,} rows x {p.shape[1]} cols")
    r26 = p[p["year"] == 2026]
    for c in ["spei_era5_1", "spei_era5_4", "spei_harg_4", "gdd_kharif",
              "sdd", "mai"]:
        log(f"    {c:14s} {100*r26[c].notna().mean():5.1f}% present (2026)")
    log("  wrote features_lgd.csv")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
