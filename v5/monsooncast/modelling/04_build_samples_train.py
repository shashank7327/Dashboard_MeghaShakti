r"""
v5/monsooncast/modelling/04_build_samples_train.py  —  STEP 4: build daily issue-date
samples and retrain the 7/14-day rainfall-departure models on the IMD/LGD
feature set (CHIRPS and TerraClimate fully replaced by IMD).

TARGET   dep_H = 100 * (sum rain[t+1..t+H] - sum normal[t+1..t+H]) / sum normal,
         the forward district rainfall departure over the next H days.
FEATURES daily antecedents (rain/departure over 7/14/30/60/90 d, wet-spell,
         extremes, dry-spell, seasonality) + the monthly covariate block
         (SPEI, MAI, GDD/SDD, anomalies, soil moisture, ENSO/IOD, ENSO-phase
         signature) for the issue month + district identity.
SPLIT    chronological: train <=2016, val 2017-2019, test 2020-2026.
MODEL    HistGradientBoosting (handles NaN natively); RandomForest as a check.

OUTPUTS -> v5/data_lgd/
  model_dep7_lgd.pkl, model_dep14_lgd.pkl
  skill_lgd.json, per_district_calib_lgd.csv, feature_importance_lgd.csv

Run:  py -3.13 -X utf8 "v5/monsooncast/modelling/04_build_samples_train.py"
"""
import json
import pathlib
import pickle
import sys

import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
V5 = HERE.parents[1]
IMD = V5.parent / "IMD_Data"
OUTD = V5 / "data_lgd"
sys.path.insert(0, str(V5))
from common_v5 import log  # noqa

HORIZONS = [7, 14]
ANT = [7, 14, 30, 60, 90]
STRIDE = 5                          # sample an issue date every 5 days
MIN_NORMAL = 5.0
TRAIN_END, VAL_END = 2016, 2019


def imd_anchor():
    """Scale factor that puts our normals on IMD's published LPA (step 01)."""
    try:
        r = json.loads((OUTD / "merge_report_lgd.json").read_text())
        return float(r.get("imd_lpa_anchor", 1.0))
    except Exception:
        return 1.0


def daily_normal(rain):
    """DOY climatology (1981-2025), +/-2 day smoothed, per district, scaled to
    IMD's published LPA so the target is on the same scale as the features."""
    base = rain[(rain.index.year >= 1971) & (rain.index.year <= 2020)]
    doy = base.index.dayofyear
    per = base.groupby(doy).mean().reindex(range(1, 367)).to_numpy()
    sm = np.nanmean(np.stack([np.roll(per, k, axis=0)
                              for k in range(-2, 3)]), axis=0)
    nrm = pd.DataFrame(sm * imd_anchor(), index=range(1, 367),
                       columns=rain.columns)
    idx_doy = rain.index.dayofyear
    return pd.DataFrame(nrm.reindex(idx_doy).to_numpy(), index=rain.index,
                        columns=rain.columns)


def fwd_sum(mat, H):
    """sum of the next H days (t+1..t+H), per column, as a DataFrame."""
    rev = mat.iloc[::-1]
    s = rev.rolling(H, min_periods=H).sum().shift(1).iloc[::-1]
    return s


def back_sum(mat, H):
    return mat.rolling(H, min_periods=H).sum()


def main():
    log("=" * 70)
    log("STEP 4 — daily samples + retrain 7/14-day models on IMD/LGD features")
    log("=" * 70)
    rain = pd.read_pickle(OUTD / "daily_rain_lgd.pkl")
    rain.columns = rain.columns.astype(int)
    nrm = daily_normal(rain)
    log(f"  daily rain {rain.shape}, normals built")

    # ---- forward targets --------------------------------------------------
    tgt = {}
    for H in HORIZONS:
        fr, fn = fwd_sum(rain, H), fwd_sum(nrm, H)
        with np.errstate(invalid="ignore", divide="ignore"):
            dep = np.where(fn >= MIN_NORMAL, (fr - fn) / fn * 100.0, np.nan)
        tgt[H] = pd.DataFrame(dep, index=rain.index, columns=rain.columns)

    # ---- daily antecedent features ---------------------------------------
    feats = {}
    for H in ANT:
        ar, an = back_sum(rain, H), back_sum(nrm, H)
        feats[f"ant_rain_{H}"] = ar
        with np.errstate(invalid="ignore", divide="ignore"):
            feats[f"ant_dep_{H}"] = pd.DataFrame(
                np.where(an >= MIN_NORMAL, (ar - an) / an * 100.0, np.nan),
                index=rain.index, columns=rain.columns)
    wet = (rain > 1.0).astype(float)
    feats["wet30"] = back_sum(wet, 30)
    p95 = rain.where(rain > 1.0).quantile(0.95)
    feats["extreme30"] = back_sum((rain > p95).astype(float), 30)
    # dry-spell length (days since last wet day), with missing data treated
    # as missing rather than as dry: `rain > 1.0` is False for NaN, so a
    # district the IMD land grid does not cover would otherwise accumulate a
    # dry spell of the whole record. The audit found seven doing exactly that
    # at 20,304 days -- the Andaman, Nicobar and Lakshadweep islands among
    # them. See the fuller note in 08_model_bakeoff.py.
    arr = rain.to_numpy()
    wet_d = arr > 1.0
    miss = ~np.isfinite(arr)
    dsw = np.zeros(arr.shape)
    for c in range(arr.shape[1]):
        run = 0
        for r in range(arr.shape[0]):
            if miss[r, c]:
                run = 0
                dsw[r, c] = np.nan
            else:
                run = 0 if wet_d[r, c] else run + 1
                dsw[r, c] = run
    feats["dsw"] = pd.DataFrame(dsw, index=rain.index, columns=rain.columns)
    log(f"  antecedent + extreme features built")

    # ---- sample every STRIDE days, melt to long --------------------------
    rows = rain.index
    keep = rows[(rows.dayofyear % STRIDE == 0)
                & (rows.year >= 1981)]
    def m(df):
        return (df.loc[keep].reset_index()
                .melt(id_vars="date", var_name="district_id", value_name="v"))
    base = m(rain.loc[keep] * np.nan)          # skeleton (date, district_id)
    base = base.drop(columns="v")
    base["district_id"] = base["district_id"].astype(int)
    for name, df in feats.items():
        base[name] = m(df)["v"].to_numpy()
    base["doy"] = base["date"].dt.dayofyear
    base["doy_sin"] = np.sin(2 * np.pi * base["doy"] / 365.25)
    base["doy_cos"] = np.cos(2 * np.pi * base["doy"] / 365.25)
    for H in HORIZONS:
        base[f"dep_{H}"] = m(tgt[H])["v"].to_numpy()
    base["year"] = base["date"].dt.year
    base["month"] = base["date"].dt.month
    log(f"  samples: {len(base):,}")

    # ---- join monthly covariates -----------------------------------------
    feat = pd.read_csv(OUTD / "features_lgd.csv", low_memory=False)
    mcols = ["spei_era5_1", "spei_era5_4", "spei_era5_12", "spei_harg_4",
             "mai", "sdd", "gdd_kharif", "gdd_rabi", "tmax_anom", "tmin_anom",
             "swvl1", "swvl2", "swvl3", "swvl4", "aet_mm", "pet_mm",
             "oni", "nino34_anom", "iod_dmi", "iod_z", "enso_iod_interact",
             "enso_warm", "enso_cold", "enso_developing", "enso_decaying",
             "oni_mam", "pct_departure_signature", "mai_signature",
             "tmax_anom_signature", "is_jk_ladakh"]
    mcols = [c for c in mcols if c in feat.columns]
    # Monthly features come from the PREVIOUS month, not the containing one:
    # joining on (year, month) hands a mid-month issue date the whole of that
    # month's SPEI/MAI/degree-days, which are not knowable until the month ends.
    # Same fix as 08_model_bakeoff.py — see the note there.
    fmo = (base["date"].values.astype("datetime64[M]")
           - np.timedelta64(1, "M"))
    base["fyear"] = (fmo.astype("datetime64[Y]").astype(int) + 1970).astype(np.int16)
    base["fmonth"] = (fmo.astype("datetime64[M]").astype(int) % 12 + 1).astype(np.int8)
    base = base.merge(
        feat[["year", "month", "district_id"] + mcols].rename(
            columns={"year": "fyear", "month": "fmonth"}),
        on=["fyear", "fmonth", "district_id"], how="left")

    XCOLS = ([f"ant_rain_{h}" for h in ANT] + [f"ant_dep_{h}" for h in ANT]
             + ["wet30", "extreme30", "dsw", "doy_sin", "doy_cos"]
             + mcols + ["district_id"])
    from sklearn.ensemble import HistGradientBoostingRegressor

    skill = {}
    calib_frames = []
    imp_frames = []
    for H in HORIZONS:
        d = base.dropna(subset=[f"dep_{H}"])
        tr = d[d.year <= TRAIN_END]
        te = d[(d.year >= 2020)]
        Xtr, ytr = tr[XCOLS], tr[f"dep_{H}"]
        Xte, yte = te[XCOLS], te[f"dep_{H}"]
        log(f"  H={H}: train {len(tr):,}  test {len(te):,}")
        mdl = HistGradientBoostingRegressor(
            max_iter=500, learning_rate=0.05, max_leaf_nodes=63,
            l2_regularization=1.0, early_stopping=True,
            validation_fraction=0.12, random_state=0)
        mdl.fit(Xtr, ytr)
        pred = mdl.predict(Xte)
        rmse = float(np.sqrt(np.mean((pred - yte) ** 2)))
        clim_rmse = float(np.sqrt(np.mean(yte ** 2)))
        mse_skill = 1 - (rmse ** 2) / (clim_rmse ** 2)
        corr = float(np.corrcoef(pred, yte)[0, 1])
        skill[f"dep_{H}"] = {"rmse": round(rmse, 1),
                             "clim_rmse": round(clim_rmse, 1),
                             "mse_skill": round(mse_skill, 3),
                             "corr": round(corr, 3),
                             "n_test": int(len(te))}
        log(f"    RMSE {rmse:.1f}  clim {clim_rmse:.1f}  "
            f"skill {mse_skill:+.3f}  corr {corr:.3f}")
        with open(OUTD / f"model_dep{H}_lgd.pkl", "wb") as f:
            pickle.dump({"model": mdl, "cols": XCOLS}, f)
        # per-district calibration on test
        te = te.assign(pred=pred)
        cal = (te.groupby("district_id")
               .apply(lambda g: pd.Series({
                   "b": np.polyfit(g["pred"], g[f"dep_{H}"], 1)[0]
                        if len(g) > 20 and g["pred"].std() > 1 else 1.0,
                   "n": len(g)}), include_groups=False).reset_index())
        cal["horizon"] = H
        calib_frames.append(cal)

    pd.concat(calib_frames).to_csv(OUTD / "per_district_calib_lgd.csv",
                                   index=False)
    (OUTD / "skill_lgd.json").write_text(json.dumps(skill, indent=1),
                                         encoding="utf-8")
    log(f"\n  wrote models, skill_lgd.json, per_district_calib_lgd.csv")
    log(f"  SKILL vs climatology: " + ", ".join(
        f"{k} {v['mse_skill']:+.3f} (r={v['corr']})" for k, v in skill.items()))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
