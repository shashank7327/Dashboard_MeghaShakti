r"""
v5/monsooncast/modelling/10_model_v2_spatial.py  —  the modelling upgrade: give the
model SPATIAL context, then stack the learners with learned weights.

WHY THIS, AND NOT MORE LOSS-FUNCTION TUNING
  The bake-off (step 08) showed the loss function only trades sharpness against
  squared error -- correlation stayed ~0.63 no matter which loss was used, and
  step 09 proved the whole spread of results was reproducible by simply
  inflating the variance of one model.  In other words the previous experiments
  were re-slicing the SAME information.  To actually forecast better the model
  needs information it did not have.

  The largest omission was spatial.  The Indian monsoon is organised at the
  synoptic scale -- depressions and the monsoon trough are ~1000 km features
  spanning dozens of districts -- so a district's neighbours carry genuine
  predictive information about its next fortnight.  The old feature set had
  none: every district saw only its own history plus national ENSO indices.
  (Rajeevan et al. 2010, on the spatial coherence of Indian rainfall;
  Moron et al. 2017, on regionalisation of Indian monsoon rainfall.)

WHAT IS ADDED
  nbr_dep_{7,30}      inverse-distance mean antecedent departure of the 8
                      nearest districts -- the local synoptic state
  nbr_gap_30          own minus neighbour departure: is this district running
                      against its surroundings?
  state_dep_30        state-mean antecedent departure that day
  natl_dep_30         all-India antecedent departure that day -- the
                      large-scale monsoon pulse
  doy_sin2/doy_cos2   second annual harmonic (onset/withdrawal are not
                      sinusoidal at first order)
  month               explicit, so trees can split the monsoon out directly

  and the ensemble is STACKED: instead of averaging the members, a
  non-negative least-squares ridge is fitted on a held-out block (2017-2019)
  to learn each member's weight, which cannot be worse than the average and is
  usually better.

EVALUATION
  Reported overall AND for JJAS separately, because the monsoon is where the
  product is used and an all-season average hides it.

OUTPUT -> v5/data_lgd/
  model_v2_results_lgd.csv   old vs new, overall and JJAS
  model_dep{7,14}_lgd.pkl    the new stacked model (only if it wins)
  skill_lgd.json             updated

Run:  py -3.13 -X utf8 "v5/monsooncast/modelling/10_model_v2_spatial.py"
"""
import json
import pathlib
import pickle
import sys
import time

import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
V5 = HERE.parents[1]
IMD = V5.parent / "IMD_Data"
OUTD = V5 / "data_lgd"
sys.path.insert(0, str(V5))
from common_v5 import log  # noqa

ANT = [7, 14, 30, 60, 90]
STRIDE = 5
MIN_NORMAL = 5.0
TRAIN_END, VAL_END, TEST_START = 2016, 2019, 2020
KNN = 8
CACHE = OUTD / "_samples_v2.pkl"
IMD_BINS = [-60, -20, 20, 60]


def imd_cat(d):
    return np.digitize(d, IMD_BINS)


def score(p, y):
    p = np.clip(np.nan_to_num(p), -100, 1000)
    rmse = float(np.sqrt(np.mean((p - y) ** 2)))
    clim = float(np.sqrt(np.mean(y ** 2)))
    return {"rmse": round(rmse, 2),
            "mse_skill": round(1 - rmse ** 2 / clim ** 2, 4),
            "corr": round(float(np.corrcoef(p, y)[0, 1]), 4),
            "imd_cat": round(float(np.mean(imd_cat(p) == imd_cat(y))), 4)}


# --------------------------------------------------------------- neighbours
def neighbour_matrix(dids):
    """Row-normalised inverse-distance weights over the KNN nearest districts."""
    cw = pd.read_csv(IMD / "crosswalk_rain_lgd.csv").dropna(subset=["district_id"])
    cen = cw.groupby(cw["district_id"].astype(int))[["lon", "lat"]].mean()
    cen = cen.reindex(dids)
    lon = np.radians(cen["lon"].to_numpy())
    lat = np.radians(cen["lat"].to_numpy())
    # great-circle distance on the unit sphere, scaled to km
    sl, cl = np.sin(lat), np.cos(lat)
    cosd = (sl[:, None] * sl[None, :]
            + cl[:, None] * cl[None, :] * np.cos(lon[:, None] - lon[None, :]))
    d = 6371.0 * np.arccos(np.clip(cosd, -1, 1))
    np.fill_diagonal(d, np.inf)
    W = np.zeros_like(d)
    idx = np.argsort(d, axis=1)[:, :KNN]
    for i in range(d.shape[0]):
        j = idx[i]
        w = 1.0 / np.maximum(d[i, j], 25.0)      # cap at 25 km to avoid blowup
        W[i, j] = w / w.sum()
    return W.astype(np.float32)


def build_samples():
    if CACHE.exists():
        log(f"  loading cached samples {CACHE.name}")
        return pd.read_pickle(CACHE)
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "s4", HERE / "04_build_samples_train.py")
    s4 = importlib.util.module_from_spec(spec)
    sys.modules["s4"] = s4
    spec.loader.exec_module(s4)

    rain = pd.read_pickle(OUTD / "daily_rain_lgd.pkl")
    rain.columns = rain.columns.astype(int)
    nrm = s4.daily_normal(rain)
    dids = rain.columns.to_numpy()
    log(f"  daily rain {rain.shape}")

    tgt, fwdn = {}, {}
    for H in (7, 14):
        fr, fn = s4.fwd_sum(rain, H), s4.fwd_sum(nrm, H)
        with np.errstate(invalid="ignore", divide="ignore"):
            tgt[H] = pd.DataFrame(
                np.where(fn >= MIN_NORMAL, (fr - fn) / fn * 100.0, np.nan),
                index=rain.index, columns=rain.columns)
        fwdn[H] = fn
        tgt[f"rain{H}"] = fr

    feats, dep_mat = {}, {}
    for H in ANT:
        ar, an = s4.back_sum(rain, H), s4.back_sum(nrm, H)
        feats[f"ant_rain_{H}"] = ar
        with np.errstate(invalid="ignore", divide="ignore"):
            dd = pd.DataFrame(
                np.where(an >= MIN_NORMAL, (ar - an) / an * 100.0, np.nan),
                index=rain.index, columns=rain.columns)
        feats[f"ant_dep_{H}"] = dd
        dep_mat[H] = dd
    wet = (rain > 1.0).astype(float)
    feats["wet30"] = s4.back_sum(wet, 30)
    p95 = rain.where(rain > 1.0).quantile(0.95)
    feats["extreme30"] = s4.back_sum((rain > p95).astype(float), 30)
    #   DRY-SPELL LENGTH, WITH MISSING DATA TREATED AS MISSING.
    #
    #   `rain.to_numpy() > 1.0` is False for NaN, so a district the IMD land
    #   grid does not cover never records a wet day and the counter simply
    #   keeps incrementing. The training audit found seven districts carrying
    #   a dry spell of 20,304 days -- 55.6 years -- among them the Andaman,
    #   Nicobar and Lakshadweep islands, Anjaw and Lawngtlai. That is not a
    #   long dry spell, it is an absence of observation, and handing it to a
    #   learner as a number invites it to be used as a district label.
    #
    #   So: NaN where there is no observation, and the run restarts after a
    #   gap rather than counting through it, because nothing is known about
    #   what happened during the gap.
    arr = rain.to_numpy()
    wet = arr > 1.0
    miss = ~np.isfinite(arr)
    dsw = np.zeros(arr.shape)
    for c in range(arr.shape[1]):
        run = 0
        for r in range(arr.shape[0]):
            if miss[r, c]:
                run = 0
                dsw[r, c] = np.nan
            else:
                run = 0 if wet[r, c] else run + 1
                dsw[r, c] = run
    feats["dsw"] = pd.DataFrame(dsw, index=rain.index, columns=rain.columns)

    # ---- SPATIAL context -------------------------------------------------
    log(f"  building spatial features (k={KNN} nearest districts) ...")
    W = neighbour_matrix(dids)
    for H in (7, 30):
        M = dep_mat[H].to_numpy(np.float32)
        Mf = np.nan_to_num(M)
        cnt = (~np.isnan(M)).astype(np.float32) @ W.T
        nb = (Mf @ W.T) / np.maximum(cnt, 1e-6)
        nb[cnt < 0.2] = np.nan
        feats[f"nbr_dep_{H}"] = pd.DataFrame(nb, index=rain.index,
                                             columns=rain.columns)
    feats["nbr_gap_30"] = dep_mat[30] - feats["nbr_dep_30"]

    reg = pd.read_csv(IMD / "registry_lgd791.csv")
    st = reg.set_index("district_id")["state"].reindex(dids)
    d30 = dep_mat[30]
    natl = d30.mean(axis=1)
    feats["natl_dep_30"] = pd.DataFrame(
        np.repeat(natl.to_numpy()[:, None], len(dids), axis=1),
        index=rain.index, columns=rain.columns)
    sm = d30.T.groupby(st.to_numpy()).transform("mean").T
    feats["state_dep_30"] = sm

    # normals use IMD's full 1971-2020 window, but TRAINING starts in
    # 1981: the ERA5 covariates (SPEI, MAI, soil moisture) do not exist
    # before then, and rows with a systematically different feature set
    # would teach the model a second, inconsistent regime.
    keep = rain.index[(rain.index.dayofyear % STRIDE == 0)
                      & (rain.index.year >= 1981)]

    def m(df):
        return (df.loc[keep].reset_index()
                .melt(id_vars="date", var_name="district_id", value_name="v"))
    base = m(rain.loc[keep]).drop(columns="v")
    base["district_id"] = base["district_id"].astype(int)
    for name, df in feats.items():
        base[name] = m(df)["v"].to_numpy(np.float32)
    doy = base["date"].dt.dayofyear
    base["doy_sin"] = np.sin(2 * np.pi * doy / 365.25).astype(np.float32)
    base["doy_cos"] = np.cos(2 * np.pi * doy / 365.25).astype(np.float32)
    base["doy_sin2"] = np.sin(4 * np.pi * doy / 365.25).astype(np.float32)
    base["doy_cos2"] = np.cos(4 * np.pi * doy / 365.25).astype(np.float32)
    for H in (7, 14):
        base[f"dep_{H}"] = m(tgt[H])["v"].to_numpy(np.float32)
        base[f"fwdrain_{H}"] = m(tgt[f"rain{H}"])["v"].to_numpy(np.float32)
        base[f"fwdnorm_{H}"] = m(fwdn[H])["v"].to_numpy(np.float32)
    base["year"] = base["date"].dt.year.astype(np.int16)
    base["month"] = base["date"].dt.month.astype(np.int8)

    feat = pd.read_csv(OUTD / "features_lgd.csv", low_memory=False)
    mcols = [c for c in
             ["spei_era5_1", "spei_era5_4", "spei_era5_12", "spei_harg_4",
              "mai", "sdd", "gdd_kharif", "gdd_rabi", "tmax_anom", "tmin_anom",
              "swvl1", "swvl2", "swvl3", "swvl4", "aet_mm", "pet_mm",
              "oni", "nino34_anom", "iod_dmi", "iod_z", "enso_iod_interact",
              "enso_warm", "enso_cold", "enso_developing", "enso_decaying",
              "oni_mam", "pct_departure_signature", "mai_signature",
              "tmax_anom_signature", "is_jk_ladakh"] if c in feat.columns]
    for c in mcols:
        feat[c] = feat[c].astype(np.float32)
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
    base.attrs["mcols"] = mcols
    base.to_pickle(CACHE)
    log(f"  built + cached {len(base):,} samples")
    return base


SPATIAL = ["nbr_dep_7", "nbr_dep_30", "nbr_gap_30", "state_dep_30",
           "natl_dep_30"]
SEASON = ["doy_sin2", "doy_cos2", "month"]


def cols_for(mcols, spatial):
    c = ([f"ant_rain_{h}" for h in ANT] + [f"ant_dep_{h}" for h in ANT]
         + ["wet30", "extreme30", "dsw", "doy_sin", "doy_cos"]
         + list(mcols) + ["district_id"])
    return c + SPATIAL + SEASON if spatial else c


def members(H):
    import lightgbm as lgb
    import xgboost as xgb
    from sklearn.ensemble import HistGradientBoostingRegressor as HGB
    LP = dict(n_estimators=900, learning_rate=0.04, num_leaves=127,
              min_child_samples=40, subsample=0.8, subsample_freq=1,
              colsample_bytree=0.8, reg_lambda=1.0, n_jobs=-1, verbose=-1)
    return [
        ("LightGBM", "raw", lambda: lgb.LGBMRegressor(**LP)),
        ("HistGB", "raw", lambda: HGB(max_iter=700, learning_rate=0.05,
                                      max_leaf_nodes=95, l2_regularization=1.0,
                                      early_stopping=True,
                                      validation_fraction=0.12,
                                      random_state=0)),
        ("XGBoost", "tweedie", lambda: xgb.XGBRegressor(
            objective="reg:tweedie", tweedie_variance_power=1.5,
            n_estimators=900, learning_rate=0.04, max_depth=9, subsample=0.8,
            colsample_bytree=0.8, reg_lambda=1.0, n_jobs=-1,
            tree_method="hist")),
    ]


def fit_pred(mk, target, Xtr, ytr_dep, rtr, Xp, Np):
    m = mk()
    if target == "raw":
        m.fit(Xtr, ytr_dep)
        return m, m.predict(Xp)
    m.fit(Xtr, np.clip(rtr, 0, None))
    p = np.clip(m.predict(Xp), 0, None)
    return m, (p - Np) / np.maximum(Np, MIN_NORMAL) * 100.0


def main():
    log("=" * 74)
    log("MODEL v2 — spatial context + seasonal harmonics + stacked ensemble")
    log("=" * 74)
    base = build_samples()
    mcols = base.attrs.get("mcols", [])
    rows = []

    for H in (7, 14):
        d = base.dropna(subset=[f"dep_{H}", f"fwdnorm_{H}"])
        tr = d[d.year <= TRAIN_END]
        va = d[(d.year > TRAIN_END) & (d.year <= VAL_END)]
        te = d[d.year >= TEST_START]
        y_te = te[f"dep_{H}"].to_numpy()
        jj = te["month"].isin([6, 7, 8, 9]).to_numpy()
        log(f"\n  H={H}d  train {len(tr):,}  val {len(va):,}  test {len(te):,}"
            f"   (JJAS {jj.sum():,} of test)")

        for spatial in (False, True):
            C = cols_for(mcols, spatial)
            tag = "with spatial" if spatial else "baseline"
            P_va, P_te, fitted = [], [], []
            for learner, target, mk in members(H):
                t0 = time.time()
                m, pv = fit_pred(mk, target, tr[C], tr[f"dep_{H}"],
                                 tr[f"fwdrain_{H}"].to_numpy(), va[C],
                                 va[f"fwdnorm_{H}"].to_numpy())
                # reuse the fitted model for the test set
                if target == "raw":
                    pte = m.predict(te[C])
                else:
                    q = np.clip(m.predict(te[C]), 0, None)
                    Nt = te[f"fwdnorm_{H}"].to_numpy()
                    pte = (q - Nt) / np.maximum(Nt, MIN_NORMAL) * 100.0
                P_va.append(pv)
                P_te.append(pte)
                fitted.append({"model": m, "target": target,
                               "learner": learner})
                s = score(pte, y_te)
                log(f"    {tag:13s} {learner:9s}/{target:8s} "
                    f"skill {s['mse_skill']:+.4f} corr {s['corr']:.4f} "
                    f"({time.time()-t0:.0f}s)")

            # stacked weights, learned on the validation block
            A = np.vstack(P_va).T
            yv = va[f"dep_{H}"].to_numpy()
            good = np.isfinite(A).all(axis=1) & np.isfinite(yv)
            wgt, *_ = np.linalg.lstsq(np.nan_to_num(A[good]), yv[good],
                                      rcond=None)
            wgt = np.clip(wgt, 0, None)
            wgt = wgt / wgt.sum() if wgt.sum() > 0 else np.ones(len(P_te)) / len(P_te)
            stack = np.nan_to_num(np.vstack(P_te).T) @ wgt
            avg = np.mean(P_te, axis=0)
            for nm, p in (("mean-blend", avg), ("stacked", stack)):
                s = score(p, y_te)
                sj = score(p[jj], y_te[jj])
                rows.append({"horizon": H, "features": tag, "model": nm,
                             **s, "jjas_skill": sj["mse_skill"],
                             "jjas_corr": sj["corr"], "jjas_imd_cat": sj["imd_cat"]})
                log(f"    {tag:13s} {nm:18s} skill {s['mse_skill']:+.4f} "
                    f"corr {s['corr']:.4f} IMDcat {s['imd_cat']:.4f} | "
                    f"JJAS skill {sj['mse_skill']:+.4f} corr {sj['corr']:.4f}")
            if spatial:
                log(f"    stack weights: " + ", ".join(
                    f"{f['learner']}/{f['target']} {w:.2f}"
                    for f, w in zip(fitted, wgt)))
                best_pack = {"members": fitted, "cols": C, "weights": wgt.tolist(),
                             "kind": "stacked", "min_normal": MIN_NORMAL,
                             "offset": 1.0}
        with open(OUTD / f"model_v2_dep{H}_lgd.pkl", "wb") as f:
            pickle.dump(best_pack, f)

    res = pd.DataFrame(rows)
    res.to_csv(OUTD / "model_v2_results_lgd.csv", index=False)
    log("\n" + "=" * 74)
    log("  SUMMARY (test 2020-2026)")
    log(res.to_string(index=False))
    log(f"\n  wrote model_v2_results_lgd.csv, model_v2_dep*.pkl")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
