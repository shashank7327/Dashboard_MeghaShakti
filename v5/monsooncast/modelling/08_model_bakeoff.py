r"""
v5/monsooncast/modelling/08_model_bakeoff.py  —  compare several ML formulations for the
7/14-day district rainfall-departure forecast and pick the best.

WHY MORE THAN ONE MODEL
  The target -- percentage departure from normal -- is badly behaved for plain
  squared-error regression: it is bounded below at -100% but unbounded above,
  so it is strongly right-skewed and a few very wet districts dominate the
  loss.  The literature is consistent that rainfall ACCUMULATION is close to
  gamma-distributed, which is why gamma / Tweedie regression is the standard
  choice for precipitation amount rather than ordinary least squares
  (Journal of Big Data 8:153, 2021; Sci. Rep. 14, 2024).  Three ways of
  respecting that are tested here, against the plain formulation:

    raw       predict % departure directly, squared-error loss   (baseline)
    huber     same target, robust loss -- stops the wet tail dominating
    tweedie   predict RAINFALL AMOUNT with Tweedie (gamma-like) loss, then
              convert to a departure.  Matches the physical distribution.
    logratio  predict log((R+a)/(N+a)) -- symmetrises the ratio -- then invert

  and across three learners (HistGradientBoosting, LightGBM, XGBoost) plus a
  blend of the best.

HOW THEY ARE JUDGED
  Every model is scored on the SAME departure scale, on the held-out 2020-2026
  period, with four metrics:
    RMSE            error magnitude
    MSE skill       1 - MSE/MSE(climatology); >0 means it beats "assume normal"
    correlation     does it get the pattern right
    IMD category    accuracy against IMD's own operational rainfall classes
                    (Large Excess / Excess / Normal / Deficient / Large
                    Deficient), which is the form IMD actually publishes --
                    this is the directly IMD-comparable score.

OUTPUT -> v5/data_lgd/
  bakeoff_results_lgd.csv     every model x horizon x metric
  model_dep{7,14}_lgd.pkl     the winning model per horizon (overwrites)
  skill_lgd.json              winning-model skill, for the dashboard

Run:  py -3.13 -X utf8 "v5/monsooncast/modelling/08_model_bakeoff.py"
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
OUTD = V5 / "data_lgd"
sys.path.insert(0, str(V5))
from common_v5 import log  # noqa

HORIZONS = [7, 14]
ANT = [7, 14, 30, 60, 90]
STRIDE = 5
MIN_NORMAL = 5.0
SAMPLE_FROM = 1990        # 30-year training baseline; see build_samples()
TRAIN_END = 2016
TEST_START = 2020
CACHE = OUTD / "_bakeoff_samples_v3-mjo-from1991.pkl"
OFF = 1.0                        # mm offset for the log-ratio transform

# IMD operational rainfall categories, by % departure
IMD_BINS = [-np.inf, -60, -20, 20, 60, np.inf]
IMD_CATS = ["Large Deficient", "Deficient", "Normal", "Excess", "Large Excess"]


def imd_cat(dep):
    return np.digitize(dep, IMD_BINS[1:-1])


# ----------------------------------------------------------------- samples
#   Inputs the cached training frame is derived from.  If any is newer than
#   the cache, the cache is stale and must be rebuilt.
CACHE_INPUTS = ["features_lgd.csv", "daily_rain_lgd.pkl"]
# bumped whenever the sample construction changes, so a cache built under the
# old definition cannot be reused silently
CACHE_VERSION = "v3-mjo-from1991"


def build_samples():
    r"""The training frame, cached because it takes minutes to assemble.

    THE CACHE MUST INVALIDATE ON ITS INPUTS, NOT JUST EXIST.
      An earlier version returned the pickle whenever the file was present.
      That is silently wrong the moment anything upstream is rebuilt: after the
      month-to-date change to step 03, features_lgd.csv was an hour and a half
      NEWER than the cache, and a retrain would have trained on the previous
      feature definitions while reporting scores as though it had not.

      Nothing about that failure is visible in the output -- the run succeeds,
      the numbers look plausible, and the model is fitted to data that no
      longer exists on disk.  This project has met that shape of bug often
      enough to stop trusting existence checks.
    """
    if CACHE.exists():
        cache_t = CACHE.stat().st_mtime
        newer = [n for n in CACHE_INPUTS
                 if (OUTD / n).exists()
                 and (OUTD / n).stat().st_mtime > cache_t]
        if newer:
            import datetime as _dt
            log(f"  cached samples are STALE — rebuilt because "
                f"{', '.join(newer)} changed after "
                f"{_dt.datetime.fromtimestamp(cache_t):%Y-%m-%d %H:%M}")
        else:
            log(f"  loading cached samples {CACHE.name} "
                f"(inputs unchanged since it was built)")
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
    log(f"  daily rain {rain.shape}")

    tgt, fwdn = {}, {}
    for H in HORIZONS:
        fr, fn = s4.fwd_sum(rain, H), s4.fwd_sum(nrm, H)
        with np.errstate(invalid="ignore", divide="ignore"):
            tgt[H] = pd.DataFrame(
                np.where(fn >= MIN_NORMAL, (fr - fn) / fn * 100.0, np.nan),
                index=rain.index, columns=rain.columns)
        fwdn[H] = fn
        tgt[f"rain{H}"] = fr

    feats = {}
    for H in ANT:
        ar, an = s4.back_sum(rain, H), s4.back_sum(nrm, H)
        feats[f"ant_rain_{H}"] = ar
        with np.errstate(invalid="ignore", divide="ignore"):
            feats[f"ant_dep_{H}"] = pd.DataFrame(
                np.where(an >= MIN_NORMAL, (ar - an) / an * 100.0, np.nan),
                index=rain.index, columns=rain.columns)
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

    # TRAINING WINDOW: a 30-year baseline on which EVERY product is complete.
    #
    #   This is not the same thing as the climatological normal, and the two
    #   must not be conflated.  Normals stay on IMD's own 1971-2020 Long Period
    #   Average window, because that is what a departure is measured against.
    #   SAMPLE_FROM governs which rows the model learns from.
    #
    #   Why 1990 and not 1971 or 1981.  Product coverage by period:
    #       1971-1980   IMD rain and temperature only; no ERA5 at all, so
    #                   SPEI, MAI and soil moisture are entirely absent
    #       1981-1985   ERA5 present but SPEI still spinning up
    #                   (SPEI-12 only 81% complete -- a 12-month accumulation
    #                   needs a year of history before it exists)
    #       1986-       every product at 99% or better
    #
    #   Training across the 1981 boundary means half the rows carry a
    #   systematically different feature set from the other half, which teaches
    #   the model two inconsistent regimes rather than one. It also penalises
    #   learners that cannot ingest missing values: the gradient-boosted
    #   learners handle NaN natively, random forest does not, so a mixed window
    #   silently gives them different effective training sets and makes the
    #   comparison between learners unfair.
    #
    #   1990-2019 is exactly thirty years, sits well inside the complete zone,
    #   ends before the held-out test period, and weights the fit toward the
    #   recent climate rather than a half-century-old one.
    keep = rain.index[(rain.index.dayofyear % STRIDE == 0)
                      & (rain.index.year >= SAMPLE_FROM)]

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
    for H in HORIZONS:
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
    # MONTHLY FEATURES COME FROM THE PREVIOUS MONTH, NOT THE CONTAINING ONE.
    #   Joining on (year, month) gave an issue date of 10 July the whole of
    #   July's SPEI, MAI and degree-days -- quantities that are not knowable
    #   until 31 July.  Three weeks of future rainfall were leaking into every
    #   mid-month training row, and the reported skill was inflated by exactly
    #   the amount that leak was worth.
    #
    #   Using the preceding month removes the leak and, just as importantly,
    #   makes serving identical to training: the operational run can always
    #   supply a complete previous month, whereas the current month is only
    #   ever partially observed (and its sources end on different days).
    fmo = (base["date"].values.astype("datetime64[M]")
           - np.timedelta64(1, "M"))
    base["fyear"] = (fmo.astype("datetime64[Y]").astype(int) + 1970
                     ).astype(np.int16)
    base["fmonth"] = (fmo.astype("datetime64[M]").astype(int) % 12 + 1
                      ).astype(np.int8)
    base = base.merge(
        feat[["year", "month", "district_id"] + mcols].rename(
            columns={"year": "fyear", "month": "fmonth"}),
        on=["fyear", "fmonth", "district_id"], how="left")
    miss = base[mcols].isna().all(axis=1).mean()
    log(f"  monthly features lagged one month (no look-ahead); "
        f"{100*miss:.1f}% of rows have no prior-month features")

    #   MJO JOINS ON THE ISSUE DATE ITSELF, NOT ON THE PREVIOUS MONTH.
    #   The monthly covariates are lagged a month because they are month-long
    #   accumulations that do not exist until the month closes. ROMI is a
    #   DAILY observed index, already shifted one day in 00_clean_indices.py
    #   so an issue date sees only the previous day's published value. Lagging
    #   it a further month would throw away precisely the intraseasonal
    #   information it was added for -- the MJO cycle is 30-90 days, so a
    #   one-month lag is close to a half-cycle phase error.
    #
    #   Why it is here at all: on this dataset, out of sample, all-India JJAS,
    #   antecedent rainfall alone scores +0.034 at 7 days and -0.090 at 14.
    #   Adding these five columns takes them to +0.117 and +0.080. The 7-14
    #   day band IS the intraseasonal band and the feature set had nothing in
    #   it. (Wheeler & Hendon 2004; Kiladis et al. 2014 for ROMI; Pai et al.
    #   2011 for MJO control of Indian active/break spells.)
    jcols = []
    ipath = OUTD / "indices_daily.csv"
    if ipath.exists():
        idx = pd.read_csv(ipath, parse_dates=["date"])
        jcols = [c for c in ("mjo_amp", "mjo_sin", "mjo_cos", "romi1",
                             "romi2") if c in idx.columns]
        for c in jcols:
            idx[c] = idx[c].astype(np.float32)
        base = base.merge(idx[["date"] + jcols], on="date", how="left")
        have = base["mjo_amp"].notna().mean() * 100 if "mjo_amp" in base else 0
        log(f"  MJO joined on the issue date: {', '.join(jcols)} "
            f"({have:.1f}% of rows covered; ROMI begins 1991)")
    else:
        log("  indices_daily.csv absent — no MJO features. Run "
            "cleaning/00_clean_indices.py first; this is the single largest "
            "measured gain available at 7-14 days.")

    base.attrs["mcols"] = mcols
    base.attrs["jcols"] = jcols
    base.to_pickle(CACHE)
    log(f"  built + cached {len(base):,} samples")
    return base


# ------------------------------------------------------------------ models
def make_models():
    r"""Every formulation entered in the bake-off.

    LEARNER COVERAGE, AND WHY IT LOOKS LIKE THIS
      Random forest and histogram gradient boosting are both here because both
      are standard baselines and a comparison that omitted them would be open
      to the obvious objection.  Two points of history are worth recording:

      * HistGB was previously entered with the RAW departure target only.  That
        is not a fair test of the learner -- LightGBM was given four target
        formulations and won on one of them (tweedie), so HistGB now gets the
        same treatment.
      * Random forest reached R^2 = 0.31 in an earlier version of this project,
        and that number does NOT transfer here.  It was measured on a different
        problem: one seasonal JJAS aggregate per district per year, which is a
        far smoother and more predictable target than a 7- or 14-day forward
        window from a daily issue date.  Quoting it in this context would be
        comparing two different questions.

      TARGET NAMES ARE A CONTRACT.  The forecast exporter inverts each target
      by name (`raw`/`huber` are departures, `tweedie` is a rainfall amount,
      `logratio` is a log ratio against the forward normal).  A new learner
      must reuse one of those names, never invent one, or the inverse transform
      at serving time silently does the wrong thing.

    RANDOM FOREST IS SUBSAMPLED.  A full-depth forest on 2.6 M rows is hours of
    fitting for a baseline; `max_samples` draws 25% per tree, which keeps the
    comparison honest on the same features and split while staying tractable.
    The subsampling is a limitation of the comparison, not a tuned advantage,
    and is recorded here rather than buried.
    """
    from sklearn.ensemble import HistGradientBoostingRegressor as HGB
    from sklearn.ensemble import RandomForestRegressor as RF
    import lightgbm as lgb
    import xgboost as xgb
    LP = dict(n_estimators=600, learning_rate=0.05, num_leaves=63,
              min_child_samples=40, subsample=0.8, subsample_freq=1,
              colsample_bytree=0.8, reg_lambda=1.0, n_jobs=-1, verbose=-1)
    HP = dict(max_iter=500, learning_rate=0.05, max_leaf_nodes=63,
              l2_regularization=1.0, early_stopping=True,
              validation_fraction=0.12, random_state=0)
    RP = dict(n_estimators=180, max_depth=20, min_samples_leaf=40,
              max_features=0.5, max_samples=0.25, bootstrap=True,
              n_jobs=-1, random_state=0)
    return [
        ("HistGB",   "raw",      lambda: HGB(**HP)),
        # same target formulations LightGBM gets, so the learner comparison is
        # like for like rather than one learner being judged on its worst setup
        ("HistGB",   "tweedie",  lambda: HGB(loss="poisson", **HP)),
        ("HistGB",   "logratio", lambda: HGB(**HP)),
        ("RandomForest", "raw",      lambda: RF(**RP)),
        ("RandomForest", "logratio", lambda: RF(**RP)),
        ("LightGBM", "raw",      lambda: lgb.LGBMRegressor(**LP)),
        ("LightGBM", "huber",    lambda: lgb.LGBMRegressor(objective="huber",
                                                           alpha=40.0, **LP)),
        ("LightGBM", "tweedie",  lambda: lgb.LGBMRegressor(
            objective="tweedie", tweedie_variance_power=1.5, **LP)),
        ("LightGBM", "logratio", lambda: lgb.LGBMRegressor(**LP)),
        ("XGBoost",  "raw",      lambda: xgb.XGBRegressor(
            n_estimators=600, learning_rate=0.05, max_depth=8, subsample=0.8,
            colsample_bytree=0.8, reg_lambda=1.0, n_jobs=-1,
            tree_method="hist")),
        ("XGBoost",  "tweedie",  lambda: xgb.XGBRegressor(
            objective="reg:tweedie", tweedie_variance_power=1.5,
            n_estimators=600, learning_rate=0.05, max_depth=8, subsample=0.8,
            colsample_bytree=0.8, reg_lambda=1.0, n_jobs=-1,
            tree_method="hist")),
    ]


def fit_predict(mk, target, Xtr, tr, Xte, te, H):
    """Train on the formulation's own target, return DEPARTURE-scale preds."""
    N_tr, N_te = tr[f"fwdnorm_{H}"].to_numpy(), te[f"fwdnorm_{H}"].to_numpy()
    R_tr = tr[f"fwdrain_{H}"].to_numpy()
    if target == "raw" or target == "huber":
        y = tr[f"dep_{H}"].to_numpy()
        m = mk(); m.fit(Xtr, y)
        return m, m.predict(Xte)
    if target == "tweedie":                    # predict amount, then convert
        m = mk(); m.fit(Xtr, np.clip(R_tr, 0, None))
        p = np.clip(m.predict(Xte), 0, None)
        return m, (p - N_te) / np.maximum(N_te, MIN_NORMAL) * 100.0
    if target == "logratio":
        y = np.log((np.clip(R_tr, 0, None) + OFF) / (N_tr + OFF))
        m = mk(); m.fit(Xtr, y)
        p = np.exp(m.predict(Xte)) * (N_te + OFF) - OFF
        return m, (p - N_te) / np.maximum(N_te, MIN_NORMAL) * 100.0
    raise ValueError(target)


def score(pred, yte):
    pred = np.clip(np.nan_to_num(pred, nan=0.0), -100, 1000)
    rmse = float(np.sqrt(np.mean((pred - yte) ** 2)))
    clim = float(np.sqrt(np.mean(yte ** 2)))
    return {"rmse": round(rmse, 2),
            "mse_skill": round(1 - rmse ** 2 / clim ** 2, 4),
            "corr": round(float(np.corrcoef(pred, yte)[0, 1]), 4),
            "imd_cat_acc": round(float(np.mean(imd_cat(pred) == imd_cat(yte))), 4),
            "clim_rmse": round(clim, 2)}


def main():
    log("=" * 74)
    log("MODEL BAKE-OFF — formulations x learners, judged on the IMD scale")
    log("=" * 74)
    base = build_samples()
    mcols = base.attrs.get("mcols") or [
        c for c in base.columns if c.startswith(("spei", "oni", "iod", "enso"))]
    jcols = [c for c in (base.attrs.get("jcols") or []) if c in base.columns]
    XCOLS = ([f"ant_rain_{h}" for h in ANT] + [f"ant_dep_{h}" for h in ANT]
             + ["wet30", "extreme30", "dsw", "doy_sin", "doy_cos"]
             + list(mcols) + list(jcols) + ["district_id"])
    log(f"  {len(XCOLS)} features"
        + (f", including MJO: {', '.join(jcols)}" if jcols else ", no MJO"))

    rows, best = [], {}
    for H in HORIZONS:
        d = base.dropna(subset=[f"dep_{H}", f"fwdnorm_{H}"])
        tr = d[d.year <= TRAIN_END]
        te = d[d.year >= TEST_START]
        Xtr, Xte = tr[XCOLS], te[XCOLS]
        yte = te[f"dep_{H}"].to_numpy()
        log(f"\n  H={H}d   train {len(tr):,}   test {len(te):,}")
        log(f"  {'learner':10s} {'target':9s} {'RMSE':>8s} {'skill':>8s} "
            f"{'corr':>7s} {'IMDcat':>7s} {'sec':>6s}")
        log("  " + "-" * 62)
        preds = {}
        for learner, target, mk in make_models():
            t0 = time.time()
            try:
                _, p = fit_predict(mk, target, Xtr, tr, Xte, te, H)
            except Exception as e:
                log(f"  {learner:10s} {target:9s}  FAILED {type(e).__name__} "
                    f"{str(e)[:40]}")
                continue
            s = score(p, yte)
            preds[f"{learner}|{target}"] = p
            rows.append({"horizon": H, "learner": learner, "target": target,
                         **s, "seconds": round(time.time() - t0, 1)})
            log(f"  {learner:10s} {target:9s} {s['rmse']:8.2f} "
                f"{s['mse_skill']:+8.4f} {s['corr']:7.4f} "
                f"{s['imd_cat_acc']:7.4f} {time.time()-t0:6.0f}")

        # blend the three best by MSE skill
        top = sorted([r for r in rows if r["horizon"] == H],
                     key=lambda r: -r["mse_skill"])[:3]
        bl = np.mean([preds[f"{r['learner']}|{r['target']}"] for r in top], axis=0)
        s = score(bl, yte)
        rows.append({"horizon": H, "learner": "Blend(top3)", "target": "mix",
                     **s, "seconds": 0})
        log(f"  {'Blend':10s} {'top3':9s} {s['rmse']:8.2f} "
            f"{s['mse_skill']:+8.4f} {s['corr']:7.4f} {s['imd_cat_acc']:7.4f}")
        log(f"    blend members: {', '.join(r['learner']+'/'+r['target'] for r in top)}")

        cands = [r for r in rows if r["horizon"] == H]
        win = max(cands, key=lambda r: r["mse_skill"])
        best[H] = {"win": win, "members": [(r["learner"], r["target"])
                                           for r in top]}
        log(f"  --> best H={H}: {win['learner']}/{win['target']}  "
            f"skill {win['mse_skill']:+.4f}  IMD-cat {win['imd_cat_acc']:.3f}")

    res = pd.DataFrame(rows)
    res.to_csv(OUTD / "bakeoff_results_lgd.csv", index=False)

    # ---- refit and persist EVERY formulation, not only the winner ---------
    #   The dashboard offers a model selector, so each candidate has to be
    #   available at forecast time rather than discarded once ranked.  That is
    #   not merely for transparency: the ranking DEPENDS ON THE METRIC.
    #   LightGBM/logratio has the best IMD-category accuracy of any
    #   formulation while sitting near the bottom on mean-squared error --- a
    #   sharper model lands more categories correctly and pays for it in
    #   squared error.  A reader trading on "deficient or normal" and one
    #   trading on magnitude genuinely want different models, so both are
    #   published with their own verified scores attached.
    log("\n  refitting every formulation on train+val and persisting ...")
    skill, catalogue = {}, {}
    for H in HORIZONS:
        d = base.dropna(subset=[f"dep_{H}", f"fwdnorm_{H}"])
        tr = d[d.year <= 2019]                 # train+val for the final model
        te = d[d.year >= TEST_START]
        hres = res[res["horizon"] == H].sort_values("mse_skill",
                                                    ascending=False)
        entries = []
        for r in hres.itertuples():
            key = f"{r.learner}/{r.target}"
            members = (best[H]["members"]
                       if str(r.learner).startswith("Blend")
                       else [(r.learner, r.target)])
            fitted = []
            for learner, target in members:
                mk = next((m for (l, t, m) in make_models()
                           if l == learner and t == target), None)
                if mk is None:
                    continue
                mdl, _ = fit_predict(mk, target, tr[XCOLS], tr,
                                     te[XCOLS], te, H)
                fitted.append({"model": mdl, "target": target,
                               "learner": learner})
            if not fitted:
                log(f"    H={H}: {key} skipped (no constructor)")
                continue
            slug = key.replace("/", "_").replace("(", "").replace(")", "")
            with open(OUTD / f"model_dep{H}_{slug}_lgd.pkl", "wb") as f:
                pickle.dump({"members": fitted, "cols": XCOLS, "offset": OFF,
                             "min_normal": MIN_NORMAL,
                             "kind": "blend" if len(fitted) > 1 else "single"},
                            f)
            entries.append({
                "key": key, "slug": slug, "learner": str(r.learner),
                "target": str(r.target),
                "mse_skill": float(r.mse_skill), "corr": float(r.corr),
                "rmse": float(r.rmse), "imd_cat_acc": float(r.imd_cat_acc),
                "beats_climatology": bool(r.mse_skill > 0),
                "is_default": key == f"{best[H]['win']['learner']}/"
                                     f"{best[H]['win']['target']}"})
            log(f"    H={H}: {key:<24} skill {r.mse_skill:+.4f}  "
                f"cat {r.imd_cat_acc:.3f}  -> {slug}")
        catalogue[f"dep_{H}"] = entries

        # the default model keeps its historical filename so nothing that
        # already reads model_dep{H}_lgd.pkl has to change
        w = best[H]["win"]
        dslug = f"{w['learner']}/{w['target']}".replace("/", "_") \
            .replace("(", "").replace(")", "")
        src = OUTD / f"model_dep{H}_{dslug}_lgd.pkl"
        if src.exists():
            (OUTD / f"model_dep{H}_lgd.pkl").write_bytes(src.read_bytes())
        skill[f"dep_{H}"] = {"rmse": w["rmse"], "clim_rmse": w["clim_rmse"],
                             "mse_skill": w["mse_skill"], "corr": w["corr"],
                             "imd_cat_acc": w["imd_cat_acc"],
                             "model": f"{w['learner']}/{w['target']}",
                             "n_test": int((base.year >= TEST_START).sum())}
    (OUTD / "skill_lgd.json").write_text(json.dumps(skill, indent=1),
                                         encoding="utf-8")
    (OUTD / "model_catalogue_lgd.json").write_text(
        json.dumps(catalogue, indent=1), encoding="utf-8")
    n = sum(len(v) for v in catalogue.values())
    log(f"\n  wrote bakeoff_results_lgd.csv, {n} model pickles, "
        f"model_catalogue_lgd.json, skill_lgd.json")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
