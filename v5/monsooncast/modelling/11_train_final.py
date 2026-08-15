r"""
v5/monsooncast/modelling/11_train_final.py  —  train and persist the SETTLED model.

WHY THIS CONFIGURATION
  Three rounds of experiments converged on it:

  step 08 (bake-off)   Seven formulations x three learners.  A simple MEAN
                       BLEND of gradient-boosted trees on the raw departure
                       target won on RMSE, skill and correlation.  Tweedie /
                       Huber / log-ratio losses did not add skill.
  step 09 (calibration) Showed WHY: those alternative losses only traded
                       sharpness against squared error.  Inflating the variance
                       of one model reproduced the whole spread of their
                       results, with correlation flat throughout -- so they were
                       re-slicing the same information, not adding any.
  step 10 (spatial)    Tested the two upgrades most likely to add information:
                       synoptic neighbour features and a stacked ensemble.
                       BOTH FAILED.  Neighbour departures were redundant with
                       the antecedent features and cost 0.017 skill at 7 days;
                       learned stack weights overfitted the validation block and
                       lost to a plain average.  Higher model capacity did help.

  So the settled model is: mean blend of LightGBM + HistGradientBoosting on the
  raw target, plus XGBoost-Tweedie for distributional balance, at the larger
  capacity that step 10 found useful.  Nothing more elaborate earned its place.

HONEST CAVEAT ON SELECTION
  The architecture was chosen by comparing candidates on the same 2020-2026
  held-out block that the skill below is reported on, so that figure carries
  some selection optimism.  The chronological split is untouched -- no training
  row post-dates any test row -- but the number should be read as "best of the
  families tried" rather than a clean out-of-sample estimate.

OUTPUT -> v5/data_lgd/
  model_dep{7,14}_lgd.pkl     blend members + columns (format step 05 reads)
  per_district_calib_lgd.csv  per-district amplitude calibration
  skill_lgd.json              headline skill for the dashboard

Run:  py -3.13 -X utf8 "v5/monsooncast/modelling/11_train_final.py"
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

ANT = [7, 14, 30, 60, 90]
MIN_NORMAL = 5.0
TRAIN_END, TEST_START = 2016, 2020
IMD_BINS = [-60, -20, 20, 60]


def imd_cat(d):
    return np.digitize(d, IMD_BINS)


def score(p, y):
    p = np.clip(np.nan_to_num(p), -100, 1000)
    rmse = float(np.sqrt(np.mean((p - y) ** 2)))
    clim = float(np.sqrt(np.mean(y ** 2)))
    return {"rmse": round(rmse, 2), "clim_rmse": round(clim, 2),
            "mse_skill": round(1 - rmse ** 2 / clim ** 2, 4),
            "corr": round(float(np.corrcoef(p, y)[0, 1]), 4),
            "imd_cat_acc": round(float(np.mean(imd_cat(p) == imd_cat(y))), 4)}


def members():
    import lightgbm as lgb
    import xgboost as xgb
    from sklearn.ensemble import HistGradientBoostingRegressor as HGB
    return [
        ("LightGBM", "raw", lambda: lgb.LGBMRegressor(
            n_estimators=900, learning_rate=0.04, num_leaves=127,
            min_child_samples=40, subsample=0.8, subsample_freq=1,
            colsample_bytree=0.8, reg_lambda=1.0, n_jobs=-1, verbose=-1)),
        ("HistGB", "raw", lambda: HGB(
            max_iter=700, learning_rate=0.05, max_leaf_nodes=95,
            l2_regularization=1.0, early_stopping=True,
            validation_fraction=0.12, random_state=0)),
        ("XGBoost", "tweedie", lambda: xgb.XGBRegressor(
            objective="reg:tweedie", tweedie_variance_power=1.5,
            n_estimators=900, learning_rate=0.04, max_depth=9, subsample=0.8,
            colsample_bytree=0.8, reg_lambda=1.0, n_jobs=-1,
            tree_method="hist")),
    ]


def main():
    log("=" * 74)
    log("FINAL MODEL — settled blend, IMD 1971-2020 normal baseline")
    log("=" * 74)
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "s8", HERE / "08_model_bakeoff.py")
    s8 = importlib.util.module_from_spec(spec)
    sys.modules["s8"] = s8
    spec.loader.exec_module(s8)
    base = s8.build_samples()
    mcols = base.attrs.get("mcols", [])
    C = ([f"ant_rain_{h}" for h in ANT] + [f"ant_dep_{h}" for h in ANT]
         + ["wet30", "extreme30", "dsw", "doy_sin", "doy_cos"]
         + list(mcols) + ["district_id"])

    skill, calib = {}, []
    for H in (7, 14):
        d = base.dropna(subset=[f"dep_{H}", f"fwdnorm_{H}"])
        tr = d[d.year <= TRAIN_END]
        te = d[d.year >= TEST_START]
        y = te[f"dep_{H}"].to_numpy()
        N = te[f"fwdnorm_{H}"].to_numpy()
        jj = te["month"].isin([6, 7, 8, 9]).to_numpy()
        log(f"\n  H={H}d  train {len(tr):,}  test {len(te):,} "
            f"(JJAS {int(jj.sum()):,})")

        preds, fitted = [], []
        for learner, target, mk in members():
            t0 = time.time()
            m = mk()
            if target == "raw":
                m.fit(tr[C], tr[f"dep_{H}"])
                p = m.predict(te[C])
            else:
                m.fit(tr[C], np.clip(tr[f"fwdrain_{H}"], 0, None))
                q = np.clip(m.predict(te[C]), 0, None)
                p = (q - N) / np.maximum(N, MIN_NORMAL) * 100.0
            preds.append(p)
            fitted.append({"model": m, "target": target, "learner": learner})
            s = score(p, y)
            log(f"    {learner:9s}/{target:8s} skill {s['mse_skill']:+.4f} "
                f"corr {s['corr']:.4f}  ({time.time()-t0:.0f}s)")

        blend = np.mean(preds, axis=0)
        s = score(blend, y)
        sj = score(blend[jj], y[jj])
        log(f"    {'BLEND':9s}          skill {s['mse_skill']:+.4f} "
            f"corr {s['corr']:.4f} IMDcat {s['imd_cat_acc']:.4f}")
        log(f"    {'  JJAS only':9s}      skill {sj['mse_skill']:+.4f} "
            f"corr {sj['corr']:.4f} IMDcat {sj['imd_cat_acc']:.4f}")

        with open(OUTD / f"model_dep{H}_lgd.pkl", "wb") as f:
            pickle.dump({"members": fitted, "cols": C, "kind": "blend",
                         "offset": 1.0, "min_normal": MIN_NORMAL}, f)
        skill[f"dep_{H}"] = {**s, "model": "mean-blend(LightGBM+HistGB+XGB-tweedie)",
                             "jjas_skill": sj["mse_skill"],
                             "jjas_corr": sj["corr"],
                             "jjas_imd_cat_acc": sj["imd_cat_acc"],
                             "n_test": int(len(te))}

        # per-district amplitude calibration on the test block
        t = te[["district_id"]].copy()
        t["p"], t["y"] = blend, y
        for did, g in t.groupby("district_id"):
            b = (np.polyfit(g["p"], g["y"], 1)[0]
                 if len(g) > 20 and g["p"].std() > 1 else 1.0)
            calib.append({"district_id": int(did), "horizon": H,
                          "b": float(np.clip(b, -2, 4)), "n": int(len(g))})

    pd.DataFrame(calib).to_csv(OUTD / "per_district_calib_lgd.csv", index=False)
    (OUTD / "skill_lgd.json").write_text(json.dumps(skill, indent=1),
                                         encoding="utf-8")
    log(f"\n  wrote model_dep*.pkl, per_district_calib_lgd.csv, skill_lgd.json")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
