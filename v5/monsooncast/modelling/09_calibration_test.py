r"""
v5/monsooncast/modelling/09_calibration_test.py  —  is the RMSE-vs-IMD-category
trade-off fundamental, or is the model simply over-smoothed?

THE OBSERVATION
  In the bake-off, every formulation that hurt RMSE improved agreement with
  IMD's operational rainfall categories, monotonically:

      LightGBM raw        RMSE 113.5   IMD-category 0.392
      LightGBM tweedie    RMSE 116.4   IMD-category 0.401
      LightGBM huber      RMSE 117.7   IMD-category 0.435
      LightGBM logratio   RMSE 123.6   IMD-category 0.448

  That is the classic signature of SHRINKAGE.  Squared-error loss is minimised
  by predicting close to the conditional mean, so a model facing an
  irreducibly noisy target hedges toward the middle.  Its predictions then have
  too little variance: they cluster in the "Normal" band and rarely commit to
  "Deficient" or "Excess", which is exactly what IMD publishes.

THE TEST
  Take the RMSE-optimal prediction and re-inflate its variance about its own
  mean, then re-score.  Two standard calibrations (von Storch 1999, on
  inflation of statistically downscaled forecasts; Wilks, *Statistical Methods
  in the Atmospheric Sciences*, on reliability vs sharpness):

    variance inflation   p' = mean(p) + f * (p - mean(p))
    quantile mapping     rank-preserving map of p onto the observed
                         distribution -- maximal sharpness, correlation
                         unchanged by construction

  If inflation recovers the category skill, the trade-off is a calibration
  choice and can be made deliberately.  If it does not, the alternative losses
  were genuinely extracting different information.

OUTPUT -> v5/data_lgd/calibration_test_lgd.csv

Run:  py -3.13 -X utf8 "v5/monsooncast/modelling/09_calibration_test.py"
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

ANT = [7, 14, 30, 60, 90]
TRAIN_END, TEST_START = 2016, 2020
IMD_BINS = [-60, -20, 20, 60]


def imd_cat(d):
    return np.digitize(d, IMD_BINS)


def score(p, y):
    p = np.clip(np.nan_to_num(p), -100, 1000)
    rmse = float(np.sqrt(np.mean((p - y) ** 2)))
    clim = float(np.sqrt(np.mean(y ** 2)))
    return (rmse, 1 - rmse ** 2 / clim ** 2,
            float(np.corrcoef(p, y)[0, 1]),
            float(np.mean(imd_cat(p) == imd_cat(y))))


def main():
    log("=" * 74)
    log("CALIBRATION TEST — is the RMSE / IMD-category trade-off fixable?")
    log("=" * 74)
    base = pd.read_pickle(OUTD / "_bakeoff_samples.pkl")
    mcols = base.attrs.get("mcols", [])
    XCOLS = ([f"ant_rain_{h}" for h in ANT] + [f"ant_dep_{h}" for h in ANT]
             + ["wet30", "extreme30", "dsw", "doy_sin", "doy_cos"]
             + list(mcols) + ["district_id"])
    import lightgbm as lgb

    rows = []
    for H in (7, 14):
        d = base.dropna(subset=[f"dep_{H}", f"fwdnorm_{H}"])
        tr, te = d[d.year <= TRAIN_END], d[d.year >= TEST_START]
        y = te[f"dep_{H}"].to_numpy()
        m = lgb.LGBMRegressor(n_estimators=600, learning_rate=0.05,
                              num_leaves=63, min_child_samples=40,
                              subsample=0.8, subsample_freq=1,
                              colsample_bytree=0.8, reg_lambda=1.0,
                              n_jobs=-1, verbose=-1)
        m.fit(tr[XCOLS], tr[f"dep_{H}"])
        p = m.predict(te[XCOLS])

        sd_p, sd_y = float(np.std(p)), float(np.std(y))
        log(f"\n  H={H}d   sd(prediction) {sd_p:6.2f}   sd(observed) {sd_y:6.2f}"
            f"   ratio {sd_p/sd_y:.3f}")
        log(f"    -> the model uses only {100*sd_p/sd_y:.0f}% of the observed "
            f"spread; that is the shrinkage.")
        log(f"    {'calibration':22s} {'RMSE':>8s} {'skill':>8s} {'corr':>7s} "
            f"{'IMDcat':>7s}")
        log("    " + "-" * 56)

        mu = p.mean()
        for f in (1.0, 1.2, 1.4, sd_y / sd_p, 1.8, 2.0):
            r = score(mu + f * (p - mu), y)
            tag = ("inflate x%.2f" % f) + (" (full sd)" if abs(f - sd_y / sd_p)
                                           < 1e-9 else "")
            rows.append({"horizon": H, "calibration": tag, "rmse": round(r[0], 2),
                         "mse_skill": round(r[1], 4), "corr": round(r[2], 4),
                         "imd_cat_acc": round(r[3], 4)})
            log(f"    {tag:22s} {r[0]:8.2f} {r[1]:+8.4f} {r[2]:7.4f} "
                f"{r[3]:7.4f}")

        # quantile mapping: rank-preserving onto the observed distribution
        order = np.argsort(np.argsort(p))
        qm = np.sort(y)[np.clip((order / (len(p) - 1) * (len(y) - 1))
                                .astype(int), 0, len(y) - 1)]
        r = score(qm, y)
        rows.append({"horizon": H, "calibration": "quantile map",
                     "rmse": round(r[0], 2), "mse_skill": round(r[1], 4),
                     "corr": round(r[2], 4), "imd_cat_acc": round(r[3], 4)})
        log(f"    {'quantile map':22s} {r[0]:8.2f} {r[1]:+8.4f} {r[2]:7.4f} "
            f"{r[3]:7.4f}")

    pd.DataFrame(rows).to_csv(OUTD / "calibration_test_lgd.csv", index=False)
    log(f"\n  wrote calibration_test_lgd.csv")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
