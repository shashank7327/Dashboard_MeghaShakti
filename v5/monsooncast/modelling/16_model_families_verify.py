r"""
v5/monsooncast/modelling/16_model_families_verify.py  —  test model families that have
NOT been tried, and verify every candidate against actual observed outcomes.

WHAT HAS ALREADY BEEN RULED OUT (so it is not repeated here)
  step 08  seven loss/learner formulations -- a mean blend of gradient-boosted
           trees on the raw departure target won; Tweedie/Huber/log-ratio
           added no skill.
  step 09  and the reason: those losses only traded sharpness against squared
           error.  Inflating one model's variance reproduced their whole spread
           with correlation flat, so they were re-slicing the same information.
  step 10  spatial neighbour features and a stacked ensemble.  BOTH FAILED --
           neighbours were redundant with the antecedent terms; learned stack
           weights overfitted the validation block.

WHAT IS NEW HERE
  1. A NEURAL NET (MLP).  Every model tried so far has been an axis-aligned
     tree ensemble, so they share an inductive bias and their errors are
     correlated.  A network learns smooth interactions instead, which is a
     genuinely different hypothesis class and can add to a blend even when it
     loses on its own.
  2. DENSER SAMPLING.  Issue dates were sampled every 5 days purely for speed.
     Stride 3 gives ~1.7x the training rows at no methodological cost.
  3. SEASON-SPECIALISED MODELS.  Monsoon and non-monsoon rainfall are
     different processes -- one is organised convection under a reliable
     moisture supply, the other is sporadic.  One model must compromise
     between them; a JJAS-only model can specialise.

VERIFICATION AGAINST OBSERVED OUTCOMES
  Skill is measured on the held-out 2020-2026 block, against what actually
  happened, and reported in the ways that can hide a weakness:
    * against BOTH baselines -- climatology (assume normal) AND persistence
      (the last H days repeat).  Beating climatology while losing to
      persistence would mean the model adds nothing a forecaster could not do
      by looking backwards.
    * split by season (JJAS vs the rest), by ENSO phase, and by region.
    * a RELIABILITY check: bin the forecasts and compare the mean prediction
      in each bin with the mean observed outcome.  A model can correlate well
      and still be systematically over- or under-confident.

OUTPUT -> v5/data_lgd/
  model_families_lgd.csv     every candidate x metric
  verification_lgd.csv       the winner verified by season / phase / region
  reliability_lgd.csv        predicted-vs-observed calibration bins

Run:  py -3.13 -X utf8 "v5/monsooncast/modelling/16_model_families_verify.py"
"""
import json
import pathlib
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
STRIDE = 3
MIN_NORMAL = 5.0
TRAIN_END, TEST_START = 2016, 2020
CACHE = OUTD / "_samples_s3.pkl"
IMD_BINS = [-60, -20, 20, 60]


def imd_cat(d):
    return np.digitize(d, IMD_BINS)


def score(p, y, base=None):
    p = np.clip(np.nan_to_num(p), -100, 1000)
    rmse = float(np.sqrt(np.mean((p - y) ** 2)))
    clim = float(np.sqrt(np.mean(y ** 2)))
    out = {"rmse": round(rmse, 2), "mse_skill": round(1 - rmse ** 2 / clim ** 2, 4),
           "corr": round(float(np.corrcoef(p, y)[0, 1]), 4),
           "imd_cat": round(float(np.mean(imd_cat(p) == imd_cat(y))), 4),
           "mae": round(float(np.mean(np.abs(p - y))), 2)}
    if base is not None:
        b = np.clip(np.nan_to_num(base), -100, 1000)
        brmse = float(np.sqrt(np.mean((b - y) ** 2)))
        out["skill_vs_persistence"] = round(1 - rmse ** 2 / brmse ** 2, 4)
    return out


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
    log(f"  daily rain {rain.shape}, stride {STRIDE}")

    tgt, fwdn = {}, {}
    for H in (7, 14):
        fr, fn = s4.fwd_sum(rain, H), s4.fwd_sum(nrm, H)
        with np.errstate(invalid="ignore", divide="ignore"):
            tgt[H] = pd.DataFrame(
                np.where(fn >= MIN_NORMAL, (fr - fn) / fn * 100.0, np.nan),
                index=rain.index, columns=rain.columns)
        fwdn[H] = fn
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
    dsw = np.zeros(rain.shape)
    arr = rain.to_numpy() > 1.0
    for c in range(arr.shape[1]):
        run = 0
        for r in range(arr.shape[0]):
            run = 0 if arr[r, c] else run + 1
            dsw[r, c] = run
    feats["dsw"] = pd.DataFrame(dsw, index=rain.index, columns=rain.columns)

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
    for H in (7, 14):
        base[f"dep_{H}"] = m(tgt[H])["v"].to_numpy(np.float32)
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
              "oni_mam", "pct_departure_signature"] if c in feat.columns]
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


def mlp_fit_predict(Xtr, ytr, Xte, seed=0):
    """A small MLP on standardised inputs -- a different hypothesis class from
    the tree ensembles, so its errors are not perfectly correlated with theirs."""
    import torch
    import torch.nn as nn
    torch.manual_seed(seed)
    med = np.nanmedian(Xtr, axis=0)
    Xtr = np.where(np.isfinite(Xtr), Xtr, med).astype(np.float32)
    Xte = np.where(np.isfinite(Xte), Xte, med).astype(np.float32)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    Xtr = (Xtr - mu) / sd
    Xte = (Xte - mu) / sd
    ysd = float(ytr.std()) or 1.0
    net = nn.Sequential(nn.Linear(Xtr.shape[1], 256), nn.ReLU(),
                        nn.Dropout(0.1), nn.Linear(256, 128), nn.ReLU(),
                        nn.Linear(128, 1))
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    lossf = nn.HuberLoss(delta=1.0)     # on the standardised target
    Xt = torch.from_numpy(Xtr)
    yt = torch.from_numpy((ytr / ysd).astype(np.float32)).unsqueeze(1)
    n, bs = len(Xt), 8192
    for ep in range(6):
        perm = torch.randperm(n)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            loss = lossf(net(Xt[idx]), yt[idx])
            loss.backward()
            opt.step()
    net.eval()
    with torch.no_grad():
        out = []
        Xe = torch.from_numpy(Xte)
        for i in range(0, len(Xe), 65536):
            out.append(net(Xe[i:i + 65536]).squeeze(1).numpy())
    return np.concatenate(out) * ysd


def main():
    log("=" * 74)
    log("MODEL FAMILIES + verification against observed outcomes")
    log("=" * 74)
    base = build_samples()
    mcols = base.attrs.get("mcols", [])
    C = ([f"ant_rain_{h}" for h in ANT] + [f"ant_dep_{h}" for h in ANT]
         + ["wet30", "extreme30", "dsw", "doy_sin", "doy_cos"]
         + list(mcols) + ["district_id"])
    import lightgbm as lgb
    from sklearn.ensemble import HistGradientBoostingRegressor as HGB

    reg = pd.read_csv(IMD / "registry_lgd791.csv")[["district_id", "state"]]
    rows, best = [], {}
    for H in (7, 14):
        d = base.dropna(subset=[f"dep_{H}", f"fwdnorm_{H}"])
        tr = d[d.year <= TRAIN_END]
        te = d[d.year >= TEST_START]
        y = te[f"dep_{H}"].to_numpy()
        pers = te[f"ant_dep_{H}"].to_numpy()      # persistence baseline
        log(f"\n  H={H}d  train {len(tr):,}  test {len(te):,}  "
            f"(stride {STRIDE})")
        log(f"  {'candidate':26s} {'RMSE':>8s} {'skill':>8s} {'vs pers':>8s} "
            f"{'corr':>7s} {'IMDcat':>7s} {'sec':>6s}")
        log("  " + "-" * 72)
        P = {}

        def run(tag, fn):
            t0 = time.time()
            p = fn()
            P[tag] = p
            s = score(p, y, base=pers)
            rows.append({"horizon": H, "candidate": tag, **s,
                         "seconds": round(time.time() - t0)})
            log(f"  {tag:26s} {s['rmse']:8.2f} {s['mse_skill']:+8.4f} "
                f"{s['skill_vs_persistence']:+8.4f} {s['corr']:7.4f} "
                f"{s['imd_cat']:7.4f} {time.time()-t0:6.0f}")

        LP = dict(n_estimators=900, learning_rate=0.04, num_leaves=127,
                  min_child_samples=40, subsample=0.8, subsample_freq=1,
                  colsample_bytree=0.8, reg_lambda=1.0, n_jobs=-1, verbose=-1)
        run("LightGBM (stride3)", lambda: lgb.LGBMRegressor(**LP)
            .fit(tr[C], tr[f"dep_{H}"]).predict(te[C]))
        run("HistGB (stride3)", lambda: HGB(
            max_iter=700, learning_rate=0.05, max_leaf_nodes=95,
            l2_regularization=1.0, early_stopping=True,
            validation_fraction=0.12, random_state=0)
            .fit(tr[C], tr[f"dep_{H}"]).predict(te[C]))
        run("MLP (neural)", lambda: mlp_fit_predict(
            tr[C].to_numpy(np.float32), tr[f"dep_{H}"].to_numpy(np.float32),
            te[C].to_numpy(np.float32)))

        # season-specialised: a JJAS-only model applied to JJAS rows
        jj_tr = tr[tr["month"].isin([6, 7, 8, 9])]
        jj_te_mask = te["month"].isin([6, 7, 8, 9]).to_numpy()

        def seasonal():
            mj = lgb.LGBMRegressor(**LP).fit(jj_tr[C], jj_tr[f"dep_{H}"])
            mo = lgb.LGBMRegressor(**LP).fit(
                tr[~tr["month"].isin([6, 7, 8, 9])][C],
                tr[~tr["month"].isin([6, 7, 8, 9])][f"dep_{H}"])
            out = np.empty(len(te))
            out[jj_te_mask] = mj.predict(te[jj_te_mask][C])
            out[~jj_te_mask] = mo.predict(te[~jj_te_mask][C])
            return out
        run("Season-specialised LGBM", seasonal)

        run("Blend GBM x2", lambda: np.mean(
            [P["LightGBM (stride3)"], P["HistGB (stride3)"]], axis=0))
        run("Blend GBM x2 + MLP", lambda: np.mean(
            [P["LightGBM (stride3)"], P["HistGB (stride3)"], P["MLP (neural)"]],
            axis=0))

        cands = [r for r in rows if r["horizon"] == H]
        win = max(cands, key=lambda r: r["mse_skill"])
        best[H] = (win, P[win["candidate"]], te, y, pers)
        log(f"  --> best H={H}: {win['candidate']}  skill {win['mse_skill']:+.4f}")

    pd.DataFrame(rows).to_csv(OUTD / "model_families_lgd.csv", index=False)

    # ------------------------------------------------ verification & reliability
    ver, rel = [], []
    for H, (win, p, te, y, pers) in best.items():
        te = te.merge(reg, on="district_id", how="left")
        jj = te["month"].isin([6, 7, 8, 9]).to_numpy()
        phase = np.where(te["enso_warm"].to_numpy() == 1, "El Nino",
                         np.where(te["enso_cold"].to_numpy() == 1, "La Nina",
                                  "Neutral"))
        groups = [("all", np.ones(len(y), bool)),
                  ("JJAS", jj), ("non-JJAS", ~jj)]
        for ph in ("El Nino", "La Nina", "Neutral"):
            groups.append((f"ENSO {ph}", phase == ph))
        top = te["state"].value_counts().head(8).index
        for st in top:
            groups.append((f"state {st.title()}", (te["state"] == st).to_numpy()))
        for name, m in groups:
            if m.sum() < 500:
                continue
            s = score(p[m], y[m], base=pers[m])
            ver.append({"horizon": H, "model": win["candidate"], "group": name,
                        "n": int(m.sum()), **s})
        # reliability: bin the forecast, compare bin mean prediction vs observed
        qs = np.quantile(p, np.linspace(0, 1, 11))
        qs[0], qs[-1] = -1e9, 1e9
        b = np.digitize(p, qs[1:-1])
        for k in range(10):
            mk = b == k
            if mk.sum() < 100:
                continue
            rel.append({"horizon": H, "bin": k + 1, "n": int(mk.sum()),
                        "mean_forecast": round(float(p[mk].mean()), 2),
                        "mean_observed": round(float(y[mk].mean()), 2)})
    pd.DataFrame(ver).to_csv(OUTD / "verification_lgd.csv", index=False)
    R = pd.DataFrame(rel)
    R.to_csv(OUTD / "reliability_lgd.csv", index=False)

    log("\n" + "=" * 74)
    log("  VERIFICATION vs observed (winner per horizon)")
    V = pd.DataFrame(ver)
    for H in (7, 14):
        log(f"\n  H={H}d — {best[H][0]['candidate']}")
        for r in V[V.horizon == H].itertuples():
            log(f"    {r.group:22s} n={r.n:7,}  skill {r.mse_skill:+.4f}  "
                f"vs-pers {r.skill_vs_persistence:+.4f}  corr {r.corr:.3f}")
    log("\n  RELIABILITY (decile of forecast: mean forecast vs mean observed)")
    for H in (7, 14):
        sub = R[R.horizon == H]
        log(f"    H={H}d  " + "  ".join(
            f"{r.mean_forecast:+.0f}/{r.mean_observed:+.0f}" for r in sub.itertuples()))
    log("\n  wrote model_families_lgd.csv, verification_lgd.csv, "
        "reliability_lgd.csv")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
