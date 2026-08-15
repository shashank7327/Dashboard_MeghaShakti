r"""v5/monsooncast/modelling/08b_persist_models.py  —  refit and persist the
bake-off's models, resumably, without re-scoring anything.

WHY THIS IS A SEPARATE STEP
  08_model_bakeoff.py scores every formulation and THEN refits each one on
  train+validation to persist it. The scoring is the cheap half; the refit is
  a second full training pass over 26 fits and takes hours. When that pass was
  interrupted the run left behind exactly the state it should not:

      bakeoff_results_lgd.csv   the NEW scores
      skill_lgd.json            the OLD scores
      model_dep7_*              six new, the rest old
      model_dep{7,14}_lgd.pkl   both old

  Nothing about that combination raises. Each pickle carries its own column
  list, so an old model quietly selects the 45 features it knows and serves a
  forecast, while the dashboard prints skill numbers measured on a different
  model. That is the project's recurring failure shape: a plausible number.

WHAT THIS DOES DIFFERENTLY
  * DEFAULTS FIRST. The blend that actually gets served is refitted before any
    alternative, so an interruption leaves a correct default rather than a
    correct list of also-rans.
  * RESUMABLE BY FINGERPRINT, not by timestamp. A pickle is considered current
    only if its stored `cols` match the feature set the cache carries. Mtimes
    lie after a partial run; the column list cannot.
  * WRITES THE CATALOGUE INCREMENTALLY, so skill_lgd.json and
    model_catalogue_lgd.json never describe models that are not on disk.

Run:  py -3.13 -X utf8 "v5/monsooncast/modelling/08b_persist_models.py"
      py -3.13 -X utf8 "v5/monsooncast/modelling/08b_persist_models.py" --defaults-only
"""
import importlib.util
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

spec = importlib.util.spec_from_file_location("bo", HERE / "08_model_bakeoff.py")
bo = importlib.util.module_from_spec(spec)
sys.modules["bo"] = bo
spec.loader.exec_module(bo)

TRAIN_END_FINAL = 2019      # train + validation for the final fit


def current(path, cols):
    """Is this pickle from the CURRENT feature set?

    Timestamps cannot answer this after a partial run -- some files are new,
    some are old, and both look like files. The stored column list can: a
    model fitted before the MJO join has no mjo_amp in it, whatever its mtime
    says.
    """
    if not path.exists():
        return False
    try:
        with open(path, "rb") as f:
            mp = pickle.load(f)
        return list(mp.get("cols", [])) == list(cols)
    except Exception:
        return False


def main():
    defaults_only = "--defaults-only" in sys.argv
    log("=" * 74)
    log("STEP 8b — refit and persist, defaults first, resumable")
    log("=" * 74)

    res = pd.read_csv(OUTD / "bakeoff_results_lgd.csv")
    base = bo.build_samples()
    mcols = base.attrs.get("mcols") or []
    jcols = [c for c in (base.attrs.get("jcols") or []) if c in base.columns]
    XCOLS = ([f"ant_rain_{h}" for h in bo.ANT]
             + [f"ant_dep_{h}" for h in bo.ANT]
             + ["wet30", "extreme30", "dsw", "doy_sin", "doy_cos"]
             + list(mcols) + list(jcols) + ["district_id"])
    log(f"  {len(XCOLS)} features"
        + (f", MJO present: {', '.join(jcols)}" if jcols else ", no MJO"))

    skill, catalogue = {}, {}
    for H in bo.HORIZONS:
        d = base.dropna(subset=[f"dep_{H}", f"fwdnorm_{H}"])
        tr = d[d.year <= TRAIN_END_FINAL]
        te = d[d.year >= bo.TEST_START]
        hres = res[res["horizon"] == H].sort_values("mse_skill",
                                                    ascending=False)
        if hres.empty:
            log(f"  H={H}: no rows in bakeoff_results_lgd.csv — skipped")
            continue
        win = hres.iloc[0]
        # the blend's members are the next best three single formulations
        members_of_blend = [(r.learner, r.target) for r in
                            hres[~hres["learner"].astype(str)
                                 .str.startswith("Blend")]
                            .head(3).itertuples()]
        log(f"\n  H={H}d  train+val {len(tr):,}  test {len(te):,}")
        log(f"    default: {win["learner"]}/{win["target"]} "
            f"(skill {win["mse_skill"]:+.4f})")
        if str(win["learner"]).startswith("Blend"):
            log(f"    blend members: "
                + ", ".join(f"{a}/{b}" for a, b in members_of_blend))

        entries = []
        for r in hres.itertuples():
            key = f"{r.learner}/{r.target}"
            slug = key.replace("/", "_").replace("(", "").replace(")", "")
            path = OUTD / f"model_dep{H}_{slug}_lgd.pkl"
            members = (members_of_blend
                       if str(r.learner).startswith("Blend")
                       else [(r.learner, r.target)])

            if current(path, XCOLS):
                log(f"    = {key:<24} already current, skipped")
            else:
                t0 = time.time()
                fitted = []
                for learner, target in members:
                    mk = next((m for (l, t, m) in bo.make_models()
                               if l == learner and t == target), None)
                    if mk is None:
                        continue
                    mdl, _ = bo.fit_predict(mk, target, tr[XCOLS], tr,
                                            te[XCOLS], te, H)
                    fitted.append({"model": mdl, "target": target,
                                   "learner": learner})
                if not fitted:
                    log(f"    ! {key:<24} no constructor — skipped")
                    continue
                with open(path, "wb") as f:
                    pickle.dump({"members": fitted, "cols": XCOLS,
                                 "offset": bo.OFF,
                                 "min_normal": bo.MIN_NORMAL,
                                 "kind": "blend" if len(fitted) > 1
                                         else "single"}, f)
                log(f"    + {key:<24} fitted and written "
                    f"({time.time() - t0:.0f}s)")

            entries.append({
                "key": key, "slug": slug, "learner": str(r.learner),
                "target": str(r.target), "mse_skill": float(r.mse_skill),
                "corr": float(r.corr), "rmse": float(r.rmse),
                "imd_cat_acc": float(r.imd_cat_acc),
                "beats_climatology": bool(r.mse_skill > 0),
                "is_default": key == f"{win["learner"]}/{win["target"]}"})

            #   The served default is copied out as soon as it exists, so an
            #   interruption from here on leaves a CORRECT default rather than
            #   a stale one paired with fresh scores.
            if entries[-1]["is_default"] and path.exists():
                (OUTD / f"model_dep{H}_lgd.pkl").write_bytes(path.read_bytes())
                #   BRACKETS, NOT ATTRIBUTES.  `win` is a Series, and
                #   `win.corr` resolves to Series.corr -- the correlation
                #   METHOD -- not to the column named "corr".  Python happily
                #   hands back the bound method and float() then raises
                #   "argument must be a real number, not 'method'".  Any
                #   column whose name collides with a pandas method (corr,
                #   count, min, max, mean, rank, size...) has this problem, so
                #   every field here is read by key.
                skill[f"dep_{H}"] = {
                    "rmse": float(win["rmse"]),
                    "clim_rmse": float(win["clim_rmse"]),
                    "mse_skill": float(win["mse_skill"]),
                    "corr": float(win["corr"]),
                    "imd_cat_acc": float(win["imd_cat_acc"]),
                    "model": f"{win['learner']}/{win['target']}",
                    "n_test": int(len(te))}
                (OUTD / "skill_lgd.json").write_text(
                    json.dumps(skill, indent=1), encoding="utf-8")
                log(f"    -> default written; skill_lgd.json updated "
                    f"({win["mse_skill"]:+.4f})")
                if defaults_only:
                    break

            catalogue[f"dep_{H}"] = entries
            (OUTD / "model_catalogue_lgd.json").write_text(
                json.dumps(catalogue, indent=1), encoding="utf-8")

    n = sum(len(v) for v in catalogue.values())
    log(f"\n  skill_lgd.json and model_catalogue_lgd.json now describe "
        f"{n} model(s) that are actually on disk")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
