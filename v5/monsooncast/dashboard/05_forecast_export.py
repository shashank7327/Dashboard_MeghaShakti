r"""
v5/monsooncast/dashboard/05_forecast_export.py  —  STEP 5: forecast the latest issue
date on all 791 LGD units and export the dashboard data (+ a district GeoJSON).

Rebuilds the antecedent features at the last observed date, joins the monthly
covariates, runs the retrained models with per-district calibration, and writes
a self-contained data payload the React dashboard consumes.

OUTPUT -> v5/dashboard_lgd/
  data.json          forecasts + monthly feature layers + ENSO state + registry
  districts.geojson  simplified LGD geometry keyed by district_id (791 incl J&K)

Run:  py -3.13 -X utf8 "v5/monsooncast/dashboard/05_forecast_export.py"
"""
import json
import pathlib
import pickle
import sys

import numpy as np
import pandas as pd
import shapefile
from shapely.geometry import shape, mapping
from shapely.ops import unary_union

HERE = pathlib.Path(__file__).resolve().parent
V5 = HERE.parents[1]
IMD = V5.parent / "IMD_Data"
DATA = V5 / "data_lgd"
OUT = V5 / "dashboard_lgd"
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(V5))
from common_v5 import log  # noqa

ANT = [7, 14, 30, 60, 90]
MIN_NORMAL = 5.0
#   A layer listed here is published only if its column actually exists in
#   the feature panel — see the filter in main(). The optional GEE products
#   (vegetation, ERA5 temperature) can therefore be declared up front and
#   simply not appear until their first export lands, rather than needing a
#   code change on the day the data arrives.
LAYERS = {"pct_departure": "Rainfall vs normal (%)",
          "spei_era5_4": "Drought index (SPEI-4)",
          "spei_era5_12": "Drought index (SPEI-12)",
          "mai": "Soil-moisture adequacy",
          "swvl2": "Root-zone soil moisture",
          "tmax_anom": "Temperature anomaly (degC)",
          "sdd_anom": "Heat stress vs normal (degC-days)",
          "ndvi": "Vegetation greenness (NDVI)",
          "evi": "Vegetation index (EVI)",
          "pct_departure_signature": "El Nino rainfall signature (pp)"}


def imd_anchor():
    try:
        import json as _j
        return float(_j.loads((DATA / "merge_report_lgd.json").read_text()).get("imd_lpa_anchor",1.0))
    except Exception:
        return 1.0


def daily_normal(rain):
    base = rain[(rain.index.year >= 1971) & (rain.index.year <= 2020)]
    per = base.groupby(base.index.dayofyear).mean().reindex(
        range(1, 367)).to_numpy()
    sm = np.nanmean(np.stack([np.roll(per, k, axis=0) for k in range(-2, 3)]),
                    axis=0)
    nrm = pd.DataFrame(sm * imd_anchor(), index=range(1, 367), columns=rain.columns)
    return pd.DataFrame(nrm.reindex(rain.index.dayofyear).to_numpy(),
                        index=rain.index, columns=rain.columns)


def forecast():
    rain = pd.read_pickle(DATA / "daily_rain_lgd.pkl")
    rain.columns = rain.columns.astype(int)
    nrm = daily_normal(rain)
    last = rain.index.max()
    dids = rain.columns.to_numpy()

    # ---- WHICH UNITS MAY BE FORECAST AT ALL --------------------------------
    #   IMD's 0.25 deg gauge grid has no valid cell over the Andaman, Nicobar
    #   and Lakshadweep islands in ANY year, so four units (0 NICOBARS, 1 NORTH
    #   AND MIDDLE ANDAMAN, 2 SOUTH ANDAMANS, 341 Lakshadweep) are all-NaN
    #   across the whole 1971-2026 record.  Every antecedent feature below is
    #   built with a bare .sum(), and pandas sums an all-NaN window to 0.0 --
    #   not NaN.  Left alone the model is therefore told, as fact, that those
    #   districts had 0 mm in the last 7/14/30/60/90 days, 0 wet days in 30, and
    #   (the backward loop never breaking) a dry spell running the entire
    #   20,300-day record.  That is the most extreme drought signature in the
    #   country, and the departure conversion then divides it by a forward
    #   normal that is itself 0.  The result was a published +73.8% 14-day
    #   forecast for Nicobars and +79.1% for Lakshadweep, pinned to the top of
    #   the map's colour scale, for places this system has never observed.
    #
    #   The observed layers already export null for these units.  The forecast
    #   must do the same: no observation in, no number out.  The mask is
    #   deliberately computed from the DATA rather than hard-coded to the four
    #   ids, so a unit that loses its gauge coverage later is caught too.
    valid = rain.iloc[-30:].notna().any().to_numpy()
    if not valid.all():
        log(f"  no rainfall observed in the last 30 days for "
            f"{(~valid).sum()} unit(s) -> "
            f"{', '.join(str(int(i)) for i in dids[~valid])}; "
            f"their forecasts will be exported as null, not fabricated")

    row = {"district_id": dids}
    for H in ANT:
        ar = rain.iloc[-H:].sum()
        an = nrm.iloc[-H:].sum()
        row[f"ant_rain_{H}"] = ar.to_numpy()
        with np.errstate(invalid="ignore", divide="ignore"):
            row[f"ant_dep_{H}"] = np.where(an >= MIN_NORMAL,
                                           (ar - an) / an * 100, np.nan)
    wet = (rain > 1.0).astype(float)
    row["wet30"] = wet.iloc[-30:].sum().to_numpy()
    p95 = rain.where(rain > 1.0).quantile(0.95)
    row["extreme30"] = (rain.iloc[-30:] > p95).sum().to_numpy()
    dsw = (rain > 1.0).to_numpy()
    d = np.zeros(len(dids))
    for c in range(len(dids)):
        run = 0
        for r in range(len(dsw) - 1, -1, -1):
            if dsw[r, c]:
                break
            run += 1
        d[c] = run
    row["dsw"] = d
    doy = last.dayofyear
    X = pd.DataFrame(row)
    X["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    X["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)

    feat = pd.read_csv(DATA / "features_lgd.csv", low_memory=False)
    # Serving must mirror training exactly: step 08 lags the monthly features by
    # one month so nothing unknowable at issue time can reach the model.  Taking
    # the containing month here instead would hand the model July's SPEI on a
    # July issue date -- and in July 2026 that value does not even exist, since
    # step 03 withholds accumulating features for an incomplete month.
    fmo = (pd.Timestamp(last.year, last.month, 1) - pd.offsets.MonthBegin(1))
    fm = feat[(feat.year == fmo.year) & (feat.month == fmo.month)]
    log(f"  monthly features taken from {fmo:%Y-%m} "
        f"(the complete month before the {last:%Y-%m-%d} issue date)")
    X = X.merge(fm.drop(columns=[c for c in ("date", "state", "district")
                                 if c in fm.columns]),
                on="district_id", how="left")

    #   MJO AT THE ISSUE DATE — the same join step 08 makes when training.
    #   If this is missing while the model was trained with it, every district
    #   is served NaN for five features the learners lean on, and the forecast
    #   quietly degrades toward the mean with nothing in the output to say so.
    #   ROMI is a national index, so one row broadcasts to all 791 districts.
    ipath = DATA / "indices_daily.csv"
    if ipath.exists():
        idx = pd.read_csv(ipath, parse_dates=["date"])
        jc = [c for c in ("mjo_amp", "mjo_sin", "mjo_cos", "romi1", "romi2")
              if c in idx.columns]
        row = idx[idx["date"] <= last].tail(1)
        if len(row) and jc:
            age = (last - row["date"].iloc[0]).days
            for c in jc:
                X[c] = float(row[c].iloc[0])
            log(f"  MJO at issue: amp {X['mjo_amp'].iloc[0]:.2f}, "
                f"phase-vector ({X['mjo_sin'].iloc[0]:+.2f}, "
                f"{X['mjo_cos'].iloc[0]:+.2f}), "
                f"from {row['date'].iloc[0]:%Y-%m-%d} ({age}d before issue)")
            if age > 3:
                log(f"    ! ROMI is {age} days stale — refresh "
                    f"Indices/romi.cpcolr.1x.txt before trusting the forecast")
        else:
            log("  ! no ROMI row at or before the issue date — MJO features "
                "will be NaN at serving")
    else:
        log("  ! indices_daily.csv absent — MJO features NaN at serving")

    # forward normal for each horizon: the model members trained on rainfall
    # AMOUNT (tweedie / logratio) return millimetres, which only become a
    # departure once divided by the normal for the SAME forward window.  nrm is
    # a day-of-year climatology, so the future window is available.
    doy_nrm = nrm.groupby(nrm.index.dayofyear).first()
    fwd_norm = {}
    for H in (7, 14):
        days = pd.date_range(last + pd.Timedelta(days=1), periods=H, freq="D")
        fwd_norm[H] = doy_nrm.reindex(days.dayofyear).sum().to_numpy()

    cal = pd.read_csv(DATA / "per_district_calib_lgd.csv")

    def run_model(path, H):
        """One persisted formulation -> a district departure forecast."""
        with open(path, "rb") as f:
            mp = pickle.load(f)
        Xc = X[mp["cols"]]
        N = fwd_norm[H]
        off = mp.get("offset", 1.0)
        mn = mp.get("min_normal", 5.0)
        preds = []
        for mem in mp["members"]:
            raw = mem["model"].predict(Xc)
            t = mem["target"]
            if t in ("raw", "huber"):
                preds.append(raw)
            elif t == "tweedie":
                preds.append((np.clip(raw, 0, None) - N)
                             / np.maximum(N, mn) * 100.0)
            elif t == "logratio":
                amt = np.exp(raw) * (N + off) - off
                preds.append((amt - N) / np.maximum(N, mn) * 100.0)
        pred = np.mean(preds, axis=0)
        cb = cal[cal.horizon == H].set_index("district_id")["b"]
        b = X["district_id"].map(cb).fillna(1.0).to_numpy()
        b = np.clip(0.5 + 0.5 * b, 0.3, 2.0)          # shrunk calibration
        out = np.clip(pred * b, -100, 1000)
        # Withhold, rather than publish, wherever the inputs are not real: no
        # gauge observation in the antecedent window, or no climatological
        # normal to divide by (N == 0 makes the departure conversion above
        # meaningless even where recent rain does exist).  Applied HERE so it
        # covers the two headline horizons and every alternative formulation in
        # the model selector by the same rule.
        return np.where(valid & np.isfinite(N) & (N > 0), out, np.nan)

    out = {"district_id": dids}
    for H in (7, 14):
        out[f"fc{H}"] = run_model(DATA / f"model_dep{H}_lgd.pkl", H)

    # ---- every other formulation, for the dashboard's model selector ------
    #   Which model is "best" depends on the metric: the tweedie fit wins on
    #   mean-squared error while the log-ratio fit lands the most IMD
    #   categories correctly and scores near the bottom on MSE.  Publishing
    #   only the MSE winner hides that trade-off, so each candidate is run and
    #   shipped WITH ITS OWN verified scores rather than presented as
    #   interchangeable.
    alts, cat = {}, {}
    cp = DATA / "model_catalogue_lgd.json"
    if cp.exists():
        cat = json.loads(cp.read_text(encoding="utf-8"))
        for hk, entries in cat.items():
            H = int(hk.split("_")[1])
            for e in entries:
                p = DATA / f"model_dep{H}_{e['slug']}_lgd.pkl"
                if not p.exists():
                    continue
                try:
                    alts.setdefault(hk, {})[e["key"]] = run_model(p, H)
                except Exception as ex:      # a bad pickle must not kill the run
                    log(f"    ! {e['key']} (H={H}) failed to predict: {ex}")
        n = sum(len(v) for v in alts.values())
        log(f"  alternative models: {n} forecasts across "
            f"{len(alts)} horizons")
    else:
        log("  (no model_catalogue_lgd.json — run 08_model_bakeoff to enable "
            "the model selector)")
    return last, pd.DataFrame(out), feat, alts, cat


def build_geojson(reg):
    r = shapefile.Reader(str(IMD / "lgd_shapefile" / "india_districts_lgd.shp"))
    fld = [f[0] for f in r.fields[1:]]
    iD, iS = fld.index("DISTRICT"), fld.index("STATE_UT")

    def norm(s):
        import re
        s = re.sub(r"[^A-Z ]", " ", str(s).upper())
        return re.sub(r"\s+", " ", s).strip()
    lut = {(norm(x.state), norm(x.district)): x.district_id
           for x in reg.itertuples()}
    # DISSOLVE BY DISTRICT — one district must be exactly one feature.
    #   The LGD shapefile carries some districts as several records (Purba
    #   Medinipur in West Bengal is two).  Emitting one feature per record gave
    #   792 features for 791 districts, and since the dashboard keys its map
    #   paths on the district id, React saw two children with key 787 and warned
    #   that one "may be duplicated and/or omitted" -- a district could silently
    #   vanish from the map.  The production build in the standalone HTML never
    #   surfaced it; the Vite dev build did.
    #   Union first, simplify after, so a shared internal border is dissolved
    #   rather than simplified twice into a sliver.
    parts = {}
    for sh, rec in zip(r.iterShapes(), r.iterRecords()):
        did = lut.get((norm(rec[iS]), norm(rec[iD])))
        if did is None:
            continue
        parts.setdefault(int(did), []).append(shape(sh.__geo_interface__))
    multi = {k: len(v) for k, v in parts.items() if len(v) > 1}
    if multi:
        log(f"  geojson: dissolved {len(multi)} district(s) held as multiple "
            f"shapefile records -> " + ", ".join(
                f"{k} ({n} parts)" for k, n in sorted(multi.items())))
    feats = []
    for did, gs in sorted(parts.items()):
        g = gs[0] if len(gs) == 1 else unary_union(gs)
        g = g.simplify(0.01, preserve_topology=True)
        feats.append({"type": "Feature", "properties": {"id": int(did)},
                      "geometry": mapping(g)})
    log(f"  geojson: {len(feats)} features for {len(parts)} districts")
    return {"type": "FeatureCollection", "features": feats}


# layers that are a RATIO of two extensive quantities: they cannot be averaged
# across districts, they have to be rebuilt from their own numerator and
# denominator.  Value -> (numerator column, denominator column).
RATIO_LAYERS = {"pct_departure": ("rain_mm", "normal_mm")}


def build_aggregates(feat, fm, reg, months, mlayers):
    r"""All-India and per-state values for every layer, computed HERE rather
    than in the browser.

    WHY THE BROWSER CANNOT BE TRUSTED WITH THIS
      The dashboard was averaging the district numbers straight:

          natMean = mean(district values)

      For an intensive index that is merely crude -- it lets a 200 km2 district
      count as much as a 30,000 km2 one.  For `pct_departure` it is simply the
      wrong quantity.  Averaging per-district PERCENTAGES answers "what is the
      typical district's departure", not "how much rain did India get against
      normal".  For July 2026 the two differ enormously:

          mean of district percentages     -1.3%     <- what was displayed
          median district                 -10.9%
          ratio of sums, area-weighted     +4.6%     <- the real departure

      A small desert district at +300% cancels several large deficient ones.
      IMD publishes the ratio of sums, and so must we.

    THE RULE
      * ratio layers  ->  100 * (sum(numerator) - sum(denominator))
                                / sum(denominator),   area-weighted
      * everything else -> area-weighted mean

    Area weighting uses district geographic area from the LGD registry.  It is
    not perfect for a rainfall total -- gauge density varies -- but it is
    unambiguous and it is what "all-India" means spatially, whereas an
    unweighted mean silently means "per administrative unit".
    """
    area = reg.set_index("district_id")["area_km2"].astype(float)
    st = reg.set_index("district_id")["state"]

    def wmean(df, col):
        v = pd.to_numeric(df[col], errors="coerce")
        w = df["_area"]
        m = v.notna() & w.notna() & (w > 0)
        return float((v[m] * w[m]).sum() / w[m].sum()) if m.any() else None

    def ratio(df, num, den):
        n = pd.to_numeric(df[num], errors="coerce")
        d = pd.to_numeric(df[den], errors="coerce")
        w = df["_area"]
        m = n.notna() & d.notna() & w.notna() & (w > 0)
        if not m.any():
            return None
        dn = (d[m] * w[m]).sum()
        return float(100.0 * ((n[m] * w[m]).sum() - dn) / dn) if dn else None

    def one(df, lyr):
        if lyr in RATIO_LAYERS:
            num, den = RATIO_LAYERS[lyr]
            if num in df.columns and den in df.columns:
                return ratio(df, num, den)
            return None                      # never silently fall back to mean
        return wmean(df, lyr) if lyr in df.columns else None

    def pack(df, lyr):
        v = one(df, lyr)
        s = {k: one(g, lyr) for k, g in df.groupby("_state")}
        return ({"national": None if v is None else round(v, 2),
                 "state": {k: round(x, 2) for k, x in s.items()
                           if x is not None}})

    f = feat.copy()
    f["_area"] = f["district_id"].map(area)
    f["_state"] = f["district_id"].map(st)
    g = fm.copy()
    g["_area"] = g["district_id"].map(area)
    g["_state"] = g["district_id"].map(st)

    out = {"method": {lyr: ("area-weighted ratio of sums"
                            if lyr in RATIO_LAYERS else "area-weighted mean")
                      for lyr in set(mlayers) | {"fc7", "fc14"}},
           "current": {}, "monthly": {}}
    for lyr in list(LAYERS) + ["fc7", "fc14"]:
        out["current"][lyr] = pack(g, lyr)
    for lyr in mlayers:
        out["monthly"][lyr] = {}
        for ym, s in f[f["ym"].isin(months)].groupby("ym"):
            out["monthly"][lyr][ym] = pack(s, lyr)

    #   A LOG LINE MUST NOT BE ABLE TO KILL THE EXPORT.
    #   This one did: when the running month had too few observed days to
    #   produce a rainfall total, `nat` came back None and the f-string raised
    #   TypeError -- after every aggregate had already been computed. The work
    #   was done and thrown away because of a format specifier.
    nat = out["current"].get("pct_departure", {}).get("national")
    naive = pd.to_numeric(g.get("pct_departure"), errors="coerce").mean()
    how = "ratio of sums" if "pct_departure" in RATIO_LAYERS else "mean"
    if nat is None:
        log(f"  ! all-India pct_departure is undefined for the current month "
            f"— no district has both a rainfall total and a normal. The "
            f"monthly rainfall layers will be blank; the daily layers are "
            f"unaffected.")
    else:
        log(f"  all-India aggregates computed by area-weighted {how}: "
            f"pct_departure {nat:+.2f}% "
            f"(an unweighted mean of district percentages would read "
            f"{naive:+.2f}% — a different and wrong quantity)")
    return out


def main():
    log("=" * 70)
    log("STEP 5 — forecast latest date + export dashboard data (791 units)")
    log("=" * 70)
    reg = pd.read_csv(IMD / "registry_lgd791.csv")
    last, fc, feat, alts, cat = forecast()
    log(f"  forecast issued at {last:%Y-%m-%d}: "
        f"fc7 mean {fc['fc7'].mean():+.1f}%  fc14 mean {fc['fc14'].mean():+.1f}%")

    #   Publish only the layers the panel actually has. The optional GEE
    #   products (vegetation, ERA5 temperature) are declared in LAYERS up
    #   front so that the day their first export lands they appear without a
    #   code change — but until then a declared-and-absent layer would ship
    #   as a column of nulls, and a map that is uniformly blank reads as a
    #   broken product rather than an absent one.
    global LAYERS
    absent = [k for k in LAYERS if k not in feat.columns]
    if absent:
        log(f"  layers declared but not in the panel, omitted: {absent}")
        LAYERS = {k: v for k, v in LAYERS.items() if k in feat.columns}
    log(f"  publishing {len(LAYERS)} layers: {', '.join(LAYERS)}")

    fm = feat[(feat.year == last.year) & (feat.month == last.month)].copy()
    fm = fm.merge(fc, on="district_id", how="left")

    #   ECMWF ENS, offered ALONGSIDE the statistical forecast rather than in
    #   place of it. It is a genuine forecast -- explicit lead time, 50
    #   members, a real distribution -- which is exactly why it is labelled
    #   differently from the CFSv2 layers, whose Earth Engine feed has no
    #   lead-time axis at all.
    #
    #   It is NOT verified here: ECMWF open data carries no reforecast
    #   archive, so no skill score exists for it and none is shown. The run
    #   time travels with it so the reader can see how old it is instead of
    #   assuming it is current.
    ecm = DATA / "ecmwf_forecast_latest.csv"
    ecmeta = {}
    if ecm.exists():
        e = pd.read_csv(ecm)
        fm = fm.merge(e, on="district_id", how="left")
        mp = DATA / "ecmwf_meta.json"
        if mp.exists():
            ecmeta = json.loads(mp.read_text(encoding="utf-8"))
            LAYERS.update(ecmeta.get("layers", {}))
            flag = " (STALE)" if ecmeta.get("stale") else ""
            log(f"  ECMWF ENS merged: run {ecmeta.get('run')}{flag}, "
                f"{ecmeta.get('members')} members, leads "
                f"{ecmeta.get('steps_h')} h")
            if ecmeta.get("stale"):
                log(f"    ! {ecmeta.get('age_hours', 0)/24:.1f} days old — "
                    f"re-run forecast_input/25_ecmwf_opendata_lgd.py")
    else:
        log("  (no ecmwf_forecast_latest.csv — run features/27_ecmwf_layers.py "
            "to offer the ECMWF ensemble as a forecast product)")
    # features_lgd already carries state, district and is_jk_ladakh

    def clean(v):
        return None if v is None or not np.isfinite(v) else round(float(v), 2)
    districts = []
    for r in fm.itertuples():
        rec = {"id": int(r.district_id), "d": r.district, "s": r.state,
               "jk": int(r.is_jk_ladakh), "fc7": clean(r.fc7),
               "fc14": clean(r.fc14)}
        for lyr in LAYERS:
            rec[lyr] = clean(getattr(r, lyr, None))

        #   HEAT STRESS TRAVELS WITH THE PAIR IT CAME FROM.
        #   sdd_anom alone is unreadable: +0.7 degC-days is a large departure
        #   in August, when the normal itself is under 1, and noise in May,
        #   when it is over 100. The briefing shows actual and normal beside
        #   the anomaly so the reader can tell those two cases apart.
        #
        #   A PERCENT-OF-NORMAL IS DELIBERATELY NOT OFFERED. sdd_normal is
        #   exactly 0 for 52% of the record and under 1 degC-d right through
        #   the monsoon, so a ratio would be a division by roughly nothing --
        #   the same trap MIN_MM guards against on rainfall departure. The
        #   difference in degree-days is the honest form.
        #
        #   The normal used is the one the anomaly was actually built from:
        #   day-matched to-date for an incomplete month, full-month otherwise.
        nrm = getattr(r, "sdd_todate_normal", None)
        if nrm is None or not np.isfinite(nrm):
            nrm = getattr(r, "sdd_normal", None)
        rec["sdd"] = clean(getattr(r, "sdd", None))
        rec["sdd_norm"] = clean(nrm)
        districts.append(rec)

    # ---- monthly layers for the date slider (last 24 months) --------------
    #   The date slider carries only layers that HAVE a monthly series in the
    #   feature panel. Two kinds are excluded:
    #     - the ENSO signature, a static per-district climatology that would
    #       render 24 identical frames
    #     - the ECMWF leads, which are a single forward snapshot from one
    #       model run and have no history to scrub through at all
    #   Without the `in feat.columns` test the slider would ask for a monthly
    #   series that does not exist and the export would die building it.
    mlayers = [k for k in LAYERS
               if k in feat.columns and k != "pct_departure_signature"]
    feat = feat.copy()
    feat["ym"] = feat["date"].astype(str).str.slice(0, 7)
    months = sorted(feat["ym"].unique())[-24:]
    monthly = {}
    for lyr in mlayers:
        monthly[lyr] = {}
        for ym, s in feat[feat["ym"].isin(months)].groupby("ym"):
            monthly[lyr][ym] = {int(i): clean(v) for i, v in
                                zip(s["district_id"], s[lyr])
                                if clean(v) is not None}
    log(f"  monthly layers exported for {len(months)} months "
        f"({months[0]}..{months[-1]})")

    #   THE MILLIMETRES BEHIND THE PERCENTAGE.
    #   A rainfall departure is a ratio, and a ratio on a small denominator is
    #   arithmetically correct and practically unreadable: April 2026 has a
    #   district at +570% because 36.4 mm fell where 5.4 mm is normal, and
    #   August has one at +641% on 177.8 mm against a 23.99 mm normal. Neither
    #   is an error -- the second is a genuinely wet month in a dry district --
    #   but a reader shown only "+570%" cannot tell a cloudburst from a
    #   rounding artefact.
    #
    #   Raising MIN_NORMAL_MM does not fix this and was measured before being
    #   rejected: even at a 50 mm floor the largest surviving departure is
    #   still 1072%, because the extremes are real rainfall in dry districts
    #   rather than small-normal noise. Suppressing them would delete true
    #   signal to tidy the display.
    #
    #   So the millimetres travel WITH the percentage and the interface shows
    #   both. This is the same reasoning as the heat-stress pair: an anomaly
    #   without the quantities it came from cannot be judged.
    context = {}
    for ym, s in feat[feat["ym"].isin(months)].groupby("ym"):
        context[ym] = {int(i): [clean(a), clean(n)]
                       for i, a, n in zip(s["district_id"], s["rain_mm"],
                                          s["normal_mm"])
                       if clean(a) is not None or clean(n) is not None}
    nctx = sum(len(v) for v in context.values())
    log(f"  rainfall context (actual + normal, mm) for {nctx:,} district-months")

    agg = build_aggregates(feat, fm, reg, months, mlayers)

    # Each layer's own most recent published month.  They differ now: rainfall
    # departure is day-matched and runs to the current month, while SPEI and the
    # degree-days are withheld for any month their sources do not fully cover,
    # so they stop one month earlier.  The dashboard falls back to this per
    # layer rather than drawing an empty map.
    layer_last = {}
    for lyr in mlayers:
        have = [ym for ym in months if monthly.get(lyr, {}).get(ym)]
        layer_last[lyr] = have[-1] if have else None
    lag = {k: v for k, v in layer_last.items() if v and v != months[-1]}
    if lag:
        log("  layers withheld for the incomplete month (shown at their own "
            "latest): " + ", ".join(f"{k} -> {v}" for k, v in sorted(lag.items())))

    # Which month, if any, is still running — so the interface can label a
    # month-to-date value instead of presenting it as a full month.  The
    # accumulating indices for that month are computed over the days observed
    # and compared against the same window in every prior year, so they are
    # valid; they are simply answering "so far this month" rather than
    # "for this month".
    mtd = None
    if "is_month_to_date" in feat.columns:
        q = feat[feat["is_month_to_date"] == 1]
        if len(q):
            ym_p = str(q["ym"].iloc[0])
            mtd = {"ym": ym_p,
                   "days_observed": int(q["days_observed"].iloc[0])
                   if "days_observed" in q.columns else None,
                   "days_in_month": int(q["days_in_month"].iloc[0])
                   if "days_in_month" in q.columns else None}
            log(f"  {ym_p} is month-to-date "
                f"({mtd['days_observed']} of {mtd['days_in_month']} days); "
                f"accumulating indices computed on that window against the "
                f"same window in prior years")

    # ---- crop stress: the season in progress, per crop ---------------------
    crops = {}
    cs_path = DATA / "crop_stress_current_lgd.csv"
    if cs_path.exists():
        cs = pd.read_csv(cs_path)
        if len(cs):
            sy = int(cs["season_year"].max())
            cs = cs[cs["season_year"] == sy]
            order = (cs.groupby("crop")["csi_to_date"].mean()
                     .sort_values(ascending=False))
            for crop in order.index:
                g = cs[cs["crop"] == crop]
                # area actually at risk, for a commercial reader: sown area
                # in districts scoring High or Severe, not a district count
                risk = g[g["csi_class"].isin(["High", "Severe"])]
                area_ha = g["district_area_ha"].sum(min_count=1)
                risk_ha = risk["district_area_ha"].sum(min_count=1)
                by_state = (g.groupby("state")
                            .apply(lambda x: pd.Series({
                                "csi": x["csi_to_date"].mean(),
                                "ha": x["district_area_ha"].sum(min_count=1)}),
                                include_groups=False)
                            .sort_values("csi", ascending=False).head(6))
                crops[crop] = {
                    "season": str(g["season"].iloc[0]),
                    "season_year": sy,
                    "stages_elapsed": int(g["stages_elapsed"].max()),
                    "stages_total": int(g["stages_total"].max()),
                    "mean": clean(g["csi_to_date"].mean()),
                    "districts": int(g["district_id"].nunique()),
                    "high": int(g["csi_class"].isin(["High", "Severe"]).sum()),
                    "area_mha": clean((area_ha or 0) / 1e6),
                    "risk_mha": clean((risk_ha or 0) / 1e6),
                    "risk_share": clean(100 * (risk_ha or 0) / area_ha)
                    if area_ha else None,
                    "top_states": [{"s": s, "csi": clean(r.csi),
                                    "mha": clean((r.ha or 0) / 1e6)}
                                   for s, r in by_state.iterrows()],
                    "v": {int(i): clean(v) for i, v in
                          zip(g["district_id"], g["csi_to_date"])
                          if clean(v) is not None},
                }
            # stage-by-stage cycle view: where in the crop cycle the stress is
            st_p = DATA / "crop_stage_stress_lgd.csv"
            if st_p.exists():
                st = pd.read_csv(st_p, low_memory=False)
                st = st[st["season_year"] == sy]
                for crop, g in st.groupby("crop"):
                    if crop not in crops:
                        continue
                    order = ["Sowing", "Vegetative", "Growing", "Harvesting"]
                    cyc = (g.groupby("stage")
                           .agg(csi=("stage_csi", "mean"),
                                heat=("heat_stress", "mean"),
                                water=("water_stress", "mean"),
                                ky=("ky", "max"),
                                months=("stage_months", "first")).reindex(order))
                    crops[crop]["cycle"] = [
                        {"stage": s, "csi": clean(r.csi), "heat": clean(r.heat),
                         "water": clean(r.water), "ky": clean(r.ky),
                         "months": (None if pd.isna(r.months) else str(r.months)),
                         "done": bool(pd.notna(r.csi))}
                        for s, r in cyc.iterrows()]
            log(f"  crop stress: {len(crops)} crops, season {sy}, "
                f"{int(cs['stages_elapsed'].max())}/"
                f"{int(cs['stages_total'].max())} stages elapsed")

            # ---- real-time sowing, from 19_sowing_dynamics -------------------
            # Stress says how the crop in the ground is faring; sowing says how
            # much of it is in the ground at all, and whether that is ahead of
            # or behind last year.  A trader reads the two together: a severe
            # index over a crop that is 40% planted is a different exposure from
            # the same index over one that is fully planted.
            sw_p = DATA / "sowing_status_national.csv"
            if sw_p.exists():
                sw = pd.read_csv(sw_p)
                hit = 0
                for crop, k in crops.items():
                    m = sw[(sw["crop"] == crop) & (sw["season"] == k["season"])]
                    if not len(m):
                        continue
                    r = m.iloc[0]
                    k["sowing"] = {
                        "as_of": str(r["week_end"])[:10],
                        "sown_mha": clean(r["area_to_date_ha"] / 1e6),
                        "ly_mha": clean(r["ly_same_week_ha"] / 1e6),
                        "yoy_pct": clean(r["yoy_pct"]),
                        "coverage_pct": clean(r["coverage_pct"]),
                        "pace_pp": clean(r["pace_pp"]),
                        "status": str(r["pace_status"]),
                        # the share of the crop the ratios actually rest on;
                        # the dashboard prints it so a partial basis is visible
                        "basis_pct": clean(r["yoy_basis_pct"]),
                    }
                    hit += 1
                log(f"  sowing status attached to {hit}/{len(crops)} crops "
                    f"(UPAg, as of {sw['week_end'].max()[:10]})")
            else:
                log("  (no sowing_status_national.csv — run 19_sowing_dynamics)")

    enso = pd.read_csv(V5 / "data" / "enso_monthly_v5.csv")
    #   THE CURRENT MONTH USUALLY HAS NO ENSO ROW YET.
    #   ONI and the DMI are monthly means published after the month closes, so
    #   for the first days of every month there is no row for `last.month` and
    #   an .iloc[0] on the empty selection raises "single positional indexer is
    #   out-of-bounds". That is a guaranteed failure on a predictable calendar
    #   date, not an edge case -- it took the whole export down on 4 August
    #   with every other layer already computed.
    #
    #   So: take the most recent month AT OR BEFORE the issue date, and carry
    #   how old it is, because a two-month-old ocean state shown as "current"
    #   is worse than one shown with its date.
    ekey = enso.year * 100 + enso.month
    avail = enso[ekey <= last.year * 100 + last.month]
    if avail.empty:
        raise SystemExit("05: enso_monthly_v5.csv has no month at or before "
                         f"{last:%Y-%m} — refresh the ONI/DMI cache")
    e = avail.sort_values(["year", "month"]).iloc[-1]
    enso_age = ((last.year * 12 + last.month)
                - (int(e["year"]) * 12 + int(e["month"])))
    if enso_age:
        log(f"  ENSO state from {int(e['year'])}-{int(e['month']):02d}, "
            f"{enso_age} month(s) before the issue date — the current month's "
            f"ONI is not published yet")

    # ---- what to CALL the current ocean state ------------------------------
    # `enso_phase` follows CPC's strict rule: five consecutive overlapping
    # seasons at ONI >= +0.5 before an event is declared.  In July 2026 the ONI
    # has been at or above +0.5 for four months (0.51, 0.98, 0.98, 0.98), so the
    # rule returns "Neutral" -- correct by the definition, and badly misleading
    # on a dashboard when the ministry is already attributing a 6% sowing
    # shortfall to El Nino.
    #
    # enso_phase is left exactly as it is: the models are trained on it, and
    # relabelling it would silently change what they learned.  This is a
    # DISPLAY-ONLY status that separates "conditions present now" from "event
    # formally declared", which is the same distinction CPC itself draws
    # between an El Nino Advisory and an episode in the historical record.
    es = enso.sort_values(["year", "month"])
    es = es[(es.year < last.year)
            | ((es.year == last.year) & (es.month <= last.month))]
    run = 0
    for v in reversed(es["oni"].tolist()):
        if pd.notna(v) and v >= 0.5:
            run += 1
        else:
            break
    cold_run = 0
    for v in reversed(es["oni"].tolist()):
        if pd.notna(v) and v <= -0.5:
            cold_run += 1
        else:
            break
    if str(e.enso_phase) != "Neutral":
        status, detail = str(e.enso_phase), "CPC five-season rule met"
    elif run:
        status = "El Nino conditions"
        detail = (f"ONI has been at or above +0.5 for {run} month"
                  f"{'s' if run > 1 else ''}; CPC declares an event at 5")
    elif cold_run:
        status = "La Nina conditions"
        detail = (f"ONI has been at or below -0.5 for {cold_run} month"
                  f"{'s' if cold_run > 1 else ''}; CPC declares an event at 5")
    else:
        status, detail = "Neutral", "ONI within +/-0.5"
    log(f"  ENSO: phase '{e.enso_phase}' (model feature), display status "
        f"'{status}' — {detail}")
    skill = json.loads((DATA / "skill_lgd.json").read_text())

    # ---- assemble the model selector -------------------------------------
    # Each entry pairs a per-district forecast with the scores that model
    # actually earned on the held-out period.  `beats_climatology` is carried
    # explicitly because most of these formulations do NOT, and a selector that
    # hid that would be worse than no selector at all.
    models_payload = {}
    ids = [d["id"] for d in districts]
    for hk, byname in alts.items():
        H = int(hk.split("_")[1])
        # `me`, not `e`: `e` is the ENSO row assigned above, and shadowing it
        # here made the payload build fail on e.oni several lines later
        meta = {m["key"]: m for m in cat.get(hk, [])}
        lst = []
        for key, arr in byname.items():
            me = meta.get(key, {})
            vals = {int(i): clean(v) for i, v in zip(fc["district_id"], arr)}
            lst.append({
                "key": key, "learner": me.get("learner", ""),
                "target": me.get("target", ""),
                "mse_skill": me.get("mse_skill"), "corr": me.get("corr"),
                "imd_cat_acc": me.get("imd_cat_acc"), "rmse": me.get("rmse"),
                "beats_climatology": bool(me.get("beats_climatology", False)),
                "is_default": bool(me.get("is_default", False)),
                "v": {i: vals.get(i) for i in ids if vals.get(i) is not None},
            })
        lst.sort(key=lambda x: -(x["mse_skill"] if x["mse_skill"] is not None
                                 else -9))
        models_payload[f"fc{H}"] = lst
    if models_payload:
        for k, v in models_payload.items():
            nb = sum(1 for m in v if m["beats_climatology"])
            log(f"  model selector {k}: {len(v)} formulations, "
                f"{nb} beat climatology")

    payload = {
        "issued": f"{last:%Y-%m-%d}", "n_units": len(districts),
        "layers": LAYERS, "months": months, "monthly": monthly,
        # rain_mm and normal_mm per district-month, so a percentage departure
        # can be read against the quantities it was formed from
        "context": context,
        "crops": crops, "agg": agg, "layer_last": layer_last,
        #   Run time, member count and lead list travel with the forecast so
        #   the dashboard can state WHEN it was issued rather than implying it
        #   is current. An ensemble forecast with no run time on it is the
        #   easiest thing on the page to misread.
        "ecmwf_meta": ecmeta,
        "month_to_date": mtd,
        # The model selector: every formulation's forecast, each carrying its
        # OWN verified scores so a user can never mistake a confident-looking
        # map for a skilful one.  Ordered by MSE skill, the default first.
        "models": models_payload,
        # The two headline KPIs live HERE, in data.json, not in the HTML
        # builder.  They used to be computed in step 06 and injected only into
        # the inlined HTML payload, so the React project -- which reads this
        # file straight off disk -- had no such keys and displayed "–%".
        "mean_fc7": agg["current"]["fc7"]["national"],
        "mean_fc14": agg["current"]["fc14"]["national"],
        "enso": {"oni": clean(e.oni), "iod": clean(e.iod_dmi),
                 "phase": str(e.enso_phase), "nino34": clean(e.nino34_anom),
                 "status": status, "status_detail": detail,
                 "warm_run_months": run},
        "skill": skill,
        "districts": districts,
    }
    (OUT / "data.json").write_text(json.dumps(payload), encoding="utf-8")
    gj = build_geojson(reg)
    (OUT / "districts.geojson").write_text(json.dumps(gj), encoding="utf-8")
    log(f"  data.json: {len(districts)} districts, {len(LAYERS)} layers")
    log(f"  districts.geojson: {len(gj['features'])} polygons "
        f"({(OUT/'districts.geojson').stat().st_size/1e6:.1f} MB)")
    log(f"  ENSO: ONI {payload['enso']['oni']}, IOD {payload['enso']['iod']}, "
        f"{payload['enso']['phase']}")
    log(f"  wrote data.json + districts.geojson")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
