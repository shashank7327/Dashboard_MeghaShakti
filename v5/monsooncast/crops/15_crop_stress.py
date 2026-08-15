r"""
v5/monsooncast/crops/15_crop_stress.py  —  crop stress by PHENOLOGICAL STAGE on the
791 LGD districts, from the IMD/ERA5 feature panel.

This is the FAO-33 / FAO-56 stage model rebuilt on the new data.  The
agronomy is unchanged from the earlier version -- only the inputs are
(IMD gauge temperature and the ERA5 water balance on 791 LGD districts,
normals on IMD's 1971-2020 window).

WHY STAGES, AND WHY A PRODUCT
  1. A crop is not equally sensitive throughout its life.  FAO-33 (Doorenbos &
     Kassam 1979) quantifies this with the yield response factor Ky:

         (1 - Ya/Ym) = Ky * (1 - ETa/ETm)

     Ky peaks at flowering for almost every crop -- maize is 0.4 vegetative
     and 1.5 at silking, so the same deficit costs nearly four times as much a
     month later.  A season-average index averages that away.

  2. Heat damage is stage-specific and the REPRODUCTIVE threshold is the
     lowest.  Rice spikelet sterility begins near 33.7 C at anthesis
     (Jagadish, Craufurd & Wheeler 2007, J. Exp. Bot. 58:1627); maize suffers
     above ~33 C at silking (Schlenker & Roberts 2009, PNAS 106:15594); wheat
     loses grain weight above ~31 C during filling (Asseng et al. 2015,
     Nature Clim. Change 5:143).

  3. Water outweighs heat in a rainfed-dominant system -- about 55% of India's
     gross cropped area is unirrigated -- so W_WATER 0.65 against W_HEAT 0.35.

     CSI_stage  = W_WATER * water_stress + W_HEAT * heat_stress
     CSI_season = 1 - PRODUCT over stages of [ 1 - Ky_s * CSI_s ]

  The season combination is MULTIPLICATIVE as FAO-33 specifies.  A Ky-weighted
  MEAN was tried in the earlier version and is wrong: calm stages dilute a
  severe one, so a crop destroyed at flowering scored only "moderate".

PARTIAL SEASONS
  For a season still in progress only some stages have elapsed, so the product
  is bounded below its eventual range and a raw CSI understates the situation.
  csi_to_date rescales the partial product by the WORST outcome still
  attainable from the elapsed stages -- "how bad is this, given how far in we
  are" -- and is the field the dashboard shows for a live season.

WATER STRESS BY SEASON
  Kharif is rainfed, so absolute moisture adequacy against the crop's
  requirement is the operative constraint.  Rabi is a dry-season, largely
  IRRIGATED crop: an absolute rainfed threshold marks all of India stressed
  every winter and carries no information, so Rabi water stress is measured
  RELATIVE to that district-month's own 1971-2020 climatology.

OUTPUT -> v5/data_lgd/
  crop_params_lgd.csv            the crop table, stage calendar and Ky
  crop_stage_stress_lgd.csv      district x crop x season-year x STAGE
  crop_stress_history_lgd.csv    district x crop x season-year (FAO-33 combined)
  crop_stress_snapshot_lgd.csv   latest COMPLETE season per crop
  crop_stress_current_lgd.csv    the season in progress

Run:  py -3.13 -X utf8 "v5/monsooncast/crops/15_crop_stress.py"
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

# Kharif is rainfed-dominant: the water deficit is the primary yield
# determinant (FAO-33) and heat modifies it.  Rabi is grown on irrigation in
# the dry season, so its water term is a SUPPLY question rather than a rainfall
# question, and TERMINAL HEAT is the dominant Indian risk -- mean temperature
# above ~31 C during grain filling (Asseng et al. 2015, Nature Clim. Change
# 5:143; corroborated for Indian wheat by IIWBR/ICAR work on late-sown crops,
# where harvest of the preceding rice pushes wheat into a hotter March).
# Heat therefore carries more weight in Rabi than in Kharif.
W_KHARIF = {"water": 0.65, "heat": 0.35}
W_RABI = {"water": 0.45, "heat": 0.55}
SDD_REF = 90.0            # degree-days above threshold that saturate the index
REF_LO, REF_HI = 1971, 2020        # reference climatology (IMD's own window)

# How much of a monsoon deficit irrigation can offset for a Kharif crop.  Not
# 1.0: even fully-equipped districts depend on the monsoon to fill the
# reservoirs and recharge the groundwater they irrigate from, so irrigation
# buffers a bad monsoon, it does not cancel one.
KHARIF_IRRIG_OFFSET = 0.55
# Rabi water supply is irrigation plus the soil moisture the monsoon left
# behind at sowing; this splits the two.
RABI_IRRIG_WEIGHT = 0.65

# stages: (name, months, Ky, t_upper_C)
CROPS = [
    # ---------------------------------------------------------- KHARIF
    {"crop": "Rice", "season": "Kharif", "t_base": 10, "water_adeq": 0.80,
     "ky_total": 1.10, "ky_source": "FAO-33 rice (paddy)",
     "stages": [("Sowing", [6], 0.40, 38), ("Vegetative", [7], 1.00, 38),
                ("Growing", [8, 9], 1.30, 34), ("Harvesting", [10], 0.50, 36)]},
    {"crop": "Maize", "season": "Kharif", "t_base": 10, "water_adeq": 0.70,
     "ky_total": 1.25, "ky_source": "FAO-33 maize",
     "stages": [("Sowing", [6], 0.40, 38), ("Vegetative", [7], 0.40, 37),
                ("Growing", [8], 1.50, 33), ("Harvesting", [9, 10], 0.50, 36)]},
    {"crop": "Soybean", "season": "Kharif", "t_base": 10, "water_adeq": 0.65,
     "ky_total": 0.85, "ky_source": "FAO-33 soybean",
     "stages": [("Sowing", [6], 0.40, 38), ("Vegetative", [7], 0.60, 37),
                ("Growing", [8], 1.00, 34), ("Harvesting", [9, 10], 0.50, 36)]},
    {"crop": "Tur", "season": "Kharif", "t_base": 7, "water_adeq": 0.60,
     "ky_total": 0.90, "ky_source": "FAO-33 pulses analogue",
     "stages": [("Sowing", [6], 0.40, 38), ("Vegetative", [7, 8], 0.60, 38),
                ("Growing", [9, 10], 1.20, 34), ("Harvesting", [11], 0.40, 36)]},
    {"crop": "Bajra", "season": "Kharif", "t_base": 10, "water_adeq": 0.50,
     "ky_total": 0.90, "ky_source": "FAO-33 sorghum/millet analogue",
     "stages": [("Sowing", [6], 0.30, 42), ("Vegetative", [7], 0.50, 42),
                ("Growing", [8], 1.10, 38), ("Harvesting", [9], 0.40, 40)]},
    {"crop": "Groundnut", "season": "Kharif", "t_base": 10, "water_adeq": 0.65,
     "ky_total": 0.70, "ky_source": "FAO-33 groundnut",
     "stages": [("Sowing", [6], 0.40, 38), ("Vegetative", [7], 0.60, 37),
                ("Growing", [8], 0.80, 34), ("Harvesting", [9, 10], 0.60, 36)]},
    {"crop": "Castor", "season": "Kharif", "t_base": 12, "water_adeq": 0.45,
     "ky_total": 0.80, "ky_source": "oilseed analogue (FAO-33 sunflower)",
     "stages": [("Sowing", [6, 7], 0.30, 42), ("Vegetative", [8], 0.50, 40),
                ("Growing", [9, 10], 1.00, 36), ("Harvesting", [11], 0.40, 38)]},
    {"crop": "Sugarcane", "season": "Kharif", "t_base": 12, "water_adeq": 0.75,
     "ky_total": 1.20, "ky_source": "FAO-33 sugarcane",
     "stages": [("Sowing", [6], 0.75, 40), ("Vegetative", [7, 8], 1.20, 40),
                ("Growing", [9, 10], 0.90, 38), ("Harvesting", [11], 0.50, 38)]},
    {"crop": "Cotton", "season": "Kharif", "t_base": 15, "water_adeq": 0.60,
     "ky_total": 0.85, "ky_source": "FAO-33 cotton",
     "stages": [("Sowing", [6], 0.40, 40), ("Vegetative", [7, 8], 0.60, 38),
                ("Growing", [9, 10], 1.00, 34), ("Harvesting", [11], 0.40, 38)]},
    {"crop": "Guar Seed", "season": "Kharif", "t_base": 10, "water_adeq": 0.45,
     "ky_total": 0.70, "ky_source": "legume analogue",
     "stages": [("Sowing", [7], 0.30, 44), ("Vegetative", [8], 0.50, 42),
                ("Growing", [9], 1.00, 38), ("Harvesting", [10], 0.40, 40)]},
    {"crop": "Ragi", "season": "Kharif", "t_base": 8, "water_adeq": 0.55,
     "ky_total": 0.90, "ky_source": "FAO-33 millet analogue",
     "stages": [("Sowing", [6], 0.30, 38), ("Vegetative", [7, 8], 0.50, 38),
                ("Growing", [9], 1.10, 34), ("Harvesting", [10], 0.40, 36)]},
    {"crop": "Sesame", "season": "Kharif", "t_base": 15, "water_adeq": 0.45,
     "ky_total": 0.75, "ky_source": "oilseed analogue",
     "stages": [("Sowing", [6], 0.30, 42), ("Vegetative", [7], 0.50, 42),
                ("Growing", [8], 1.00, 38), ("Harvesting", [9], 0.40, 40)]},
    # ---------------------------------------------------------- RABI
    {"crop": "Wheat", "season": "Rabi", "t_base": 5, "water_adeq": 0.65,
     "ky_total": 1.00, "ky_source": "FAO-33 winter wheat",
     "stages": [("Sowing", [11], 0.20, 32), ("Vegetative", [12, 1], 0.60, 30),
                ("Growing", [2], 0.60, 28), ("Harvesting", [3, 4], 0.50, 31)]},
    {"crop": "Mustard", "season": "Rabi", "t_base": 5, "water_adeq": 0.55,
     "ky_total": 0.90, "ky_source": "FAO-33 rapeseed analogue",
     "stages": [("Sowing", [10], 0.20, 32), ("Vegetative", [11, 12], 0.55, 30),
                ("Growing", [1], 1.00, 27), ("Harvesting", [2, 3], 0.50, 30)]},
    {"crop": "Chana", "season": "Rabi", "t_base": 5, "water_adeq": 0.55,
     "ky_total": 0.90, "ky_source": "FAO-33 pulses analogue",
     "stages": [("Sowing", [10], 0.20, 32), ("Vegetative", [11, 12], 0.50, 30),
                ("Growing", [1, 2], 1.10, 28), ("Harvesting", [3], 0.45, 32)]},
    {"crop": "Maize (Rabi)", "season": "Rabi", "t_base": 10, "water_adeq": 0.65,
     "ky_total": 1.25, "ky_source": "FAO-33 maize",
     "stages": [("Sowing", [11], 0.40, 34), ("Vegetative", [12, 1], 0.40, 33),
                ("Growing", [2], 1.50, 30), ("Harvesting", [3], 0.50, 33)]},
    {"crop": "Barley", "season": "Rabi", "t_base": 5, "water_adeq": 0.55,
     "ky_total": 1.00, "ky_source": "FAO-33 barley",
     "stages": [("Sowing", [11], 0.20, 32), ("Vegetative", [12, 1], 0.55, 30),
                ("Growing", [2], 0.60, 28), ("Harvesting", [3, 4], 0.45, 31)]},
]


def stress_class(c):
    return np.select([c < 0.25, c < 0.5, c < 0.75],
                     ["Low", "Moderate", "High"], "Severe")


def season_year_offset(month, season):
    """Kharif: calendar year.  Rabi: Jan-Apr belong to the season that began
    the PREVIOUS calendar year."""
    if season == "Kharif":
        return 0
    return np.where(month <= 4, -1, 0)


def main():
    log("=" * 72)
    log("CROP STRESS by phenological stage — 791 LGD districts, IMD/ERA5")
    log("=" * 72)
    log(f"  Kharif  water {W_KHARIF['water']:.2f} / heat {W_KHARIF['heat']:.2f}"
        f"   — monsoon-driven, irrigation offsets up to "
        f"{KHARIF_IRRIG_OFFSET:.0%} of the deficit")
    log(f"  Rabi    water {W_RABI['water']:.2f} / heat {W_RABI['heat']:.2f}"
        f"   — irrigation + residual soil moisture; terminal heat dominant")
    log(f"  season combination multiplicative (FAO-33)")

    p = pd.read_csv(OUTD / "features_lgd.csv", low_memory=False,
                    usecols=["district_id", "state", "district", "year",
                             "month", "tmax_c", "tmin_c", "mai", "swvl2",
                             "pct_departure"])
    p = p.rename(columns={"tmax_c": "tmax", "tmin_c": "tmin"})
    irr = pd.read_csv(OUTD / "irrigation_fraction_lgd.csv")[
        ["district_id", "irrig_index"]]
    p = p.merge(irr, on="district_id", how="left")
    p["irrig_index"] = p["irrig_index"].fillna(p["irrig_index"].median())
    log(f"  irrigation buffer joined: median index "
        f"{p['irrig_index'].median():.2f}")
    p["date"] = pd.to_datetime(dict(year=p["year"], month=p["month"], day=1))
    lt = p.dropna(subset=["tmax"])["date"].max()
    lw = p.dropna(subset=["mai"])["date"].max()
    log(f"  temperature to {lt:%Y-%m}; water (MAI) to {lw:%Y-%m}")

    # Rabi reference: that district-month's own 1971-2020 MAI
    base = p[(p["year"] >= REF_LO) & (p["year"] <= REF_HI)]
    ref = (base.groupby(["district_id", "month"])["mai"]
           .agg(mai_ref="mean", mai_sd="std").reset_index())
    p = p.merge(ref, on=["district_id", "month"], how="left")
    # residual root-zone moisture, referenced to the district's own month
    sref = (base.groupby(["district_id", "month"])["swvl2"]
            .agg(sw_ref="mean", sw_sd="std").reset_index())
    p = p.merge(sref, on=["district_id", "month"], how="left")

    stage_rows, prm = [], []
    for c in CROPS:
        for sname, months, ky, t_up in c["stages"]:
            prm.append({"crop": c["crop"], "season": c["season"],
                        "stage": sname, "ky": ky, "t_upper_c": t_up,
                        "months": "-".join(map(str, months)),
                        "t_base": c["t_base"], "water_adeq": c["water_adeq"],
                        "ky_total": c["ky_total"],
                        "ky_source": c["ky_source"]})
            cm = p[p["month"].isin(months)].copy()
            if cm.empty:
                continue
            d = cm["date"].dt.days_in_month
            cm["season_year"] = cm["year"] + season_year_offset(cm["month"],
                                                                c["season"])
            cm["heat_stress"] = np.clip(
                d * np.maximum(0.0, cm["tmax"] - t_up) / SDD_REF, 0, 1)
            cm["gdd"] = d * np.maximum(
                0.0, (cm["tmax"] + cm["tmin"]) / 2 - c["t_base"])
            if c["season"] == "Kharif":
                # MONSOON PERSPECTIVE.  The crop is fed by the rains, so the
                # deficit is absolute moisture adequacy against the crop's
                # requirement, sharpened by the season's rainfall departure.
                # Irrigation then BUFFERS it -- partially, because the same
                # monsoon fills the reservoirs and aquifers that irrigate.
                raw = np.clip(1 - cm["mai"] / c["water_adeq"], 0, 1)
                dep = cm["pct_departure"]
                rain_pen = np.clip(-dep / 60.0, 0, 1)      # 60% short -> full
                raw = np.clip(0.75 * raw + 0.25 * rain_pen.fillna(raw), 0, 1)
                cm["water_stress"] = np.clip(
                    raw * (1 - KHARIF_IRRIG_OFFSET * cm["irrig_index"]), 0, 1)
                W = W_KHARIF
            else:
                # IRRIGATION PERSPECTIVE.  Rabi is a dry-season crop: winter
                # rainfall is not the supply, so a rainfall-departure term
                # carries almost no signal.  Supply is (a) whether the district
                # can irrigate at all and (b) the soil moisture the monsoon
                # left behind at sowing.  Deficiency in either is the stress.
                short_irrig = 1 - cm["irrig_index"]
                zsw = ((cm["sw_ref"] - cm["swvl2"])
                       / cm["sw_sd"].replace(0, np.nan))
                short_sw = np.clip(zsw / 2.0, 0, 1)
                cm["water_stress"] = np.clip(
                    RABI_IRRIG_WEIGHT * short_irrig
                    + (1 - RABI_IRRIG_WEIGHT) * short_sw.fillna(short_irrig),
                    0, 1)
                W = W_RABI

            hh, hw = cm["heat_stress"].notna(), cm["water_stress"].notna()
            cm["stage_csi"] = np.where(
                hh & hw,
                W["water"] * cm["water_stress"] + W["heat"] * cm["heat_stress"],
                np.where(hw, cm["water_stress"],
                         np.where(hh, cm["heat_stress"], np.nan)))
            cm["components"] = np.where(
                hh & hw, "heat+water",
                np.where(hw, "water-only", np.where(hh, "heat-only", "none")))
            cm = cm.dropna(subset=["stage_csi"])
            if cm.empty:
                continue
            g = cm.groupby(["district_id", "state", "district", "season_year"],
                           as_index=False)
            agg = g.agg(heat_stress=("heat_stress", "max"),      # acute
                        water_stress=("water_stress", "mean"),   # cumulative
                        stage_csi=("stage_csi", "mean"),
                        gdd=("gdd", "sum"),
                        n_months=("stage_csi", "size"),
                        components=("components", "min"))
            agg["crop"], agg["season"], agg["stage"] = c["crop"], c["season"], sname
            agg["ky"], agg["t_upper_c"] = ky, t_up
            agg["stage_months"] = "-".join(map(str, months))
            stage_rows.append(agg)

    pd.DataFrame(prm).to_csv(OUTD / "crop_params_lgd.csv", index=False)
    ST = pd.concat(stage_rows, ignore_index=True)
    for col in ("stage_csi", "heat_stress", "water_stress"):
        ST[col] = ST[col].round(3)
    ST["stage_class"] = stress_class(ST["stage_csi"].to_numpy())

    # ---- crop-area mask ---------------------------------------------------
    mk = pd.read_csv(OUTD / "crop_area_mask_lgd.csv")
    grown = mk[mk["grown"] == 1][["district_id", "crop"]].drop_duplicates()
    before = len(ST)
    ST = ST.merge(grown, on=["district_id", "crop"], how="inner")
    log(f"  crop-area mask: {before:,} -> {len(ST):,} district-crop-stage rows "
        f"({100*(1-len(ST)/before):.1f}% removed as not grown)")
    ST.to_csv(OUTD / "crop_stage_stress_lgd.csv", index=False)

    # ---- FAO-33 multiplicative combination over stages --------------------
    ST["_pc"] = 1.0 - (ST["ky"] * ST["stage_csi"]).clip(0, 1)
    ST["_ph"] = 1.0 - (ST["ky"] * ST["heat_stress"]).clip(0, 1)
    ST["_pw"] = 1.0 - (ST["ky"] * ST["water_stress"]).clip(0, 1)
    ST["_pmax"] = 1.0 - ST["ky"].clip(0, 1)      # worst attainable, this stage

    key = ["district_id", "state", "district", "crop", "season", "season_year"]
    H = (ST.groupby(key, as_index=False)
         .agg(_pc=("_pc", "prod"), _ph=("_ph", "prod"), _pw=("_pw", "prod"),
              _pmax=("_pmax", "prod"), stages_elapsed=("stage", "nunique"),
              gdd=("gdd", "sum")))
    nstage = {c["crop"]: len(c["stages"]) for c in CROPS}
    H["stages_total"] = H["crop"].map(nstage)
    H["season_complete"] = H["stages_elapsed"] >= H["stages_total"]

    # A season can be incomplete for two very different reasons, and only one
    # of them is "in progress".  Rabi season-year Y spans Oct Y..Apr Y+1, so
    # with the record starting in January 1971 the 1970 Rabi season is missing
    # its first three months -- a RECORD-BOUNDARY artefact, not a live season.
    # Only the most recent season-year of each crop can genuinely be underway;
    # earlier incomplete ones are truncated by the start of the data and are
    # dropped rather than reported as if they were still running.
    latest = H.groupby("crop")["season_year"].transform("max")
    edge = (~H["season_complete"]) & (H["season_year"] < latest)
    if edge.any():
        n_edge = int(edge.sum())
        yrs = sorted(H.loc[edge, "season_year"].unique())
        H = H[~edge].copy()
        log(f"  dropped {n_edge:,} rows from season-years truncated by the "
            f"start of the record (season-year {yrs[0]}"
            f"{'..' + str(yrs[-1]) if len(yrs) > 1 else ''})")

    H["csi"] = (1.0 - H["_pc"]).round(3)
    H["csi_max_possible"] = (1.0 - H["_pmax"]).round(3)
    # csi_to_date: the partial product rescaled by the worst outcome the
    # elapsed stages could have produced -- "how bad, given how far in we are"
    H["csi_to_date"] = np.where(
        H["csi_max_possible"] > 1e-9,
        (H["csi"] / H["csi_max_possible"]).clip(0, 1), np.nan).round(3)
    H["heat_stress"] = (1.0 - H["_ph"]).round(3)
    H["water_stress"] = (1.0 - H["_pw"]).round(3)
    H["csi_class"] = np.where(
        ~H["season_complete"],
        stress_class(H["csi_to_date"].fillna(0).to_numpy()),
        stress_class(H["csi"].to_numpy()))
    H = H.drop(columns=["_pc", "_ph", "_pw", "_pmax"])

    # attach the allocated sown area for the stress-weighted ranking
    # The mask is season-aware now, so Kharif rice and Rabi rice carry their own
    # areas instead of both taking the larger.  Crops whose stress season has no
    # matching mask season fall back to the crop's largest allocated area.
    mg = mk[mk["grown"] == 1]
    area_s = (mg.groupby(["district_id", "crop", "season"], as_index=False)
              ["district_area_ha"].max())
    area_c = (mg.groupby(["district_id", "crop"], as_index=False)
              ["district_area_ha"].max()
              .rename(columns={"district_area_ha": "_area_any"}))
    H = H.merge(area_s, on=["district_id", "crop", "season"], how="left")
    H = H.merge(area_c, on=["district_id", "crop"], how="left")
    H["district_area_ha"] = H["district_area_ha"].fillna(H["_area_any"])
    H = H.drop(columns="_area_any")
    H.to_csv(OUTD / "crop_stress_history_lgd.csv", index=False)
    log(f"  history: {len(H):,} district-crop-season rows, "
        f"{H['season_year'].min()}..{H['season_year'].max()}")

    # ---- snapshots --------------------------------------------------------
    comp = H[H["season_complete"]]
    snap = (comp.sort_values("season_year")
            .groupby(["crop", "season"], as_index=False).tail(1)
            .merge(comp, on=list(comp.columns), how="inner")) if len(comp) else comp
    last_complete = (comp.groupby(["crop", "season"])["season_year"].max()
                     .rename("last_year").reset_index())
    snap = comp.merge(last_complete, on=["crop", "season"])
    snap = snap[snap["season_year"] == snap["last_year"]].drop(columns="last_year")
    snap.to_csv(OUTD / "crop_stress_snapshot_lgd.csv", index=False)

    cur = H[~H["season_complete"]]
    cur.to_csv(OUTD / "crop_stress_current_lgd.csv", index=False)
    log(f"  latest complete season per crop: {len(snap):,} rows; "
        f"in-progress: {len(cur):,} rows")

    if len(cur):
        log("\n  SEASON IN PROGRESS — mean csi_to_date by crop:")
        s = (cur.groupby(["crop", "season", "season_year"], as_index=False)
             .agg(mean_to_date=("csi_to_date", "mean"),
                  districts=("district_id", "nunique"),
                  stages=("stages_elapsed", "max"),
                  of=("stages_total", "max"),
                  high=("csi_class", lambda v: int((v.isin(["High", "Severe"])).sum())))
             .sort_values("mean_to_date", ascending=False))
        for r in s.itertuples():
            log(f"    {r.crop:14s} {r.season:6s} {int(r.season_year)}  "
                f"csi_to_date {r.mean_to_date:.3f}  "
                f"{r.stages}/{r.of} stages  {r.districts} districts  "
                f"{r.high} High/Severe")
    log("\n  wrote crop_params_lgd.csv, crop_stage_stress_lgd.csv, "
        "crop_stress_history_lgd.csv, crop_stress_snapshot_lgd.csv, "
        "crop_stress_current_lgd.csv")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
