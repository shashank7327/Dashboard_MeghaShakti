r"""
v5/monsooncast/crops/19_sowing_dynamics.py  —  real-time sowing coverage and the
year-on-year comparison, from the UPAg weekly releases.

WHAT THIS ADDS THAT THE CROP MASK DID NOT
  14_build_crop_mask.py used UPAg for one purpose only: deciding WHERE a crop is
  grown, so that stress is not reported for crops a district does not sow.  It
  then collapsed season_year away entirely, so the mask says "rice is grown in
  this district" without ever asking whether rice has actually been sown THIS
  year, or how this year compares with last.  Two questions a trader or an FMCG
  buyer asks first were therefore unanswerable from our own outputs:

      how much of the crop is in the ground so far?
      is that ahead of or behind last year at the same point?

  This script answers both, at the weekly resolution UPAg publishes.

WHY THE COMPARISON IS TO LAST SEASON AND NOT TO A "NORMAL"
  UPAg's `NormalValue`, `PreviousValue`, `CurrentYearChange` and
  `PreviousYearChange` columns exist in the files and are 100% EMPTY in every
  row of both releases -- so the comparison cannot be read off and has to be
  computed from the weekly series.  The archive holds 2024 (thin: 8 crops),
  2025 (complete: 14 crops) and 2026 (in progress).  Two prior seasons, one of
  them partial, is not a climatological normal, and calling it one would be
  dishonest.  Every comparison here is therefore explicitly against the PRIOR
  SEASON, and the column names say so.

THE TWO MEASURES
  coverage_pct   area sown so far, as a share of the prior season's FINAL area.
                 "Kharif rice is 62% planted" -- progress through the season.

  pace_pp        coverage_pct now, minus what coverage_pct was at the SAME WEEK
                 last season.  This is the one that carries signal: a crop can
                 be at 62% and be either well ahead or badly behind, depending
                 on the calendar.  Positive = ahead of last year's pace.

  Comparing like-for-like requires matching on ISO week within the same season
  segment, which is what makes the season segmentation in upag_common.py a
  prerequisite rather than a nicety.

CAVEAT CARRIED THROUGH TO THE OUTPUT
  UPAg reports at STATE level.  District values are the state figure allocated
  evenly, exactly as in the crop mask, so district rows are only meaningful
  BETWEEN states, never within one.  Every district row carries
  `area_allocation` saying so.

OUTPUT -> v5/data_lgd/
  sowing_dynamics_state.csv     weekly panel, state x crop x season, with YoY
  sowing_status_current.csv     live snapshot per state x crop
  sowing_status_national.csv    the same, summed to all-India
  sowing_status_district.csv    allocated to the 791 LGD units

Run:  py -3.13 -X utf8 "v5/monsooncast/crops/19_sowing_dynamics.py"
"""
import pathlib
import sys

import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
V5 = HERE.parents[1]
ROOT = V5.parent
IMD = ROOT / "IMD_Data"
OUTD = V5 / "data_lgd"
sys.path.insert(0, str(V5))
sys.path.insert(0, str(IMD))
sys.path.insert(0, str(HERE.parent / "lib"))
from common_v5 import log  # noqa
from build_crosswalk import norm_state  # noqa
from upag_common import load_upag  # noqa

# a season is treated as still running if its last report is within this many
# days of the newest report anywhere in the file
LIVE_DAYS = 45
AHEAD, BEHIND = 3.0, -3.0        # percentage points, either side of "on pace"
MIN_PRIOR_WEEKS = 4              # weeks a prior season needs to define a final
MIN_BASIS_PCT = 50.0             # matched share below which a ratio is withheld
MAX_COVERAGE_PCT = 200.0         # above this the prior-season denominator is bad


def rnd(df, n=2):
    """Round the numeric columns only -- DataFrame.round warns and no-ops on
    the datetime week_end column."""
    df = df.copy()
    num = df.select_dtypes("number").columns
    df[num] = df[num].round(n)
    return df


def pace_label(pp):
    if not np.isfinite(pp):
        return "no prior season"
    if pp >= AHEAD:
        return "Ahead of last year"
    if pp <= BEHIND:
        return "Behind last year"
    return "On pace"


def main():
    log("=" * 74)
    log("SOWING DYNAMICS — season-to-date coverage and year-on-year pace")
    log("=" * 74)

    st, nat = load_upag()
    if st is None or st.empty:
        log("  ! no UPAg Sowing*.csv found — nothing to do")
        return
    asof = st["week_end"].max()
    log(f"  UPAg state panel: {len(st):,} state-crop-week rows, "
        f"latest report {asof:%Y-%m-%d}")
    log(f"  seasons present: " + ", ".join(
        f"{int(y)} {s} ({n} crops)" for (y, s), n in
        st.groupby(["season_year", "season"])["crop"].nunique().items()))

    # ---- weekly panel with the same-week-last-season join -----------------
    st["ks"] = norm_state(st["state_raw"])
    W = st[["ks", "state_raw", "crop", "season", "season_year", "wk",
            "week_end", "area_ha"]].copy()
    ly = W.rename(columns={"area_ha": "ly_same_week_ha"}).copy()
    ly["season_year"] = ly["season_year"] + 1
    W = W.merge(ly[["ks", "crop", "season", "season_year", "wk",
                    "ly_same_week_ha"]],
                on=["ks", "crop", "season", "season_year", "wk"], how="left")
    W["yoy_pct"] = np.where(
        W["ly_same_week_ha"].fillna(0) > 0,
        100.0 * (W["area_ha"] / W["ly_same_week_ha"] - 1.0), np.nan)

    # prior season's FINAL area, the denominator for coverage
    # A prior season reported for only a week or two has no meaningful "final"
    # area -- it was caught mid-ramp -- and dividing by it produces coverage
    # figures over 100% that look like a bumper year and are pure artefact.
    fin = (W.groupby(["ks", "crop", "season", "season_year"], as_index=False)
           .agg(final_ha=("area_ha", "max"), n_weeks=("wk", "nunique")))
    pf = fin[fin["n_weeks"] >= MIN_PRIOR_WEEKS].copy()
    pf["season_year"] = pf["season_year"] + 1
    pf = pf.rename(columns={"final_ha": "prev_final_ha",
                            "n_weeks": "prev_weeks"})
    W = W.merge(pf, on=["ks", "crop", "season", "season_year"], how="left")
    W["coverage_pct"] = np.where(
        W["prev_final_ha"].fillna(0) > 0,
        100.0 * W["area_ha"] / W["prev_final_ha"], np.nan)
    # Where last season stood at this SAME week, as a share of its own final.
    # Deriving this from last year's coverage_pct would need the season BEFORE
    # last as its denominator -- two prior seasons -- and the archive holds only
    # one for Kharif, which is why the first attempt left pace entirely empty.
    # Both terms below come from the prior season itself, so one suffices.
    W["ly_coverage_pct"] = np.where(
        W["prev_final_ha"].fillna(0) > 0,
        100.0 * W["ly_same_week_ha"] / W["prev_final_ha"], np.nan)
    W["pace_pp"] = W["coverage_pct"] - W["ly_coverage_pct"]
    W.sort_values(["state_raw", "crop", "season_year", "week_end"]).to_csv(
        OUTD / "sowing_dynamics_state.csv", index=False)
    log(f"  weekly panel: {len(W):,} rows, "
        f"{int(W['yoy_pct'].notna().sum()):,} with a same-week prior year")

    # ---- live snapshot -----------------------------------------------------
    live = W[(asof - W["week_end"]).dt.days <= LIVE_DAYS]
    if live.empty:
        log(f"  ! no season reported within {LIVE_DAYS} days of {asof:%Y-%m-%d}")
        live = W[W["season_year"] == W["season_year"].max()]
    cur = (live.sort_values("week_end")
           .groupby(["ks", "state_raw", "crop", "season", "season_year"],
                    as_index=False).tail(1)
           .rename(columns={"area_ha": "area_to_date_ha"}))
    cur["pace_status"] = [pace_label(p) for p in cur["pace_pp"]]
    cur = cur.sort_values(["crop", "state_raw"])
    keep = ["state_raw", "crop", "season", "season_year", "wk", "week_end",
            "area_to_date_ha", "ly_same_week_ha", "yoy_pct", "prev_final_ha",
            "coverage_pct", "ly_coverage_pct", "pace_pp", "pace_status"]
    rnd(cur[keep]).to_csv(OUTD / "sowing_status_current.csv", index=False)
    seasons = ", ".join(f"{s} {int(y)}" for y, s in
                        cur[["season_year", "season"]].drop_duplicates()
                        .itertuples(index=False))
    log(f"\n  live snapshot as of {asof:%Y-%m-%d}: {seasons}")
    log(f"    {len(cur):,} state-crop pairs, {cur['crop'].nunique()} crops, "
        f"{cur['state_raw'].nunique()} states")

    # ---- national roll-up --------------------------------------------------
    # A ratio is only honest over states present on BOTH sides.  Summing this
    # year over 33 states and last year over the 20 that happened to report
    # makes every crop look like a record -- which is what produced a spurious
    # +74% for maize and a 1020% "coverage" for summer rice.  The totals below
    # are full-coverage; the ratios are computed on the matched subset only, and
    # `yoy_basis_pct` says how much of the crop that subset represents.
    cur = cur.copy()
    cur["_ly_ok"] = cur["ly_same_week_ha"].fillna(0) > 0
    cur["_cv_ok"] = cur["prev_final_ha"].fillna(0) > 0
    cur["_a_ly"] = cur["area_to_date_ha"].where(cur["_ly_ok"])
    cur["_a_cv"] = cur["area_to_date_ha"].where(cur["_cv_ok"])
    N = (cur.groupby(["crop", "season", "season_year"], as_index=False)
         .agg(area_to_date_ha=("area_to_date_ha", "sum"),
              matched_area_ha=("_a_ly", "sum"),
              ly_same_week_ha=("ly_same_week_ha", "sum"),
              cov_area_ha=("_a_cv", "sum"),
              prev_final_ha=("prev_final_ha", "sum"),
              week_end=("week_end", "max"), states=("state_raw", "nunique"),
              states_matched=("_ly_ok", "sum")))
    N["yoy_pct"] = np.where(N["ly_same_week_ha"] > 0,
                            100 * (N["matched_area_ha"] / N["ly_same_week_ha"]
                                   - 1), np.nan)
    N["yoy_basis_pct"] = np.where(N["area_to_date_ha"] > 0,
                                  100 * N["matched_area_ha"]
                                  / N["area_to_date_ha"], np.nan)
    N["cov_basis_pct"] = np.where(N["area_to_date_ha"] > 0,
                                  100 * N["cov_area_ha"]
                                  / N["area_to_date_ha"], np.nan)
    N["coverage_pct"] = np.where(N["prev_final_ha"] > 0,
                                 100 * N["cov_area_ha"] / N["prev_final_ha"],
                                 np.nan)
    # Coverage resting on a thin slice of the crop is not a national statement.
    # Summer rice is the case in point: the states sowing it in 2026 barely
    # reported a 2025 summer segment, so the ratio came out at 1007% -- an
    # artefact of the denominator, not a tenfold expansion.  Withhold it rather
    # than publish a number that reads as a record crop.
    # Two ways the denominator fails, and both must be caught.  A THIN basis is
    # one where few states have a prior season at all.  The subtler one is a
    # basis that looks complete but whose prior areas are tiny: summer rice
    # passes the basis test at ~100% yet still divides 1.18 M ha by a prior
    # "final" of 0.12 M ha.  No crop expands tenfold in a year at national
    # scale, so a coverage above MAX_COVERAGE_PCT is read as a broken
    # denominator and withheld.
    thin = ((N["cov_basis_pct"].fillna(0) < MIN_BASIS_PCT)
            | (N["coverage_pct"] > MAX_COVERAGE_PCT))
    N.loc[thin, "coverage_pct"] = np.nan
    thin_yoy = N["yoy_basis_pct"].fillna(0) < MIN_BASIS_PCT
    N.loc[thin_yoy, "yoy_pct"] = np.nan
    # national pace: area-weighted mean of the state pace, so a big state moves
    # the number more than a small one
    cp = cur.dropna(subset=["pace_pp"]).copy()
    cp["_w"] = cp["prev_final_ha"].fillna(0) + 1.0
    cp["_wp"] = cp["pace_pp"] * cp["_w"]
    # keyed on crop AND season: merging on crop alone handed the Kharif rice
    # pace to summer rice as well
    wm = cp.groupby(["crop", "season"], as_index=False).agg(_wp=("_wp", "sum"),
                                                           _w=("_w", "sum"))
    wm["pace_pp"] = wm["_wp"] / wm["_w"]
    N = N.merge(wm[["crop", "season", "pace_pp"]], on=["crop", "season"],
                how="left")
    N.loc[thin, "pace_pp"] = np.nan
    N["pace_status"] = [pace_label(p) for p in N["pace_pp"]]
    rnd(N.sort_values("area_to_date_ha", ascending=False)).to_csv(
        OUTD / "sowing_status_national.csv", index=False)

    log(f"\n  ALL-INDIA SOWING, season to date ({asof:%d %b %Y}):")
    log(f"    {'crop':<11}{'season':<8}{'sown M ha':>10}{'LY same wk':>12}"
        f"{'YoY':>9}{'basis':>7}{'covered':>9}{'pace':>9}  status")
    for r in N.sort_values("area_to_date_ha", ascending=False).itertuples():
        ly_s = (f"{r.ly_same_week_ha/1e6:.2f}"
                if np.isfinite(r.ly_same_week_ha) and r.ly_same_week_ha > 0
                else "--")
        yo = f"{r.yoy_pct:+.1f}%" if np.isfinite(r.yoy_pct) else "--"
        bs = f"{r.yoy_basis_pct:.0f}%" if np.isfinite(r.yoy_basis_pct) else "--"
        cv = f"{r.coverage_pct:.0f}%" if np.isfinite(r.coverage_pct) else "--"
        pc = f"{r.pace_pp:+.1f}pp" if np.isfinite(r.pace_pp) else "--"
        log(f"    {r.crop:<11}{r.season:<8}{r.area_to_date_ha/1e6:>10.2f}"
            f"{ly_s:>12}{yo:>9}{bs:>7}{cv:>9}{pc:>9}  {r.pace_status}")

    # cross-check against UPAg's own published national row
    if nat is not None and not nat.empty:
        nl = (nat[(asof - nat["week_end"]).dt.days <= LIVE_DAYS]
              .sort_values("week_end")
              .groupby(["crop", "season"], as_index=False).tail(1))
        chk = N.merge(nl[["crop", "season", "area_ha"]], on=["crop", "season"],
                      how="inner")
        if len(chk):
            d = 100 * (chk["area_to_date_ha"] / chk["area_ha"] - 1)
            log(f"\n    cross-check vs UPAg's own all-India row "
                f"({len(chk)} crops): median difference {d.median():+.1f}%, "
                f"max |diff| {d.abs().max():.1f}%")

    # ---- allocate to the 791 LGD districts ---------------------------------
    reg = pd.read_csv(IMD / "registry_lgd791.csv")
    reg["ks"] = norm_state(reg["state"])
    D = reg[["district_id", "state", "district", "ks"]].merge(
        cur, on="ks", how="inner")
    n_d = reg.groupby("ks")["district_id"].transform("size")
    cnt = reg.groupby("ks")["district_id"].size().rename("n_districts")
    D = D.merge(cnt, on="ks", how="left")
    for c in ("area_to_date_ha", "ly_same_week_ha", "prev_final_ha"):
        D[c.replace("_ha", "_district_ha")] = D[c] / D["n_districts"]
    D["area_allocation"] = "state area / districts in state"
    out = D[["district_id", "state", "district", "crop", "season",
             "season_year", "wk", "week_end",
             "area_to_date_district_ha", "ly_same_week_district_ha",
             "prev_final_district_ha", "yoy_pct", "coverage_pct",
             "ly_coverage_pct", "pace_pp", "pace_status", "area_allocation"]]
    rnd(out).to_csv(OUTD / "sowing_status_district.csv", index=False)
    log(f"\n  district allocation: {len(out):,} district-crop rows across "
        f"{out['district_id'].nunique()} of {len(reg)} LGD units "
        f"({out['crop'].nunique()} crops)")
    miss = sorted(set(reg["ks"]) - set(cur["ks"]))
    if miss:
        log(f"    states with no live UPAg report ({len(miss)}): "
            f"{', '.join(m.title() for m in miss[:8])}"
            f"{' …' if len(miss) > 8 else ''}")
    log(f"\n  wrote sowing_dynamics_state.csv, sowing_status_current.csv, "
        f"sowing_status_national.csv, sowing_status_district.csv")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
