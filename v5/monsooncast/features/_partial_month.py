r"""
v5/monsooncast/features/_partial_month.py  —  valid indices for a month that is
still running, instead of a blank map.

THE PROBLEM
  SPEI and the degree-days are ACCUMULATIONS.  A 25-day accumulation compared
  against a distribution fitted on 31-day months is meaningless, and the
  previous build handled that by withholding the value.  Statistically sound,
  operationally poor: the running month is the one a user actually cares about,
  and an empty map reads as a broken product rather than a careful one.

THE FIX — COMPARE LIKE WITH LIKE RATHER THAN NOT COMPARING
  For the trailing incomplete month with D days observed, the quantity is
  recomputed over the FIRST D DAYS OF THAT CALENDAR MONTH IN EVERY YEAR of the
  record.  Current value and reference distribution then cover the same window,
  so the standardised result is directly interpretable: "how unusual is this
  month-to-date against the same window in the past fifty years".  Degree-days
  gain a day-matched normal for the same reason.

  Only the trailing month is affected.  Complete months keep their full-month
  values unchanged, so no historical figure moves.

WHY HARGREAVES PET AND NOT ERA5 FOR THIS ONE CASE
  The water balance is precipitation minus potential evapotranspiration, and
  the two sources behind it end on different days — ERA5-Land runs about a week
  behind the IMD gauge analysis.  Mixing them is what previously made the
  running month score WETTER than a normal July during a season 13% below
  normal.  Hargreaves (1985) PET needs only daily temperature extremes and
  extraterrestrial radiation, so it ends on exactly the same day as the
  rainfall.  Carrying a second, independent PET estimate was always intended as
  a hedge against the reanalysis lagging; this is that hedge being used.

  The two PET formulations differ in level, which is why the partial-window
  SPEI is fitted on its OWN Hargreaves-based series across all years rather
  than being spliced onto the ERA5-based distribution.  Like is compared with
  like on both axes: same days, same PET model.
"""
import numpy as np
import pandas as pd

# FAO-56 mid-month extraterrestrial radiation, mm/day equivalent
_DOY = np.array([15, 46, 74, 105, 135, 166, 196, 227, 258, 288, 319, 349])


def _ra(lat_deg, doy):
    phi = np.deg2rad(np.asarray(lat_deg, dtype=float))
    dr = 1 + 0.033 * np.cos(2 * np.pi * doy / 365.0)
    dec = 0.409 * np.sin(2 * np.pi * doy / 365.0 - 1.39)
    x = np.clip(-np.tan(phi) * np.tan(dec), -1, 1)
    ws = np.arccos(x)
    return (24 * 60 / np.pi) * 0.0820 * dr * (
        ws * np.sin(phi) * np.sin(dec)
        + np.cos(phi) * np.cos(dec) * np.sin(ws)) * 0.408


def align(*frames):
    """Common date index and column set across frames.

    The three IMD daily pickles do not always carry identical indices — tmax
    and tmin ended one day apart in the reference build — so every arithmetic
    combination has to be aligned first rather than assuming a shared shape.
    """
    idx = frames[0].index
    cols = frames[0].columns
    for f in frames[1:]:
        idx = idx.intersection(f.index)
        cols = cols.intersection(f.columns)
    return [f.reindex(index=idx, columns=cols) for f in frames]


def daily_hargreaves(tmax, tmin, lat_by_district):
    """Daily Hargreaves PET, same shape as the daily temperature frames."""
    tmax, tmin = align(tmax, tmin)
    lat = np.asarray([lat_by_district.get(int(c), np.nan)
                      for c in tmax.columns], dtype=float)
    doy = tmax.index.dayofyear.to_numpy()
    ra = np.vstack([_ra(lat, dd) for dd in doy])          # days x districts
    tr = (tmax.to_numpy() - tmin.to_numpy())
    tr = np.where(np.isfinite(tr) & (tr > 0), tr, np.nan)
    tm = (tmax.to_numpy() + tmin.to_numpy()) / 2.0
    pet = 0.0023 * ra * np.sqrt(tr) * (tm + 17.8)
    return pd.DataFrame(np.clip(pet, 0, None), index=tmax.index,
                        columns=tmax.columns)


def window_sums(daily, month, ndays, min_count=1):
    """Sum the first `ndays` of `month`, per year. -> year x district."""
    s = daily[(daily.index.month == month) & (daily.index.day <= ndays)]
    if not len(s):
        return None
    return s.groupby(s.index.year).sum(min_count=min_count)


def partial_spei(rain_daily, tmax_daily, tmin_daily, lat, month, year, ndays,
                 spei_fn, scales=(1, 4, 12), lo=1971):
    r"""Month-to-date SPEI for one (year, month), fitted on the same window
    across every year.

    Returns {scale: {district_id: value}}.  For scales beyond 1 the preceding
    months are taken whole from the same daily data, so the accumulation is
    "the last (s-1) complete months plus this month to date" — consistently
    defined for every year in the fit.
    """
    pet = daily_hargreaves(tmax_daily, tmin_daily, lat)
    rain_a, pet_a = align(rain_daily, pet)
    wb = rain_a - pet_a
    wb = wb[wb.index.year >= lo]

    part = window_sums(wb, month, ndays)
    if part is None or len(part) < 20:
        return {}

    out = {}
    for s in scales:
        acc = part.copy()
        ok = True
        for k in range(1, s):
            mm, back = month - k, 0
            while mm <= 0:
                mm += 12
                back += 1
            full = wb[wb.index.month == mm]
            if not len(full):
                ok = False
                break
            fs = full.groupby(full.index.year).sum(min_count=15)
            fs.index = fs.index + back          # align to the anchor year
            acc = acc.add(fs.reindex(index=acc.index, columns=acc.columns))
        if not ok:
            continue
        if year not in acc.index:
            continue
        i = list(acc.index).index(year)
        vals = {}
        for did in acc.columns:
            col = acc[did].to_numpy(float)
            if np.isfinite(col).sum() < 20:
                continue
            z = spei_fn(col)
            if i < len(z) and np.isfinite(z[i]):
                vals[int(did)] = round(float(z[i]), 4)
        if vals:
            out[s] = vals
    return out


def partial_thermal(tmax_daily, tmin_daily, month, year, ndays,
                    lo=1971, hi=2020):
    r"""Month-to-date degree-days and their day-matched normals.

    Returns {column: {district_id: value}} for the to-date sums and for the
    matching normals, so a partial total can be read against what is normal by
    this point in the month rather than against a whole one.
    """
    tmax_daily, tmin_daily = align(tmax_daily, tmin_daily)
    tmean = (tmax_daily + tmin_daily) / 2.0
    res = {}
    defs = [("gdd_kharif", tmean, 10.0), ("gdd_rabi", tmean, 5.0),
            ("sdd", tmax_daily, 34.0)]
    for name, frame, base in defs:
        w = window_sums((frame - base).clip(lower=0), month, ndays)
        if w is None or year not in w.index:
            continue
        res[name] = {int(c): round(float(v), 3)
                     for c, v in w.loc[year].items() if np.isfinite(v)}
        ref = w[(w.index >= lo) & (w.index <= hi)]
        if len(ref):
            mn = ref.mean()
            res[name + "_todate_normal"] = {
                int(c): round(float(v), 3) for c, v in mn.items()
                if np.isfinite(v)}
    return res
