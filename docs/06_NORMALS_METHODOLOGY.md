# Normals — methodology, and what it was validated against

How I compute the rainfall and temperature normals, how that compares with
IMD's own practice, and the measured size of every place the two differ.

Every number on this page is reproduced by
`v5/monsooncast/validation/07_validate_features.py`, which runs as step 07 of
the pipeline. This document records what I concluded; the script is what
enforces it, so if I ever drift the tests say so rather than the page going
quietly stale.

Validated **2026-08-06** against IMD's published figures.

---

## 1. What IMD does

| Quantity | IMD's baseline | Source |
|---|---|---|
| Rainfall Long Period Average | **1971–2020**, fifty years, adopted April 2022 from 4,132 gauges | IMD LPA revision |
| Temperature normals | **1981–2010**, thirty-year WMO CLINO | *Climatological Normals 1981–2010*, IMD Pune |

Note that these are **two different windows**, by IMD's own practice. The WMO
1991–2020 standard normal is designed for climate-change monitoring and IMD
deliberately does not use it for operational rainfall departure.

IMD's published all-India monthly LPA (1971–2020), in mm:

| Jan | Feb | Mar | Apr | May | Jun | Jul | Aug | Sep | Oct | Nov | Dec | JJAS |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 16.1 | 20.5 | 25.0 | 32.5 | 60.5 | 165.3 | 280.4 | 254.9 | 167.9 | 75.4 | 29.4 | 14.0 | 868.6 |

## 2. What this system does

**Rainfall — I match IMD exactly.** Normals are the 1971–2020 mean, and I
extended the record back to 1971 specifically so the window could be IMD's own.
`NORM_LO/NORM_HI` in `IMD_Data/build_imd_lgd_csvs.py` and `WMO_LO/WMO_HI` in
`v5/monsooncast/cleaning/01_clean_merge_panel.py` both hold 1971–2020, and they
have to stay in step with each other.

**Temperature — I use 1971–2020 where IMD uses 1981–2010.** That is a deliberate
deviation: one window across every variable keeps the panel internally
consistent. Section 5 measures exactly what the choice costs, because I did not
want to make it on instinct.

**Departures are ratios of sums, never means of percentages.** For any
aggregate — state or all-India — the actual and the normal are each
area-weighted and summed, and the ratio taken once:

```
departure % = 100 × (Σ wᵢ·actualᵢ − Σ wᵢ·normalᵢ) / Σ wᵢ·normalᵢ
```

The mean of district percentages is a different quantity and is wrong: for
July 2026 it reads −1.3% where the correct figure is +3.4%.

**Part-months are day-matched.** A running month is compared against a normal
covering exactly the days observed, computed over the same 1–D day window in
every year 1971–2020. This is not a refinement — it is the difference between
a number and nonsense. On 6 August 2026:

| Basis | Normal | Departure |
|---|---|---|
| Day-matched (1–6 Aug, 1971–2020) | 52.6 mm | **+0.0%** |
| Full-month August normal | 247.7 mm | −78.7% |

The second is pure arithmetic error: six days is not thirty-one.

---

## 3. Rainfall normals vs IMD, month by month

All-India, area-weighted, both on the identical 1971–2020 window.

| Month | Ours (mm) | IMD (mm) | Diff | Diff % |
|---|---|---|---|---|
| Jan | 15.8 | 16.1 | −0.3 | −1.6% |
| Feb | 21.3 | 20.5 | +0.8 | +3.8% |
| Mar | 27.8 | 25.0 | +2.8 | +11.4% |
| **Apr** | **38.8** | **32.5** | **+6.3** | **+19.4%** |
| May | 59.7 | 60.5 | −0.8 | −1.4% |
| Jun | 158.9 | 165.3 | −6.4 | −3.9% |
| Jul | 269.2 | 280.4 | −11.2 | −4.0% |
| Aug | 247.7 | 254.9 | −7.2 | −2.8% |
| Sep | 162.7 | 167.9 | −5.2 | −3.1% |
| Oct | 74.2 | 75.4 | −1.2 | −1.6% |
| Nov | 29.7 | 29.4 | +0.3 | +1.0% |
| Dec | 15.0 | 14.0 | +1.0 | +7.3% |
| **JJAS** | **838.5** | **868.6** | **−30.0** | **−3.5%** |

**This is a spatial-estimator difference, not a baseline error.** Both series
sit on IMD's own window, so the window cannot be the explanation. Mine is an
area-weighted mean of 0.25° grid cells over 791 districts; IMD's operational
all-India figure comes off their subdivision-weighted gauge network. The two
track the same signal at a small offset — correlation with IMD's published JJAS
%LPA is **0.99** across 2015–2025, mean absolute difference **2.3 pp**.

The monsoon months run 3–4% low and the dry months high. April is the one
check that does not pass its 8% tolerance (+19.4%), and it is reported rather
than suppressed: on a 32 mm normal a 6 mm difference is large in percentage
and small in millimetres, which is exactly where a grid-vs-gauge estimator
difference shows up worst. It has no bearing on monsoon departure.

---

## 4. Cross-validation against the 2026 season

| Period | Ours | Published | Gap |
|---|---|---|---|
| June 2026 | 106.2 mm, **−33.2%** | 99.5 mm, ≈ −40% (IMD) | 6.7 mm / ~7 pp |
| July 2026 | 278.2 mm, **+3.4%** | "near-normal" after 21–24 Jul rain | consistent |
| 1 Jun – 28 Jul | **−13.7%** | −15% (Skymet) | 1.3 pp |
| 1 Jun – 6 Aug | **−9.1%** | — | — |
| August to date (6 days) | **+0.0%** | — | — |

June is the widest gap and the reason is the one named above: our grid puts
June 2026 6.7 mm wetter than IMD's gauge network, which on a deficit month
turns into ~7 pp of departure. Both agree it was a severe deficit — IMD called
it the fifth-lowest June since 1901. The season-to-date figure, where the
percentage rests on a larger total, agrees to 1.3 pp.

**Read these as the same signal on a slightly different instrument, not as
agreement to the decimal.** Anything quoted publicly should cite IMD's own
figure for all-India and use this system for the district detail IMD does not
publish daily.

---

## 5. The temperature baseline, and what the deviation costs

Switching the temperature normals from 1971–2020 to IMD's 1981–2010 CLINO
moves the district JJAS normals by:

| | mean shift | median | p5 … p95 | max | districts beyond 0.2 °C |
|---|---|---|---|---|---|
| Tmax | +0.004 °C | +0.017 | −0.198 … +0.159 | 0.296 | 5.4% |
| Tmin | +0.015 °C | +0.005 | −0.075 … +0.132 | 0.263 | 1.1% |

A typical district anomaly moves by about **0.02 °C** and the worst by
**0.3 °C**. That is well inside the observational spread of a 1° gauge grid
interpolated to district polygons, so the choice of window is not what limits
the temperature layer's accuracy.

**My decision: keep 1971–2020, and record why.** Aligning to 1981–2010 would
buy no measurable accuracy, would put two different baselines in one panel, and
would move every temperature anomaly the models were fitted on — a retrain for a
hundredth of a degree. Step 07 re-measures the gap on every run, so if it ever
grows the tests will tell me instead of the assumption quietly going stale.

---

## 6. What is checked automatically

`v5/monsooncast/validation/07_validate_features.py`, step 07 of the pipeline:

| Test | What it asserts |
|---|---|
| 1–3 | JJAS LPA and year-by-year %LPA vs IMD published, area-weighted |
| 5 | all-India monthly normal vs IMD's published monthly LPA, all 12 months |
| 6 | the running month's departure equals a day-matched calculation from the daily grid |
| 7 | how far the temperature baseline sits from IMD's CLINO |
| 4 | definitional self-consistency: SPEI ~ N(0,1), MAI ∈ [0, 1.5], degree-days ≥ 0, anomalies mean ~0 |

Current status: **43 of 44 pass**, the exception being the April monthly LPA
described in section 3, which is reported as REVIEW rather than passed
silently.
