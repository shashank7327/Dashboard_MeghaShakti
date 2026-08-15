# `v5/monsooncast/` — the pipeline

All cleaning, feature-building, crop, modelling, validation and dashboard code
for MonsoonCast. Run it with [`run_all.py`](run_all.py); see
[`docs/03_RUNBOOK.md`](../../docs/03_RUNBOOK.md) for VS Code setup.

## Layout

| Folder | Scripts | What it does |
|---|---|---|
| `lib/` | `upag_common.py` | shared UPAg sowing loader: unit repair, national-row removal, season segmentation |
| `cleaning/` | `00`, `01`, `02` | MJO/ENSO/IOD indices; IMD + ERA5 + ENSO → master panel; ENSO-phase climatology |
| `features/` | `03`, `17`, `18`, `21`, `26`, `27` | SPEI-1/4/12, GDD, SDD, MAI; reliability; irrigated fraction; daily layers; CFSv2 and ECMWF products |
| `crops/` | `14`, `15`, `19`, `29` | crop-area mask; FAO-33 stress by stage; sowing coverage & YoY pace; DA&FW national sowing |
| `modelling/` | `04`, `08`, `08b`, `09`–`11`, `16` | model bake-off, resumable persist, calibration, spatial experiment, verification |
| `validation/` | `07`, `12`, `20`, `28` | features vs IMD, districts vs IMD, sowing vs DA&FW, training-data audit |
| `dashboard/` | `05`, `06`, `13` | forecast export, dashboard build, master CSV/XLSX |

Filenames keep their step numbers, so execution order stays readable no matter
which folder a script lives in. `run_all.py --list` prints the order.

## Conventions worth knowing

**Paths are absolute, derived from `__file__`.** No script depends on the
current working directory. `V5 = HERE.parents[1]` throughout; anything reaching
outside v5 (IMD_Data, UPAJ, Chirps) goes through `ROOT = HERE.parents[2]`.

**Python 3.13 specifically.** Bare `py` is 3.14 here with no packages. Always
`py -3.13`.

**Observed features are computed, never fitted to IMD.** IMD's district
bulletins are station averages with their own sampling error and do not cover
every LGD unit, so they are used as a cross-check with quantified spread
(`17_feature_reliability.py`), not as ground truth to match.

**Ratios with a thin basis are withheld, not published.** Where a year-on-year
comparison rests on less than half a crop's area, `19_sowing_dynamics.py` emits
nothing and records why. `20_validate_sowing_official.py` reads the published
value rather than recomputing, so the two can never disagree.

**Normals are day-matched for an incomplete month.** The running month is
compared against a normal covering exactly the days observed. Comparing 23 days
of rain against a 31-day normal reported −22% where the true departure was
+4.8%. Step 07 now proves this every run: it recomputes the running month's
departure straight from the daily grid and fails if the panel disagrees by more
than a point.

**Normals are validated against IMD month by month, not just for the season.**
`validation/07_validate_features.py` checks all twelve monthly all-India
normals against IMD's published 1971–2020 LPA, the running month against a
day-matched calculation, and the temperature baseline against IMD's 1981–2010
CLINO. 43 of 44 checks pass; the exception is the April normal (+19.4% vs IMD),
reported as REVIEW because on a 32 mm normal a 6 mm estimator difference is
large in percent and irrelevant in millimetres. Conclusions and the 2026
cross-validation live in [`docs/06_NORMALS_METHODOLOGY.md`](../../docs/06_NORMALS_METHODOLOGY.md).

**`enso_phase` is a model feature and is left alone.** The dashboard shows a
separate display status distinguishing "El Niño conditions present" from CPC's
formally declared event; relabelling the trained feature would silently change
what the models learned.

**The MJO is in the feature set, and it is why the forecast improved.** The
7–14 day target sits in the intraseasonal band, and until `00_clean_indices.py`
existed nothing in the features occupied it: the antecedent block carries
amplitude but not phase, and the monthly and seasonal blocks are too slow to
turn over inside a fortnight. Adding ROMI took the blend from +0.0484 to
**+0.0617** at 7 days and +0.0269 to **+0.0345** at 14, with 23 of 24
model-horizon combinations improving and none regressing.

Phase is encoded on the **unit circle**, never as the octant number: phase 8 and
phase 1 are adjacent, and the integer asserts they are seven apart. Amplitude is
a separate column because it is the confidence — below about 1 the oscillation
is incoherent and the phase means nothing.

**The MJO joins on the issue date, not the previous month.** Every other monthly
covariate is lagged a month because it is an accumulation that does not exist
until the month closes. ROMI is a daily index, already lagged one day for its
own revision window. Lagging it a further month would be a half-cycle phase
error on a 30–90 day oscillation.

**Second-opinion products are additions, never substitutions, and are labelled
by what they actually are.** CFSv2 is served by Earth Engine indexed by valid
time with no lead axis, so it is a model *analysis* and is labelled one; MOS
cannot be trained on it. ECMWF ENS has real lead times and 50 members, so it is
a forecast — but an unverified one, because open data carries no reforecast
archive.

**Run the audit before trusting a retrain.** `validation/28_audit_training_data.py`
checks the sample matrix for leakage, degeneracy, missingness that differs
between train and test, and physical range violations. It found two real
defects the scores could not have revealed: dry-spell length counting
unobserved days as dry (20,304-day spells for seven districts), and ERA5-Land
PET reaching 47 mm/day in the tail.

**`refresh_status.py` says what needs updating today.** Each product is judged
against its own publication lag, so CHIRPS 20 days behind reads CURRENT while
ERA5-Land 20 days behind reads DUE.
