# 3. Runbook — which file to run, in what order

The short answer:

```bash
py -3.13 -X utf8 build_dashboard.py
```

That runs everything below in the right order and stops at the first failure.
I wrote the rest of this page for when you want to run one stage on its own, or
you want to know what that command is actually doing.

---

## Four rules that will save you most of the trouble I had

**Always `py -3.13`, never bare `py`.** On my machine bare `py` resolves to
Python 3.14, which has none of the libraries installed, and the error you get is
a `ModuleNotFoundError` for something you know perfectly well you installed.

**Always `-X utf8`.** The scripts print °C, ± and district names with
non-English characters, and without this flag a Windows terminal throws
`UnicodeEncodeError` halfway through a run.

**Paths are absolute, worked out from each script's own location.** You can run
any script from any directory; you never need to `cd` first.

**A step that exits cleanly has not necessarily finished.** These scripts write
large CSVs in one go, so a run that is killed part-way leaves a file that looks
perfectly valid and is simply short — and every later step then reads it without
complaint. That happened to me once and produced a dashboard covering 288
districts instead of 791, with every single stage reporting success. If you
interrupt a run, **re-run the interrupted step from the start** rather than the
one after it, and check the row counts.

---

## The order

### Stage A — acquire (only when refreshing data)

| # | Command | Produces |
|---|---|---|
| A1 | `Indices/fetch_indices.py` | MJO, ONI, Niño-3.4, IOD series |
| A2 | `IMD_Data/build_imd_lgd_csvs.py` | district daily rain/tmax/tmin |
| A3 | `v5/monsooncast/forecast_input/25_ecmwf_opendata_lgd.py` | ECMWF ensemble |
| A4 | *(Earth Engine, manual)* | ERA5-Land, MODIS, CFSv2, CHIRPS |

### Stage B — the pipeline

Run by `v5/monsooncast/run_all.py`. In order:

| # | Script | What it does |
|---|---|---|
| 43 | `v5/43_build_enso_features.py` | ENSO state features from the index cache. **Run this after A1** — it is outside `run_all.py` and the panel consumes its output |
| 01 | `cleaning/01_clean_merge_panel.py` | Merges IMD + ERA5-Land + ocean indices into the master district-month panel |
| 02 | `cleaning/02_enso_phase_climatology.py` | Per-district El Niño / La Niña climate signatures, fitted on training years only |
| 03 | `features/03_build_features.py` | SPEI-1/4/12, moisture adequacy, growing and stress degree-days |
| 21 | `features/21_daily_features.py` | Daily rolling layers: 7-day, 30-day and season-to-date departures |
| 17 | `features/17_feature_reliability.py` | Per-district data support and observational spread |
| 18 | `features/18_irrigation_fraction.py` | Irrigated fraction per district |
| 07 | `validation/07_validate_features.py` | 44 checks against IMD's published figures |
| 08 | `modelling/08_model_bakeoff.py` | Trains and selects the forecast model. **Slow (~40 min)** — skipped by `--fast` |
| 14 | `crops/14_build_crop_mask.py` | Which crops are grown where |
| 19 | `crops/19_sowing_dynamics.py` | Sowing coverage and year-on-year pace |
| 29 | `crops/29_dafw_sowing.py` | Parses the Ministry's weekly release |
| 20 | `validation/20_validate_sowing_official.py` | Our sowing totals vs the Ministry's |
| 15 | `crops/15_crop_stress.py` | FAO-33 stage-weighted crop stress, 12 crops |
| 26 | `features/26_cfsv2_layers.py` | CFSv2 second-opinion layers |
| 27 | `features/27_ecmwf_layers.py` | ECMWF ensemble as a selectable forecast |
| 05 | `dashboard/05_forecast_export.py` | Runs the forecast, writes `data.json` |
| 06 | `dashboard/06_build_dashboard.py` | Generates the React application source |
| 13 | `dashboard/13_export_masters.py` | Spreadsheet exports (CSV + Excel) |
| 30 | `validation/30_audit_display.py` | Checks every layer the dashboard offers actually has values |

### Stage C — confirm

```bash
py -3.13 -X utf8 "v5/monsooncast/validation/30_audit_display.py"
```

This walks the same chain a reader does — file on disk, then column in the
panel, then layer in `data.json`, then values per month, then the button in the
built app — and tells you the first place a layer stops having values and why. It
should end with **PASS**.

---

## Common tasks

**Daily refresh with fresh data:**
```bash
py -3.13 -X utf8 build_dashboard.py --fetch
```

**Rebuild after editing the dashboard's appearance** (seconds, not minutes):
```bash
py -3.13 -X utf8 "v5/monsooncast/dashboard/06_build_dashboard.py"
```

**Resume after a failure at step NN:**
```bash
py -3.13 -X utf8 "v5/monsooncast/run_all.py" --from NN --fast
```

**See the step list without running anything:**
```bash
py -3.13 -X utf8 "v5/monsooncast/run_all.py" --list
```

**Retrain the models** (only needed when features change, not for new data):
```bash
py -3.13 -X utf8 build_dashboard.py --full
```

---

## Viewing and developing the dashboard

**To view the built dashboard** — no Node, no install:

```bash
py -3.13 view_dashboard.py
```

**To develop the interface**, with live reload as you edit — needs
[Node.js](https://nodejs.org/) 18 or newer:

```bash
npm --prefix v5/dashboard_react install
npm --prefix v5/dashboard_react run dev
```

Then open the address it prints (usually `http://localhost:5180`). Remember
that `src/App.jsx` is generated: edit `dashboard/06_build_dashboard.py` and
re-run step 06, or your changes are lost.

**To publish a new static build** into `dashboard/`:

```bash
npm --prefix v5/dashboard_react run build
```

`build_dashboard.py` does this for you at the end of a full run.

---

## When something goes wrong

| Symptom | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError` | ran bare `py` instead of `py -3.13` | use `py -3.13` |
| `UnicodeEncodeError` | missing `-X utf8` | add the flag |
| A layer is blank on the map | its source has not been downloaded | `refresh_status.py`, then `02_DATA_EXTRACTION.md` |
| District count below 791 | a truncated write from an interrupted run | re-run the interrupted step from scratch |
| `13_export_masters.py` fails | a master workbook is open in Excel | close it |
| Blank page after double-clicking `dashboard/index.html` | browsers block JavaScript apps opened from disk | use `py -3.13 view_dashboard.py` |
| `npm` not found during a build | Node.js is not installed | install Node 18+, or ignore — the data pipeline still runs |
