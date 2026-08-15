# GEE_scripts — Earth Engine exports for MonsoonCast

Every layer that does not come from IMD is exported from Earth Engine onto the
**791 LGD district units** and dropped into a product folder that the pipeline
reads. This directory holds one script per product, each self-contained and
ready to paste into the Earth Engine code editor.

IMD rainfall and temperature are **not** here — those come down through Python
(`IMD_Data/download_imd.py`), because `imdlib` talks to IMD's own servers rather
than to Earth Engine.

---

## Layout

| Folder | Script | Writes to | Status |
|---|---|---|---|
| `_shared/` | `00_config.js` | — | the header every script starts with |
| `01_evapotranspiration/` | `era5land_aet_pet_LGD.js` | `Evapotranspiration_LGD/` | **operational** |
| `02_soil_moisture/` | `era5land_soilmoisture_LGD.js` | `SM_LGD/` | **operational** |
| `03_temperature_era5/` | `era5land_t2m_LGD.js` | `TemperatureERA5_LGD/` | cross-check |
| `04_rainfall_crosscheck/` | `chirps_daily_LGD.js` | `Chirps_LGD/` | validation |
| `05_forecast_cfsv2/` | `cfsv2_daily_LGD.js` | `CFSv2_LGD/` | roadmap |
| `06_ocean_indices/` | `oisst_nino34_iod.js` | `Indices/` | cross-check |
| `07_vegetation/` | `modis_ndvi_evi_LGD.js` | `Vegetation_LGD/` | validation |
| `08_static_layers/` | irrigation, cropland | `GlobalIrrigation/` | static, rarely re-run |
| `_archive/` | superseded GAUL-era scripts | — | kept for provenance |

Also here: `verify_exports.py` (run it after every download — see the warning
below) and `LGD_MIGRATION_README.md`, which records why the whole system moved
off GAUL-2024's 701 units onto the 791 LGD boundaries.

**Operational** means the master panel breaks without it. Everything else is a
check, a comparison or a future input — useful, and nothing downstream fails if
it is stale.

---

## The daily routine

Start here every day. It reads every product on disk and prints only what is
actually behind, with the command to fix each one:

```bash
py -3.13 -X utf8 "refresh_status.py"
```

Each product is judged against **its own publication lag**, so the report
distinguishes three things a bare date cannot:

| Status | Meaning |
|---|---|
| `CURRENT` | inside its normal lag — nothing to do |
| `DUE` | the source has new data you do not have — run the action |
| `STALE` | further behind than the source's lag explains — something failed |
| `MISSING` | the folder does not exist yet |

That distinction matters because the lags differ by more than two weeks between
products. CHIRPS sitting 20 days behind is normal; ERA5-Land sitting 20 days
behind means an export was never re-run.

Then, in order:

1. Run whatever the report lists (GEE exports below, IMD via Python).
2. `py -3.13 -X utf8 "GEE_scripts/verify_exports.py"` — refuses overlapping files.
3. `py -3.13 -X utf8 "v5/monsooncast/run_all.py" --update-imd --fast` — rebuild.
4. `py -3.13 -X utf8 "refresh_status.py"` again — the list should be empty.

## The refresh workflow

Each script has a `MODE` switch near the top:

```js
var MODE = 'UPDATE';     // 'UPDATE' | 'YEAR' | 'BACKFILL'
```

| Mode | What it queues | When |
|---|---|---|
| `UPDATE` | the current year to date, one task | the routine refresh |
| `YEAR` | one nominated year (`var YEAR = 2026`) | repairing a specific year |
| `BACKFILL` | a year range, one task per year | first build, or a definition change |

**`UPDATE` re-exports the whole current year, not just the new days.** That is
deliberate — see the warning below.

### Step by step

1. Open the script in the [Earth Engine code editor](https://code.earthengine.google.com/).
2. Read the two lines it prints before you queue anything:
   ```
   ERA5-Land daily — latest available: 2026-07-27
   ERA5-Land daily — days behind today: 7
   ```
   Every reanalysis lags real time and the lag is not constant. Knowing it up
   front is the difference between *"the export is broken"* and *"the data does
   not exist yet"*.
3. Leave `MODE = 'UPDATE'`, run, and start the task from the **Tasks** tab.
4. When it finishes, download the CSV from Drive.
5. Copy it into the product folder, **replacing** the file of the same name.
6. Verify, then run the pipeline:
   ```bash
   py -3.13 -X utf8 "GEE_scripts/verify_exports.py"
   ```
   ```bash
   py -3.13 -X utf8 "v5/monsooncast/run_all.py" --update-imd --fast
   ```

---

## ⚠ The one mistake that silently corrupts the panel

**Never put two files covering the same month in one product folder.**

`v5/monsooncast/cleaning/01_clean_merge_panel.py` globs `*.csv`, aggregates each
file to monthly, and then combines the files. For an extensive quantity that
combination is a **sum**:

```python
out.groupby(["date", "district_id"])[valcols].sum()      # if how == "sum"
```

So `ERA5L_aet_pet_district_daily_2026.csv` sitting next to a hand-named
`..._2026_july_update.csv` makes July's evapotranspiration **double**. Nothing
raises. PET doubles, the SPEI water balance `rain − PET` goes sharply negative,
and the dashboard shows a nationwide drought that is not there.

Two things protect against it:

- **`UPDATE` mode emits the canonical year filename**, so a download replaces the
  file already on disk instead of joining it. This is why it re-exports the whole
  year rather than only the new days — a partial-window file would need a new
  name, and a new name is the bug.
- **`verify_exports.py` fails on overlap** before the pipeline ever runs it.

Products the pipeline **sums** (overlap is fatal): evapotranspiration, CHIRPS,
CFSv2 precipitation.
Products it **averages** (overlap is wrong but less destructive): soil moisture,
temperature, vegetation.

---

## What stays identical across every script

The shared header is duplicated verbatim into each file because the Earth Engine
editor has no cross-file import unless the scripts live in a Git-backed EE
repository, and these are pasted in. If you change one, change all of them.

```js
var ASSET_ID          = 'projects/krishisutra/assets/india_districts_lgd';
var USE_INTERPOLATION = true;
var INTERP_SCALE      = 1000;
reducer:  ee.Reducer.mean()
```

Every layer here is eventually **differenced against another** — the SPEI water
balance is rainfall minus PET, moisture adequacy is AET over PET. If two products
land on even slightly different district geometry, or one is bilinearly resampled
and the other is not, that difference carries a systematic artefact which looks
exactly like signal.

The one deliberate exception is MODIS, reduced at its native 250 m: it is already
finer than the districts, so resampling would only cost time.

---

## Expected latency, by product

| Product | Native lag behind real time | Consequence |
|---|---|---|
| IMD rainfall / temperature | 1–2 days | the operational frontier |
| ERA5-Land (ET, soil moisture, t2m) | ~5–8 days | month-to-date indices switch to Hargreaves PET so the water balance is not built across a window only one source covers |
| MODIS NDVI/EVI | ~8–16 days | 16-day composites; the window start is the date |
| OISST | ~1–2 days daily, but monthly means need a complete month | the script stops at the last **complete** month |
| CHIRPS final | 2–3 weeks | validation only; cannot be operational |
| CFSv2 | ~1 day | archive back to 1979 is the point, not the latency |

The ERA5 lag is why `et_days` is recorded per month in `merge_report_lgd.json`
and why the running month is computed month-to-date on a single consistent PET
model. It is the dataset behaving normally, not a failed export.

---

## Creating the new folders

Three of these products have no folder yet. Create them next to the existing
ones, at the repository root:

```bash
mkdir -p TemperatureERA5_LGD Chirps_LGD CFSv2_LGD Indices Vegetation_LGD
```

`Chirps_LGD/` is distinct from the existing `Chirps/`, which holds the older
GAUL-era exports on the previous boundary vintage. Do not mix them: they key on a
different district registry.
