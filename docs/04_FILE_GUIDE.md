# 4. File guide — every file, explained

I have grouped these by job. If you only read one section, read the one
matching whatever you have been asked to change.

Every script also explains itself in the comment block at the top — what it
does, why it exists, and what breaks if you skip it. I wrote those as I went and
they are the real documentation; this page is just the map to them.

---

## Top level

| File | What it is |
|---|---|
| `README.md` | Start here. What the system is, how to open it, how to rebuild it |
| `build_dashboard.py` | **The one command.** Checks the environment and data, runs the whole pipeline, then rebuilds and publishes the dashboard |
| `view_dashboard.py` | Serves the built dashboard and opens your browser. Uses only the Python standard library |
| `refresh_status.py` | "What is out of date today?" Judges each dataset against its own publication lag and prints the exact command to fix it |
| `requirements.txt` | The Python libraries to install |
| `noaa_indices_cache.csv` | El Niño (ONI, Niño-3.4) and IOD values by month, maintained by the index fetcher |
| `.gitignore` | Tells Git to ignore downloaded data and generated files — they are far too large for a repository and are all reproducible |

---

## `dashboard/` — the product

| File | What it is |
|---|---|
| `index.html` | The page the browser loads |
| `assets/index-*.js` | **The dashboard itself** — the React application with all its data compiled in: every layer for every district for 24 months, the forecast, ENSO state, crop stress and the district shapes |
| `assets/index-*.css` | Styling |

Served by `view_dashboard.py`. This folder is a normal static website: it can
also be uploaded to any web host, an S3 bucket or GitHub Pages as-is.

---

## `IMD_Data/` — rain gauges and district boundaries

The observational backbone: India Meteorological Department gridded data,
converted from a grid to districts.

### Scripts

| File | What it does |
|---|---|
| `build_imd_lgd_csvs.py` | **The one you run.** Downloads new IMD days and aggregates the grid onto the 791 districts. Incremental — only fetches what is missing, plus the last 10 days because IMD revises recent values |
| `download_imd.py` | The lower-level downloader for the raw grids, used for the historical archive |
| `build_crosswalk.py` | Builds the grid-cell → district lookup table. Run once |
| `build_areal_crosswalk.py` | The **area-weighted** version of that lookup: each district's value is a weighted average of the grid cells overlapping it, weighted by how much of the cell falls inside. This matters because many districts are smaller than one IMD grid cell |
| `build_lgd_system.py` | Builds the 791-district registry from the official Local Government Directory |
| `make_simplified_shapefile.py` | Shrinks the district polygons for browser use — the full boundary file is 167 MB, the simplified one is 2.4 MB and looks identical on screen |
| `validate_imd.py` | Sanity checks on the downloaded IMD data |

### Data files

| File | What it is |
|---|---|
| `registry_lgd791.csv` | **The master district list.** district_id, name, state, area. Every other file joins on `district_id` |
| `crosswalk_rain.csv`, `crosswalk_temp.csv` | Which grid cells belong to which district, for the rainfall (0.25°) and temperature (1°) grids |
| `crosswalk_*_areal.csv` | The same, with area weights |
| `crosswalk_meta.json`, `lgd_system_meta.json` | Grid dimensions and provenance for the above |
| `_district_daily_*_lgd.pkl` | **Generated.** The aggregated daily district series — the pipeline's actual input. Not in Git; produced by `build_imd_lgd_csvs.py` |

### `lgd_shapefile/` — the map geometry

A shapefile is **five files that have to stay together**. Copying only the
`.shp` is the classic mistake — I have done it — and you get either "file not
found" or a blank map with no explanation.

| File | Holds |
|---|---|
| `india_districts_lgd.shp` | The polygon coordinates — the shapes themselves |
| `india_districts_lgd.shx` | An index into the .shp, so software can jump to a district |
| `india_districts_lgd.dbf` | The attributes: district name, state, codes |
| `india_districts_lgd.prj` | The coordinate system the numbers are in |
| `india_districts_lgd.cpg` | The text encoding of the .dbf |

---

## `GEE_scripts/` — the satellite data

JavaScript, not Python. These run **inside Google Earth Engine's web editor**
rather than on your machine, which is the whole point: Earth Engine holds
petabytes of satellite imagery and does the averaging on Google's servers, so
what comes back is one small CSV per year instead of terabytes of images.

| Folder | Dataset | Gives us |
|---|---|---|
| `01_evapotranspiration/` | ERA5-Land | How much water the atmosphere pulls from the soil (AET, PET) — the demand side of every drought index |
| `02_soil_moisture/` | ERA5-Land | Soil water at four depths |
| `03_temperature_era5/` | ERA5-Land | 2 m air temperature |
| `04_rainfall_crosscheck/` | CHIRPS | An independent rainfall estimate to disagree with IMD |
| `05_forecast_cfsv2/` | NOAA CFSv2 | A coupled model's view of rain, temperature, humidity and wind |
| `06_ocean_indices/` | NOAA OISST | Sea-surface temperature in the Niño-3.4 and Indian Ocean boxes |
| `07_vegetation/` | MODIS MOD13Q1 | Vegetation greenness (NDVI, EVI) at 250 m |
| `08_static_layers/` | LGRIP30, C3S | Cropland extent and irrigated fraction — these do not change yearly |
| `_shared/` | — | Common helper code the scripts import |
| `verify_exports.py` | — | **Run after downloading.** Refuses files whose date ranges overlap, the error that otherwise double-counts a month invisibly |

---

## `Indices/` — ocean and atmosphere

| File | What it is |
|---|---|
| `fetch_indices.py` | **The one you run.** Downloads the MJO, ONI, Niño-3.4 and IOD series over plain HTTP. Compares against what is on disk and refuses a feed that has lost data |
| `romi.cpcolr.1x.txt` | The MJO index (ROMI), daily since 1991 |
| `nino34.long.anom.csv` | Niño-3.4 sea-surface temperature anomaly, monthly since 1870 |
| `dmi.had.long.csv`, `dmieast.had.long.csv` | The Indian Ocean Dipole and its eastern pole. Kept separate because the monsoon responds differently to each |
| `OISST_nino34_iod_monthly.csv` | Raw sea-surface temperatures for the same ocean boxes |

---

## `v5/monsooncast/` — the pipeline (the "back end")

All the processing. Filenames keep their step numbers so the execution order is
readable regardless of which folder a script sits in.

### `cleaning/` — build the panel

| File | What it does |
|---|---|
| `00_clean_indices.py` | Parses the MJO, ENSO and IOD files into two clean tables. Encodes MJO phase on a **circle** rather than as a number 1–8, because phase 8 and phase 1 are neighbours and a plain number claims they are seven apart |
| `01_clean_merge_panel.py` | **The heart of the data work.** Merges IMD rainfall and temperature, ERA5-Land, vegetation and ocean indices into one district-month table. Computes rainfall normals over IMD's own 1971–2020 window, and day-matches them for an incomplete month |
| `02_enso_phase_climatology.py` | Works out how each district behaves in El Niño vs La Niña years, fitted on training years only so later validation stays honest |

### `features/` — the indices

| File | What it computes |
|---|---|
| `03_build_features.py` | **The index suite.** SPEI at 1, 4 and 12 months (rainfall minus evaporative demand, standardised); a second SPEI from a temperature-only formula as a cross-check; moisture adequacy (AET/PET); growing degree-days; stress degree-days above 34 °C |
| `21_daily_features.py` | The daily rolling layers: 7-day, 30-day and season-to-date departures |
| `17_feature_reliability.py` | How well-observed each district is — how many grid cells it actually contains, and how far our figure sits from CHIRPS |
| `18_irrigation_fraction.py` | Irrigated fraction per district, which decides how much a rainfall deficit actually matters |
| `26_cfsv2_layers.py` | CFSv2 layers, compared against **CFSv2's own** climatology — comparing a model against IMD's normal would report the model's bias as a rainfall anomaly |
| `27_ecmwf_layers.py` | The ECMWF ensemble as a selectable forecast, with its spread |
| `_partial_month.py` | Helper: computes a normal over exactly the days observed so far |

### `crops/` — crop stress and sowing

| File | What it does |
|---|---|
| `14_build_crop_mask.py` | Which crops are grown in which districts, and how much area |
| `15_crop_stress.py` | **The crop-stress model.** FAO-33 method: water deficit weighted by how sensitive the crop is at its current growth stage. 12 kharif crops, 4 stages each |
| `19_sowing_dynamics.py` | How much has been sown so far, and how that compares with last year |
| `29_dafw_sowing.py` | Parses the Ministry's weekly release, which carries a genuine *normal* the UPAg files lack |

### `modelling/` — the forecast

| File | What it does |
|---|---|
| `04_build_samples_train.py` | Builds the training matrix |
| `08_model_bakeoff.py` | **Trains and picks the model.** Seven combinations of algorithm and loss function; keeps the best three and blends them. The slow step |
| `08b_persist_models.py` | Saves the trained models so later runs reuse them |
| `09_calibration_test.py` | Diagnostic: are the predicted probabilities honest? |
| `10_model_v2_spatial.py` | An experiment using neighbouring districts. Did not beat the simpler model; kept as a record |
| `11_train_final.py` | Fits the chosen model on all data |
| `16_model_families_verify.py` | Cross-checks the model families |

### `validation/` — the checks

| File | What it checks |
|---|---|
| `07_validate_features.py` | 44 checks against IMD's published figures and against each index's own definition — SPEI must be normally distributed, moisture adequacy must lie in range, degree-days cannot be negative |
| `12_validate_district_vs_imd.py` | District by district against IMD bulletins |
| `20_validate_sowing_official.py` | Our sowing totals against the Ministry's |
| `28_audit_training_data.py` | Looks for leakage and degeneracy in the training matrix before a retrain is trusted |
| `30_audit_display.py` | **Run this last.** Walks the whole chain to the buttons in the built HTML and reports the first place a layer stops having values |

### `dashboard/` — building the output

| File | What it does |
|---|---|
| `05_forecast_export.py` | Runs the forecast for every district and writes `data.json` and the map geometry |
| `06_build_dashboard.py` | **Builds the interface.** Contains the entire user interface as one React component, and emits it twice: as a self-contained HTML file and as a Vite project. One component, so the two builds can never drift apart |
| `13_export_masters.py` | Spreadsheet exports, CSV and Excel |

### `lib/` and runners

| File | What it does |
|---|---|
| `lib/upag_common.py` | Shared loader for the government sowing files, handling the five defects in them |
| `run_all.py` | Runs the pipeline in order, stopping at the first failure |
| `run_post_training.py` | The steps that must follow a retrain, in the order they must happen |

### `forecast_input/`

| File | What it does |
|---|---|
| `25_ecmwf_opendata_lgd.py` | Downloads the ECMWF ensemble and aggregates it to districts |

---

## `v5/` root

| File | What it is |
|---|---|
| `common_v5.py` | Shared paths and logging, imported by 31 pipeline scripts. **Do not move it** |
| `43_build_enso_features.py` | Turns the raw El Niño index into the features the panel uses — trailing windows only, so a June forecast never sees the following winter |

---

## `v5/dashboard_react/` — the web application (the "front end")

| File | What it is |
|---|---|
| `src/App.jsx` | The user interface: map, layer buttons, sliders, district briefing, tables. **Generated** by `06_build_dashboard.py` — edit that script, not this file, or your changes are overwritten on the next build |
| `src/main.jsx` | The few lines that start the React application |
| `src/theme.css` | Colours, fonts and layout |
| `index.html` | The page the browser loads first |
| `package.json` | Which JavaScript libraries are needed |
| `vite.config.js` | Build settings, including the development server port |
| `src/data/` | **Generated.** The data files the app reads |

### How the interface is produced

I wrote the whole interface **once**, inside
`v5/monsooncast/dashboard/06_build_dashboard.py`, and it is emitted as the React
project in `v5/dashboard_react/`. Running `npm run build` there compiles it into
the static bundle in `dashboard/`.

The indirection is deliberate. The interface has to be generated rather than
hand-maintained, because the layer list, the colour scales and the captions all
depend on what the pipeline actually produced that day — a hand-written version
would go out of step the first time a data source went missing.

> That script also writes a single-file HTML version next to the React project.
> I am not shipping it here. It pulls React from a content-delivery network, so
> it needs internet to render — which made my own description of it as "one
> self-contained file" untrue in exactly the situation the claim was for, a demo
> on a machine with no network. The React build in `dashboard/` has no external
> dependencies at all, so I ship that instead.

---

## Generated folders (empty until you build)

| Folder | Fills with |
|---|---|
| `v5/data_lgd/` | The panel, the features, the trained models, the daily layers |
| `v5/masters_lgd/` | The spreadsheet exports |
| `v5/dashboard_lgd/` | The freshly built dashboard |
| `Evapotranspiration_LGD/`, `SM_LGD/`, `Vegetation_LGD/`, `CFSv2_LGD/`, `Chirps_LGD/`, `TemperatureERA5_LGD/`, `GlobalIrrigation/` | Earth Engine downloads |
| `ECMWF_LGD/` | The ECMWF forecast |
| `UPAJ/` | Sowing releases you download |
| `IMD_Data/raw/` | Raw IMD grid files |

None of these are in Git — they are large and all reproducible. That is what
`.gitignore` is for.
