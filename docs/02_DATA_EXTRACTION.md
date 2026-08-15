# 2. Data extraction — how to actually get the data

There are three kinds of source here, in increasing order of effort:

1. **Automatic** — a script downloads it. Run one command.
2. **Google Earth Engine** — a free Google account and a few clicks per dataset.
3. **Manual portal** — a human logs in and downloads a file.

Check what you currently have at any time with:

```bash
py -3.13 -X utf8 refresh_status.py
```

That prints every dataset, how far behind it is, and whether that is normal. I
made it judge each product against **its own publication lag**, so CHIRPS 20
days behind reads CURRENT while ERA5-Land 20 days behind reads DUE. Those are
simply the delays those two sources normally have, and one shared cut-off would
have been wrong for both.

---

## 1. Automatic downloads

### 1a. IMD rainfall and temperature — the backbone

```bash
py -3.13 -X utf8 "IMD_Data/build_imd_lgd_csvs.py"
```

Downloads every day published since the last run and aggregates the grid to the
791 districts. First run downloads the 1971–2025 archive and takes a few hours;
afterwards it is incremental and takes minutes.

**Two things that caught me out, so they are worth knowing.**

IMD's most recent values are *preliminary and get revised*, so I re-fetch the
last 10 days on every run. Revisions of 10–30 mm on a district-day are normal
during the monsoon. The script reports them rather than hiding them, because I
wanted to know when a number I had already looked at had moved underneath me.

Today's maximum temperature usually is not published when today's rainfall
already is. That one-day lag on Tmax belongs to the source, not to the script —
it took me a while to be sure of that the first time it happened.

**Output:** `IMD_Data/_district_daily_{rain,tmax,tmin}_lgd.pkl`

### 1b. Climate indices — El Niño, the IOD and the MJO

```bash
py -3.13 -X utf8 "Indices/fetch_indices.py"
```

Pulls four series over plain HTTP: ROMI (the MJO index) from NOAA PSL, ONI and
Niño-3.4 from NOAA CPC, and the IOD dipole and its east pole from NOAA PSL.

It compares every download against what is already on disk and **refuses a feed
that has lost data**, and it reports revisions instead of quietly taking them.
Add `--check` to see what would change without writing anything.

> The old CPC address for ROMI now returns "not found"; PSL serves the same
> product and is the one that is current. Already handled in the script.

**Output:** `Indices/*.csv`, `Indices/romi.cpcolr.1x.txt`,
`noaa_indices_cache.csv`

### 1c. ECMWF ensemble forecast

```bash
py -3.13 -X utf8 "v5/monsooncast/forecast_input/25_ecmwf_opendata_lgd.py" --run 0 --steps 24,48,72,96,120,144,168,240,336,360
```

Downloads today's 00Z run of the 50-member ensemble and aggregates it to
districts. About 460 MB and 10–20 minutes on a normal connection.

The `--steps` list restricts the download to the ten lead times the dashboard
actually publishes. Without it you pull all 24 steps — 1.1 GB — for nothing you
can see on screen.

> ECMWF open data is a **rolling real-time feed, not an archive**: only recent
> runs exist. If you want to build a corrected forecast later, archive this
> file daily starting now; after two or three monsoon seasons there is enough
> paired forecast/observation history to fit a correction.

**Output:** `ECMWF_LGD/ecmwf_ens_district_YYYYMMDDHH.csv`

---

## 2. Google Earth Engine — the satellite data

Four datasets come from Earth Engine and **cannot be scripted from here** — the
Python API needs an interactive Google sign-in, so this part stays manual. It is
not difficult, and it takes about ten minutes per dataset, most of that spent
waiting for the export.

### One-time setup

1. Go to [earthengine.google.com](https://earthengine.google.com/) and
   **sign up** — free for research and non-commercial use.
2. Open the Code Editor at
   [code.earthengine.google.com](https://code.earthengine.google.com/).
3. Upload the district boundary as an Earth Engine asset:
   - **Assets** → **New** → **Shape files**
   - select all five files from `IMD_Data/lgd_shapefile/`
     (`.shp`, `.shx`, `.dbf`, `.prj`, `.cpg` — all of them, they are one
     dataset in five parts)
   - note the asset path it gives you, e.g.
     `projects/your-project/assets/india_districts_lgd`

### For each dataset

1. Open the matching script from `GEE_scripts/` in a text editor:

   | Dataset | Script |
   |---|---|
   | ERA5-Land evapotranspiration | `01_evapotranspiration/era5land_aet_pet_LGD.js` |
   | ERA5-Land soil moisture | `02_soil_moisture/era5land_soilmoisture_LGD.js` |
   | ERA5-Land 2 m temperature | `03_temperature_era5/era5land_t2m_LGD.js` |
   | CHIRPS rainfall | `04_rainfall_crosscheck/chirps_daily_LGD.js` |
   | CFSv2 model layers | `05_forecast_cfsv2/cfsv2_daily_LGD.js` |
   | Ocean SST boxes | `06_ocean_indices/oisst_nino34_iod.js` |
   | MODIS NDVI / EVI | `07_vegetation/modis_ndvi_evi_LGD.js` |
   | Cropland & irrigation | `08_static_layers/*.js` |

2. Copy its whole contents into the Code Editor.
3. Near the top, set the asset path to the one you noted, and set
   `MODE = 'UPDATE'` to fetch only new dates (or `'FULL'` for the whole
   archive the first time).
4. Click **Run**, then open the **Tasks** tab on the right and click **Run** on
   each queued task. Exports go to your Google Drive.
5. Download the CSVs from Drive into the matching folder here:

   | Script | Download into |
   |---|---|
   | evapotranspiration | `Evapotranspiration_LGD/` |
   | soil moisture | `SM_LGD/` |
   | 2 m temperature | `TemperatureERA5_LGD/` |
   | CHIRPS | `Chirps_LGD/` |
   | CFSv2 | `CFSv2_LGD/` |
   | ocean indices | `Indices/` |
   | vegetation | `Vegetation_LGD/` |
   | static layers | `GlobalIrrigation/` |

6. Verify before building:

   ```bash
   py -3.13 -X utf8 "GEE_scripts/verify_exports.py"
   ```

   This refuses files whose date ranges overlap — the failure that otherwise
   double-counts a month and is invisible in the output.

> **If you skip this section entirely** the build still works. You simply will
> not get the soil-moisture, vegetation, drought or CHIRPS cross-check layers. I
> made the pipeline drop a layer whose source is missing rather than fail, or
> worse, invent a value for it.

---

## 3. Manual portal downloads — sowing data

### UPAg weekly sowing

1. Sign in at [upag.gov.in](https://upag.gov.in/).
2. Download the weekly **area sown** release.
3. Save it into `UPAJ/`.

> The UPAg files have five defects in them that quietly corrupt national
> totals, and I found every one of them the hard way: national rows labelled
> both "India" and "All India" so they double-count, an unlabelled cumulative
> counter that restarts each season, shifted value and unit columns,
> inconsistent year stamping, and seasons that merge when no reset fires. All
> five are handled in `v5/monsooncast/lib/upag_common.py`. I would not write a
> fresh loader for these files without reading that one first — wheat came out
> at 66.8 M ha against a true 33.4 before I caught the first defect.

### DA&FW weekly kharif progress

The Ministry's "All India Cropwise Progressive Area Sown" XLSX. Save into
`UPAJ/`. These carry a genuine **normal** column, which the UPAg files do not,
and are what the national sowing figures are validated against.

---

## After any download

```bash
py -3.13 -X utf8 build_dashboard.py
```

or, if you prefer to drive the pipeline directly, see
[`03_RUNBOOK.md`](03_RUNBOOK.md).
