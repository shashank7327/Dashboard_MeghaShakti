# 5. Architecture — how the pieces fit together

This is for a technical reader who needs to extend or maintain what I built. If
you only need to *run* it, [`03_RUNBOOK.md`](03_RUNBOOK.md) is enough.

---

## The shape of it

```
   ELEVEN PUBLIC SOURCES                    ACQUISITION
   IMD gauge grids ─────────────┐      IMD_Data/build_imd_lgd_csvs.py
   ERA5-Land, MODIS, CFSv2 ─────┤      GEE_scripts/*.js  (manual export)
   CHIRPS, OISST ───────────────┤      Indices/fetch_indices.py
   ECMWF ENS ───────────────────┤      forecast_input/25_ecmwf_opendata_lgd.py
   ONI, Niño-3.4, IOD, MJO ─────┤      UPAg portal (manual)
   UPAg / DA&FW sowing ─────────┘
                                        │
                                        ▼
   ┌──────────────────────────────────────────────────────────────┐
   │  BACK END — Python, v5/monsooncast/                          │
   │                                                              │
   │  cleaning/   grid → district, merge, normals, ENSO state     │
   │      ▼                                                       │
   │  features/   SPEI · moisture adequacy · degree-days ·        │
   │              daily rolling windows · reliability             │
   │      ▼                                                       │
   │  crops/      FAO-33 stage-weighted stress · sowing           │
   │      ▼                                                       │
   │  modelling/  bake-off → blended 7/14-day forecast            │
   │      ▼                                                       │
   │  validation/ 44 checks vs IMD · display audit                │
   │      ▼                                                       │
   │  dashboard/  05 writes data.json → 06 builds the interface   │
   └──────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
                          v5/dashboard_react/  (React + Vite)
                                        │  npm run build
                                        ▼
                          dashboard/  — static site, no dependencies
                          served by view_dashboard.py
```

---

## The spine: one district registry

Everything joins on `district_id` from `IMD_Data/registry_lgd791.csv`. **791
units on the Local Government Directory boundary.**

This matters more than it sounds. The GAUL boundary that most global datasets
ship on has 701 Indian units and **leaves out Jammu & Kashmir, Ladakh and the
island territories entirely**. Building on LGD means those districts appear. It
also means I have to crosswalk every external dataset onto LGD rather than use
it as delivered, which is real extra work I took on deliberately.

`IMD_Data/build_areal_crosswalk.py` does that for the gauge grids, weighting
each cell by how much of it falls inside each district. That weighting is not
optional: IMD's temperature grid is 1°, about 110 km, and a great many Indian
districts are smaller than a single cell. Assigning cells by whichever district
holds their centre would have given small districts nothing at all.

---

## The data model

**The panel** — `v5/data_lgd/features_lgd.csv`. One row per district per month,
1971 to now: **528,388 rows × 84 columns**. Every monthly layer on the dashboard
is a column here. Long format, joined on `(district_id, year, month)`.

**The daily layers** — `daily_layers_lgd.json`. A 120-day rolling window at
district resolution, stored as positional arrays against one shared id list
rather than as `{id: value}` maps, because 120 days × 791 districts × 8 layers
is about 760,000 numbers and the map form roughly triples the payload for no
extra information.

**The dashboard payload** — `data.json`, about 3 MB. Monthly layers for 24
months, the current forecast, per-district records, ENSO state, crop stress,
skill scores and aggregates. This is the contract between the back end and the
front end: step 05 writes it, the interface reads it, and nothing else crosses
that boundary.

---

## Principles the code actually follows

These are not aspirations. Each one is here because a specific bug taught me to
do it that way.

**Paths are absolute, derived from `__file__`.** No script depends on the
current working directory, so everything runs identically from an editor, a
terminal or a scheduled task.

**A layer whose source is missing is dropped, never faked.** The toolbar is
built from what the payload actually contains. A product whose export has not
landed simply does not appear, rather than sitting behind a button rendering a
blank map — because a blank map reads as a broken product rather than an absent
one.

**Every published figure is checked against something external.** Rainfall
against IMD's published Long Period Average, sowing against the Ministry's own
release, our own indices against their mathematical definitions. 44 checks, run
on every build.

**Normals are day-matched for an incomplete month.** A running month is
compared against a normal covering exactly the days observed. Getting this wrong
is not subtle: on 6 August 2026 the full-month normal reports −78.7% where the
correct figure is +0.0%.

**Aggregates are ratios of sums, never means of percentages.** For any
all-India or state figure, the actual and the normal are each area-weighted and
summed, and the ratio taken once. The mean of district percentages is a
different quantity and is wrong — for July 2026 it reads −1.3% where the correct
figure is +3.4%.

**Features are causal.** An El Niño event is conventionally named for the winter
peak it eventually reaches. Using that label in a June forecast leaks
information that does not exist in June, so every ENSO feature is built from a
strictly trailing window and district composites are fitted on training years
only.

**The interface is generated, not hand-maintained.** It lives once inside
`06_build_dashboard.py` and is emitted as the React project, which Vite compiles
into the static bundle in `dashboard/`. It has to be generated because the layer
list, the colour scales and the captions all depend on what the pipeline
actually produced on a given day.

---

## The front end

Deliberately plain — no mapping library, no chart library, no CSS framework. I
wanted the built file to have nothing it could fail to download.

The map is **inline SVG**: `districts.geojson` is projected to screen
coordinates in Python at export time, and the browser draws 791 `<path>`
elements with a fill colour per district. There are no map tiles to fetch and no
external service to reach, which is why the built dashboard works with no
internet at all.

State is plain React hooks: selected layer, selected month, selected day,
selected district, scope (district or state), theme. Changing a layer
recomputes the fill colours from `data.json`; no request is made because all
the data is already in the page.

The colour scales are fixed per layer and chosen to mean something. Rainfall
departure runs ±60% because those are **IMD's own Large Deficient / Large
Excess boundaries** — a district at the end of that scale is in IMD's extreme
category, which is a statement, not a rendering artefact. Heat stress is the
exception: its range collapses seasonally (degree-days above 34 °C are near zero
during the monsoon and large before it), so its scale is fitted to the values on
screen and the legend says so.

---

## Where to make a change

| To change | Edit | Then run |
|---|---|---|
| How a layer looks, or the interface | `dashboard/06_build_dashboard.py` | step 06 (seconds) |
| Which layers are published | `LAYERS` in `dashboard/05_forecast_export.py` | steps 05, 06 |
| An index formula | `features/03_build_features.py` | steps 03 onward |
| Crop parameters or stages | `crops/15_crop_stress.py` | steps 15, 05, 06 |
| Model families or losses | `modelling/08_model_bakeoff.py` | full rebuild |
| A validation threshold | `validation/07_validate_features.py` | step 07 |

**Never edit `v5/dashboard_react/src/App.jsx` directly.** It is generated. Edit
`06_build_dashboard.py`, which is the actual source of the interface — anything
typed into `App.jsx` is silently overwritten on the next build, and I lost an
afternoon to that once.

---

## Known limitations, technically stated

- **Forecast skill is small** — +0.062 and +0.035 MSE skill against
  climatology at 7 and 14 days. Real, measured on held-out 2020–26 data,
  published beside the forecast, and roughly what the literature offers at this
  range and scale. The observed layers are the stronger product.
- **No reforecast archive**, so the ECMWF ensemble cannot be bias-corrected or
  verified here. Archiving the daily district file from now on is the cheapest
  route to fixing this.
- **Irrigation is on the wrong boundary** — the export is 701 GAUL units, so
  the ~90 LGD districts with no GAUL equivalent fall back to a state median.
  This affects the Kharif buffer and the whole Rabi water term for those
  districts.
- **Four island districts have no rainfall data** and are blank by design.
- **Crop thresholds are literature values**, not calibrated to Indian yield
  records.
