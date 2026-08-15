# 1. Data sources

Every number on the dashboard comes from one of the sources below. For each one
I have put down who publishes it, what it measures, how quickly it arrives, and
what its licence permits. That last one decides whether this can ever be sold,
so I checked it properly rather than assuming.

---

## At a glance

| # | Source | What it gives us | Resolution | Record | Arrives after |
|---|---|---|---|---|---|
| 1 | **IMD** gridded gauge | Daily rainfall | 0.25° (~28 km) | 1901– (we use 1971–) | 1–2 days |
| 2 | **IMD** gridded gauge | Daily max/min temperature | 1.0° (~110 km) | 1951– (we use 1971–) | 1–2 days |
| 3 | **ERA5-Land** (Copernicus) | Evapotranspiration, soil moisture | ~9 km | 1981– | ~7 days |
| 4 | **MODIS** MOD13Q1 (NASA) | Vegetation greenness (NDVI, EVI) | 250 m | 2000– | 16-day composites |
| 5 | **CHIRPS** (UCSB/USGS) | Rainfall, independent cross-check | 0.05° | 1981– | 2–3 weeks |
| 6 | **NOAA CFSv2** | Coupled-model rain, temp, humidity, wind | ~22 km | 1989– | 2–3 days |
| 7 | **ECMWF ENS** open data | 15-day rainfall forecast, 50 members | 0.25° | rolling | 7–9 hours |
| 8 | **NOAA CPC** | ONI, Niño-3.4 (El Niño indices) | index | 1950– | monthly |
| 9 | **NOAA PSL** | IOD dipole and its two poles | index | 1870– | monthly |
| 10 | **CPC ROMI** | Madden–Julian Oscillation phase | index | 1991– | 1–3 days |
| 11 | **UPAg / DA&FW** | Weekly area sown, by crop | state / national | 2024– | weekly |
| 12 | **C3S land cover** | Irrigated area fraction | 300 m | 1992–2022 | static |
| 13 | **NEX-GDDP-CMIP6** | Climate *projection* to 2100 | 25 km | 2026–2100 | held, not used |

---

## What each one is actually for

**IMD rainfall and temperature** are the backbone. They are gauge-based —
measured by instruments on the ground and interpolated onto a grid — so I treat
them as the observational truth for every figure I publish. Everything else
either supports them or gets compared against them.

**ERA5-Land** is a *reanalysis* — a physics model run back over history and
constrained by observations. I use it for the things no gauge network measures
directly: how much water the atmosphere is pulling out of the soil
(evapotranspiration) and how wet the root zone is. The drought indices need
both.

**MODIS NDVI/EVI** is what a satellite sees of the crop canopy. I use it to
confirm a stress call rather than to make one, because it lags a water deficit
by one to three weeks and saturates over dense canopy. I carry `n_clear_frac`
alongside every value — the fraction of the district that was not under cloud —
because a monsoon NDVI built from two clear pixels is not the same measurement
as one built from two thousand, and I did not want that difference buried.

**CHIRPS** is in here purely to disagree with IMD. Where two independent
rainfall estimates diverge, that disagreement tells you how well the rainfall is
actually known in that district, which is worth having.

**CFSv2 and ECMWF** are the two model opinions, and I label them differently on
purpose. Earth Engine serves CFSv2 indexed by *valid time* with no lead-time
axis, so it is the model's **analysis** of the atmosphere and calling it a
forecast would be wrong. ECMWF ENS has real lead times and 50 members, so it
genuinely is a forecast — but an unverified one, because the free feed carries
no historical archive to score it against, and I would rather say that than
claim skill I cannot measure.

**The ocean and atmosphere indices** give the seasonal and intraseasonal
context: El Niño state (ONI, Niño-3.4), the Indian Ocean Dipole, and the MJO — a
wave that circles the tropics every 30–90 days and organises the monsoon's
active and break spells. The MJO turned out to be the single most valuable thing
I added to the forecast.

**UPAg and DA&FW** are the government's own sowing returns: how much area has
actually been planted, by crop, week by week.

---

## Licensing — read this before selling anything

| Source | Licence | Commercial use |
|---|---|---|
| **IMD** rainfall & temperature | IMD Data Supply Portal v4.0 (2021), chargeable | **PAID LICENCE REQUIRED** |
| ERA5-Land | CC-BY-4.0 (since 2 July 2025) | Yes, with attribution |
| ECMWF open data | CC-BY-4.0 | Yes, with attribution |
| NOAA (CFSv2, OISST, ONI, ROMI) | US Government public domain | Yes |
| MODIS (NASA LP DAAC) | NASA open data, no restriction on reuse or resale | Yes |
| CHIRPS | Open, unrestricted | Yes |
| UPAg / DA&FW | GODL-India | Yes, with attribution |
| District boundaries (LGD) | GODL-India / NIC | Yes, with attribution |

### The constraint that matters

**IMD data is licensed, not open.** IMD supplies it on a chargeable basis
through its Data Supply Portal, at rates that differ for commercial and
non-commercial users plus 18% GST, and the non-commercial terms state
explicitly that the data may not be used for commercial purposes or to earn
consultancy fees.

IMD rainfall and temperature are the backbone of everything here. The normals,
every departure, the drought indices, the degree-days and the crop-stress layer
all come off them. So:

- **Internal use, research and evaluation as built today: fine.**
- **Selling access, or bundling these outputs into a paid product or a
  consultancy deliverable: requires a commercial data licence from IMD**,
  negotiated with the National Data Centre, Pune, *before* the product is
  offered.

The clause I would read twice is whether **derived products** — a departure, an
index — are covered by whatever licence you buy, because derived products are
the entire output of this system.

*That is my reading of the published terms as an engineer, not legal advice.*

---

## Attribution text to reproduce

If you publish anything built from this system:

> Contains modified Copernicus Climate Change Service information (ERA5-Land),
> and ECMWF open data, both under CC-BY-4.0. Rainfall and temperature data
> © India Meteorological Department. Vegetation indices courtesy of NASA
> LP DAAC (MODIS MOD13Q1). Sowing data from UPAg / DA&FW, Government of India,
> under GODL-India. Neither ECMWF nor the Copernicus programme is responsible
> for any use of this information.
