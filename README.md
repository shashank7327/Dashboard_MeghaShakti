# MonsoonCast

District-level monsoon, drought and crop-stress intelligence for all 791
districts of India.

I built this during my internship at Provenance Agri Supply Chain Solutions,
Gurugram, between 8 June and 8 August 2026. This repository holds the whole
thing: the data pipeline, the models, the indices and the dashboard.

---

## Just want to see it? One command

Clone the repository, then run:

```bash
py -3.13 view_dashboard.py
```

Your browser opens on the dashboard. Ctrl+C in the terminal stops it.

That is the whole procedure. Nothing gets installed, nothing is uploaded, and
it doesn't touch the internet — the dashboard and all its data are already in
`dashboard/`, and the script just serves them from your own machine.

> **Why not simply double-click the HTML file?** I tried that, and it doesn't
> work. Browsers refuse to load a modern JavaScript application straight off
> the disk — Chrome, Edge and Firefox all block it, and what you get is a blank
> screen with no error at all. The page has to be *served*, not opened. That
> script does it using nothing but what already ships with Python, which is why
> I wrote it rather than leaving you to figure the blank page out.

**What you're looking at:** a map of India split into 791 districts, coloured by
whichever measurement you choose on the left — rainfall against the seasonal
normal, soil dryness, heat stress on the crops, how far sowing has got, and a 7-
and 14-day rainfall outlook. Click any district for a written briefing on it.

---

## The problem I was trying to solve

The monsoon behaves very differently from one district to the next. The official
picture is published for the country and for large subdivisions, but a trader,
an insurer or a supply-chain planner needs to know about **Vidarbha**
specifically, or **Kalahandi** specifically, and they need it today.

So I took eleven public data sources — rain gauges, satellites, ocean sensors,
weather models and government sowing returns — and built one consistent
district-level picture out of them, rebuilt daily, where every number can be
traced back to where it came from.

---

## The documentation

I'd read these in order. I've written them for someone who doesn't already know
climate science or Python.

| Document | What it covers |
|---|---|
| [`docs/01_DATA_SOURCES.md`](docs/01_DATA_SOURCES.md) | Where every number comes from, who publishes it, how current it is, and what the licence lets you do with it |
| [`docs/02_DATA_EXTRACTION.md`](docs/02_DATA_EXTRACTION.md) | How to actually get each dataset — the automatic downloads and the manual ones, step by step |
| [`docs/03_RUNBOOK.md`](docs/03_RUNBOOK.md) | Which file to run, in what order, and what each one produces |
| [`docs/04_FILE_GUIDE.md`](docs/04_FILE_GUIDE.md) | Every file in here, explained one by one |
| [`docs/05_ARCHITECTURE.md`](docs/05_ARCHITECTURE.md) | How the pieces fit together — the back end, the front end, and what flows between them |
| [`docs/06_NORMALS_METHODOLOGY.md`](docs/06_NORMALS_METHODOLOGY.md) | How I compute rainfall and temperature normals, and how they check out against IMD's published figures |

---

## Rebuilding it with newer data

The dashboard in `dashboard/` is already built, so you only need this if you
want to regenerate it.

**1. Install Python 3.13** from [python.org](https://www.python.org/downloads/).
On Windows, tick "Add Python to PATH" during setup.

**2. Install the libraries:**

```bash
py -3.13 -m pip install -r requirements.txt
```

**3. See what you have before changing anything:**

```bash
py -3.13 -X utf8 build_dashboard.py --check
```

This looks at your machine and prints a report — which libraries are installed,
which datasets are present, how old each one is, and for anything missing, the
exact command or document that gets it. It changes nothing. I'd run this first
every time.

**4. Build:**

```bash
py -3.13 -X utf8 build_dashboard.py
```

About 35 minutes. Add `--fetch` to download fresh data first. When it finishes
the refreshed dashboard is published into `dashboard/`; view it with
`py -3.13 view_dashboard.py`.

Rebuilding the dashboard itself needs [Node.js](https://nodejs.org/) 18 or
newer, because the interface is a React app. If Node isn't installed the data
pipeline still runs fine and the previous dashboard stays where it is — the
build tells you that rather than failing.

**What will be missing.** Four datasets come from Google Earth Engine and can't
be downloaded automatically; they need a free Google account and a few clicks in
a web editor, which [`docs/02_DATA_EXTRACTION.md`](docs/02_DATA_EXTRACTION.md)
walks through. Skip them and the build still works — the soil-moisture and
vegetation layers just don't appear. I made the pipeline drop a layer whose
source is missing rather than fail or fill in a guess.

---

## What's in here

```
Dashboard_final/
├── dashboard/            the built app — served by view_dashboard.py
├── view_dashboard.py     opens it in your browser
├── build_dashboard.py    one command to rebuild everything
├── refresh_status.py     what data is out of date today
├── requirements.txt      the Python libraries
├── docs/                 the six guides above
├── IMD_Data/             rain-gauge downloads + district boundaries
├── GEE_scripts/          the Google Earth Engine satellite scripts
├── Indices/              ocean and atmosphere index downloads
└── v5/
    ├── monsooncast/      the pipeline — all the processing code
    ├── dashboard_react/  the web app source (React + Vite)
    └── data_lgd/         generated files, empty until you build
```

The folder layout is the same as the one I developed in, on purpose. Every
script finds its neighbours by relative path, so moving things around breaks the
build. The docs are the map instead.

---

## What this can't do

I'd rather put the limits up front than have someone find them later.

- **The 7/14-day forecast only just beats assuming normal rainfall.** Measured
  skill is +0.062 at 7 days and +0.035 at 14. That's useful for ranking
  districts by risk, and it is not a promise about any one district. The score
  sits on the dashboard next to the forecast, deliberately.
- **The observed layers are the strong part of this.** Rainfall departure,
  drought, heat stress and crop stress are computed from measurements rather
  than predicted, and they match IMD's published figures at 0.994 correlation.
- **Four island districts have no rainfall data at all.** IMD's gauge grid has
  no valid cell over Nicobar, North & Middle Andaman, South Andaman or
  Lakshadweep, so I show them blank rather than estimating something.
- **The crop thresholds are literature values**, not calibrated against Indian
  yield records.
- **IMD data is licensed.** Free for research and internal use, but selling this
  or bundling it into a paid service needs a commercial licence from IMD first.
  [`docs/01_DATA_SOURCES.md`](docs/01_DATA_SOURCES.md) has the detail.

---

## If you get stuck

Every script explains itself at the top — what it does, why it exists, and what
breaks if you skip it. I wrote those notes as I went, and they're the real
documentation; these guides are just the map to them.
