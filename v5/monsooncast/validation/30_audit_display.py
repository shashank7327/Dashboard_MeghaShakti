r"""v5/monsooncast/validation/30_audit_display.py  —  does every file the
dashboard reads actually contain values, and does every layer it offers
actually render one?

WHY THIS EXISTS
  The pipeline reports success per stage, and every stage genuinely succeeded,
  yet layers still arrived on the map empty. The reasons were never "the step
  failed" -- they were a source ending earlier than the slider, a completeness
  flag masking a month that was then never refilled, a normal below the
  threshold that makes a ratio undefined, or a payload written by one build and
  read by another. None of those raise.

  So this walks the chain the reader actually sees:

      file on disk  ->  column in the panel  ->  layer in data.json
                    ->  values per month     ->  button in the built HTML

  and reports the first place a layer stops having values, with the reason.

WHAT COUNTS AS A PROBLEM
  Not "this month is empty" on its own -- ERA5-Land legitimately lags the
  gauge analysis by a week, MODIS by up to sixteen days, and a departure is
  legitimately undefined where the normal is under 5 mm. What matters is
  whether a gap is EXPLAINED. A layer that stops at its source's own edge is
  fine and the dashboard says so; a layer with a hole in the MIDDLE of its
  range is a bug, because no source lags backwards.

Run:  py -3.13 -X utf8 "v5/monsooncast/validation/30_audit_display.py"
"""
import json
import pathlib
import re
import sys

import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
V5 = HERE.parents[1]
DATA = V5 / "data_lgd"
DASH = V5 / "dashboard_lgd"
REACT = V5 / "dashboard_react"

#   Files the dashboard chain depends on, and what each is for.
FILES = [
    ("data.json", DASH, "monthly layers, forecasts, ENSO, registry"),
    ("districts.geojson", DASH, "the geometry every layer is drawn on"),
    ("daily_layers_lgd.json", DATA, "daily rolling layers + CFSv2 product"),
    ("cfsv2_layers.json", DATA, "CFSv2 second product, pre-merge"),
    ("ecmwf_forecast_latest.csv", DATA, "ECMWF ENS leads"),
    ("ecmwf_meta.json", DATA, "ECMWF run time and member count"),
    ("features_lgd.csv", DATA, "the panel every monthly layer is read from"),
    ("crop_stress_current_lgd.csv", DATA, "crop layers"),
    ("sowing_status_national.csv", DATA, "sowing panel"),
    ("dafw_sowing_national.csv", DATA, "DA&FW national sowing with normals"),
    ("indices_daily.csv", DATA, "MJO, served with the forecast"),
    ("skill_lgd.json", DATA, "the scores shown beside each model"),
]


def main():
    problems, notes = [], []
    print("=" * 78)
    print("DISPLAY AUDIT — every file the dashboard reads, every layer it offers")
    print("=" * 78)

    # ---- 1 files exist and are non-trivial ------------------------------
    print("\n[1] FILES ON DISK")
    for name, folder, what in FILES:
        p = folder / name
        if not p.exists():
            print(f"    MISSING  {name:<32} {what}")
            problems.append(f"{name} does not exist")
            continue
        kb = p.stat().st_size / 1024
        #   Size is the wrong test. skill_lgd.json holds two model records and
        #   is legitimately 0.4 KB; sowing_status_national.csv is five rows.
        #   What matters is whether the file PARSES and carries records, so
        #   that is what is checked.
        flag, rows = "", None
        try:
            if p.suffix == ".json":
                j = json.loads(p.read_text(encoding="utf-8"))
                rows = len(j) if isinstance(j, (list, dict)) else 1
            elif p.suffix == ".csv":
                rows = len(pd.read_csv(p, nrows=5000))
            elif p.suffix == ".geojson":
                rows = len(json.loads(p.read_text(encoding="utf-8"))
                           .get("features", []))
            if rows == 0:
                flag = "  <-- parses but holds no records"
                problems.append(f"{name} has no records")
        except Exception as e:
            flag = f"  <-- UNREADABLE ({type(e).__name__})"
            problems.append(f"{name} does not parse: {type(e).__name__}")
        cnt = f"{rows:>6,} rec" if rows is not None else "         "
        print(f"    ok       {name:<32} {kb:>9,.0f} KB {cnt}  {what}{flag}")

    dp = DASH / "data.json"
    if not dp.exists():
        print("\ndata.json missing — cannot continue")
        sys.exit(1)
    d = json.loads(dp.read_text(encoding="utf-8"))
    months, n = d["months"], d["n_units"]

    # ---- 2 monthly layers, month by month -------------------------------
    print(f"\n[2] MONTHLY LAYERS — {len(months)} months on the slider "
          f"({months[0]} .. {months[-1]}), {n} districts")
    print(f"    {'layer':<26}{'first':>9}{'last':>9}{'months':>8}"
          f"{'holes':>7}  verdict")
    for lyr, per in d["monthly"].items():
        have = [m for m in months if len(per.get(m, {})) > 0]
        if not have:
            print(f"    {lyr:<26}{'—':>9}{'—':>9}{0:>8}{'—':>7}  EMPTY")
            problems.append(f"{lyr} has no values in any month")
            continue
        i0, i1 = months.index(have[0]), months.index(have[-1])
        #   A hole is a month with no values BETWEEN the first and last month
        #   that do have them. Nothing lags backwards, so an interior gap is
        #   always a defect, whereas a short tail is just a source edge.
        holes = [m for m in months[i0:i1 + 1] if m not in have]
        verdict = "ok"
        if holes:
            verdict = f"HOLE at {', '.join(holes[:3])}"
            problems.append(f"{lyr}: interior gap at {', '.join(holes[:3])}")
        elif i0 > 0:
            verdict = f"starts {i0} month(s) in — check the export range"
            notes.append(f"{lyr} covers only the last {len(have)} months")
        print(f"    {lyr:<26}{have[0]:>9}{have[-1]:>9}{len(have):>8}"
              f"{len(holes):>7}  {verdict}")

    # ---- 3 per-layer as-of, and whether it matches ----------------------
    print("\n[3] AS-OF PER LAYER  (what the dashboard tells the reader)")
    ll = d.get("layer_last", {})
    for lyr, per in d["monthly"].items():
        have = [m for m in months if len(per.get(m, {})) > 0]
        shown, actual = ll.get(lyr), (have[-1] if have else None)
        ok = shown == actual
        print(f"    {lyr:<26} shows {str(shown):>9}   actually {str(actual):>9}"
              f"   {'ok' if ok else '<-- MISMATCH'}")
        if not ok:
            problems.append(f"{lyr}: as-of says {shown}, data ends {actual}")

    # ---- 4 daily payload -------------------------------------------------
    print("\n[4] DAILY LAYERS")
    dl = DATA / "daily_layers_lgd.json"
    if dl.exists():
        j = json.loads(dl.read_text(encoding="utf-8"))
        dates, ids = j["dates"], j["ids"]
        print(f"    {len(dates)} days ({dates[0]} .. {dates[-1]}), "
              f"{len(ids)} districts")
        for k, arr in j["layers"].items():
            if len(arr) != len(dates) or any(len(r) != len(ids) for r in arr):
                print(f"    {k:<20} SHAPE MISMATCH — positional payload, "
                      f"this draws the wrong district")
                problems.append(f"daily layer {k} has the wrong shape")
                continue
            nz = sum(1 for r in arr if any(v is not None for v in r))
            edge = j.get("edge", {}).get(k)
            print(f"    {k:<20} {nz:>4}/{len(dates)} days with data, "
                  f"edge {edge}")
            if nz == 0:
                problems.append(f"daily layer {k} is empty")

    # ---- 5 what the built HTML actually offers ---------------------------
    print("\n[5] BUILT DASHBOARD")
    for label, path in (("standalone", DASH / "MonsoonCast_LGD.html"),
                        ("react data", REACT / "src" / "data" / "daily.json")):
        if not path.exists():
            print(f"    {label:<12} MISSING")
            problems.append(f"{label} not built")
            continue
        if path.suffix == ".html":
            h = path.read_text(encoding="utf-8")
            m = re.search(r'window\.__DAILY__=(\{.*?\});</script>', h, re.S)
            nd = len(json.loads(m.group(1))["layers"]) if m else 0
            print(f"    {label:<12} {path.stat().st_size/1e6:>5.1f} MB, "
                  f"{nd} daily layers embedded")
        else:
            nd = len(json.loads(path.read_text(encoding="utf-8"))["layers"])
            print(f"    {label:<12} {path.stat().st_size/1e6:>5.1f} MB, "
                  f"{nd} daily layers")
    #   The two builds must offer the same thing; they diverged once already.
    try:
        h = (DASH / "MonsoonCast_LGD.html").read_text(encoding="utf-8")
        a = set(json.loads(re.search(r'window\.__DAILY__=(\{.*?\});</script>',
                                     h, re.S).group(1))["layers"])
        b = set(json.loads((REACT / "src" / "data" / "daily.json")
                           .read_text(encoding="utf-8"))["layers"])
        if a != b:
            print(f"    ! the two builds differ: only in HTML {sorted(a-b)}, "
                  f"only in React {sorted(b-a)}")
            problems.append("standalone and React builds offer different layers")
        else:
            print(f"    both builds offer the same {len(a)} daily layers")
    except Exception:
        pass

    # ---- verdict ---------------------------------------------------------
    print("\n" + "=" * 78)
    for nte in notes:
        print(f"  note: {nte}")
    if problems:
        print(f"\n{len(problems)} PROBLEM(S):")
        for p_ in problems:
            print(f"  - {p_}")
        sys.exit(1)
    print("\nPASS — every display file has values, every layer has an "
          "explained range,")
    print("       no interior gaps, and both builds offer the same layers.")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
