r"""
IMD_Data/build_areal_crosswalk.py  —  replace the cell-centre crosswalk with a
proper AREA-WEIGHTED overlap between each district polygon and the IMD grid.

THE PROBLEM WITH THE OLD CROSSWALK
  It assigned a grid cell to a district if the cell's CENTRE fell inside the
  polygon, then averaged those cells equally.  That is a crude estimator of a
  district mean and it fails in three ways:

    * a district smaller than a 0.25 deg cell (~27 km) contains no centre at
      all, so it was snapped to the single nearest cell -- 34 districts for
      rainfall, 540 for the 1 deg temperature grid;
    * a cell straddling a district boundary is counted wholly in or wholly
      out, so edge rainfall leaks between neighbours;
    * every cell gets equal weight regardless of how much of it actually lies
      inside the district.

  District-level validation against IMD's own bulletins (step 12) put the
  departure MAE at ~20 pp with a regression slope near 0.58 -- our values
  compressed toward the regional mean, exactly the signature of a smoothed /
  mis-weighted spatial estimator.

WHAT THIS DOES INSTEAD
  For each district polygon, intersect it with every grid cell it touches and
  weight each cell by the FRACTION OF THE DISTRICT'S AREA that falls in it.
  The weights sum to one per district, so the result is a true areal mean --
  the standard way to move a gridded field onto polygons, and what GEE's
  reduceRegions does.  Small districts now draw from the cells that genuinely
  cover them, in proportion, instead of one arbitrary neighbour.

OUTPUT -> IMD_Data/crosswalk_{rain,temp}_areal.csv
  district_id, cell_row, cell_col, weight   (weights sum to 1 per district)

Run:  py -3.13 -X utf8 "IMD_Data/build_areal_crosswalk.py"
"""
import json
import pathlib
import sys

import numpy as np
import pandas as pd
from shapely.geometry import box
from shapely.strtree import STRtree

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[0] / "v5"))
from common_v5 import log  # noqa
import build_crosswalk as bc  # noqa


def main():
    log("=" * 70)
    log("areal (area-weighted) crosswalk: district polygons x IMD grid")
    log("=" * 70)
    reg = pd.read_csv(HERE / "registry_lgd791.csv")
    polys, meta = bc.load_lgd_polygons()
    meta = meta.copy()
    meta["ks"], meta["kd"] = bc.norm_state(meta["state"]), bc.norm_key(meta["district"])
    key = reg.copy()
    key["ks"], key["kd"] = bc.norm_state(key["state"]), bc.norm_key(key["district"])
    lut = {(r.ks, r.kd): r.district_id for r in key.itertuples()}
    did = np.array([lut.get((meta.at[i, "ks"], meta.at[i, "kd"]), -1)
                    for i in meta.index])

    # union the polygons that belong to the same registry district
    from collections import defaultdict
    groups = defaultdict(list)
    for g, d in zip(polys, did):
        if d >= 0:
            groups[int(d)].append(g)
    log(f"  {len(groups)} registry districts have geometry")

    for name, g in bc.GRIDS.items():
        lat0, lon0, step = g["lat0"], g["lon0"], g["step"]
        nlat, nlon = g["nlat"], g["nlon"]
        # build the grid cells as boxes, indexed for fast lookup
        cells, cellij = [], []
        for r in range(nlat):
            for c in range(nlon):
                x0 = lon0 + step * c - step / 2
                y0 = lat0 + step * r - step / 2
                cells.append(box(x0, y0, x0 + step, y0 + step))
                cellij.append((r, c))
        tree = STRtree(cells)

        rows, nsmall = [], 0
        for d, gs in groups.items():
            geom = gs[0] if len(gs) == 1 else None
            if geom is None:
                from shapely.ops import unary_union
                geom = unary_union(gs)
            if not geom.is_valid:
                geom = geom.buffer(0)
            hits = tree.query(geom)
            tot = 0.0
            part = []
            for h in hits:
                inter = geom.intersection(cells[h])
                if inter.is_empty:
                    continue
                a = inter.area
                if a <= 0:
                    continue
                part.append((cellij[h], a))
                tot += a
            if tot <= 0:                       # smaller than any overlap: snap
                nsmall += 1
                c = geom.centroid
                r_ = int(round((c.y - lat0) / step))
                c_ = int(round((c.x - lon0) / step))
                r_ = min(max(r_, 0), nlat - 1)
                c_ = min(max(c_, 0), nlon - 1)
                part, tot = [((r_, c_), 1.0)], 1.0
            for (r_, c_), a in part:
                rows.append({"district_id": d, "cell_row": r_, "cell_col": c_,
                             "weight": a / tot})
        cw = pd.DataFrame(rows)
        cw.to_csv(HERE / f"crosswalk_{name}_areal.csv", index=False)
        per = cw.groupby("district_id").size()
        log(f"  {name}: {cw['district_id'].nunique()} districts, "
            f"{len(cw):,} district-cell pairs, "
            f"median {int(per.median())} cells/district "
            f"(min {per.min()}, max {per.max()}); {nsmall} snapped")
    log("  wrote crosswalk_{rain,temp}_areal.csv")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
