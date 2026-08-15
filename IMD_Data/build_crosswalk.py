r"""
IMD_Data/build_crosswalk.py  —  grid-cell -> district crosswalk for the IMD
gridded products, so a 0.25 deg (rain) or 1 deg (temperature) field can be
reduced to the 701-unit v5 district registry exactly as GEE's reduceRegions
does, but locally.

WHY A CROSSWALK, COMPUTED ONCE
  The IMD grid is fixed, so the map from grid cell to district never changes
  between years.  Building it once and reusing it turns each year's district
  aggregation into a groupby, which is what makes 46 years x 3 variables
  tractable on a laptop.

BOUNDARIES
  The panel registry is GAUL-2024 (701 units) but no GAUL-2024 polygons are
  held locally.  The finest boundary in hand is the LGD district shapefile
  (791 polygons), which is also the most current (it carries the post-2023
  Rajasthan and Telangana reorganisations that a Census-2011 layer misses —
  Census-2011 still files Adilabad under Andhra Pradesh, so its STATE names
  no longer match the registry).  The shapefile is in a Lambert Conformal
  Conic projection, reprojected here to WGS84 lon/lat with pyproj so the
  0.25/1.0 deg cell centres can be tested against it.

METHOD
  1. reproject every LGD polygon LCC -> EPSG:4326
  2. for each grid, test every cell centre for containment in an LGD polygon
     (shapely STRtree); a land cell with no polygon is left unassigned
  3. map the LGD polygon to a registry district_id by name (exact, then
     token-overlap within the same state) — the same matcher used elsewhere
     in v5
  4. a registry district that captured no cell (small districts under the
     coarse temperature grid especially) is snapped to the single nearest
     cell centre, so every district that has any geometry receives a value

OUTPUT -> IMD_Data\
  crosswalk_rain.csv   cell_row, cell_col, lon, lat, district_id
  crosswalk_temp.csv   same, for the 1 deg grid
  crosswalk_meta.json  grid definitions and coverage

Run:  py -3.13 -X utf8 "IMD_Data/build_crosswalk.py"
"""
import json
import pathlib
import sys

import numpy as np
import pandas as pd
import shapefile
from pyproj import Transformer
from shapely.geometry import Point, Polygon, MultiPolygon
from shapely.strtree import STRtree

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "v5"))
from common_v5 import BASE, get_registry, log  # noqa

HERE = pathlib.Path(__file__).resolve().parent
SHP = BASE / "GEE_scripts" / "BOUNDARY GIS" / "New_DISTRICT_BOUNDARY.shp"

# IMD grid definitions (from the downloaded .grd headers)
GRIDS = {
    "rain": dict(lat0=6.5, lon0=66.5, step=0.25, nlat=129, nlon=135),
    "temp": dict(lat0=7.5, lon0=7.5, step=1.0, nlat=31, nlon=31),
}
# temperature longitude origin is 67.5, not 7.5 — set explicitly below
GRIDS["temp"]["lon0"] = 67.5


def norm_key(s):
    s = pd.Series(s).astype(str).str.upper()
    s = s.str.replace(r"[^A-Z ]", " ", regex=True)
    return s.str.replace(r"\s+", " ", regex=True).str.strip()


# LGD state spellings that differ from the GAUL-2024 registry.  A single
# mismatched state name silently drops an entire state (all 28 Chhattisgarh
# districts were lost to "CHHATISGARH" vs "CHHATTISGARH"), so these are
# aligned explicitly before matching.
STATE_ALIAS = {
    "CHHATISGARH": "CHHATTISGARH",
    "ANDAMAN AND NICOBAR ISLANDS": "ANDAMAN NICOBAR",
    "ANDAMAN AND NICOBAR": "ANDAMAN NICOBAR",
    "LAKSHADWEEP UT": "LAKSHADWEEP",
    "GUJARAT AND DNH DD ISLANDS": "GUJARAT",
    "NCT OF DELHI": "DELHI",
    "DADRA AND NAGAR HAVELI AND DAMAN AND DIU":
        "DADRA NAGAR HAVELI DAMAN DIU",
}


def norm_state(s):
    return norm_key(s).map(lambda x: STATE_ALIAS.get(x, x))


def load_lgd_polygons():
    """Reproject the LGD shapefile to lon/lat; return polygons + attributes."""
    wkt = (SHP.with_suffix(".prj")).read_text(encoding="utf-8")
    tf = Transformer.from_crs(wkt, "EPSG:4326", always_xy=True)
    r = shapefile.Reader(str(SHP))
    fld = [f[0] for f in r.fields[1:]]
    iD, iS = fld.index("DISTRICT"), fld.index("STATE_UT")
    polys, meta = [], []
    for sh, rec in zip(r.shapes(), r.records()):
        pts = sh.points
        if len(pts) < 3:
            continue
        parts = list(sh.parts) + [len(pts)]
        rings = []
        for a, b in zip(parts[:-1], parts[1:]):
            xs, ys = zip(*pts[a:b])
            lon, lat = tf.transform(xs, ys)
            rings.append(list(zip(lon, lat)))
        # largest ring is the outer boundary; the rest are holes/islands
        rings.sort(key=lambda r_: _ring_area(r_), reverse=True)
        try:
            geom = Polygon(rings[0], rings[1:] if len(rings) > 1 else None)
            if not geom.is_valid:
                geom = geom.buffer(0)
        except Exception:
            continue
        if geom.is_empty:
            continue
        polys.append(geom)
        meta.append((rec[iS], rec[iD]))
    log(f"  reprojected {len(polys)} LGD polygons to lon/lat")
    return polys, pd.DataFrame(meta, columns=["state", "district"])


def _ring_area(ring):
    a = np.array(ring)
    x, y = a[:, 0], a[:, 1]
    return abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))) / 2


def lgd_to_registry(meta, reg):
    """Map each LGD polygon to a registry district_id (exact, then token)."""
    meta = meta.copy()
    meta["ks"], meta["kd"] = norm_state(meta["state"]), norm_key(meta["district"])
    reg = reg.copy()
    reg["ks"], reg["kd"] = norm_state(reg["state"]), norm_key(reg["district"])

    ex = meta.merge(reg[["ks", "kd", "district_id"]], on=["ks", "kd"],
                    how="left")
    did = ex["district_id"].to_numpy(dtype=float, copy=True)
    n_exact = int(pd.notna(did).sum())

    reg_by_state = {k: g for k, g in reg.groupby("ks")}
    n_tok = 0
    for i in np.where(pd.isna(did))[0]:
        cand = reg_by_state.get(meta.at[i, "ks"])
        if cand is None:
            continue
        toks = set(meta.at[i, "kd"].split())
        best, sc = None, 0.0
        for r_ in cand.itertuples():
            rt = set(r_.kd.split())
            inter = len(toks & rt)
            # symmetric overlap, so a short LGD name folds into a longer
            # registry one and vice versa (Kanker <-> Uttar Bastar Kanker)
            ov = inter / max(min(len(toks), len(rt)), 1)
            if ov > sc:
                best, sc = r_.district_id, ov
        if best is not None and sc >= 0.5:
            did[i] = best
            n_tok += 1
    log(f"  LGD->registry: {n_exact} exact + {n_tok} token = "
        f"{int(pd.notna(did).sum())}/{len(meta)} polygons mapped")
    return did


def build(grid_name, g, tree, poly_did, reg_ids):
    lats = g["lat0"] + g["step"] * np.arange(g["nlat"])
    lons = g["lon0"] + g["step"] * np.arange(g["nlon"])
    rows = []
    geoms = tree.geometries  # shapely 2.x: array of the indexed geometries
    for ri, la in enumerate(lats):
        for ci, lo in enumerate(lons):
            p = Point(lo, la)
            # shapely 2.x applies the predicate as input.predicate(tree_geom),
            # so "contains" would ask point.contains(polygon) (always false);
            # "intersects" is symmetric and gives point-in-polygon
            hit = tree.query(p, predicate="intersects")
            did = np.nan
            if len(hit):
                # keep only polygons that truly contain the point, and prefer
                # one that carries a registry id
                cand = [h for h in hit if geoms[h].contains(p)] or list(hit)
                mapped = [h for h in cand if pd.notna(poly_did[h])]
                did = poly_did[mapped[0]] if mapped else poly_did[cand[0]]
            rows.append((ri, ci, round(lo, 4), round(la, 4), did))
    cw = pd.DataFrame(rows, columns=["cell_row", "cell_col", "lon", "lat",
                                     "district_id"])
    inside = cw["district_id"].notna()
    log(f"  {grid_name}: {int(inside.sum())} of {len(cw)} cells fall in a "
        f"mapped district ({cw.loc[inside, 'district_id'].nunique()} distinct)")

    # ---- snap districts that captured no cell to their nearest cell -------
    got = set(cw.loc[inside, "district_id"].dropna().astype(int))
    missing = [d for d in reg_ids if d not in got]
    if missing:
        # a district's location = centroid of the cells of any LGD polygon
        # mapped to it; if it never contained a cell, use the polygon centroid
        cent = _district_centroids(tree, poly_did)
        cx = cw["lon"].to_numpy()
        cy = cw["lat"].to_numpy()
        snapped = 0
        add = []
        for d in missing:
            if d not in cent:
                continue
            lo, la = cent[d]
            k = int(np.argmin((cx - lo) ** 2 + (cy - la) ** 2))
            add.append((cw.at[k, "cell_row"], cw.at[k, "cell_col"],
                        cw.at[k, "lon"], cw.at[k, "lat"], d))
            snapped += 1
        if add:
            cw = pd.concat([cw, pd.DataFrame(add, columns=cw.columns)],
                           ignore_index=True)
        log(f"  {grid_name}: snapped {snapped} unmatched districts to their "
            f"nearest cell; {len(missing)-snapped} have no geometry at all")
    return cw


def _district_centroids(tree, poly_did):
    out = {}
    for geom, d in zip(tree.geometries, poly_did):
        if pd.isna(d):
            continue
        c = geom.centroid
        out.setdefault(int(d), (c.x, c.y))
    return out


def main():
    log("=" * 70)
    log("IMD crosswalk — grid cell -> v5 district registry")
    log("=" * 70)
    reg = get_registry()
    reg_ids = reg["district_id"].tolist()
    polys, meta = load_lgd_polygons()
    poly_did = lgd_to_registry(meta, reg)
    tree = STRtree(polys)

    meta_out = {"grids": {}, "registry_units": len(reg_ids)}
    for name, g in GRIDS.items():
        cw = build(name, g, tree, poly_did, reg_ids)
        cw.to_csv(HERE / f"crosswalk_{name}.csv", index=False)
        cov = cw["district_id"].nunique()
        meta_out["grids"][name] = {
            **{k: g[k] for k in ("lat0", "lon0", "step", "nlat", "nlon")},
            "districts_covered": int(cov),
            "districts_total": len(reg_ids),
        }
        log(f"  wrote crosswalk_{name}.csv — {cov}/{len(reg_ids)} districts\n")

    (HERE / "crosswalk_meta.json").write_text(json.dumps(meta_out, indent=1),
                                              encoding="utf-8")
    log("  wrote crosswalk_meta.json")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
