r"""
IMD_Data/build_lgd_system.py  —  establish the LGD 791-unit district system
that the IMD-based pipeline will use, replacing the GAUL-2024 701-unit registry.

WHY
  The user has chosen the updated LGD boundaries.  These are the current
  official districts (2023/24 reorganisations included) and, crucially, they
  contain Jammu & Kashmir and Ladakh (24 units) that the GAUL-2024 exports
  never carried.  Because every registry unit is now itself an LGD polygon,
  the IMD grid -> district crosswalk is direct (cell in polygon -> that unit),
  with no cross-vintage name matching and no state-fill.

OUTPUTS -> IMD_Data\
  registry_lgd791.csv      district_id, state, district, lgd_code, area_km2,
                           is_jk_ladakh
  crosswalk_rain_lgd.csv   cell_row, cell_col, lon, lat, district_id  (0.25 deg)
  crosswalk_temp_lgd.csv   same for the 1 deg temperature grid
  lgd_system_meta.json     coverage per grid

Run:  py -3.13 -X utf8 "IMD_Data/build_lgd_system.py"
"""
import json
import pathlib
import sys

import numpy as np
import pandas as pd
import shapefile
from shapely.geometry import Point
from shapely.strtree import STRtree

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[0] / "v5"))
from common_v5 import BASE, log  # noqa
import build_crosswalk as bc  # noqa  (reuse reprojection + GRIDS + norm keys)

SHP = BASE / "GEE_scripts" / "BOUNDARY GIS" / "New_DISTRICT_BOUNDARY.shp"


def build_registry():
    r = shapefile.Reader(str(SHP))
    fld = [f[0] for f in r.fields[1:]]
    iS, iD = fld.index("STATE_UT"), fld.index("DISTRICT")
    iL = fld.index("Dist_LGD")
    iA = fld.index("Shape_Ar_1")
    # DBF fields can carry a trailing carriage-return/newline and padding
    # -- 7 district names do -- and left in place they surface with a
    # visible line break inside the name in every display and export.
    def _cl(v):
        return " ".join(str(v).split())
    rows = [(_cl(rec[iS]), _cl(rec[iD]), _cl(rec[iL]),
             float(rec[iA]) / 1e6) for rec in r.records()]
    df = pd.DataFrame(rows, columns=["state", "district", "lgd_code",
                                     "area_km2"])
    # 792 polygons -> 791 distinct units (one district is split in two records)
    g = (df.groupby(["state", "district"], as_index=False)
         .agg(lgd_code=("lgd_code", "first"), area_km2=("area_km2", "sum")))
    g = g.sort_values(["state", "district"]).reset_index(drop=True)
    g.insert(0, "district_id", np.arange(len(g), dtype=int))
    g["is_jk_ladakh"] = g["state"].str.upper().str.contains(
        "KASHMIR|LADAKH|JAMMU").astype(int)
    g.to_csv(HERE / "registry_lgd791.csv", index=False)
    log(f"  registry_lgd791.csv: {len(g)} units, {g['state'].nunique()} "
        f"states/UTs, {int(g['is_jk_ladakh'].sum())} J&K/Ladakh units")
    return g


def build_crosswalks(reg):
    polys, meta = bc.load_lgd_polygons()             # reprojected to lon/lat
    meta = meta.copy()
    meta["ks"] = bc.norm_state(meta["state"])
    meta["kd"] = bc.norm_key(meta["district"])
    key = reg.copy()
    key["ks"] = bc.norm_state(key["state"])
    key["kd"] = bc.norm_key(key["district"])
    lut = {(r.ks, r.kd): r.district_id for r in key.itertuples()}
    poly_did = np.array([lut.get((meta.at[i, "ks"], meta.at[i, "kd"]), -1)
                         for i in meta.index])
    n_unmapped = int((poly_did < 0).sum())
    log(f"  {len(polys)} polygons -> {int((poly_did >= 0).sum())} mapped to "
        f"registry ({n_unmapped} unmapped)")
    tree = STRtree(polys)
    geoms = tree.geometries

    meta_out = {"registry_units": len(reg), "grids": {}}
    for name, g in bc.GRIDS.items():
        lats = g["lat0"] + g["step"] * np.arange(g["nlat"])
        lons = g["lon0"] + g["step"] * np.arange(g["nlon"])
        rows = []
        for ri, la in enumerate(lats):
            for ci, lo in enumerate(lons):
                p = Point(lo, la)
                hit = tree.query(p, predicate="intersects")
                did = np.nan
                if len(hit):
                    cand = [h for h in hit if geoms[h].contains(p)] or list(hit)
                    good = [h for h in cand if poly_did[h] >= 0]
                    did = poly_did[good[0]] if good else poly_did[cand[0]]
                    if did < 0:
                        did = np.nan
                rows.append((ri, ci, round(lo, 4), round(la, 4), did))
        cw = pd.DataFrame(rows, columns=["cell_row", "cell_col", "lon", "lat",
                                         "district_id"])
        got = set(cw["district_id"].dropna().astype(int))
        direct = len(got)
        # snap any unit with no cell to its nearest cell centre
        cent = {}
        for geom, d in zip(geoms, poly_did):
            if d >= 0:
                c = geom.centroid
                cent.setdefault(int(d), (c.x, c.y))
        cx, cy = cw["lon"].to_numpy(), cw["lat"].to_numpy()
        add, snapped = [], 0
        for d in reg["district_id"]:
            if d in got or d not in cent:
                continue
            lo, la = cent[d]
            k = int(np.argmin((cx - lo) ** 2 + (cy - la) ** 2))
            add.append((cw.at[k, "cell_row"], cw.at[k, "cell_col"],
                        cw.at[k, "lon"], cw.at[k, "lat"], d))
            snapped += 1
        if add:
            cw = pd.concat([cw, pd.DataFrame(add, columns=cw.columns)],
                           ignore_index=True)
        cw.dropna(subset=["district_id"]).to_csv(
            HERE / f"crosswalk_{name}_lgd.csv", index=False)
        total = cw["district_id"].nunique()
        log(f"  {name}: {direct} units with a direct cell, +{snapped} snapped "
            f"= {total}/{len(reg)} covered")
        meta_out["grids"][name] = {
            **{k: g[k] for k in ("lat0", "lon0", "step", "nlat", "nlon")},
            "direct": direct, "snapped": snapped, "covered": int(total),
            "total": len(reg)}
    (HERE / "lgd_system_meta.json").write_text(json.dumps(meta_out, indent=1),
                                               encoding="utf-8")


def main():
    log("=" * 68)
    log("IMD_Data — establish the LGD 791-unit district system")
    log("=" * 68)
    reg = build_registry()
    build_crosswalks(reg)
    log("  wrote registry_lgd791.csv, crosswalk_{rain,temp}_lgd.csv, "
        "lgd_system_meta.json")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
