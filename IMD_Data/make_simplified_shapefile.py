r"""
IMD_Data/make_simplified_shapefile.py  —  produce a light, GEE-friendly copy of
the LGD district shapefile.

The original New_DISTRICT_BOUNDARY.shp is 167 MB of ultra-dense vertices in a
custom Lambert projection, which GEE's ingester rejects (Internal error 13).
This reprojects it to plain WGS84 (EPSG:4326) and simplifies each polygon by a
~400 m tolerance -- invisible against ERA5-Land's 11 km pixels but a 10-50x
size cut -- keeping DISTRICT / STATE_UT / Dist_LGD attributes and all 791
units (islands preserved as multipolygons).

Output -> IMD_Data\lgd_shapefile\india_districts_lgd.{shp,shx,dbf,prj,cpg}

Run:  py -3.13 -X utf8 "IMD_Data/make_simplified_shapefile.py"
"""
import pathlib
import sys

import shapefile
from pyproj import Transformer
from shapely.geometry import shape
from shapely.geometry.polygon import orient
from shapely.ops import transform as shp_transform

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "v5"))
from common_v5 import BASE, log  # noqa

SRC = BASE / "GEE_scripts" / "BOUNDARY GIS" / "New_DISTRICT_BOUNDARY.shp"
OUTDIR = pathlib.Path(__file__).resolve().parent / "lgd_shapefile"
OUTDIR.mkdir(parents=True, exist_ok=True)
OUT = OUTDIR / "india_districts_lgd"

TOL_DEG = 0.004          # ~440 m at Indian latitudes; << 11 km ERA5 pixel

WGS84_WKT = (
    'GEOGCS["GCS_WGS_1984",DATUM["D_WGS_1984",SPHEROID["WGS_1984",'
    '6378137.0,298.257223563]],PRIMEM["Greenwich",0.0],'
    'UNIT["Degree",0.0174532925199433]]'
)


def rings_of(geom):
    out = []
    polys = geom.geoms if geom.geom_type == "MultiPolygon" else [geom]
    for p in polys:
        if p.is_empty or p.exterior is None:
            continue
        p = orient(p, sign=-1.0)          # exterior CW, holes CCW (shp spec)
        out.append([list(c[:2]) for c in p.exterior.coords])
        for h in p.interiors:
            out.append([list(c[:2]) for c in h.coords])
    return out


def main():
    log("=" * 68)
    log("build a light WGS84 copy of the LGD district shapefile for GEE")
    log("=" * 68)
    wkt = SRC.with_suffix(".prj").read_text(encoding="utf-8")
    tf = Transformer.from_crs(wkt, "EPSG:4326", always_xy=True)

    r = shapefile.Reader(str(SRC))
    fld = [f[0] for f in r.fields[1:]]
    iD, iS = fld.index("DISTRICT"), fld.index("STATE_UT")
    iL = fld.index("Dist_LGD")

    w = shapefile.Writer(str(OUT), shapeType=shapefile.POLYGON)
    w.field("DISTRICT", "C", size=80)
    w.field("STATE_UT", "C", size=80)
    w.field("Dist_LGD", "C", size=40)

    n_in = n_out = v_in = v_out = 0
    for sh, rec in zip(r.iterShapes(), r.iterRecords()):
        n_in += 1
        try:
            g = shape(sh.__geo_interface__)
            v_in += _count(g)
            g = shp_transform(lambda x, y, z=None: tf.transform(x, y), g)
            if not g.is_valid:
                g = g.buffer(0)
            gs = g.simplify(TOL_DEG, preserve_topology=True)
            if gs.is_empty or not gs.is_valid:
                gs = g.simplify(TOL_DEG / 4, preserve_topology=True)
            rings = rings_of(gs if not gs.is_empty else g)
            if not rings:
                continue
            v_out += sum(len(rr) for rr in rings)
            w.poly(rings)
            w.record(str(rec[iD])[:80], str(rec[iS])[:80], str(rec[iL])[:40])
            n_out += 1
        except Exception as e:
            log(f"  skip record {n_in}: {type(e).__name__} {e}")
    w.close()
    OUT.with_suffix(".prj").write_text(WGS84_WKT, encoding="utf-8")
    OUT.with_suffix(".cpg").write_text("UTF-8", encoding="utf-8")

    mb = OUT.with_suffix(".shp").stat().st_size / 1e6
    log(f"  features: {n_out}/{n_in}")
    log(f"  vertices: {v_in:,} -> {v_out:,} ({100*v_out/max(v_in,1):.1f}% kept)")
    log(f"  new .shp size: {mb:.2f} MB  (was 167 MB)")
    log(f"  wrote {OUT}.{{shp,shx,dbf,prj,cpg}}")
    log("\n  upload these 4-5 files to GEE instead of the originals.")


def _count(g):
    if g.geom_type == "MultiPolygon":
        return sum(_count(p) for p in g.geoms)
    n = len(g.exterior.coords)
    return n + sum(len(h.coords) for h in g.interiors)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
