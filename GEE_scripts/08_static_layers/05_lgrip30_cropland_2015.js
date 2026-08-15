/*******************************************************************
 * 05 - GFSAD LGRIP30: IRRIGATED vs RAINFED CROPLAND AREA (nominal 2015)
 * District-wise, India
 *
 * Dataset : projects/sat-io/open-datasets/GFSAD/LGRIP30
 *           (community catalog, 30 m, Landsat-derived, nominal 2015)
 * Classes : 0 = ocean/water, 1 = non-cropland,
 *           2 = IRRIGATED cropland, 3 = RAINFED cropland
 * Output  : single CSV:
 *           state, district, irrigated_ha, rainfed_ha, cropland_ha,
 *           pct_irrigated
 *
 * NOTE 1  : Categorical data - bilinear interpolation must NOT be
 *           used. Areas are summed from 30 m pixels, which covers
 *           every district (no coverage problem at 30 m).
 * NOTE 2  : Whole-India at 30 m is a heavy export (can take hours).
 *           Set STATE_FILTER to run state-by-state if it times out.
 * NOTE 3  : LGRIP30 v2 (nominal 2020) exists on LP DAAC if you later
 *           want a second epoch. See script 06 for TIME-SERIES
 *           irrigation alternatives (1992-2022).
 *******************************************************************/

/* ================================================================
   COMMON CONFIG (boundaries only - no interpolation for classes)
   ================================================================ */
var DRIVE_FOLDER = 'GEE_ENSO_India';

var BOUNDARY_SOURCE = 'GAUL2024';   // 'GAUL2024' | 'GAUL2015' | 'ASSET'
var ASSET_ID            = 'users/YOUR_USERNAME/india_districts_lgd';
var ASSET_DISTRICT_PROP = 'district';
var ASSET_STATE_PROP    = 'state';

var districts;
if (BOUNDARY_SOURCE === 'GAUL2024') {
  districts = ee.FeatureCollection('projects/sat-io/open-datasets/FAO/GAUL/GAUL_2024_L2')
    .filter(ee.Filter.eq('gaul0_name', 'India'))
    .select(['gaul2_name', 'gaul1_name'], ['district', 'state']);
} else if (BOUNDARY_SOURCE === 'GAUL2015') {
  districts = ee.FeatureCollection('FAO/GAUL_SIMPLIFIED_500m/2015/level2')
    .filter(ee.Filter.inList('ADM0_NAME',
      ['India', 'Jammu and Kashmir', 'Arunachal Pradesh']))
    .select(['ADM2_NAME', 'ADM1_NAME'], ['district', 'state']);
} else {
  districts = ee.FeatureCollection(ASSET_ID)
    .select([ASSET_DISTRICT_PROP, ASSET_STATE_PROP], ['district', 'state']);
}

// Optional: run one state at a time for the 30 m export
var STATE_FILTER = null;            // e.g. 'Maharashtra'
if (STATE_FILTER) {
  districts = districts.filter(ee.Filter.eq('state', STATE_FILTER));
}
print('District count:', districts.size());
/* ================================================================ */

var lgrip = ee.ImageCollection('projects/sat-io/open-datasets/GFSAD/LGRIP30')
              .mosaic();

var areaHa = ee.Image.pixelArea().divide(10000);   // m2 -> hectares

var areas = areaHa.updateMask(lgrip.eq(2)).rename('irrigated_ha')
  .addBands(areaHa.updateMask(lgrip.eq(3)).rename('rainfed_ha'))
  .addBands(areaHa.updateMask(lgrip.eq(2).or(lgrip.eq(3))).rename('cropland_ha'));

var out = areas.reduceRegions({
  collection: districts,
  reducer:    ee.Reducer.sum(),
  scale:      30,
  tileScale:  16
}).map(function(f) {
  function z(p) {   // null-safe -> 0
    return ee.Number(ee.Algorithms.If(
      ee.Algorithms.IsEqual(f.get(p), null), 0, f.get(p)));
  }
  var irr  = z('irrigated_ha');
  var crop = z('cropland_ha');
  return f.set({
    irrigated_ha:  irr,
    rainfed_ha:    z('rainfed_ha'),
    cropland_ha:   crop,
    pct_irrigated: ee.Algorithms.If(crop.gt(0),
                     irr.divide(crop).multiply(100), null)
  });
});

Export.table.toDrive({
  collection:     out,
  description:    'LGRIP30_cropland_districts_2015' +
                  (STATE_FILTER ? '_' + STATE_FILTER : ''),
  folder:         DRIVE_FOLDER,
  fileNamePrefix: 'LGRIP30_cropland_districts_2015' +
                  (STATE_FILTER ? '_' + STATE_FILTER : ''),
  fileFormat:     'CSV',
  selectors: ['state', 'district', 'irrigated_ha', 'rainfed_ha',
              'cropland_ha', 'pct_irrigated']
});
