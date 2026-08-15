/*******************************************************************
 * 06 - IRRIGATED / RAINFED CROPLAND AS A TIME SERIES
 * District-wise, India. LGRIP30 is a single 2015 snapshot; these are
 * the best in-GEE substitutes with a TIMELINE:
 *
 * A) C3S/ESA-CCI LAND COVER, ANNUAL 1992-2022 (300 m)
 *    projects/sat-io/open-datasets/ESA/C3S-LC-L4-LCCS
 *    band 'lccs_class': 10 = rainfed cropland,
 *                       20 = irrigated/post-flooding cropland,
 *                       30 = cropland/vegetation mosaic
 *    -> district-wise class areas per year (this script, part A)
 *
 * B) GLOBAL IRRIGATION AREAS 2001-2015, ANNUAL (~500 m, MODIS-based)
 *    users/deepakna/global_irrigation_maps  (community catalog)
 *    band classes: 0 = no/very little irrigation,
 *                  1 = low-to-medium, 2 = high irrigation
 *    -> district-wise irrigated area per year (part B)
 *
 * Not in GEE but worth knowing (see README):
 *  - AEI: Area Equipped for Irrigation 1900-2015 (Mehta et al. 2024,
 *    Nature Water; 5-arcmin GeoTIFFs on Zenodo - easy to ingest)
 *  - ICRISAT District-Level Database: TABULAR gross/net irrigated
 *    area by district & crop, 1966-2017 - the best true district
 *    irrigation timeline for India (pairs with your APY work).
 *
 * No 1981-present gridded irrigated/rainfed product exists anywhere;
 * 1992-2022 (A) is the longest gridded series available.
 *******************************************************************/

/* ================================================================
   COMMON CONFIG (boundaries only - categorical, no interpolation)
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
print('District count:', districts.size());

var areaHa = ee.Image.pixelArea().divide(10000);

function classAreas(maskImgDict, scale, tag) {
  // maskImgDict: {bandName: maskImage}; returns FeatureCollection of sums
  var img = null;
  Object.keys(maskImgDict).forEach(function(k) {
    var b = areaHa.updateMask(maskImgDict[k]).rename(k);
    img = img === null ? b : img.addBands(b);
  });
  return img.reduceRegions({
    collection: districts,
    reducer:    ee.Reducer.sum(),
    scale:      scale,
    tileScale:  8
  }).map(function(f) { return f.set({year_tag: tag}); });
}
/* ================================================================ */

/* ---------------- A) C3S / ESA-CCI LC, 1992-2022 ---------------- */
var RUN_C3S = true;
var c3s = ee.ImageCollection('projects/sat-io/open-datasets/ESA/C3S-LC-L4-LCCS');

if (RUN_C3S) {
  for (var y = 1992; y <= 2022; y++) {
    var lc = c3s.filterDate(y + '-01-01', (y + 1) + '-01-01')
                .first().select('lccs_class');
    var fc = classAreas({
      rainfed_crop_ha:   ee.Image(lc).eq(10),
      irrigated_crop_ha: ee.Image(lc).eq(20),
      mosaic_crop_ha:    ee.Image(lc).eq(30)
    }, 300, String(y));

    Export.table.toDrive({
      collection:     fc,
      description:    'C3SLC_cropland_districts_' + y,
      folder:         DRIVE_FOLDER,
      fileNamePrefix: 'C3SLC_cropland_districts_' + y,
      fileFormat:     'CSV',
      selectors: ['year_tag', 'state', 'district',
                  'rainfed_crop_ha', 'irrigated_crop_ha', 'mosaic_crop_ha']
    });
  }
}

/* -------- B) Global irrigation areas, 2001-2015 (annual) --------- */
var RUN_GIA = true;

if (RUN_GIA) {
  for (var yy = 2001; yy <= 2015; yy++) {
    var gia = ee.Image('users/deepakna/global_irrigation_maps/' + yy);
    var fc2 = classAreas({
      irrigated_any_ha:  gia.gte(1),    // low-to-medium + high
      irrigated_high_ha: gia.eq(2)
    }, 500, String(yy));

    Export.table.toDrive({
      collection:     fc2,
      description:    'GlobalIrrigation_districts_' + yy,
      folder:         DRIVE_FOLDER,
      fileNamePrefix: 'GlobalIrrigation_districts_' + yy,
      fileFormat:     'CSV',
      selectors: ['year_tag', 'state', 'district',
                  'irrigated_any_ha', 'irrigated_high_ha']
    });
  }
}
