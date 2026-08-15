/*******************************************************************
 * ERA5-LAND SOIL MOISTURE, 4 LAYERS — daily, 791 LGD units
 *
 * FEEDS      SM_LGD/                            (drop the CSV in here)
 * CONSUMED BY v5/monsooncast/cleaning/01_clean_merge_panel.py
 *               -> swvl1..swvl4  (monthly MEAN — these are INTENSIVE)
 *             swvl2 is the root-zone layer the dashboard publishes and
 *             the ENSO composite tracks.
 *
 * Dataset : ECMWF/ERA5_LAND/DAILY_AGGR   0.1 deg (~11 km), 1950 -> now-~7d
 * Bands   : volumetric_soil_water_layer_1 .. _4
 * Units   : m3 water / m3 soil, roughly 0.0 (dry) to 0.5 (saturated).
 *           NO unit conversion and NO sign flip — unlike the ET product,
 *           these are already in the form the pipeline wants.
 *
 * THE FOUR LAYERS AND WHY ALL OF THEM ARE CARRIED
 *   layer 1   0 -  7 cm   skin/seedbed. Responds within a day, dries out
 *                         just as fast. This is what governs germination
 *                         at sowing and almost nothing after.
 *   layer 2   7 - 28 cm   ROOT ZONE for most annual crops. The one that
 *                         matters agronomically and the one published.
 *   layer 3  28 -100 cm   deep root / buffer. Drains slowly, so it carries
 *                         the memory of the previous season.
 *   layer 4 100 -289 cm   effectively the water table term. Near-constant
 *                         within a season; useful as a slow covariate.
 *
 *   All four go to the models because their DIFFERENT TIME CONSTANTS are
 *   the signal: a district where layer 1 is dry but layer 3 is wet has a
 *   short dry spell, and one where both are dry has a drought.
 *
 * Output columns: date, state, district, swvl1, swvl2, swvl3, swvl4
 *
 * ! Replace the year file, never add a second overlapping one. These are
 *   averaged rather than summed across files, so an overlap is less
 *   destructive than for ET — but it still silently reweights the month.
 *******************************************************************/

/* ==================== SHARED CONFIG (see _shared/00_config.js) ======= */
var ASSET_ID            = 'projects/krishisutra/assets/india_districts_lgd';
var ASSET_DISTRICT_PROP = 'DISTRICT';
var ASSET_STATE_PROP    = 'STATE_UT';

var districts = ee.FeatureCollection(ASSET_ID)
                  .select([ASSET_DISTRICT_PROP, ASSET_STATE_PROP],
                          ['district', 'state']);
print('District units:', districts.size());

var INDIA = ee.Geometry.Rectangle([67.0, 6.0, 98.0, 38.0], null, false);

var USE_INTERPOLATION = true;
var INTERP_SCALE      = 1000;

function districtMeans(img, nativeScale) {
  var src = USE_INTERPOLATION ? img.resample('bilinear') : img;
  return src.reduceRegions({
    collection: districts,
    reducer:    ee.Reducer.mean(),
    scale:      USE_INTERPOLATION ? INTERP_SCALE : nativeScale,
    tileScale:  4
  });
}

var MODE          = 'UPDATE';        // 'UPDATE' | 'YEAR' | 'BACKFILL'
var YEAR          = 2026;
var BACKFILL_FROM = 1981;
var BACKFILL_TO   = 2026;

function yearsToExport() {
  var now = new Date().getFullYear();
  if (MODE === 'YEAR')     return [YEAR];
  if (MODE === 'BACKFILL') {
    var ys = []; for (var y = BACKFILL_FROM; y <= BACKFILL_TO; y++) ys.push(y);
    return ys;
  }
  return [now];
}

function reportLatest(ic, label) {
  var last = ee.Date(ic.aggregate_max('system:time_start'));
  print(label + ' — latest available:', last.format('YYYY-MM-dd'));
  print(label + ' — days behind today:',
        ee.Date(Date.now()).difference(last, 'day').round());
}
/* ==================== END SHARED CONFIG ============================== */

var DRIVE_FOLDER = 'GEE_ENSO_India';
var ERA_SCALE    = 11132;

var BANDS = ['volumetric_soil_water_layer_1', 'volumetric_soil_water_layer_2',
             'volumetric_soil_water_layer_3', 'volumetric_soil_water_layer_4'];
var OUT   = ['swvl1', 'swvl2', 'swvl3', 'swvl4'];

var era = ee.ImageCollection('ECMWF/ERA5_LAND/DAILY_AGGR')
            .select(BANDS)
            .filterBounds(INDIA);

reportLatest(era, 'ERA5-Land soil moisture');

function exportYear(y) {
  var start = ee.Date.fromYMD(y, 1, 1);
  var ic    = era.filterDate(start, start.advance(1, 'year'));

  var table = ic.map(function(img) {
    img = ee.Image(img);
    var dateStr = img.date().format('YYYY-MM-dd');
    return districtMeans(img.rename(OUT), ERA_SCALE).map(function(f) {
      return ee.Feature(null, {
        date:     dateStr,
        state:    f.get('state'),
        district: f.get('district'),
        swvl1:    f.get('swvl1'),
        swvl2:    f.get('swvl2'),
        swvl3:    f.get('swvl3'),
        swvl4:    f.get('swvl4')
      });
    });
  }).flatten();

  var name = 'ERA5L_soilmoisture_district_daily_' + y;
  Export.table.toDrive({
    collection:     table,
    description:    name,
    folder:         DRIVE_FOLDER,
    fileNamePrefix: name,
    fileFormat:     'CSV',
    selectors: ['date', 'state', 'district',
                'swvl1', 'swvl2', 'swvl3', 'swvl4']
  });
  print('queued: ' + name + '.csv  ->  SM_LGD/');
}

var ys = yearsToExport();
print('MODE=' + MODE + '  exporting ' + ys.length + ' year task(s):', ys);
for (var i = 0; i < ys.length; i++) exportYear(ys[i]);
