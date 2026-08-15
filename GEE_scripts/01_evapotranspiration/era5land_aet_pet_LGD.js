/*******************************************************************
 * ERA5-LAND ACTUAL & POTENTIAL EVAPOTRANSPIRATION — daily, 791 LGD units
 *
 * FEEDS      Evapotranspiration_LGD/            (drop the CSV in here)
 * CONSUMED BY v5/monsooncast/cleaning/01_clean_merge_panel.py
 *               -> aet_mm, pet_mm, et_days  (monthly SUM)
 *             which become mai = AET/PET and the SPEI water balance
 *               D = rain - PET
 *
 * Dataset : ECMWF/ERA5_LAND/DAILY_AGGR   0.1 deg (~11 km), 1950 -> now-~7d
 * Bands   : total_evaporation_sum       -> AET
 *           potential_evaporation_sum   -> PET
 *
 * UNITS AND THE SIGN FLIP — READ THIS
 *   ECMWF stores evaporation in METRES of water equivalent and uses the
 *   downward-flux-positive convention, so evaporation is NEGATIVE. We
 *   multiply by -1000 to get mm/day with positive meaning water leaving
 *   the surface. Forget the minus and every drought index in the system
 *   inverts; forget the 1000 and PET arrives three orders of magnitude
 *   too small, which does not error — it just makes moisture adequacy
 *   pin at 1.0 everywhere.
 *
 * WHY THIS PRODUCT LAGS THE RAINFALL
 *   ERA5-Land is a reanalysis: it runs about a week behind real time,
 *   while the IMD gauge analysis is 1-2 days behind. The pipeline records
 *   et_days per month for exactly this reason and refuses to build a
 *   water balance across a window the two sources do not both cover.
 *   Expect this export to end ~7 days before the IMD data does. That is
 *   the dataset, not a failed export.
 *
 * Output columns: date, state, district, aet_mm, pet_mm
 *
 * ! Replace the year file, never add a second overlapping one. Monthly
 *   AET/PET are SUMMED across files — an overlap doubles them silently.
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

var era = ee.ImageCollection('ECMWF/ERA5_LAND/DAILY_AGGR')
            .select(['total_evaporation_sum', 'potential_evaporation_sum'])
            .filterBounds(INDIA);

reportLatest(era, 'ERA5-Land daily');

function exportYear(y) {
  var start = ee.Date.fromYMD(y, 1, 1);
  var ic    = era.filterDate(start, start.advance(1, 'year'));

  var table = ic.map(function(img) {
    img = ee.Image(img);
    var dateStr = img.date().format('YYYY-MM-dd');
    var mm = img.multiply(-1000)                    // m -> mm, ECMWF sign flip
                .rename(['aet_mm', 'pet_mm']);
    return districtMeans(mm, ERA_SCALE).map(function(f) {
      return ee.Feature(null, {
        date:     dateStr,
        state:    f.get('state'),
        district: f.get('district'),
        aet_mm:   f.get('aet_mm'),
        pet_mm:   f.get('pet_mm')
      });
    });
  }).flatten();

  var name = 'ERA5L_aet_pet_district_daily_' + y;
  Export.table.toDrive({
    collection:     table,
    description:    name,
    folder:         DRIVE_FOLDER,
    fileNamePrefix: name,                 // canonical -> replaces on download
    fileFormat:     'CSV',
    selectors: ['date', 'state', 'district', 'aet_mm', 'pet_mm']
  });
  print('queued: ' + name + '.csv  ->  Evapotranspiration_LGD/');
}

var ys = yearsToExport();
print('MODE=' + MODE + '  exporting ' + ys.length + ' year task(s):', ys);
for (var i = 0; i < ys.length; i++) exportYear(ys[i]);
