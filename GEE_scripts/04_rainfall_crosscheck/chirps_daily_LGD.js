/*******************************************************************
 * CHIRPS DAILY RAINFALL — 791 LGD units
 *
 * FEEDS      Chirps_LGD/                        (new folder — create it)
 * CONSUMED BY v5/monsooncast/features/17_feature_reliability.py
 *
 * THIS IS NOT AN OPERATIONAL LAYER AND MUST NOT BECOME ONE
 *   MonsoonCast's rainfall is IMD gauge data. CHIRPS exists here for
 *   exactly one purpose: to be a SECOND, INDEPENDENT estimate of the same
 *   quantity, so the system can say how well district rainfall is known
 *   rather than asserting a single number.
 *
 *   CHIRPS is satellite-plus-station (infrared cold-cloud duration
 *   calibrated against gauges); IMD's product is an interpolated gauge
 *   analysis. Their errors are largely independent, which is what makes
 *   the comparison worth anything. NEITHER IS ADJUSTED TOWARD THE OTHER,
 *   and no feature is ever built from a blend of the two.
 *
 *   What comes out of it: the median district shows a 22.1% difference
 *   between the two products on seasonal totals, rising to 43.6% at the
 *   90th percentile. That band is the yardstick — a disagreement with any
 *   published bulletin INSIDE it is not an error in either source, it is
 *   district rainfall being genuinely uncertain.
 *
 *   An earlier version of this project used CHIRPS as the primary
 *   rainfall input. It carries a wet bias over India relative to the
 *   gauge analysis, and replacing it with IMD is why the pipeline exists
 *   in its current form. Do not put it back.
 *
 * Dataset : UCSB-CHG/CHIRPS/DAILY   0.05 deg (~5.5 km), 1981 -> now-~3 weeks
 * Band    : precipitation, already mm/day. No conversion.
 *
 * NOTE ON LATENCY: the CHIRPS final product lags by two to three weeks.
 * It is not, and cannot be, a real-time layer. That is the second reason
 * it is a validation input rather than an operational one.
 *
 * Output columns: date, state, district, chirps_mm
 *
 * ! Replace the year file, never add a second overlapping one.
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

/* CHIRPS at 5.5 km is fine enough that bilinear resampling buys little and
   costs time — but it is left ON so this product is reduced by exactly the
   same operator as every other layer. Comparability beats micro-optimisation
   when the entire point of the file is a like-for-like comparison. */
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

var DRIVE_FOLDER  = 'GEE_ENSO_India_CHIRPS';
var CHIRPS_SCALE  = 5566;

var chirps = ee.ImageCollection('UCSB-CHG/CHIRPS/DAILY')
               .select('precipitation')
               .filterBounds(INDIA);

reportLatest(chirps, 'CHIRPS daily');

function exportYear(y) {
  var start = ee.Date.fromYMD(y, 1, 1);
  var ic    = chirps.filterDate(start, start.advance(1, 'year'));

  var table = ic.map(function(img) {
    img = ee.Image(img);
    var dateStr = img.date().format('YYYY-MM-dd');
    return districtMeans(img.rename('chirps_mm'), CHIRPS_SCALE)
      .map(function(f) {
        return ee.Feature(null, {
          date:      dateStr,
          state:     f.get('state'),
          district:  f.get('district'),
          chirps_mm: f.get('chirps_mm')
        });
      });
  }).flatten();

  var name = 'CHIRPS_district_daily_' + y;
  Export.table.toDrive({
    collection:     table,
    description:    name,
    folder:         DRIVE_FOLDER,
    fileNamePrefix: name,
    fileFormat:     'CSV',
    selectors: ['date', 'state', 'district', 'chirps_mm']
  });
  print('queued: ' + name + '.csv  ->  Chirps_LGD/');
}

var ys = yearsToExport();
print('MODE=' + MODE + '  exporting ' + ys.length + ' year task(s):', ys);
for (var i = 0; i < ys.length; i++) exportYear(ys[i]);
