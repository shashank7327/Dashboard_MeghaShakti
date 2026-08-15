/*******************************************************************
 * ERA5-LAND 2 m TEMPERATURE — daily max/min/mean, 791 LGD units
 *
 * FEEDS      TemperatureERA5_LGD/               (new folder — create it)
 * STATUS     CROSS-CHECK LAYER, not yet wired into the master panel.
 *            The panel's tmax_c / tmin_c stay on IMD gauge data. Read the
 *            next paragraph before changing that.
 *
 * WHY THIS EXISTS
 *   IMD's temperature grid is 1.0 degree — about 110 km. The reliability
 *   audit (v5/monsooncast/features/17_feature_reliability.py) found that
 *   on that grid MANY DISTRICTS HAVE AN EFFECTIVE SUPPORT NEAR 1: their
 *   whole temperature series is essentially one cell, and neighbouring
 *   districts share it. Every thermal index in the system — GDD, SDD, the
 *   crop heat-stress term, the temperature anomaly layer — inherits that
 *   coarseness. It is the single worst-resolved input in the product.
 *
 *   ERA5-Land is 0.1 degree (~11 km), a hundredfold finer in area. This
 *   export is here so the gap can be QUANTIFIED, and so a future version
 *   can either switch or blend.
 *
 * WHY IT IS NOT SIMPLY SWAPPED IN
 *   Reanalysis 2 m temperature is not a gauge measurement. Over India it
 *   carries a known cool bias in the pre-monsoon north-west and a warm
 *   bias in the Himalayan foothills, because the model's land surface and
 *   elevation do not match the station network. Swapping it in wholesale
 *   would trade a resolution problem for a bias problem and silently move
 *   every degree-day threshold the crop model depends on.
 *
 *   The defensible path is the one the rainfall layer already took:
 *   compute both, quantify the disagreement per district, publish the
 *   spread. Do that before trusting either.
 *
 * Dataset : ECMWF/ERA5_LAND/DAILY_AGGR
 * Bands   : temperature_2m_max / _min / temperature_2m
 * Units   : KELVIN in the source. Converted to deg C here (-273.15).
 *
 * Output columns:
 *   date, state, district, t2m_max_c, t2m_min_c, t2m_mean_c
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

var DRIVE_FOLDER = 'GEE_ENSO_India_Temp';
var ERA_SCALE    = 11132;

var BANDS = ['temperature_2m_max', 'temperature_2m_min', 'temperature_2m'];
var OUT   = ['t2m_max_c', 't2m_min_c', 't2m_mean_c'];

var era = ee.ImageCollection('ECMWF/ERA5_LAND/DAILY_AGGR')
            .select(BANDS)
            .filterBounds(INDIA);

reportLatest(era, 'ERA5-Land 2 m temperature');

function exportYear(y) {
  var start = ee.Date.fromYMD(y, 1, 1);
  var ic    = era.filterDate(start, start.advance(1, 'year'));

  var table = ic.map(function(img) {
    img = ee.Image(img);
    var dateStr = img.date().format('YYYY-MM-dd');
    var c = img.subtract(273.15).rename(OUT);        // K -> deg C
    return districtMeans(c, ERA_SCALE).map(function(f) {
      return ee.Feature(null, {
        date:       dateStr,
        state:      f.get('state'),
        district:   f.get('district'),
        t2m_max_c:  f.get('t2m_max_c'),
        t2m_min_c:  f.get('t2m_min_c'),
        t2m_mean_c: f.get('t2m_mean_c')
      });
    });
  }).flatten();

  var name = 'ERA5L_t2m_district_daily_' + y;
  Export.table.toDrive({
    collection:     table,
    description:    name,
    folder:         DRIVE_FOLDER,
    fileNamePrefix: name,
    fileFormat:     'CSV',
    selectors: ['date', 'state', 'district',
                't2m_max_c', 't2m_min_c', 't2m_mean_c']
  });
  print('queued: ' + name + '.csv  ->  TemperatureERA5_LGD/');
}

var ys = yearsToExport();
print('MODE=' + MODE + '  exporting ' + ys.length + ' year task(s):', ys);
for (var i = 0; i < ys.length; i++) exportYear(ys[i]);
