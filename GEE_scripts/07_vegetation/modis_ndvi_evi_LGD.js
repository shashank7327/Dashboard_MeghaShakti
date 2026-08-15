/*******************************************************************
 * MODIS NDVI / EVI — 16-day composites, 791 LGD units
 *
 * FEEDS      Vegetation_LGD/                    (new folder — create it)
 * STATUS     NEW. Not consumed by the current pipeline. This is an
 *            INDEPENDENT CHECK ON THE CROP-STRESS LAYER — see below.
 *
 * WHY THIS IS WORTH HAVING
 *   The crop-stress index (v5/monsooncast/crops/15_crop_stress.py) is a
 *   PARAMETRIC model: FAO-33 yield-response factors, crop calendars and
 *   heat thresholds applied to a water balance. Every term traces to a
 *   published source, and the seasonal split has been tested against
 *   irrigation — but the index has never been compared with anything that
 *   actually observes the crop.
 *
 *   NDVI does observe the crop. It is not a yield measurement and it is
 *   not a stress index, but a district whose CSI says "severe" while its
 *   NDVI sits at the seasonal norm is a district worth looking at. That
 *   is the check this export enables.
 *
 * WHAT IT CANNOT DO, AND WHY IT IS NOT A FEATURE
 *   1. NDVI SATURATES over dense canopy, so it compresses exactly where
 *      a healthy crop is — it discriminates poorly at the good end.
 *   2. It cannot separate CROP from other green cover. A district-mean
 *      NDVI mixes the crop with forest, plantation and weeds, and the mix
 *      changes through the season.
 *   3. It is a LAGGING indicator. Vegetation responds to water stress over
 *      one to three weeks, so NDVI confirms a deficit the rainfall data
 *      already showed rather than anticipating it. For a 7-14 day forecast
 *      that lag is the wrong way round.
 *   4. Cloud. The monsoon is the cloudiest time of the Indian year, which
 *      is precisely when the crop matters most. The 16-day composite hides
 *      this by construction — its whole job is to find one clear look per
 *      pixel per window — but in a bad monsoon fortnight the "composite"
 *      can rest on a single marginal observation.
 *
 *   Points 2 and 4 are the reasons this stays a validation layer. Using a
 *   mixed-cover, cloud-limited, lagging signal as a model feature would
 *   import all three problems into the forecast.
 *
 * Dataset : MODIS/061/MOD13Q1   250 m, 16-day, 2000-02-18 -> present
 * Bands   : NDVI, EVI — stored as int16 scaled by 10000. Divided here, so
 *           the output is the conventional -1..+1 range.
 * QA      : SummaryQA 0 (good) or 1 (marginal) kept; 2 (snow/ice) and
 *           3 (cloud) dropped. Dropping marginal as well would leave
 *           monsoon months almost empty.
 *
 * Output columns:
 *   date, state, district, ndvi, evi, n_clear_frac
 *   n_clear_frac is the share of the district's pixels that survived QA.
 *   READ IT. A district-mean NDVI built from 4% of its pixels is not a
 *   district mean, and in July it often is.
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

/* MODIS at 250 m is FINER than the districts, so bilinear resampling is
   pointless here — it is the one product in this collection reduced at
   native scale. */
var REDUCE_SCALE = 250;

var MODE          = 'UPDATE';        // 'UPDATE' | 'YEAR' | 'BACKFILL'
var YEAR          = 2026;
var BACKFILL_FROM = 2000;
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
/* ==================== END SHARED CONFIG ============================== */

var DRIVE_FOLDER = 'GEE_ENSO_India_Vegetation';

var modis = ee.ImageCollection('MODIS/061/MOD13Q1')
              .select(['NDVI', 'EVI', 'SummaryQA'])
              .filterBounds(INDIA);

var last = ee.Date(modis.aggregate_max('system:time_start'));
print('MODIS MOD13Q1 — latest composite start:', last.format('YYYY-MM-dd'));
print('MODIS — days behind today:',
      ee.Date(Date.now()).difference(last, 'day').round());

function exportYear(y) {
  var start = ee.Date.fromYMD(y, 1, 1);
  var ic    = modis.filterDate(start, start.advance(1, 'year'));

  var table = ic.map(function(img) {
    img = ee.Image(img);
    var dateStr = img.date().format('YYYY-MM-dd');

    var qa    = img.select('SummaryQA');
    var clear = qa.lte(1);                       // 0 good, 1 marginal
    var vi    = img.select(['NDVI', 'EVI'])
                   .divide(10000)
                   .rename(['ndvi', 'evi'])
                   .updateMask(clear);

    /* n_clear_frac: the mean of a 0/1 band, i.e. the share of pixels that
       passed QA. It has to be a SEPARATE band from the masked NDVI/EVI —
       if it inherited their mask the reducer would only ever see pixels
       that survived, and the answer would be 1.0 by construction.
       It still carries MOD13Q1's own footprint mask, so read it as "share
       of pixels that had a QA value and were usable", which over Indian
       land is the same thing. */
    var stack = vi.addBands(clear.rename('n_clear_frac').toFloat());

    return stack.reduceRegions({
      collection: districts,
      reducer:    ee.Reducer.mean(),
      scale:      REDUCE_SCALE,
      tileScale:  8                    // 250 m over 791 polygons is heavy
    }).map(function(f) {
      return ee.Feature(null, {
        date:         dateStr,
        state:        f.get('state'),
        district:     f.get('district'),
        ndvi:         f.get('ndvi'),
        evi:          f.get('evi'),
        n_clear_frac: f.get('n_clear_frac')
      });
    });
  }).flatten();

  var name = 'MODIS_ndvi_evi_district_16day_' + y;
  Export.table.toDrive({
    collection:     table,
    description:    name,
    folder:         DRIVE_FOLDER,
    fileNamePrefix: name,
    fileFormat:     'CSV',
    selectors: ['date', 'state', 'district', 'ndvi', 'evi', 'n_clear_frac']
  });
  print('queued: ' + name + '.csv  ->  Vegetation_LGD/');
}

var ys = yearsToExport();
print('MODE=' + MODE + '  exporting ' + ys.length + ' year task(s):', ys);
print('NOTE: 250 m over 791 districts is the heaviest reduction here. '
    + 'If a task fails on memory, raise tileScale to 16.');
for (var i = 0; i < ys.length; i++) exportYear(ys[i]);
