/*******************************************************************
 * NCEP CFSv2 DAILY DYNAMICAL COVARIATES — 791 LGD units
 *
 * FEEDS      CFSv2_LGD/                         (new folder — create it)
 * STATUS     ROADMAP INPUT. Not consumed by the current pipeline.
 *
 * WHY THIS DATASET IS THE ONE THAT COULD MOVE THE FORECAST
 *   MonsoonCast's 7/14-day skill is +0.048 and +0.027 against climatology.
 *   That is not a tuning failure — it is the ceiling of the information
 *   available. At 7-14 days district rainfall is WEATHER: monsoon
 *   low-pressure systems, their tracks, and the phase of the MJO.
 *   Antecedent rainfall, last month's land surface and slowly varying
 *   ocean indices do not determine where a monsoon low will form.
 *
 *   CFSv2 is a coupled atmosphere-ocean-land MODEL. Its fields carry the
 *   circulation, humidity transport and thermal state that actually drive
 *   monsoon rainfall at that range, and — unlike ECMWF open data — Earth
 *   Engine holds it as a LONG, UNIFORM ARCHIVE back to 1979.
 *
 *   That archive property is the whole point. Model Output Statistics
 *   needs many years of past forecasts paired with what was observed, so
 *   the correction can be trained and then verified. ECMWF open data is a
 *   real-time feed with no reforecast and cannot supply it; CFSv2 can.
 *
 * Dataset : NOAA/CFSV2/FOR6H   ~0.2 deg (~22 km), 6-hourly, 1979-03-31 ->
 *
 * DATE RANGE — 1989, chosen deliberately
 *   The forecast models train on 1990-2019 (every product complete).
 *   Starting at 1989 gives one full year of lead-in so 12-month
 *   accumulations are defined from the first training year, without
 *   hauling down eleven years that would never be used.
 *
 * 6-HOURLY -> DAILY
 *   Precipitation is a RATE (kg/m2/s). The daily total is
 *   mean(rate) x 86400, which equals sum(rate) x 21600 but does not depend
 *   on exactly four records existing on every day.
 *
 *   ALWAYS READ n_records. A full day has 4. The LAST day of a current-year
 *   export usually has fewer, because the run stops wherever the live edge
 *   happens to fall, and mean(rate) x 86400 then EXTRAPOLATES a part-day to
 *   a full-day total — a day holding only the 00:00 record is reported as
 *   if that rate ran for 24 hours. The column is there so the consumer can
 *   drop or flag those rows; nothing upstream does it for you.
 *
 * RUNTIME WARNING
 *   ~13,500 days x 791 districts on BACKFILL. One export task per year;
 *   submitting the whole range at once will exceed the user memory limit.
 *   Expect several minutes per year-task. Use MODE 'UPDATE' for refreshes.
 *
 * Output columns:
 *   date, state, district, cfs_precip_mm, cfs_tmean_c, cfs_tmax_c,
 *   cfs_tmin_c, cfs_q, cfs_wind_ms, cfs_srad, n_records
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

/* CFSv2 is ~22 km and many LGD districts are smaller than one cell.
   Reducing at native scale leaves those districts empty, so the reduction
   runs at a finer scale over the interpolated image — same treatment the
   ERA5 products get. */
var USE_INTERPOLATION = true;
var REDUCE_SCALE      = 5000;

function districtMeans(img, nativeScale) {
  var src = USE_INTERPOLATION ? img.resample('bilinear') : img;
  return src.reduceRegions({
    collection: districts,
    reducer:    ee.Reducer.mean(),
    scale:      USE_INTERPOLATION ? REDUCE_SCALE : nativeScale,
    tileScale:  4
  });
}

var MODE          = 'UPDATE';        // 'UPDATE' | 'YEAR' | 'BACKFILL'
var YEAR          = 2026;
var BACKFILL_FROM = 1989;            // one year of lead-in before 1990
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

var DRIVE_FOLDER = 'GEE_ENSO_India_CFSv2';

var PREC = 'Precipitation_rate_surface_6_Hour_Average';
var TEMP = 'Temperature_height_above_ground';
var SPFH = 'Specific_humidity_height_above_ground';
var UWND = 'u-component_of_wind_height_above_ground';
var VWND = 'v-component_of_wind_height_above_ground';
var SRAD = 'Downward_Short-Wave_Radiation_Flux_surface_6_Hour_Average';

var cfs = ee.ImageCollection('NOAA/CFSV2/FOR6H')
            .select([PREC, TEMP, SPFH, UWND, VWND, SRAD])
            .filterBounds(INDIA);

reportLatest(cfs, 'CFSv2 6-hourly');

/*  THE TEMPERATURE BAND CHANGED UNITS MID-COLLECTION
 *
 *  NOAA/CFSV2/FOR6H delivered `Temperature_height_above_ground` in KELVIN
 *  until 20 October 2025 and in DEGREES CELSIUS from 21 October 2025. The
 *  band was not renamed and nothing in the metadata announces it.
 *
 *  Subtracting 273.15 unconditionally therefore converts the newer records
 *  a SECOND time: a real 27 C is published as 27, and the script reports
 *  -246.15. Every value after the cutover is wrong by exactly 273.15, and
 *  it looks like plausible-shaped data — same spatial pattern, same daily
 *  range — just shifted, which is why it survived a download.
 *
 *  The cutover lands MID-DAY on 20 Oct 2025, so that day's four 6-hourly
 *  records are in different units from each other. Any code that takes the
 *  daily mean/max/min BEFORE converting mixes Kelvin with Celsius inside a
 *  single statistic and produces a number no arithmetic can undo: on 20 Oct
 *  the national tmax read 30.3 (max picked a Kelvin record) while tmin read
 *  -253.4 (min picked a Celsius one).
 *
 *  So: convert EVERY RECORD FIRST, then aggregate. A 2 m air temperature
 *  above 100 can only be Kelvin, so the test is per pixel and needs no
 *  hard-coded cutover date — it keeps working if the provider switches back.
 */
function toCelsius(img) {
  return ee.Image(img).subtract(ee.Image(img).gt(100).multiply(273.15))
           .copyProperties(img, ['system:time_start']);
}

function dailyImage(date) {
  var d0  = ee.Date(date);
  var day = cfs.filterDate(d0, d0.advance(1, 'day'));

  var tC = ee.ImageCollection(day.select(TEMP).map(toCelsius));

  var precip = day.select(PREC).mean().multiply(86400).rename('cfs_precip_mm');
  var tmean  = tC.mean().rename('cfs_tmean_c');
  var tmax   = tC.max().rename('cfs_tmax_c');
  var tmin   = tC.min().rename('cfs_tmin_c');
  var q      = day.select(SPFH).mean().rename('cfs_q');
  var wind   = day.select(UWND).mean().hypot(day.select(VWND).mean())
                  .rename('cfs_wind_ms');
  var srad   = day.select(SRAD).mean().rename('cfs_srad');

  return precip.addBands([tmean, tmax, tmin, q, wind, srad])
               .set('system:time_start', d0.millis())
               .set('date', d0.format('YYYY-MM-dd'))
               .set('n_records', day.size());
}

var COLS = ['date', 'state', 'district', 'cfs_precip_mm', 'cfs_tmean_c',
            'cfs_tmax_c', 'cfs_tmin_c', 'cfs_q', 'cfs_wind_ms', 'cfs_srad',
            'n_records'];

/*  THE CURRENT YEAR IS A PARTIAL YEAR, AND THAT USED TO KILL THE EXPORT
 *
 *  Generating all 365 days of the current year and calling dailyImage on
 *  each one fails, because for any day past the live edge
 *  `cfs.filterDate(d0, d1)` is EMPTY, and `.mean()` of an empty
 *  ImageCollection returns an image with ZERO BANDS. The next line is
 *  `.multiply(86400)`, and Earth Engine rejects it:
 *
 *      Image.multiply: If one image has no bands, the other must also
 *      have no bands. Got 0 and 1. (Error code: 3)
 *
 *  Every completed year has records on all 365 days, so this NEVER fires
 *  on 1989-2025 — the whole backfill succeeds and only the current year
 *  dies, about 45 seconds in, after burning real compute on the months
 *  that do exist. That asymmetry is what makes it look like a problem
 *  with "this year's data" rather than a bug in the loop.
 *
 *  So the day sequence is clamped to the last day the collection actually
 *  carries. No empty day is ever constructed.
 */
function exportYear(y) {
  var start = ee.Date.fromYMD(y, 1, 1);
  var yStop = start.advance(1, 'year');

  // midnight of the last day that has any record at all
  var lastDay = ee.Date(ee.Date(ee.Number(
    cfs.aggregate_max('system:time_start'))).format('YYYY-MM-dd'));
  // exclusive end = whichever comes first, next January or the day after
  // the live edge
  var end = ee.Date(lastDay.advance(1, 'day').millis().min(yStop.millis()));
  var nDays = end.difference(start, 'day');

  print(y + ': exporting', nDays, 'day(s), through',
        end.advance(-1, 'day').format('YYYY-MM-dd'));

  var daily = ee.ImageCollection(
    ee.List.sequence(0, nDays.subtract(1)).map(function(i) {
      return dailyImage(start.advance(ee.Number(i), 'day'));
    })
  );

  var table = daily.map(function(img) {
    img = ee.Image(img);
    return districtMeans(img, 22000).map(function(f) {
      return f.set('date', img.get('date'))
              .set('n_records', img.get('n_records'));
    });
  }).flatten();

  var name = 'CFSv2_district_daily_' + y;
  Export.table.toDrive({
    collection:     table,
    description:    name,
    folder:         DRIVE_FOLDER,
    fileNamePrefix: name,
    fileFormat:     'CSV',
    selectors:      COLS
  });
  print('queued: ' + name + '.csv  ->  CFSv2_LGD/');
}

var ys = yearsToExport();
print('MODE=' + MODE + '  exporting ' + ys.length + ' year task(s):', ys);
if (ys.length > 4) {
  print('NOTE: ' + ys.length + ' CFSv2 year-tasks queued. These are heavy — '
        + 'let them finish in batches rather than all at once.');
}
for (var i = 0; i < ys.length; i++) exportYear(ys[i]);
