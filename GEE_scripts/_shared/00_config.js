/*******************************************************************
 * _shared/00_config.js  —  the header every district export starts with.
 *
 * The Earth Engine code editor has no cross-file import unless the scripts
 * live in a Git-backed EE repository, and these do not: they are pasted in.
 * So this block is DUPLICATED verbatim at the top of every product script.
 * If you change something here — the asset id, the interpolation setting —
 * change it in all of them, or your products stop being comparable.
 *
 * WHAT MUST STAY IDENTICAL ACROSS PRODUCTS, AND WHY
 *   Every layer in MonsoonCast is eventually differenced against another:
 *   the SPEI water balance is rainfall minus PET, moisture adequacy is
 *   AET over PET. If two products are reduced onto even slightly different
 *   district geometry, or one is bilinearly resampled and the other is not,
 *   the difference carries a systematic artefact that looks like signal.
 *   That is why boundary, reducer and resampling live in one block rather
 *   than being set per script.
 *
 * THE MODE SWITCH
 *   UPDATE    the current year to date, written with the canonical year
 *             filename so the download REPLACES the file already on disk
 *   YEAR      one nominated year, same filename
 *   BACKFILL  a year range, one export task per year
 *
 *   UPDATE is the daily/weekly refresh. It re-exports the whole current
 *   year rather than only the new days, on purpose — see the warning below.
 *
 * ! NEVER ADD AN OVERLAPPING FILE TO A PRODUCT FOLDER !
 *   v5/monsooncast/cleaning/01_clean_merge_panel.py globs *.csv in the
 *   product folder, aggregates each file to months, and then combines the
 *   files. For an EXTENSIVE quantity that combination is a SUM. Two files
 *   both covering July 2026 therefore make July's evapotranspiration
 *   DOUBLE. Nothing errors; the month simply comes out twice as thirsty
 *   and SPEI goes dry across the country.
 *
 *   So: one file per year per folder, always. UPDATE mode emits the
 *   canonical name so the new download overwrites the old file instead of
 *   sitting beside it. After copying a file in, run
 *       py -3.13 -X utf8 "GEE_scripts/verify_exports.py"
 *   which fails loudly on overlapping coverage.
 *******************************************************************/

/* ---------------- boundary: the 791 LGD district units ---------------- */
var ASSET_ID            = 'projects/krishisutra/assets/india_districts_lgd';
var ASSET_DISTRICT_PROP = 'DISTRICT';     // field name in the shapefile
var ASSET_STATE_PROP    = 'STATE_UT';

var districts = ee.FeatureCollection(ASSET_ID)
                  .select([ASSET_DISTRICT_PROP, ASSET_STATE_PROP],
                          ['district', 'state']);
print('District units:', districts.size());     // expect 791

/* India bounding box: filterBounds on this before reducing is the single
   biggest speed win — it stops the reducer touching global tiles. */
var INDIA = ee.Geometry.Rectangle([67.0, 6.0, 98.0, 38.0], null, false);

/* ---------------- reduction ------------------------------------------ */
/* Bilinear resampling before reduceRegions matters for the coarse
   reanalysis grids: ERA5-Land is ~11 km, and a district smaller than a
   cell otherwise inherits one cell's value wholesale. INTERP_SCALE is the
   scale the reducer works at, NOT the native resolution of the data. */
var USE_INTERPOLATION = true;
var INTERP_SCALE      = 1000;

function districtMeans(img, nativeScale) {
  var src = USE_INTERPOLATION ? img.resample('bilinear') : img;
  return src.reduceRegions({
    collection: districts,
    reducer:    ee.Reducer.mean(),
    scale:      USE_INTERPOLATION ? INTERP_SCALE : nativeScale,
    tileScale:  4                       // raise to 8 or 16 on memory errors
  });
}

/* ---------------- what to export ------------------------------------- */
var MODE           = 'UPDATE';          // 'UPDATE' | 'YEAR' | 'BACKFILL'
var YEAR           = 2026;              // used by MODE 'YEAR'
var BACKFILL_FROM  = 1981;              // used by MODE 'BACKFILL'
var BACKFILL_TO    = 2026;

function yearsToExport() {
  var now = new Date().getFullYear();
  if (MODE === 'YEAR')     return [YEAR];
  if (MODE === 'BACKFILL') {
    var ys = [];
    for (var y = BACKFILL_FROM; y <= BACKFILL_TO; y++) ys.push(y);
    return ys;
  }
  return [now];                          // UPDATE
}

/* MAP OVER THE COLLECTION, NOT OVER A CALENDAR
 *
 *   ic.filterDate(...).map(fn)          safe: only real images are visited
 *   ee.List.sequence(...).map(byDate)   DANGEROUS for the current year
 *
 *   The second form visits every day you generate, including days past the
 *   live edge. On those, filterDate returns an EMPTY collection, .mean()
 *   of an empty collection is an image with ZERO BANDS, and the next
 *   arithmetic call fails with
 *
 *       Image.multiply: If one image has no bands, the other must also
 *       have no bands. Got 0 and 1. (Error code: 3)
 *
 *   Completed years have data on every day, so a backfill of 1989-2025
 *   succeeds and only the current year dies — which reads as a problem
 *   with this year's data rather than a bug in the loop. It cost a real
 *   CFSv2 export, 45 seconds and 1,875 EECU-seconds, before failing.
 *
 *   Only build a calendar when you must aggregate sub-daily records up to
 *   days (CFSv2) or days up to months (OISST). When you do, CLAMP the
 *   sequence to the collection's own last date — both of those scripts do.
 */

/* Print the last date the collection actually carries, before you queue
   anything. Reanalysis lags real time and the lag is not constant; knowing
   it up front is the difference between "the export is broken" and "the
   data does not exist yet". */
function reportLatest(ic, label) {
  var last = ee.Date(ic.aggregate_max('system:time_start'));
  print(label + ' — latest available:', last.format('YYYY-MM-dd'));
  print(label + ' — days behind today:',
        ee.Date(Date.now()).difference(last, 'day').round());
}
