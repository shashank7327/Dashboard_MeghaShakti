/*******************************************************************
 * NIÑO 3.4 & IOD BOX SSTs — monthly, single CSV (not district-wise)
 *
 * FEEDS      Indices/                           (new folder — create it)
 * STATUS     CROSS-CHECK. The pipeline's operational ONI/DMI come from
 *            NOAA CPC and NOAA PSL, NOT from this export. Read why below.
 *
 * WHY THE OFFICIAL SERIES ARE USED INSTEAD OF THESE
 *   ONI is not simply "the Niño 3.4 SST anomaly". CPC computes it from
 *   ERSSTv5 against a ROLLING 30-YEAR base period that is re-centred every
 *   five years, precisely so that a warming trend does not gradually turn
 *   every year into an El Niño. Reproducing that from OISST here would
 *   give a similar-looking series that is not the one anybody else quotes,
 *   and the ENSO phase label is a TRAINED MODEL FEATURE — changing its
 *   definition silently changes what the models learned.
 *
 *   So this export exists to CHECK the cached official series, not to
 *   replace it. If the two diverge materially, something is wrong with the
 *   cache and that is worth knowing.
 *
 * Dataset : NOAA/CDR/OISST/V2_1   0.25 deg daily, from 1981-09-01
 *           band 'sst', stored as int16 with a 0.01 scale factor -> deg C
 *
 * BOXES (the standard definitions)
 *   Niño 3.4   5S-5N, 170W-120W    ENSO state
 *   WTIO       10S-10N, 50E-70E    IOD western pole
 *   SETIO      10S-0,   90E-110E   IOD eastern pole
 *
 * POST-PROCESSING (pandas, not part of this script)
 *   anomaly = sst - calendar-month climatology on a stated baseline
 *   DMI     = wtio_anom - setio_anom
 *   ONI-ish = 3-month centred running mean of the Niño 3.4 anomaly
 *
 *   Note that a CENTRED running mean is not knowable in real time. The
 *   pipeline's enso_phase uses a strictly TRAILING classification for
 *   exactly that reason — a centred definition would let a June forecast
 *   know what the ocean does in September.
 *
 * Output columns: month, nino34_sst_c, wtio_sst_c, setio_sst_c
 *
 * One file, rewritten whole each run. No overlap hazard here.
 *******************************************************************/

var DRIVE_FOLDER = 'GEE_ENSO_India_Indices';

var MODE       = 'UPDATE';       // 'UPDATE' = 1981-09 -> now (one CSV)
var START      = '1981-09-01';   // OISST V2_1 begins here

var sst = ee.ImageCollection('NOAA/CDR/OISST/V2_1').select('sst');

var last = ee.Date(sst.aggregate_max('system:time_start'));
print('OISST — latest available:', last.format('YYYY-MM-dd'));
print('OISST — days behind today:',
      ee.Date(Date.now()).difference(last, 'day').round());

var NINO34 = ee.Geometry.Rectangle([-170, -5, -120,  5], null, false);
var WTIO   = ee.Geometry.Rectangle([  50, -10,   70, 10], null, false);
var SETIO  = ee.Geometry.Rectangle([  90, -10,  110,  0], null, false);

var start   = ee.Date(START);
/* Stop at the last COMPLETE month. A part-month box mean is a different
   quantity from a monthly mean and would land in the series looking like
   a real value. */
var lastFull = last.advance(-1, 'month').update(null, null, 1);
var nMonths  = lastFull.difference(start, 'month').floor().add(1);

var table = ee.FeatureCollection(
  ee.List.sequence(0, nMonths.subtract(1)).map(function(m) {
    var t0  = start.advance(ee.Number(m), 'month');
    var t1  = t0.advance(1, 'month');
    var img = sst.filterDate(t0, t1).mean().multiply(0.01);   // -> deg C

    function boxMean(geom) {
      return img.reduceRegion({
        reducer:   ee.Reducer.mean(),
        geometry:  geom,
        scale:     27830,          // OISST native ~0.25 deg
        maxPixels: 1e9,
        bestEffort: true
      }).get('sst');
    }

    return ee.Feature(null, {
      month:         t0.format('YYYY-MM'),
      nino34_sst_c:  boxMean(NINO34),
      wtio_sst_c:    boxMean(WTIO),
      setio_sst_c:   boxMean(SETIO)
    });
  })
);

var name = 'OISST_nino34_iod_monthly';
Export.table.toDrive({
  collection:     table,
  description:    name,
  folder:         DRIVE_FOLDER,
  fileNamePrefix: name,
  fileFormat:     'CSV',
  selectors: ['month', 'nino34_sst_c', 'wtio_sst_c', 'setio_sst_c']
});
print('queued: ' + name + '.csv  ->  Indices/');
print('Complete months in the export:', nMonths);
