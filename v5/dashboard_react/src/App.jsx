
import React, {useState, useMemo, useRef} from "react";
import * as XLSX from "xlsx";
import DATA from "./data/data.json";
import GEO from "./data/geo.json";
import DAILY_JSON from "./data/daily.json";
if (typeof window !== "undefined") window.__DAILY__ = DAILY_JSON;
const DAILY = window.__DAILY__ || null;
const MTD = DATA.month_to_date || null;      // the month still running, if any
// Every trained formulation, each with the scores it actually earned.
// Publishing only the MSE winner would hide a real trade-off: the log-ratio
// fit lands the most IMD categories correctly while scoring near the bottom on
// mean-squared error. Which model is "best" depends on what you are trading.
const MODELS = DATA.models || {};

// DAILY LAYERS
//   Every daily layer is a ROLLING or CUMULATIVE window ending on the selected
//   day, never a single day's departure: most district-days are dry, so a
//   one-day ratio is either -100% or a huge positive number and the map would
//   just flicker.  Because each window's normal covers exactly the same
//   calendar days as its actual, these are day-matched by construction and the
//   partial-month problem that affects the monthly indices cannot arise.
const RAW_DGROUPS = [
  ["Rainfall", [["dep_7d","Last 7 days","%","div",80],
    ["dep_30d","Last 30 days","%","div",60],
    ["dep_season","Season to date","%","div",60],
    ["rain_mm","Rain that day","mm","blu",[0,60]]]],
  ["Heat", [["tmax_anom_d","Tmax anomaly","°C","tmp",4],
    ["tmin_anom_d","Tmin anomaly","°C","tmp",4],
    // adaptive for the same reason as the monthly anomaly: a 7-day degree-day
    // sum above 34 degC is ~0 across almost all of India in the monsoon
    ["sdd_7d","Heat stress, 7d","","hot",[0,40],true]]],
  ["Water", [["swvl2_d","Root-zone moisture","","blu",[0.1,0.42]]]],
  // CFSv2 is a SECOND PRODUCT, not a replacement. Earth Engine serves this
  // collection indexed by VALID TIME with no lead-time axis, so these are the
  // coupled model's analysis of the atmosphere, not a forecast — the group is
  // labelled "CFSv2 model" rather than "forecast" for exactly that reason.
  // Its departures are computed against CFSv2's OWN climatology: differencing
  // a model actual against IMD's normal would report the model's wet bias
  // over India as a rainfall anomaly.
  ["CFSv2 model", [["cfs_dep_7d","CFSv2 rain, 7d vs own normal","%","div",80],
    ["cfs_dep_30d","CFSv2 rain, 30d vs own normal","%","div",60],
    ["cfs_precip_mm","CFSv2 rain that day","mm","blu",[0,60]],
    ["cfs_tmax_c","CFSv2 Tmax","°C","tmp",45],
    ["cfs_wind_ms","CFSv2 wind speed","m/s","blu",[0,12]]]],
];

// Same rule the monthly toolbar uses: offer only what the payload carries, so
// a product whose export has not landed does not sit behind a button
// rendering a blank map.
const DHAVE = (DAILY && DAILY.layers) ? DAILY.layers : {};
const DGROUPS = RAW_DGROUPS
  .map(function(g){ return [g[0], g[1].filter(function(l){ return l[0] in DHAVE; })]; })
  .filter(function(g){ return g[1].length > 0; });
const DLAY = {}; DGROUPS.forEach(([g,ls])=>ls.forEach(l=>DLAY[l[0]]={grp:g,label:l[1],unit:l[2],kind:l[3],dom:l[4],adaptive:!!l[5]}));

const RAW_GROUPS = [
  ["Forecast", [["fc7","7-day","%","div",30],["fc14","14-day","%","div",30]]],
  ["Climate", [["pct_departure","Rainfall vs normal","%","div",60],
    ["spei_era5_4","SPEI-4","σ","div",2],["spei_era5_12","SPEI-12","σ","div",2],
    ["tmax_anom","Temp anomaly","°C","tmp",3]]],
  ["Water", [["mai","Soil-moisture adequacy","","adeq",[0,1]],
    ["swvl2","Root-zone moisture","","blu",[0.1,0.42]]]],
  // Heat stress is shown as an ANOMALY on a diverging scale, not as a raw
// degree-day sum. The raw sum maps where it is hot -- the same places
// every year -- and says nothing about whether this month is unusual,
// which is the only question a stress layer answers. 52% of the raw
// record is exactly zero and the peak months pin any fixed scale.
  // The 6th field marks the domain ADAPTIVE — see adaptDomain() below. Heat
// stress needs it because its spread collapses seasonally: SDD counts
// degree-days above 34 degC, and through the monsoon Tmax rarely reaches 34,
// so August's real range is about +/-5 while April-June runs past +/-100. On
// the fixed +/-60 scale August put 95% of districts inside the neutral band
// and the layer read as broken when it was merely small.
  ["Heat", [["sdd_anom","Heat stress vs normal","°C-d","tmp",60,true]]],
  // NDVI saturates over dense canopy and mixes crop with every other green
  // thing in the district, so it is shown as an observation to compare
  // against the modelled crop stress — not as a stress measure itself.
  ["Vegetation", [["ndvi","Greenness (NDVI)","","grn",[0.05,0.85]],
    ["evi","Vegetation index (EVI)","","grn",[0.02,0.65]]]],
  ["ENSO", [["pct_departure_signature","El Niño signature","pp","div",50]]],
];

// Offer only what the export actually published. LAYERS in
// 05_forecast_export.py drops any layer missing from the panel, so a product
// whose first GEE export has not landed yet simply does not appear in the
// toolbar. Listing it anyway would put a uniformly blank map behind a button,
// and a blank map reads as a broken product rather than an absent one.
const HAVE = Object.assign({fc7:1, fc14:1}, DATA.layers || {});

// The ECMWF group is BUILT FROM THE DATA, not hard-coded, because which lead
// times exist depends on what the last fetch actually retrieved — the run in
// hand had only 24 h and 48 h where the fetcher defaults out to 360 h. A
// static list would either show buttons for leads that were never downloaded
// or hide leads that were. Keys are ecmwf_{hours}h_mean; sorted numerically so
// +1d comes before +10d, which a lexical sort gets wrong.
// ONE ECMWF ENTRY PLUS A LEAD TOGGLE, NOT TEN BUTTONS.
//
// The leads are not ten different quantities — they are one quantity at ten
// horizons, and accumulated rainfall grows monotonically with lead (the +15d
// national mean is ten times the +1d one). Ten buttons on the same toolbar
// row as the statistical forecasts implied ten independent choices and made
// the group unreadable. The lead is a property of the ECMWF layer, so it
// belongs on its own control with its own scale.
const ECMWF_LEADS = Object.keys(DATA.layers || {})
  .filter(function(k){ return /^ecmwf_\d+h_mean$/.test(k); })
  .map(function(k){ return parseInt(k.match(/\d+/)[0], 10); })
  .sort(function(a,b){ return a - b; });

// Each lead needs its OWN domain. A single 0-80 mm scale would leave +1d
// almost blank and +15d uniformly saturated, since the accumulation runs from
// ~15 mm to ~156 mm nationally. Scaling by lead keeps every horizon legible.
function ecmwfLayer(h){
  return ["ecmwf_" + h + "h_mean", "ECMWF ENS +" + (h/24) + "d", "mm", "blu",
          [0, Math.max(30, Math.round(h/24 * 22))]];
}
const ECMWF = ECMWF_LEADS.length ? [ecmwfLayer(ECMWF_LEADS[0])] : [];

const GROUPS = RAW_GROUPS
  .map(function(g){
    // ECMWF is a real forecast — explicit lead time, 50 members, a genuine
    // distribution — so it sits in the Forecast group beside the statistical
    // models as a CHOICE. It is not verified here: open data carries no
    // reforecast archive, so no skill score exists for it and none is shown.
    if (g[0] === "Forecast" && ECMWF.length) return [g[0], g[1].concat(ECMWF)];
    return g;
  })
  .map(function(g){ return [g[0], g[1].filter(function(l){ return l[0] in HAVE; })]; })
  .filter(function(g){ return g[1].length > 0; });
// Crop stress is its own layer family: one entry per crop, values coming from
// DATA.crops[crop].v rather than from the district record, because the index
// exists only where that crop is actually sown.
const CROPS = DATA.crops || {};
const CROPKEYS = Object.keys(CROPS);
const LAY = {}; GROUPS.forEach(([g,ls])=>ls.forEach(l=>LAY[l[0]]={grp:g,label:l[1],unit:l[2],kind:l[3],dom:l[4],adaptive:!!l[5]}));

// ADAPTIVE DOMAINS — for layers whose spread collapses with the season.
//
// Most layers have a stable range and a fixed domain is right: it keeps the
// colour of a district comparable as you scrub the slider, which is the whole
// point of a fixed scale. Heat stress does not have a stable range. SDD counts
// degree-days above 34 degC, so it is large before the monsoon and ~0 during
// it, and the same fixed domain cannot serve both. On +/-60 the August map put
// 95% of districts inside the neutral band.
//
// So the domain is taken from the values ACTUALLY ON SCREEN, snapped to a
// readable step, and the legend prints it — the reader is never left guessing
// what the colours mean. The floor stops a genuinely uniform month from being
// magnified into a map of rounding noise.
const NICE = [1,2,3,5,10,15,20,30,40,60,80,100,150,200,300];
function adaptDomain(L, vals){
  if(!L || !L.adaptive) return L;
  const a = [];
  for(const k in vals){ const v = vals[k];
    if(v != null && isFinite(v)) a.push(Math.abs(v)); }
  if(a.length < 8) return L;                       // too few to infer a scale
  a.sort(function(x,y){ return x - y; });
  const p95 = a[Math.floor(0.95 * (a.length - 1))];
  let lim = NICE.find(function(n){ return n >= p95; });
  if(lim == null) lim = Math.ceil(p95);
  if(Array.isArray(L.dom)){
    // one-sided ramps (0 -> hi): only the top moves
    const floor = Math.max(1, (L.dom[1] - L.dom[0]) * 0.05);
    return Object.assign({}, L, {dom:[L.dom[0], Math.max(L.dom[0] + floor, lim)],
                                 domAuto:true});
  }
  return Object.assign({}, L, {dom: Math.max(1, lim), domAuto:true});
}
// Every ECMWF lead has to be in the registry even though only the shortest
// appears in the toolbar: the lead toggle sets `layer` to any of them, and a
// layer absent from LAY resolves to undefined and takes the render down.
ECMWF_LEADS.forEach(function(h){
  const l = ecmwfLayer(h);
  LAY[l[0]] = {grp:"Forecast", label:l[1], unit:l[2], kind:l[3], dom:l[4]};
});

function hx(h){h=h.replace('#','');return[0,2,4].map(i=>parseInt(h.slice(i,i+2),16));}
function mix(a,b,t){const x=hx(a),y=hx(b);return`rgb(${x.map((v,i)=>Math.round(v+(y[i]-v)*t)).join(',')})`;}
// nodata is passed as a literal colour, not a CSS variable: the map is
// serialised to PNG, and a detached SVG cannot resolve var(--nodata).
// COLOUR: MAGNITUDE MUST READ AS INTENSITY, AND THE HUE MUST MATCH THE THING
//
// The sequential ramps used to run DARK -> BRIGHT, which inverted every
// reading on a dark map. Zero rainfall was painted deep navy (#0d366b) and
// heavy rain pale blue; bare soil was dark forest green and dense canopy
// bright green; no heat stress was near-black brown. The low end of every
// scale looked like the strongest possible value, and worse, "no rain" was
// almost exactly the no-data colour, so absence of rain and absence of
// measurement were indistinguishable — the confusion the crop layer goes out
// of its way to avoid.
//
// Now every sequential ramp starts at a pale, low-saturation base and
// deepens toward a hue that means the quantity: blue for water, green for
// vegetation, red for heat, and brown->green for adequacy so that "dry" is
// brown rather than a dark version of "wet". Diverging scales are unchanged:
// deficit red / excess blue and cold blue / hot red are the conventions IMD
// and every met service already publish, and breaking them to be internally
// tidy would be worse than the problem.
const RAMPS = {
  // low (pale, reads as "little")            high (saturated, reads as "much")
  blu:  ["#e8f1fb", "#0b4f9e"],   // water: pale sky -> deep blue
  grn:  ["#f2ecdd", "#166534"],   // vegetation: bare tan -> dense green
  hot:  ["#fff2d9", "#a5150a"],   // heat load: pale straw -> deep red
  adeq: ["#8c5a2b", "#166534"],   // adequacy: brown (stressed) -> green (met)
};
function color(v,L,mid,nod){
  if(v==null||!isFinite(v))return nod||"#4a5568";
  if(L.kind==="div"||L.kind==="tmp"){const lim=L.dom;let t=Math.max(-1,Math.min(1,v/lim));
    const neg=L.kind==="tmp"?"#3b82f6":"#ff5a4d", pos=L.kind==="tmp"?"#ff5a4d":"#3b82f6";
    return t<0?mix(mid,neg,-t):mix(mid,pos,t);}
  const[lo,hi]=L.dom;let t=Math.max(0,Math.min(1,(v-lo)/(hi-lo)));
  if(L.kind==="cropstress"){
    return t<0.5 ? mix("#1a9850","#fedc78",t*2) : mix("#fedc78","#aa1228",(t-0.5)*2);
  }
  const stops=RAMPS[L.kind]||RAMPS.blu;
  return mix(stops[0],stops[1],t);
}
const f1=v=>v==null||!isFinite(v)?"–":(v>0&&/div|tmp/.test("x")?"":"" )+(Math.round(v*10)/10);
const sg=v=>v==null||!isFinite(v)?"–":(v>0?"+":"")+(Math.round(v*10)/10);
// unsigned, one decimal — for magnitudes like a degree-day total and its
// normal, where a leading "+" would read as an anomaly
const sg1=v=>v==null||!isFinite(v)?"–":(Math.round(v*10)/10)+"";

// The fixed rows of the district briefing. Each is resolved at its OWN latest
// published month by briefVal() in the component, because these features have
// different source lags and do not share an edge.
const BRIEF_ROWS=[["pct_departure","Rainfall vs normal","%"],
  ["spei_era5_4","Drought SPEI-4","σ"],["spei_era5_12","Drought SPEI-12","σ"],
  ["mai","Soil-moisture adequacy",""],["swvl2","Root-zone soil moisture",""],
  ["tmax_anom","Temperature anomaly","°C"],
  ["sdd_anom","Heat stress vs normal","°C-d"],
  ["ndvi","Greenness (NDVI)",""],["evi","Vegetation index (EVI)",""],
  ["pct_departure_signature","El Niño signature","pp"]];

const B=(()=>{let a=[999,999,-999,-999];GEO.features.forEach(f=>{const gg=f.geometry,ps=gg.type==="Polygon"?[gg.coordinates]:gg.coordinates;
 ps.forEach(p=>p.forEach(r=>r.forEach(([x,y])=>{a[0]=Math.min(a[0],x);a[1]=Math.min(a[1],y);a[2]=Math.max(a[2],x);a[3]=Math.max(a[3],y);})));});return a;})();
const MW=600,MH=660,px=x=>(x-B[0])/(B[2]-B[0])*MW,py=y=>MH-(y-B[1])/(B[3]-B[1])*MH;
function pathOf(g){const ps=g.type==="Polygon"?[g.coordinates]:g.coordinates;let d="";
 ps.forEach(p=>p.forEach(r=>{d+="M"+r.map(([x,y])=>px(x).toFixed(1)+","+py(y).toFixed(1)).join("L")+"Z";}));return d;}

// ---------------------------------------------------------------- downloads
function dl(name, blob){
  const a=document.createElement("a");
  a.href=URL.createObjectURL(blob); a.download=name;
  document.body.appendChild(a); a.click();
  setTimeout(()=>{URL.revokeObjectURL(a.href); a.remove();}, 0);
}
// One row per district with EVERY layer, so the export is the whole feature
// set rather than just whatever the map happens to be showing.
function tableRows(vals, ym){
  return DATA.districts.map(d=>{
    const o={district_id:d.id, district:d.d, state:d.s,
             jk_ladakh:d.jk?1:0, month:ym||DATA.issued,
             forecast_7d_pct:d.fc7, forecast_14d_pct:d.fc14};
    Object.keys(LAY).forEach(k=>{
      if(k==="fc7"||k==="fc14") return;
      const m = DATA.monthly && DATA.monthly[k];
      o[LAY[k].label] = (m && ym) ? (m[ym]||{})[d.id] : d[k];
    });
    return o;
  });
}
function toCSV(rs){
  const cols=Object.keys(rs[0]);
  const esc=v=>v==null?"":(/[",\n]/.test(String(v))
      ? '"'+String(v).replace(/"/g,'""')+'"' : String(v));
  return cols.join(",")+"\n"+rs.map(r=>cols.map(c=>esc(r[c])).join(",")).join("\n");
}
function saveCSV(rs, name){
  dl(name, new Blob(["﻿"+toCSV(rs)], {type:"text/csv;charset=utf-8"}));
}
function saveXLSX(rs, name){
  if(typeof XLSX==="undefined"){
    alert("The spreadsheet library did not load (it comes from a CDN, so this "
        + "needs internet). CSV export works offline.");
    return;
  }
  const ws=XLSX.utils.json_to_sheet(rs);
  const ncol=Object.keys(rs[0]).length;
  ws["!autofilter"]={ref:XLSX.utils.encode_range(
      {s:{r:0,c:0}, e:{r:rs.length, c:ncol-1}})};
  ws["!freeze"]={xSplit:0, ySplit:1};
  ws["!cols"]=Object.keys(rs[0]).map(c=>({wch:Math.max(12, c.length+2)}));
  const wb=XLSX.utils.book_new();
  XLSX.utils.book_append_sheet(wb, ws, "districts");
  XLSX.writeFile(wb, name);
}
// SVG -> canvas -> PNG.  The clone gets explicit width/height and a solid
// background, because a serialised SVG has neither.
function savePNG(name, bg){
  const src=document.querySelector("svg.map");
  if(!src){alert("Map not ready yet.");return;}
  const c2=src.cloneNode(true);
  c2.setAttribute("xmlns","http://www.w3.org/2000/svg");
  c2.setAttribute("width", MW); c2.setAttribute("height", MH);
  const s=new XMLSerializer().serializeToString(c2);
  const url=URL.createObjectURL(new Blob([s],
      {type:"image/svg+xml;charset=utf-8"}));
  const img=new Image();
  img.onload=()=>{
    const sc=2, cv=document.createElement("canvas");
    cv.width=MW*sc; cv.height=MH*sc;
    const g=cv.getContext("2d");
    g.fillStyle=bg; g.fillRect(0,0,cv.width,cv.height);
    g.drawImage(img,0,0,cv.width,cv.height);
    cv.toBlob(b=>{ dl(name,b); URL.revokeObjectURL(url); });
  };
  img.onerror=()=>{alert("Could not render the map to PNG.");
                   URL.revokeObjectURL(url);};
  img.src=url;
}

function App(){
  const [layer,setLayer]=useState("fc14");
  const [theme,setTheme]=useState("dark");
  const [sel,setSel]=useState(null);
  const [q,setQ]=useState("");
  const [sort,setSort]=useState(["fc14",1]);
  const [tip,setTip]=useState(null);
  const [scope,setScope]=useState("district");
  const [mi,setMi]=useState((DATA.months||[]).length-1);
  const byId=useMemo(()=>{const m={};DATA.districts.forEach(d=>m[d.id]=d);return m;},[]);
  React.useEffect(()=>{document.documentElement.setAttribute("data-theme",theme);},[theme]);
  const mid=theme==="light"?"#e7edf5":"#243149";
  const nod=theme==="light"?"#eef1f6":"#182338";
  // absence of a crop is a different statement from absence of data, so it
  // gets its own neutral colour rather than a point on the stress ramp
  const NOTGROWN=theme==="light"?"#d9dee7":"#3a3f4b";
  // declared before any use: see the note on Babel's const lowering
  const isCrop = layer.startsWith("crop:");
  const cropName = isCrop ? layer.slice(5) : null;
  const isDaily = !isCrop && !!DLAY[layer];
  const isEcmwf = layer.indexOf("ecmwf_")===0;
  const L0 = isCrop
    ? {grp:"Crops", label:cropName+" stress", unit:"", kind:"cropstress",
       dom:[0,1]}
    : (isDaily ? DLAY[layer] : LAY[layer]);
  const S=DATA.skill;
  // is this a monthly layer (has a time series) or a latest-only forecast layer?
  const isM = !isCrop && DATA.monthly && DATA.monthly[layer];
  const ymSel = DATA.months ? DATA.months[mi] : null;
  // A layer is withheld for any month its sources do not fully cover, so SPEI
  // and the degree-days stop a month earlier than rainfall.  Fall back to that
  // layer's own latest published month instead of painting an empty map, and
  // say so underneath.
  const ymHas = isM && ymSel && DATA.monthly[layer][ymSel]
    && Object.keys(DATA.monthly[layer][ymSel]).length > 0;
  const ymLast = ((DATA.layer_last || {})[layer]) || null;
  const ym = (isM && !ymHas && ymLast) ? ymLast : ymSel;
  const ymFellBack = isM && ym !== ymSel;
  // ---- daily layers ----------------------------------------------------
  // Stored as positional arrays against one shared id list rather than
  // {id: value} maps: 120 days x 791 districts x 8 layers is ~760k numbers and
  // the map form would roughly triple the payload for no extra information.
  const dDates = (DAILY && DAILY.dates) || [];
  const [di,setDi]=useState(Math.max(0,dDates.length-1));
  const dLast = isDaily ? ((DAILY.edge||{})[layer]||null) : null;
  // a layer whose source lags (ERA5 is about a week behind IMD) is clamped to
  // its own last day instead of drawing an empty map
  const dIdx = useMemo(()=>{
    if(!isDaily) return di;
    const cap = dLast ? dDates.indexOf(dLast) : -1;
    return (cap >= 0 && di > cap) ? cap : di; },[isDaily,di,dLast]);
  const dayShown = isDaily ? (dDates[dIdx]||null) : null;
  const dayFellBack = isDaily && dIdx !== di;
  // model selector: which formulation drives the forecast layers
  const [mdl,setMdl]=useState("");            // "" = the default (MSE winner)
  const isFc = layer==="fc7" || layer==="fc14";
  const mList = MODELS[layer] || [];
  const mSel = mdl ? mList.find(m=>m.key===mdl) : null;
  const vals = useMemo(()=>{
    if(isCrop) return (CROPS[cropName]||{}).v || {};
    if(isFc && mSel) return mSel.v || {};
    if(isDaily){
      const rows=(DAILY.layers||{})[layer]; if(!rows) return {};
      const r=rows[dIdx]||[]; const m={};
      DAILY.ids.forEach((id,k)=>{const v=r[k]; if(v!=null)m[id]=v;});
      return m; }
    if(isM) return DATA.monthly[layer][ym]||{};
    const m={}; DATA.districts.forEach(d=>m[d.id]=d[layer]); return m;
  },[layer,mi,dIdx,isDaily,mdl]);
  // Declared here, not beside L0: the domain can only be inferred once the
  // values for the selected month or day are in hand. Everything downstream
  // — colour, legend, the "most affected" bars — reads L, so they all agree.
  const L = useMemo(()=>adaptDomain(L0, vals),[L0, vals]);
  // scope: per-district values, or each district painted with its STATE mean
  // State and all-India numbers come from DATA.agg, computed in Python.
  // They are NOT averaged here on purpose: `pct_departure` is a ratio, and the
  // mean of per-district percentages is a different quantity from India's
  // rainfall against normal (-1.3% vs +4.6% for July 2026).  Everything else is
  // area-weighted, so a 200 km2 district no longer counts as much as a
  // 30,000 km2 one.  See build_aggregates() in step 05.
  const agg=useMemo(()=>{
    const A=DATA.agg||{};
    if(isCrop) return null;                       // crop layers are not aggregated
    if(isDaily){
      const a=((DAILY.agg||{})[layer])||null; if(!a) return null;
      return {national:a.national[dIdx], state:a.state[dIdx]}; }
    return isM ? ((A.monthly||{})[layer]||{})[ym] : (A.current||{})[layer];
  },[layer,ym,isM,isCrop,isDaily,dIdx]);
  const shown = useMemo(()=>{ if(scope==="district") return vals;
    const sm=(agg&&agg.state)||null;
    if(!sm) return vals;                          // no state aggregate: leave as-is
    const o={}; DATA.districts.forEach(d=>{const v=sm[d.s]; if(v!=null)o[d.id]=v;});
    return o; },[vals,scope,agg]);
  const natMean=agg?agg.national:null;
  // Which end of the scale is "worst" depends on the layer: for rainfall,
  // SPEI, moisture and the forecast a LOW value is the bad one, but for heat
  // stress, a hot temperature anomaly and crop stress it is the HIGH end.
  // Ranking every layer ascending listed the healthiest districts as "most
  // affected" on the crop layers.
  const hiBad = isCrop || L.kind==="hot" || L.kind==="tmp";
  const ranks=(()=>{let a=[...DATA.districts].map(d=>({d,v:shown[d.id]})).filter(o=>o.v!=null)
      .sort((x,y)=>hiBad ? y.v-x.v : x.v-y.v);
    if(scope==="india"){const seen={},o=[];a.forEach(r=>{if(!seen[r.d.s]){seen[r.d.s]=1;o.push(r);}});a=o;}
    return a.slice(0,12);})();
  const rows=[...DATA.districts].filter(d=>!q||((d.d+d.s).toLowerCase().includes(q.toLowerCase())))
     .sort((a,b)=>{const[k,dir]=sort;const av=a[k],bv=b[k];
        if(typeof av==="string")return dir*av.localeCompare(bv);
        return dir*((av==null?9e9:av)-(bv==null?9e9:bv));});
  const e=DATA.enso;
  // sel==null is the no-selection state, NOT falsiness: NICOBARS has district_id 0.
  const D=sel==null?null:(byId[sel]||null);
  // A briefing row's value, resolved at the latest month that feature actually
  // has. Returns {v, asOf}: asOf is null when the value is for the month the
  // slider is on, and the month string when it fell back to an earlier one.
  // The district record (D[k]) holds only the newest month, which is why a
  // lagging feature showed nothing at all before this.
  function briefVal(k){
    if(!D) return {v:null, asOf:null};
    const mm = (DATA.monthly||{})[k];
    if(!mm) return {v:D[k]==null?null:D[k], asOf:null};   // static, no series
    const cur = ymSel && mm[ymSel] ? mm[ymSel][D.id] : undefined;
    if(cur != null) return {v:cur, asOf:null};
    const last = (DATA.layer_last||{})[k];
    if(last && mm[last] && mm[last][D.id] != null)
      return {v:mm[last][D.id], asOf:last};
    return {v:null, asOf:null};
  }
  // The layer currently painted on the map, for this district — whatever kind
  // it is (monthly, daily, crop, ECMWF lead).
  const selRow = (()=>{
    if(!D) return null;
    // Suppress it when the selected layer already has a fixed row below —
    // otherwise picking "Rainfall vs normal" printed the same district value
    // twice, one line apart.
    if(BRIEF_ROWS.some(r=>r[0]===layer)) return null;
    const v = shown[D.id];
    const asOf = isDaily ? dayShown : (isM ? ym : null);
    return {lb:L.label, u:L.unit, v:(v==null?null:v), asOf:asOf};
  })();
  const rampCss=`linear-gradient(90deg,${color(L.dom.length?L.dom[0]:-L.dom,L,mid,nod)},${color(L.dom.length?(L.dom[0]+L.dom[1])/2:0,L,mid,nod)},${color(L.dom.length?L.dom[1]:L.dom,L,mid,nod)})`;
  const legTxt=L.dom.length?[L.dom[0],L.dom[1]]:["−"+L.dom,"+"+L.dom];

  return <div className="wrap">
   <div className="top">
    <div className="brand"><div className="logo">🌧️</div><div>
      <h1>MonsoonCast</h1><div className="tag">India Rainfall Intelligence · IMD gauge · {DATA.n_units} LGD districts</div></div></div>
    <div className="issue">
      <span>Issued <b>{DATA.issued}</b></span>
      <span>Rain to <b>{DATA.issued}</b></span>
      <span><b>{DATA.n_units}</b> districts</span></div>
    <button className="tbtn" onClick={()=>setTheme(theme==="light"?"dark":"light")}>
      {theme==="light"?"🌙 Dark":"☀ Light"}</button>
   </div>

   <div className="kstrip">
    <div className="kpi"><div className="l">All-India 7-day</div><div className="v" style={{color:color(DATA.mean_fc7,LAY.fc7,mid,nod)}}>{sg(DATA.mean_fc7)}%</div><div className="n">area-weighted all-India</div></div>
    <div className="kpi"><div className="l">All-India 14-day</div><div className="v" style={{color:color(DATA.mean_fc14,LAY.fc14,mid,nod)}}>{sg(DATA.mean_fc14)}%</div><div className="n">area-weighted all-India</div></div>
    <div className="kpi"><div className="l">7-day skill</div><div className="v">{S.dep_7.mse_skill>0?"+":""}{S.dep_7.mse_skill}</div><div className="n">vs climatology · r {S.dep_7.corr}</div></div>
    <div className="kpi"><div className="l">14-day skill</div><div className="v">{S.dep_14.mse_skill>0?"+":""}{S.dep_14.mse_skill}</div><div className="n">vs climatology · r {S.dep_14.corr}</div></div>
    {/* Show the STATUS, not the bare CPC phase: at ONI +0.98 in the fourth
        month above the threshold, "Neutral" is right by the five-season rule
        and wrong to anyone reading a monsoon deficit off the same page. The
        detail line carries the distinction instead of hiding it. */}
    <div className="kpi" title={e.status_detail||""}><div className="l">ENSO state</div>
      <div className="v" style={{fontSize:19,color:(e.status||"").indexOf("El Nino")>=0?"var(--hot)":((e.status||"").indexOf("La Nina")>=0?"var(--cool)":"inherit")}}>{e.status||e.phase}</div>
      <div className="n">ONI {e.oni} · IOD {e.iod}{e.status&&e.status!==e.phase?" · CPC phase "+e.phase:""}</div></div>
   </div>

   <div className="toolbar">
    {GROUPS.map(([g,ls],i)=><React.Fragment key={g}>
      {i>0&&<div className="sep"/>}<span className="lab">{g}</span>
      {ls.map(l=><button key={l[0]} className={"lbtn"+(layer===l[0]?" on":"")} onClick={()=>setLayer(l[0])}>{l[1]}</button>)}
    </React.Fragment>)}
    {/* Model selector. Only meaningful on a forecast layer, and every option
        carries the skill IT earned -- a selector that let a user pick a
        pretty-looking map without seeing that it scores worse than
        climatology would be worse than offering no choice at all. */}
    {isFc&&mList.length>0&&<React.Fragment>
      <div className="sep"/><span className="lab" title="Each formulation was trained and scored identically on held-out 2020-2026 data. Which one is 'best' depends on the metric.">Model</span>
      <select value={mdl} onChange={ev=>setMdl(ev.target.value)}
        style={{font:"inherit",fontSize:12.5,color:"var(--ink)",
                background:"var(--panel-2)",border:"1px solid var(--edge)",
                borderRadius:8,padding:"6px 8px",maxWidth:290}}>
        <option value="">Default — best MSE skill</option>
        {mList.map(m=><option key={m.key} value={m.key}>
          {m.key + "  ·  skill " + (m.mse_skill>=0?"+":"") + (m.mse_skill!=null?m.mse_skill.toFixed(3):"?")
           + "  ·  cat " + (m.imd_cat_acc!=null?(100*m.imd_cat_acc).toFixed(0)+"%":"?")
           + (m.beats_climatology?"":"  ·  below climatology")}
        </option>)}
      </select>
    </React.Fragment>}
    {DAILY&&<React.Fragment>
      <div className="sep"/><span className="lab" title="Rolling windows ending on the chosen day, day-matched against the normal for those same calendar days">Daily</span>
      {DGROUPS.map(([g,ls])=>ls.map(l=>
        <button key={l[0]} className={"lbtn"+(layer===l[0]?" on":"")}
          onClick={()=>setLayer(l[0])}>{l[1]}</button>))}
    </React.Fragment>}
    {/* ECMWF: one toggle plus a lead selector, mirroring the crop control.
        The ensemble is a genuine forecast — real lead times, 50 members, a
        real distribution — so it sits beside the statistical models as a
        choice. It is NOT verified here: open data carries no reforecast
        archive, so no skill score exists for it and none is shown. */}
    {ECMWF_LEADS.length>0&&<React.Fragment>
      <div className="sep"/><span className="lab"
        title="ECMWF operational ensemble, 0.25 deg, 50 perturbed members. Accumulated rainfall from the run time to the selected lead. Unverified: open data has no reforecast archive, so no skill score can be computed for it.">
        ECMWF forecast</span>
      <button className={"lbtn"+(isEcmwf?" on":"")}
        onClick={()=>setLayer(isEcmwf?"fc14":"ecmwf_"+ECMWF_LEADS[0]+"h_mean")}>
        {isEcmwf?"● ECMWF":"○ ECMWF"}</button>
      <select value={isEcmwf?layer:""} disabled={!isEcmwf}
        onChange={ev=>setLayer(ev.target.value)}
        style={{font:"inherit",fontSize:12.5,color:"var(--ink)",
          background:"var(--panel-2)",border:"1px solid var(--edge)",
          borderRadius:8,padding:"7px 10px",opacity:isEcmwf?1:.45}}>
        {ECMWF_LEADS.map(h=><option key={h} value={"ecmwf_"+h+"h_mean"}>
          {"+" + (h/24) + " day" + (h>24?"s":"")}
        </option>)}
      </select>
    </React.Fragment>}
    {CROPKEYS.length>0&&<React.Fragment>
      <div className="sep"/><span className="lab">Crop stress</span>
      <button className={"lbtn"+(isCrop?" on":"")}
        onClick={()=>setLayer(isCrop?"fc14":"crop:"+CROPKEYS[0])}>
        {isCrop?"● Crop view":"○ Crop view"}</button>
      <select value={isCrop?cropName:""} disabled={!isCrop}
        onChange={ev=>setLayer("crop:"+ev.target.value)}
        style={{font:"inherit",fontSize:12.5,color:"var(--ink)",
          background:"var(--panel-2)",border:"1px solid var(--edge)",
          borderRadius:8,padding:"7px 10px",opacity:isCrop?1:.45}}>
        {CROPKEYS.map(c=><option key={c} value={c}>
          {c + "  —  " + CROPS[c].season + " " + CROPS[c].season_year}
        </option>)}
      </select>
    </React.Fragment>}
   </div>

   <div className="toolbar" style={{marginTop:-8}}>
    <span className="lab">Map view</span>
    <button className={"lbtn"+(scope==="district"?" on":"")} onClick={()=>setScope("district")}>District-wise</button>
    <button className={"lbtn"+(scope==="india"?" on":"")} onClick={()=>setScope("india")}>All-India (by state)</button>
    {isDaily?<React.Fragment><div className="sep"/><span className="lab">Day</span>
      <button className="lbtn" onClick={()=>setDi(Math.max(0,di-1))}>‹</button>
      <input type="range" min="0" max={Math.max(0,dDates.length-1)} value={di}
        onChange={ev=>setDi(+ev.target.value)} style={{flex:1,minWidth:150,accentColor:"var(--accent)"}}/>
      <button className="lbtn" onClick={()=>setDi(Math.min(dDates.length-1,di+1))}>›</button>
      <button className="lbtn" onClick={()=>setDi(dDates.length-1)}>Latest</button>
      <span style={{color:"var(--accent)",fontWeight:700,minWidth:88,textAlign:"center"}}>{dayShown}</span>
      {dayFellBack&&<span className="lab" title="This layer's source has not published up to the selected day; the map shows its most recent available day."
        style={{textTransform:"none",color:"var(--warn)"}}>
        source ends {dLast} — showing that day</span>}</React.Fragment>
     :isM?<React.Fragment><div className="sep"/><span className="lab">Month</span>
      <button className="lbtn" onClick={()=>setMi(Math.max(0,mi-1))}>‹</button>
      <input type="range" min="0" max={DATA.months.length-1} value={mi}
        onChange={ev=>setMi(+ev.target.value)} style={{flex:1,minWidth:130,accentColor:"var(--accent)"}}/>
      <button className="lbtn" onClick={()=>setMi(Math.min(DATA.months.length-1,mi+1))}>›</button>
      <button className="lbtn" onClick={()=>setMi(DATA.months.length-1)}>Latest</button>
      <span style={{color:"var(--accent)",fontWeight:700,minWidth:66,textAlign:"center"}}>{ym}</span>
      {ymFellBack&&<span className="lab" title={"This layer's source has not reached "+ymSel+" yet; the map shows its most recent complete month."}
        style={{textTransform:"none",color:"var(--warn)"}}>
        source ends {ym} — showing that month</span>}
      {/* The running month is computed month-to-date against the same window
          in every prior year, so it is a valid number rather than a partial
          one — but it answers "so far this month", and says so. */}
      {!ymFellBack&&MTD&&ym===MTD.ym&&<span className="lab"
        title={"Accumulating indices for "+MTD.ym+" are computed over the "+MTD.days_observed+" days observed so far and compared against the same "+MTD.days_observed+"-day window in every prior year, so the value is directly comparable — it answers 'so far this month'."}
        style={{textTransform:"none",color:"var(--accent)"}}>
        month to date · {MTD.days_observed} of {MTD.days_in_month} days</span>}</React.Fragment>
     :<span className="lab" style={{textTransform:"none",color:"var(--ink-3)"}}>
        {isFc&&mSel
          ? <span style={{color:mSel.beats_climatology?"var(--ink-2)":"var(--hot)"}}>
              {mSel.key} — skill {(mSel.mse_skill>=0?"+":"")+mSel.mse_skill.toFixed(4)},
              correlation {mSel.corr!=null?mSel.corr.toFixed(3):"–"},
              IMD-category accuracy {mSel.imd_cat_acc!=null?(100*mSel.imd_cat_acc).toFixed(1)+"%":"–"}
              {mSel.beats_climatology?"":" · this model scores WORSE than climatology"}
            </span>
          : "Forecast layers use the latest issue date ("+DATA.issued+")."}
      </span>}
    {/* Name the statistic honestly: rainfall vs normal is a ratio of totals,
        not a mean of district percentages, and the two differ a lot. */}
    {natMean!=null&&(()=>{const meth=(isDaily?((DAILY.method||{})[layer])
        :((DATA.agg||{}).method||{})[layer])||"";
      const isRatio=meth.indexOf("ratio")>=0;
      return <span title={"All-India figure computed as an "+meth+" over the 791 districts"}
        style={{marginLeft:"auto",fontSize:12,color:"var(--ink-2)"}}>
        {isRatio?"All-India (total vs normal)":"All-India (area-weighted)"}&nbsp;
        <b style={{color:"var(--ink)"}}>{sg(natMean)}{L.unit?" "+L.unit:""}</b></span>;})()}
   </div>

   <div className="toolbar" style={{marginTop:-8}}>
    <span className="lab">Download</span>
    <button className="lbtn" title="All districts, every layer, as CSV"
      onClick={()=>saveCSV(tableRows(vals,isM?ym:null),
        `monsooncast_${isM?ym:DATA.issued}_all_features.csv`)}>⤓ CSV</button>
    <button className="lbtn" title="Same table as an Excel workbook with filters"
      onClick={()=>saveXLSX(tableRows(vals,isM?ym:null),
        `monsooncast_${isM?ym:DATA.issued}_all_features.xlsx`)}>⤓ Excel</button>
    <button className="lbtn" title="The map exactly as displayed, as a PNG"
      onClick={()=>savePNG(`monsooncast_${layer}_${isM?ym:DATA.issued}.png`,
        theme==="light"?"#ffffff":"#111a2e")}>⤓ Map PNG</button>
    <span className="lab" style={{textTransform:"none",marginLeft:6}}>
      791 districts × {Object.keys(LAY).length} layers{isM?" · "+ym:""}</span>
    <span style={{marginLeft:"auto",fontSize:11,color:"var(--ink-3)"}}>
      Full 1971–2026 history: see v5/masters_lgd/</span>
   </div>

   <div className="main">
    <div className="card"><h2><span className="dot"/>{L.label}{L.unit?` (${L.unit})`:""}{scope==="india"?" — all-India (by state)":""}</h2>
     {/* AS-OF, PER LAYER, ALWAYS VISIBLE.
         Every layer here ends on a different day because every source does:
         the IMD gauge analysis runs 1-2 days behind, ERA5-Land about a week,
         MODIS composites up to 16 days, CFSv2 a day, and the ECMWF ensemble
         is a forward forecast from a fixed run time. A single "issued" date
         in the header invited the reader to assume all of it was current as
         of that date, which was never true for more than one layer at a
         time. */}
     <div className="cap" style={{color:"var(--accent)",textTransform:"none"}}>
       {(function(){
         const k=layer;
         if(k.indexOf("ecmwf_")===0){
           const h=parseInt(k.match(/\d+/)[0],10);
           const m=DATA.ecmwf_meta||{};
           return "ECMWF ENS forecast · accumulated to +"+(h/24)+" d from the "
             +(m.run||"?")+" run · "+(m.members||"?")+" members · not verified here";
         }
         if(k.indexOf("cfs_")===0){
           const e=(DAILY&&DAILY.edge&&DAILY.edge[k])||null;
           return "CFSv2 coupled-model analysis · data to "+(e||"?")
             +" · model output at valid time, not a lead-time forecast";
         }
         if(isDaily) return "Derived from data to "+((DAILY.edge||{})[k]||"?")
             +" · rolling window ending on the selected day";
         if(k==="pct_departure_signature")
           return "Static per-district climatology · warm-phase mean minus that district's own all-year mean";
         // The statistical forecasts are not "derived from data to X" — they
         // are ISSUED on a date and valid forward from it. layer_last has no
         // entry for them, which is correct, so they need their own sentence
         // rather than falling through to an unanswerable one.
         if(isFc){
           const s=(DATA.skill||{})[k==="fc7"?"dep_7":"dep_14"]||{};
           return "Issued "+DATA.issued+" · valid over the next "
             +(k==="fc7"?"7":"14")+" days · antecedent rainfall to "
             +DATA.issued+", monthly predictors from the last complete month"
             +(s.mse_skill!=null?" · skill "+(s.mse_skill>=0?"+":"")
                 +s.mse_skill.toFixed(4)+" vs climatology":"");
         }
         if(isCrop) return "Crop stress · season to date, from the feature panel";
         // THE MONTH BEING SHOWN, not the layer's newest month. This read
         // "Derived from data to 2026-08" while the map was on April, which
         // says the April map was built from August data. `ymLast` is the
         // right answer only when the selected month has no data and the map
         // has fallen back to it — which is exactly what ymFellBack means.
         return "Showing "+(ym||"?")
           +(ymFellBack
             ? " · the selected month ("+(ymSel||"?")+") has no data for this "
               +"layer, so its latest published month is drawn"
             : "")
           +" · layer published through "+(ymLast||"?");
       })()}</div>
     <div className="cap">{scope==="india"?"Each district painted with its state mean":`District choropleth · ${DATA.n_units} LGD units incl. J&K, Ladakh, islands`} · {isM?ym:"issued "+DATA.issued}. Hover for detail, click for a briefing.</div>
     <div className="mapwrap">
      <svg viewBox={`0 0 ${MW} ${MH}`} className="map"
        onMouseLeave={()=>setTip(null)}>
       {GEO.features.map(f=>{const d=byId[f.properties.id];const v=shown[f.properties.id];
         // On a crop layer a missing value means the crop is NOT GROWN there --
         // a different statement from "no observation" -- so it is drawn in a
         // neutral colour off the stress ramp rather than as a zero-stress
         // district, which would read as "healthy".
         const ng = isCrop && (v==null || !isFinite(v));
         return <path key={f.properties.id} d={pathOf(f.geometry)}
           fill={ng ? NOTGROWN : color(v,L,mid,nod)}
           className={sel===f.properties.id?"selp":""}
           onMouseMove={ev=>d&&setTip({x:ev.clientX,y:ev.clientY,d})}
           onClick={()=>setSel(f.properties.id)}/>;})}
      </svg>
      <div className="legend"><span>{isCrop?"no stress":legTxt[0]}</span>
        <div className="ramp" style={{background:rampCss}}/>
        <span>{isCrop?"severe":legTxt[1]}</span>
        {/* +/-60 is not an arbitrary cut: it is IMD's own boundary for Large
            Excess and Large Deficient. Saying so turns a saturated map from
            "the scale is too small" into "these districts are in IMD's
            extreme categories", which is what it actually means. */}
        {layer==="pct_departure"&&<span className="lab" style={{marginLeft:8}}
          title="IMD's operational rainfall categories: Large Excess at or above +60%, Excess +20 to +59, Normal -19 to +19, Deficient -20 to -59, Large Deficient -60 to -99. Districts at the ends of this scale are in the extreme categories; values beyond +/-60% are not clipped in the data, only in the colour.">
          ends are IMD's Large Deficient / Large Excess</span>}
        {L.domAuto&&<span className="lab" style={{marginLeft:8}}
          title="This layer's range changes with the season — degree-days above 34 °C are near zero through the monsoon and large before it. The scale is fitted to the 95th percentile of what is on screen, so the map stays readable in both. It is NOT comparable month to month; read the numbers, not the colour, when scrubbing.">
          scale fitted to this {isDaily?"day":"month"}</span>}
        {isCrop&&<React.Fragment>
          <span style={{width:14,height:10,background:NOTGROWN,
            border:"1px solid var(--edge)",borderRadius:2,marginLeft:10,
            display:"inline-block"}}/>
          <span>not grown here</span></React.Fragment>}</div>
     </div>
    </div>

    <div className="card"><h2><span className="dot"/>District briefing</h2>
     <div className="cap">Search a district, or click one on the map.</div>
     <input className="search" placeholder="Search a district (e.g. Pune, Guntur, Anantnag)…"
       onChange={ev=>{const t=ev.target.value.toLowerCase();const m=DATA.districts.find(d=>d.d.toLowerCase().includes(t)&&t.length>2);if(m)setSel(m.id);}}/>
     {!D?<div className="dsub">No district selected. The map is coloured by <b>{L.label}</b>.</div>:
      <div>
       <div className="dhead"><span className="nm">{D.d}{D.jk?<span className="star"> ★</span>:""}</span><span className="stt">{D.s}</span></div>
       <div className="dsub">{D.jk?"Jammu & Kashmir / Ladakh — new coverage in this system":"LGD district"}</div>
       <div className="hz">
         <div className="b"><div className="h">7-day outlook</div><div className="val" style={{color:color(D.fc7,LAY.fc7,mid,nod)}}>{sg(D.fc7)}%</div></div>
         <div className="b"><div className="h">14-day outlook</div><div className="val" style={{color:color(D.fc14,LAY.fc14,mid,nod)}}>{sg(D.fc14)}%</div></div>
       </div>
       {/* The selected layer, whatever it is, reported first. The briefing used
           to describe a fixed set of monthly features while the map showed
           something else entirely — pick an ECMWF lead, a daily layer or a
           crop and the panel said nothing about it. */}
       {selRow && <div className="prow sel" key="__sel">
         <span className="k">{selRow.lb}<span className="lab" style={{marginLeft:6}}>shown on map</span></span>
         <span className="v" style={{color:color(selRow.v,L,mid,nod)}}>
           {selRow.v==null?"no data":sg(selRow.v)+(selRow.u?" "+selRow.u:"")}
           {selRow.asOf&&<span className="lab" style={{marginLeft:6}}>{selRow.asOf}</span>}
         </span></div>}
       {/* EVERY ROW AT ITS OWN AS-OF DATE.
           These features do not share an edge: ERA5-Land lags the gauge
           analysis by about a week and MODIS by up to sixteen days, so in an
           incomplete month soil-moisture adequacy, root-zone moisture and the
           vegetation indices have no value for the month the panel was showing
           — and the row simply read "–", which looks like a broken product
           rather than a source that has not published yet. Each row now falls
           back to that feature's own latest published month and says which
           month that is. */}
       {BRIEF_ROWS.map(([k,lb,u])=>{const r=briefVal(k);
         return <div className="prow" key={k}><span className="k">{lb}</span>
          <span className="v">
            {r.v==null?"–":sg(r.v)+(u?" "+u:"")}
            {/* A departure is a ratio; without the millimetres it came from,
                "+570%" cannot be told apart from a cloudburst. April 2026 has
                a district at +570% because 36.4 mm fell where 5.4 mm is
                normal. Both numbers, always. */}
            {r.v!=null&&k==="pct_departure"&&(()=>{
              const c=((DATA.context||{})[r.asOf||ymSel]||{})[D.id];
              if(!c||c[0]==null||c[1]==null) return null;
              return <span className="lab" style={{marginLeft:6}}
                title="Actual rainfall against the 1971-2020 day-matched normal for the same period. A percentage on a small normal moves a long way on very little rain — read the millimetres.">
                {sg1(c[0])} vs {sg1(c[1])} mm normal</span>;
            })()}
            {r.v!=null&&k==="sdd_anom"&&D.sdd!=null&&D.sdd_norm!=null&&
              <span className="lab" style={{marginLeft:6}}
                title="Degree-days above 34 °C this period, against the same window averaged over 1971–2020. Shown as a difference, not a percentage: the normal is under 1 °C-d through the monsoon and exactly 0 for half the record, so a ratio would divide by roughly nothing.">
                {sg1(D.sdd)} vs {sg1(D.sdd_norm)} normal</span>}
            {r.asOf&&<span className="lab" style={{marginLeft:6}}
              title={"This feature's source has not published for "+(ymSel||"the selected month")+"; showing its latest available month."}>
              as of {r.asOf}</span>}
          </span></div>;})}
      </div>}
    </div>
   </div>

   <div className="grid2">
    <div className="card"><h2><span className="dot"/>Forecast skill by outlook length</h2>
     <div className="cap">Mean-squared-error skill against assuming normal rainfall (higher is better), on held-out 2020–26.</div>
     {[["7-day",S.dep_7],["14-day",S.dep_14]].map(([nm,s])=>
      <div className="rk" key={nm}><span className="cn">{nm}</span>
       <div className="track"><div className="fill" style={{width:(s.mse_skill*180)+"%",background:"var(--good)"}}/></div>
       <span className="cv">+{s.mse_skill}</span></div>)}
     <table className="tv" style={{marginTop:10}}><thead><tr>
       <th>Outlook</th><th className="num">RMSE</th><th className="num">Corr</th>
       <th className="num">IMD category</th></tr></thead><tbody>
       {[["7-day",S.dep_7],["14-day",S.dep_14]].map(([nm,s])=>
        <tr key={nm}><td>{nm}</td><td className="num">{s.rmse}</td>
        <td className="num">{s.corr}</td>
        <td className="num">{s.imd_cat_acc?(100*s.imd_cat_acc).toFixed(1)+"%":"–"}</td></tr>)}
     </tbody></table>
     <div className="note">Winning model: <b>{S.dep_14.model||"gradient-boosted blend"}</b>, chosen in a
      seven-way bake-off (HistGB / LightGBM / XGBoost × squared-error, Huber, Tweedie, log-ratio) on 2.6M
      samples of IMD gauge rainfall; spatial-neighbour features and a stacked ensemble were both tested
      and rejected as no better. Normals use <b>IMD’s own 1971–2020 fifty-year window</b>, so departures
      are on the same baseline IMD publishes against. “IMD category” is agreement with IMD’s operational
      classes (Large Deficient → Large Excess). Monsoon-season skill is higher than the all-season figure
      ({S.dep_14.jjas_skill?"JJAS 14-day "+(S.dep_14.jjas_skill>0?"+":"")+S.dep_14.jjas_skill:"—"}).
      Predictions are deliberately conservative — they carry about two-thirds of the observed spread,
      which is what minimises expected error.</div>
    </div>
    {isCrop ? (()=>{const K=CROPS[cropName]||{};const cyc=K.cycle||[];
      const done=cyc.filter(s=>s.done);
      const worst=done.length?done.reduce((a,b)=>(b.csi||0)>(a.csi||0)?b:a):null;
      return <div className="card"><h2><span className="dot"/>{cropName} — season risk</h2>
      <div className="cap">{K.season} {K.season_year} · {K.stages_elapsed} of {K.stages_total} crop-cycle stages complete</div>
      <div className="hz" style={{gridTemplateColumns:"1fr 1fr 1fr"}}>
        <div className="b"><div className="h">Area at risk</div>
          <div className="val" style={{color:"var(--hot)"}}>{K.risk_mha!=null?K.risk_mha.toFixed(1):"–"}</div>
          <div className="h">M ha of {K.area_mha!=null?K.area_mha.toFixed(1):"–"} sown</div></div>
        <div className="b"><div className="h">Share at risk</div>
          <div className="val">{K.risk_share!=null?K.risk_share.toFixed(0)+"%":"–"}</div>
          <div className="h">{K.high} districts High/Severe</div></div>
        <div className="b"><div className="h">Growing districts</div>
          <div className="val">{K.districts}</div>
          <div className="h">of {DATA.n_units} nationally</div></div>
      </div>
      {K.sowing?(()=>{const W=K.sowing;
        // Sowing progress sits ABOVE the stress cycle deliberately: how much of
        // the crop is planted bounds what the stress index can cost.
        const up=W.pace_pp!=null&&W.pace_pp>0;
        const col=W.pace_pp==null?"var(--ink)":(up?"var(--good)":"var(--hot)");
        return <React.Fragment>
        <div className="cap" style={{marginTop:8}}>Sowing progress — UPAg, week ending {W.as_of}</div>
        <div className="hz" style={{gridTemplateColumns:"1fr 1fr 1fr"}}>
          <div className="b"><div className="h">Sown to date</div>
            <div className="val">{W.sown_mha!=null?W.sown_mha.toFixed(2):"–"}</div>
            {/* Show last year's figure only when the ratio itself survived the
                basis test. Printing "2.37 last year" next to a withheld YoY
                invites the reader to divide 6.79 by it and arrive at the very
                +187% the basis check exists to suppress. */}
            <div className="h">M ha{W.ly_mha!=null&&W.yoy_pct!=null?" · "+W.ly_mha.toFixed(2)+" last year":""}</div></div>
          <div className="b"><div className="h">vs last year</div>
            <div className="val" style={{color:col}}>
              {W.yoy_pct!=null?(W.yoy_pct>0?"+":"")+W.yoy_pct.toFixed(1)+"%":"–"}</div>
            <div className="h">{W.basis_pct!=null&&W.basis_pct<99
              ?"on "+W.basis_pct.toFixed(0)+"% of the crop":"same week"}</div></div>
          <div className="b"><div className="h">Planted</div>
            <div className="val">{W.coverage_pct!=null?W.coverage_pct.toFixed(0)+"%":"–"}</div>
            <div className="h">of last season's area</div></div>
        </div>
        <div className="note" style={{marginTop:2}}>
          {W.pace_pp!=null
            ? <span><b>{W.status}</b> — {Math.abs(W.pace_pp).toFixed(1)} points {up?"ahead of":"behind"} where
              the crop stood in this same week last season.</span>
            : <span>No comparable prior season in the UPAg archive for this crop, so pace is not shown.</span>}
          {" "}Areas are state-level releases; the comparison uses only states reporting in both years.
        </div></React.Fragment>;})():null}
      <div className="cap" style={{marginTop:4}}>Crop cycle — where the stress is falling</div>
      {cyc.map(s=><div className="rk" key={s.stage}>
        <span className="cn">{s.stage}{s.months?" (m"+s.months+")":""}</span>
        <div className="track"><div className="fill" style={{
          width:(s.done?Math.min(100,(s.csi||0)*100):0)+"%",
          background:s.done?color(s.csi,L,mid,nod):"transparent"}}/></div>
        <span className="cv">{s.done?(s.csi||0).toFixed(2):"ahead"}</span></div>)}
      <div className="cap" style={{marginTop:10}}>Most exposed producing states</div>
      {(K.top_states||[]).map(x=><div className="rk" key={x.s}>
        <span className="cn">{x.s?x.s.replace(/\w/g,c=>c.toUpperCase()):""}</span>
        <div className="track"><div className="fill" style={{width:Math.min(100,(x.csi||0)*100)+"%",
          background:color(x.csi,L,mid,nod)}}/></div>
        <span className="cv">{(x.csi||0).toFixed(2)}</span></div>)}
      <div className="note">{worst?<span>Stress is concentrated in the <b>{worst.stage}</b> stage
        (yield-response factor Ky {worst.ky}), where a shortfall costs the most yield.</span>:null}
        {" "}{K.season==="Kharif"
          ? "Kharif is monsoon-fed: the water term is rainfall and soil moisture against the crop's requirement, partly offset where the district is irrigated."
          : "Rabi is grown on irrigation: the water term is irrigation availability plus the soil moisture left by the monsoon, and terminal heat during grain fill carries the greater weight."}
      </div></div>;})() :
    <div className="card"><h2><span className="dot"/>Most affected — {L.label}</h2>
     <div className="cap">{hiBad?"Highest":"Lowest"} twelve {scope==="india"?"states":"districts"} on the selected layer{isM?" ("+ym+")":""}.</div>
     {ranks.map(({d,v})=>{const w=Math.min(100,Math.abs(v)/(L.dom.length?L.dom[1]:L.dom)*100);
       return <div className="rk" key={d.id} onClick={()=>setSel(d.id)} style={{cursor:"pointer"}}>
        <span className="cn">{scope==="india"?d.s:d.d}</span>
        <div className="track"><div className="fill" style={{width:w+"%",background:color(v,L,mid,nod)}}/></div>
        <span className="cv">{sg(v)}</span></div>;})}
    </div>}
   </div>

   <div className="card"><h2><span className="dot"/>All {DATA.n_units} districts</h2>
    <div className="cap">Click a row to open its briefing · sort by any column · ★ = J&amp;K / Ladakh.</div>
    <input className="search" style={{maxWidth:300}} placeholder="Filter districts or states…" value={q} onChange={ev=>setQ(ev.target.value)}/>
    <div className="scrolly"><table className="tv"><thead><tr>
      {[["d","District",0],["s","State",0],["fc7","7-day",1],["fc14","14-day",1],
        ["pct_departure","Rain%",1],["mai","MAI",1],["sdd_anom","SDD",1]].map(([k,lb,n])=>
        <th key={k} className={n?"num":""} onClick={()=>setSort([k,sort[0]===k?-sort[1]:1])}>{lb}{sort[0]===k?(sort[1]>0?" ▲":" ▼"):""}</th>)}
     </tr></thead><tbody>
      {rows.slice(0,400).map(d=><tr key={d.id} onClick={()=>setSel(d.id)}>
        <td>{d.d}{d.jk?<span className="star"> ★</span>:""}</td><td>{d.s}</td>
        <td className="num">{sg(d.fc7)}</td><td className="num">{sg(d.fc14)}</td>
        <td className="num">{sg(d.pct_departure)}</td><td className="num">{d.mai==null?"–":d.mai}</td>
        {/* one decimal, not Math.round: through the monsoon the whole column
            sits between -2 and +2 degree-days, and rounding to integers turned
            every real value into "0" or "-0" — the column looked broken. */}
        <td className="num">{d.sdd_anom==null?"–":sg(d.sdd_anom)}</td></tr>)}
     </tbody></table></div>
   </div>

   <div className="foot">MonsoonCast · IMD gauge rainfall (0.25°) &amp; temperature (1°) · ERA5-Land evapotranspiration &amp; soil moisture · NOAA/BoM ENSO indices ·
     791 LGD districts · models retrained on the IMD scale · built from the study's own data products.</div>

   {tip&&<div className="tip" style={{left:Math.min(tip.x+14,innerWidth-260),top:tip.y+14}}>
     <b>{tip.d.d}, {tip.d.s}</b>{L.label}{scope==="india"?" (state)":""}: {
       shown[tip.d.id]==null ? (isCrop?"not grown in this district":"no data")
       : sg(shown[tip.d.id])+(L.unit?" "+L.unit:"")}<br/>
     7d {sg(tip.d.fc7)}% · 14d {sg(tip.d.fc14)}%</div>}
  </div>;
}
export default App;
