/* BC Wildfire Mortgage Stress Test - frontend (M7).
 * Plain JS. Reads everything from the FastAPI backend (or its static snapshot); computes
 * nothing but histogram bins, map class breaks and the shares quoted in the Story headings. */
"use strict";

const API = (new URLSearchParams(location.search).get("api") || window.API_BASE || "").replace(/\/$/, "");
const BC = "BC";
const STORY_SCENARIO = "hot_house";
const N_CLASSES = 5;
const N_TOP = 3;           // divisions highlighted on the Story map
const TABS = ["story", "explore", "method"];

const state = { scenario: STORY_SCENARIO, metric: "el_uplift_pct", selected: null,
                sort: { key: "el_uplift", dir: -1 }, tornado: "el_uplift", tab: "story" };
const D = {};              // data from the API
const charts = {};         // Chart.js instances
const maps = {};           // Leaflet maps: { story, explore } -> { map, layer }
const lossCache = {};      // scenario -> runs
const rendered = new Set();  // tabs drawn at least once

/* ---------- helpers ---------- */
const $ = (id) => document.getElementById(id);
const cssVar = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
const esc = (s) => String(s).replace(/[&<>"']/g, (c) =>
  ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const isNil = (x) => x === null || x === undefined || Number.isNaN(x);

function money(x) {
  if (isNil(x)) return "—";
  const a = Math.abs(x), s = x < 0 ? "-" : "";
  if (a >= 1e9) return `${s}$${(a / 1e9).toFixed(2)}B`;
  if (a >= 1e6) return `${s}$${(a / 1e6).toFixed(2)}M`;
  if (a >= 1e3) return `${s}$${(a / 1e3).toFixed(1)}K`;
  return `${s}$${a.toFixed(0)}`;
}
function pct(x) {
  if (isNil(x)) return "—";
  return `${x.toFixed(Math.abs(x) < 10 ? 2 : 1)}%`;
}
const pctN = (x, d) => `${x.toFixed(d)}%`;
const millions = (x) => `$${(x / 1e6).toFixed(2)}M`;
const num = (d) => (x) => (isNil(x) ? "—" : x.toFixed(d));
const int = (x) => (isNil(x) ? "—" : Math.round(x).toLocaleString());
const sum = (xs) => xs.reduce((a, b) => a + b, 0);
// "Observed PD (BC arrears)" -> "observed PD (BC arrears)"; leaves acronyms alone.
const lcFirst = (s) => (/^[A-Z][a-z]/.test(s) ? s[0].toLowerCase() + s.slice(1) : s);
const listAnd = (xs) => (xs.length < 2 ? xs.join("") : `${xs.slice(0, -1).join(", ")} and ${xs[xs.length - 1]}`);

// Static snapshot (scripts/build_static.py): "/api/losses?scenario=x" -> "data/losses_x.json".
const staticPath = (path) => `data/${path.replace(/^\/api\//, "").replace("?scenario=", "_")}.json`;

async function getJSON(path) {
  const r = await fetch(window.API_STATIC ? staticPath(path) : API + path);
  if (!r.ok) throw new Error(`${path}: HTTP ${r.status}`);
  return r.json();
}

async function losses(scenario) {
  if (!lossCache[scenario]) {
    lossCache[scenario] = (await getJSON(`/api/losses?scenario=${encodeURIComponent(scenario)}`)).runs;
  }
  return lossCache[scenario];
}

const METRICS = {
  el_uplift_pct: { label: "Mean EL uplift vs historical (%)", fmt: pct },
  el_uplift: { label: "Mean EL uplift vs historical ($)", fmt: money },
  es_99_uplift: { label: "ES 99% uplift vs historical ($)", fmt: money, tail: true },
  hazard_score: { label: "Wildfire hazard score (historical, same in every scenario)", fmt: num(4), static: true },
};

const row = (scenario, cduid) => D.byKey.get(`${scenario}|${cduid}`);
const scenarioLabel = (id) => (D.scenarios.find((s) => s.id === id) || { label: id }).label;
const shortLabel = (id) => scenarioLabel(id).split(" (")[0];  // "Orderly (Net Zero 2050-like)" -> "Orderly"
const isRef = (id) => id === D.ref;
const divisionIds = () => Object.keys(D.props);

/* ---------- story facts (every number in the Story headings comes from here) ---------- */
function storyFacts(runs) {
  const hh = row(STORY_SCENARIO, BC), hist = row(D.ref, BC);
  const divs = divisionIds().map((id) => ({ id, name: D.props[id].cd_name, ...row(STORY_SCENARIO, id) }));
  const totalUplift = sum(divs.map((d) => d.el_uplift));
  const top = [...divs].sort((a, b) => b.el_uplift - a.el_uplift).slice(0, N_TOP);
  const gv = divs.find((d) => d.name === "Greater Vancouver");

  const nTail = Math.max(1, Math.ceil(0.01 * runs.length));
  const tail = [...runs].sort((a, b) => b.loss - a.loss).slice(0, nTail);
  const years = new Map();
  tail.forEach((r) => years.set(r.year, (years.get(r.year) || 0) + 1));
  const yearCounts = [...years.entries()].sort((a, b) => b[1] - a[1] || b[0] - a[0]);

  const sens = [...D.sensitivity].sort((a, b) => b.el_uplift_swing - a.el_uplift_swing);

  return {
    elPct: hh.el_uplift_pct, elUsd: hh.el_uplift,
    esPct: 100 * hh.es_99_uplift / hist.es_99, esUsd: hh.es_99_uplift,
    top, topShare: 100 * sum(top.map((d) => d.el_uplift)) / totalUplift,
    topEadShare: 100 * sum(top.map((d) => d.ead_total)) / hh.ead_total,
    gv: gv && { eadShare: 100 * gv.ead_total / hh.ead_total, upliftShare: 100 * gv.el_uplift / totalUplift },
    nTail, tailRuns: new Set(tail.map((r) => r.run)), yearCounts,
    drivers: sens.slice(0, 2), sensBase: sens[0].el_uplift_base, simUplift: hh.el_uplift,
  };
}

async function renderStory() {
  const runs = await losses(STORY_SCENARIO);
  const f = storyFacts(runs);
  D.story = f;

  // 1. The answer
  const avg = f.elPct < 2 ? "an average year barely changes" : `an average year costs ${pctN(f.elPct, 1)} more`;
  $("s1-head").textContent = `Under the ${shortLabel(STORY_SCENARIO)} scenario, ${avg}, but a severe fire year costs about ${pctN(f.esPct, 1)} more.`;
  $("s1-el-pct").textContent = `+${pctN(f.elPct, 1)}`;
  $("s1-el-usd").textContent = `+${millions(f.elUsd)} a year`;
  $("s1-es-pct").textContent = `+${pctN(f.esPct, 1)}`;
  $("s1-es-usd").textContent = `+${millions(f.esUsd)} in a 1-in-100 year`;

  // 2. Where the risk is
  const gvClause = f.gv
    ? `, while Greater Vancouver holds ${pctN(f.gv.eadShare, 0)} of the lending but only ${pctN(f.gv.upliftShare, 1)} of the uplift`
    : "";
  $("s2-head").textContent = `${N_TOP === 3 ? "Three" : N_TOP} divisions carry ${pctN(f.topShare, 0)} of the extra loss${gvClause}.`;
  $("s2-body").textContent = `${listAnd(f.top.map((d) => d.name))} are where fire-prone land and mortgages overlap most. ` +
    `Together they hold ${pctN(f.topEadShare, 0)} of the book's exposure. The map shows each division's increase in ` +
    `mean expected loss under ${shortLabel(STORY_SCENARIO)}; darker grey means a larger increase.`;
  renderStoryMap(f);

  // 3. Why bad years matter
  const [domYear, domN] = f.yearCounts[0];
  const verb = domN > f.nTail / 2 ? "are dominated by" : "are led by";
  $("s3-head").textContent = `The worst 1% of simulated years ${verb} the ${domYear} fire season, which ${domN} of the ${f.nTail} replay.`;
  $("s3-body").textContent = "Each simulated year replays one historical fire season, so most years land close together " +
    "and the right-hand tail is set by the few seasons when fire burned most widely. ES 99% is the average of the highlighted runs.";
  $("s3-note").textContent = `The worst ${f.nTail} of ${int(runs.length)} runs drew these fire seasons: ` +
    f.yearCounts.map(([y, n]) => `${y} (${n} ${n === 1 ? "run" : "runs"})`).join(", ") + ".";
  renderStoryHist(runs, f);

  // 4. How sure we are
  const [a, b] = f.drivers;
  $("s4-head").textContent = `The estimate is most sensitive to the ${lcFirst(a.label)} and the ${lcFirst(b.label)}.`;
  $("s4-body").textContent = "A tornado chart shows how far the result moves when one input is set to the low or high end " +
    "of its plausible range while everything else stays fixed, so the longest bars matter most. " +
    `Across their ranges, these two inputs swing the ${shortLabel(STORY_SCENARIO)} mean EL uplift by ${millions(a.el_uplift_swing)} ` +
    `and ${millions(b.el_uplift_swing)}. The chart's base, ${millions(f.sensBase)}, is the season-averaged analytic estimate, ` +
    `close to the simulated ${millions(f.simUplift)} above.`;
  renderTornado("chart-story-tornado", "el_uplift", "story");
}

/* ---------- maps ---------- */
// No basemap: division outlines on the page surface. Keyless tile services either need
// an API key (CARTO) or discourage app use (OSM), and labels compete with the choropleth.

function quantileBreaks(values) {
  const v = values.filter((x) => !isNil(x)).sort((a, b) => a - b);
  if (!v.length || v[0] === v[v.length - 1]) return [];
  const breaks = [];
  for (let i = 1; i < N_CLASSES; i++) {
    const q = v[Math.min(v.length - 1, Math.floor((i / N_CLASSES) * v.length))];
    if (!breaks.includes(q) && q > v[0]) breaks.push(q);
  }
  return breaks;
}
const classOf = (v, breaks) => breaks.filter((b) => v >= b).length;
const ramp = (prefix, k) => cssVar(`--${prefix}-${Math.min(k + 1, N_CLASSES)}`);

function makeMap(el, { onClick, tooltip, fitOpts = { padding: [4, 4] } }) {
  // trackResize off: a window resize (e.g. the scrollbar appearing on a tab switch) would
  // otherwise reach a map in a hidden panel at zero size and break it. fitVisibleMaps() resizes.
  const map = L.map(el, { scrollWheelZoom: false, zoomSnap: 0.25, attributionControl: true, trackResize: false });
  map.attributionControl.setPrefix(false);
  map.attributionControl.addAttribution("Boundaries: Statistics Canada 2021");
  const layer = L.geoJSON(D.regions, {
    onEachFeature: (f, lyr) => {
      if (tooltip) lyr.bindTooltip(() => tooltip(f.properties.cduid), { sticky: true });
      if (onClick) lyr.on("click", () => onClick(f.properties.cduid));
    },
  }).addTo(map);
  const m = { map, layer, fitOpts };
  fitMap(m);
  return m;
}

function fitMap(m) {
  m.map.fitBounds(m.layer.getBounds(), { ...m.fitOpts, animate: false });
  if (m.afterFit) m.afterFit();
}

// Names of the top divisions sit east of BC with leader lines, so small neighbours don't overlap.
const CALLOUT_GAP = 20;  // px between stacked labels
function placeCallouts(m) {
  const { map } = m;
  (m.callouts || []).forEach((l) => map.removeLayer(l));
  const x = map.latLngToContainerPoint(m.layer.getBounds().getNorthEast()).x + 14;
  const pts = m.top.map((id) => ({ id, ll: m.centers[id], p: map.latLngToContainerPoint(m.centers[id]) }))
    .sort((a, b) => a.p.y - b.p.y);
  let prev = -Infinity;
  pts.forEach((o) => { o.y = Math.max(o.p.y, prev + CALLOUT_GAP); prev = o.y; });
  const ink = cssVar("--ink-2");
  m.callouts = pts.flatMap((o) => {
    const at = map.containerPointToLatLng([x, o.y]);
    return [
      L.polyline([o.ll, at], { color: ink, weight: 1, interactive: false }),
      L.marker(at, { interactive: false, keyboard: false, icon: L.divIcon({
        className: "callout", html: esc(D.props[o.id].cd_name), iconSize: null, iconAnchor: [-4, 9] }) }),
    ];
  });
  m.callouts.forEach((l) => l.addTo(map));
}

function storyTooltip(id) {
  const p = D.props[id], r = row(STORY_SCENARIO, id);
  return `<strong>${esc(p.cd_name)}</strong><br>EL uplift: ${esc(money(r.el_uplift))} (${esc(pct(r.el_uplift_pct))})<br>` +
         `${int(p.n_loans)} loans &middot; ${money(p.ead_total)} exposure`;
}

function renderStoryMap(f) {
  const topIds = new Set(f.top.map((d) => d.id));
  if (!maps.story) {
    const w = $("story-map").clientWidth;
    const m = makeMap("story-map", { tooltip: storyTooltip,
      fitOpts: { paddingTopLeft: [4, 12], paddingBottomRight: [Math.min(180, Math.round(w * 0.48)), 12] } });
    m.top = f.top.map((d) => d.id);
    m.centers = {};
    m.layer.eachLayer((lyr) => {
      const id = lyr.feature.properties.cduid;
      if (topIds.has(id)) m.centers[id] = lyr.getBounds().getCenter();
    });
    m.afterFit = () => placeCallouts(m);
    m.map.on("zoomend", () => placeCallouts(m));
    maps.story = m;
    placeCallouts(m);
  } else {
    placeCallouts(maps.story);  // recolour leader lines after a theme change
  }
  const rest = divisionIds().filter((id) => !topIds.has(id)).map((id) => row(STORY_SCENARIO, id).el_uplift);
  const breaks = quantileBreaks(rest);
  maps.story.layer.setStyle((feat) => {
    const id = feat.properties.cduid, top = topIds.has(id);
    return { fillColor: top ? cssVar("--accent") : ramp("grey", classOf(row(STORY_SCENARIO, id).el_uplift, breaks)),
             fillOpacity: 1, color: cssVar("--page"), weight: 1 };
  });
  maps.story.layer.eachLayer((lyr) => { if (topIds.has(lyr.feature.properties.cduid)) lyr.bringToFront(); });
  const greys = Array.from({ length: N_CLASSES }, (_, k) =>
    `<span class="swatch" style="background:${ramp("grey", k)}"></span>`).join("");
  $("story-legend").innerHTML =
    `<span class="key"><span class="swatch" style="background:${cssVar("--accent")}"></span>` +
    `Top ${N_TOP} divisions (${pctN(f.topShare, 0)} of the uplift)</span>` +
    `<span class="key">Smaller ${greys} larger uplift</span>`;
}

function metricValue(cduid) {
  if (state.metric === "hazard_score") return D.props[cduid].hazard_score;
  const r = row(state.scenario, cduid);
  return r ? r[state.metric] : null;
}

function exploreTooltip(id) {
  const p = D.props[id];
  const v = metricValue(id);
  const m = METRICS[state.metric];
  const val = isNil(v) ? "not reported (fewer than 100 loans)" : m.fmt(v);
  return `<strong>${esc(p.cd_name)}</strong><br>${esc(m.label)}: ${esc(val)}<br>` +
         `${int(p.n_loans)} loans &middot; ${money(p.ead_total)} exposure`;
}

function renderExploreMap() {
  if (!maps.explore) maps.explore = makeMap("map", { onClick: select, tooltip: exploreTooltip });
  const values = divisionIds().map(metricValue);
  const breaks = quantileBreaks(values);
  maps.explore.layer.setStyle((feat) => {
    const id = feat.properties.cduid, v = metricValue(id), sel = id === state.selected;
    return { fillColor: isNil(v) ? cssVar("--no-data") : ramp("acc", classOf(v, breaks)), fillOpacity: 1,
             color: sel ? cssVar("--ink") : cssVar("--page"), weight: sel ? 2.5 : 1 };
  });
  const sel = state.selected && maps.explore.layer.getLayers().find((l) => l.feature.properties.cduid === state.selected);
  if (sel) sel.bringToFront();
  $("map-title").textContent = state.metric === "hazard_score"
    ? METRICS.hazard_score.label
    : `${METRICS[state.metric].label}, ${shortLabel(state.scenario)}`;
  renderLegend(values, breaks);
}

function renderLegend(values, breaks) {
  const m = METRICS[state.metric];
  const v = values.filter((x) => !isNil(x));
  const keys = [];
  if (!breaks.length) {
    keys.push([ramp("acc", 0), isRef(state.scenario) && !m.static
      ? `${m.fmt(v[0] ?? 0)} (historical is the reference, so its uplift is zero)` : m.fmt(v[0] ?? 0)]);
  } else {
    const edges = [Math.min(...v), ...breaks, Math.max(...v)];
    for (let k = 0; k < edges.length - 1; k++) keys.push([ramp("acc", k), `${m.fmt(edges[k])} – ${m.fmt(edges[k + 1])}`]);
  }
  if (values.some(isNil)) keys.push([cssVar("--no-data"), "Not reported (fewer than 100 loans)"]);
  $("legend").innerHTML = keys.map(([c, t]) =>
    `<span class="key"><span class="swatch" style="background:${c}"></span>${esc(t)}</span>`).join("");
}

/* ---------- detail ---------- */
function renderDetail() {
  const id = state.selected;
  if (!id) return;
  const p = D.props[id];
  const lines = D.scenarios.map((s) => {
    const r = row(s.id, id);
    return `<tr${s.id === state.scenario ? ' class="selected"' : ""}><td>${esc(shortLabel(s.id))}</td>` +
      `<td>${money(r.el_mean)}</td><td>${isRef(s.id) ? "ref" : pct(r.el_uplift_pct)}</td>` +
      `<td>${isRef(s.id) ? "ref" : money(r.es_99_uplift)}</td></tr>`;
  }).join("");
  $("detail").innerHTML = `<h3>${esc(p.cd_name)}</h3>
    <dl class="kv">
      <dt>Loans</dt><dd>${int(p.n_loans)}</dd>
      <dt>Exposure</dt><dd>${money(p.ead_total)}</dd>
      <dt>Hazard score</dt><dd>${num(4)(p.hazard_score)}</dd>
      <dt>Land burned per year</dt><dd>${pct(100 * p.annual_burn_share)}</dd>
    </dl>
    <div class="table-scroll"><table>
      <thead><tr><th>Scenario</th><th>Mean EL</th><th>Uplift</th><th>ES 99% up</th></tr></thead>
      <tbody>${lines}</tbody></table></div>
    ${p.tail_reportable ? "" : '<p class="note">Fewer than 100 loans, so tail measures are not reported: with so few loans they reflect individual loan sizes, not risk. Mean EL is still shown, and these loans count in the BC totals.</p>'}`;
}

function select(id) {
  state.selected = id;
  renderExploreMap();
  renderDetail();
  renderTable();
}

/* ---------- charts ---------- */
function chartDefaults() {
  Chart.defaults.font.family = cssVar("--font");
  Chart.defaults.font.size = 12;
  Chart.defaults.color = cssVar("--ink-2");
  Chart.defaults.borderColor = cssVar("--rule");
}
const axis = (title) => ({
  grid: { color: cssVar("--rule") }, border: { color: cssVar("--axis") },
  ticks: { color: cssVar("--ink-2") }, title: title ? { display: true, text: title, color: cssVar("--ink-2") } : undefined,
});
const tooltipStyle = () => ({
  backgroundColor: cssVar("--surface"), titleColor: cssVar("--ink"), bodyColor: cssVar("--ink"),
  borderColor: cssVar("--rule"), borderWidth: 1, padding: 10, displayColors: false,
});
function draw(key, canvasId, cfg) {
  if (charts[key]) charts[key].destroy();
  charts[key] = new Chart($(canvasId), cfg);
}

// Bracket over the bins holding the worst-1% runs, labelled with ES 99%.
const tailBracket = {
  id: "tailBracket",
  afterDatasetsDraw(chart, _args, opts) {
    const idx = opts.tail.map((n, i) => (n ? i : -1)).filter((i) => i >= 0);
    if (!idx.length) return;
    const { ctx, scales: { x, y } } = chart;
    const half = (x.getPixelForValue(1) - x.getPixelForValue(0)) / 2;
    const x0 = x.getPixelForValue(idx[0]) - half, x1 = x.getPixelForValue(idx[idx.length - 1]) + half;
    const yb = y.getPixelForValue(Math.max(...idx.map((i) => opts.tail[i] + opts.body[i]))) - 10;
    ctx.save();
    ctx.strokeStyle = ctx.fillStyle = cssVar("--accent");
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(x0, yb + 5); ctx.lineTo(x0, yb); ctx.lineTo(x1, yb); ctx.lineTo(x1, yb + 5);
    ctx.stroke();
    ctx.font = `600 12px ${cssVar("--font")}`;
    ctx.textAlign = "right";
    ctx.textBaseline = "bottom";
    // Lift the label clear of any taller bars it spans (narrow screens).
    const x0t = x1 - ctx.measureText(opts.text).width;
    const under = opts.body.map((n, i) => n + opts.tail[i])
      .filter((_, i) => x.getPixelForValue(i) + half >= x0t && x.getPixelForValue(i) - half <= x1);
    ctx.fillText(opts.text, x1, Math.min(yb - 4, y.getPixelForValue(Math.max(0, ...under)) - 6));
    ctx.restore();
  },
};

function renderStoryHist(runs, f) {
  const xs = runs.map((r) => r.loss / 1e6);
  const lo = Math.min(...xs), hi = Math.max(...xs), nb = 40, w = (hi - lo) / nb || 1;
  const bin = (x) => Math.min(nb - 1, Math.floor((x - lo) / w));
  const body = new Array(nb).fill(0), tail = new Array(nb).fill(0);
  runs.forEach((r) => { (f.tailRuns.has(r.run) ? tail : body)[bin(r.loss / 1e6)] += 1; });
  const labels = Array.from({ length: nb }, (_, i) => (lo + (i + 0.5) * w).toFixed(1));
  const ds = (label, data, color) => ({ label, data, backgroundColor: color, barPercentage: 1, categoryPercentage: 1,
                                        borderColor: cssVar("--page"), borderWidth: { left: 1, right: 0 } });
  draw("storyHist", "chart-story-hist", {
    type: "bar",
    data: { labels, datasets: [ds("Other runs", body, cssVar("--bar")), ds(`Worst 1% (${f.nTail} runs)`, tail, cssVar("--accent"))] },
    options: {
      maintainAspectRatio: false, animation: false, interaction: { mode: "index", intersect: false },
      plugins: {
        tailBracket: { tail, body, text: `Worst 1%: ES 99% = ${millions(row(STORY_SCENARIO, BC).es_99)}` },
        legend: { display: true, position: "top", align: "start", labels: { boxWidth: 10, boxHeight: 10, color: cssVar("--ink-2") } },
        tooltip: { ...tooltipStyle(), callbacks: {
          title: (c) => `Annual loss ≈ $${c[0].label}M`,
          label: (c) => (c.raw ? `${c.dataset.label}: ${c.raw}` : null) } },
      },
      scales: {
        x: { ...axis(`Simulated annual BC portfolio loss, ${shortLabel(STORY_SCENARIO)} ($M)`), stacked: true,
             grid: { display: false }, ticks: { maxTicksLimit: 8, color: cssVar("--ink-2"), maxRotation: 0 } },
        y: { ...axis("Simulated years"), stacked: true, beginAtZero: true },
      },
    },
    plugins: [tailBracket],
  });
}

function renderScenarioChart() {
  const scen = D.scenarios.filter((s) => !isRef(s.id));
  const hist = row(D.ref, BC);
  const narrow = $("chart-scenarios").parentElement.clientWidth < 560;
  const pick = (s, strong) => (s.id === state.scenario
    ? cssVar(strong ? "--accent" : "--accent-soft") : cssVar(strong ? "--bar" : "--bar-soft"));
  const ds = (label, field, base, strong, seField) => ({
    label, backgroundColor: scen.map((s) => pick(s, strong)), maxBarThickness: 40,
    data: scen.map((s) => 100 * row(s.id, BC)[field] / base),
    usd: scen.map((s) => row(s.id, BC)[field]),
    se: seField ? scen.map((s) => row(s.id, BC)[seField]) : null,
  });
  draw("scenarios", "chart-scenarios", {
    type: "bar",
    // Narrow screens: one word per line so neighbouring labels don't collide.
    data: { labels: scen.map((s) => [...(narrow ? shortLabel(s.id).split(" ") : [shortLabel(s.id)]), `m = ${s.multiplier}`]),
            datasets: [ds("Mean EL uplift", "el_uplift", hist.el_mean, false, "el_uplift_se"),
                       ds("ES 99% uplift", "es_99_uplift", hist.es_99, true, null)] },
    options: {
      maintainAspectRatio: false, animation: false,
      plugins: { legend: { display: false }, tooltip: { ...tooltipStyle(), callbacks: {
        label: (c) => {
          const d = c.dataset, i = c.dataIndex;
          const se = d.se ? ` ± ${money(1.96 * d.se[i])} (95% MC)` : "";
          return `${d.label}: +${c.raw.toFixed(1)}% (${money(d.usd[i])}${se})`;
        } } } },
      scales: { x: { ...axis(), grid: { display: false }, ticks: { color: cssVar("--ink-2"), maxRotation: 0, autoSkip: false } },
                y: { ...axis("% above historical"), beginAtZero: true, ticks: { color: cssVar("--ink-2"), callback: (v) => `${v}%` } } },
    },
  });
  $("scenario-legend").innerHTML =
    `<span class="key"><span class="swatch" style="background:${cssVar("--bar-soft")}"></span>Mean EL uplift (an average year)</span>` +
    `<span class="key"><span class="swatch" style="background:${cssVar("--bar")}"></span>ES 99% uplift (a 1-in-100 year)</span>` +
    (isRef(state.scenario) ? "" : `<span class="key"><span class="swatch" style="background:${cssVar("--accent")}"></span>Selected: ${esc(shortLabel(state.scenario))}</span>`);
}

const barEndLabels = {
  id: "barEndLabels",
  afterDatasetsDraw(chart) {
    const { ctx } = chart;
    ctx.save();
    ctx.font = `11px ${cssVar("--font")}`;
    ctx.fillStyle = cssVar("--ink-2");
    ctx.textBaseline = "middle";
    const xs = chart.scales.x;
    chart.data.datasets.forEach((ds, di) => {
      chart.getDatasetMeta(di).data.forEach((bar, i) => {
        const lab = ds.endLabels[i];
        if (!lab || ds.data[i] === null) return;
        const right = ds.direction === "up";
        const x = xs.getPixelForValue(ds.data[i][1]);  // the bar's far end, not its base
        ctx.textAlign = right ? "left" : "right";
        ctx.fillText(lab, right ? x + 4 : x - 4, bar.y);
      });
    });
    ctx.restore();
  },
};

function renderTornado(canvasId, m, key) {
  const rows = [...D.sensitivity].sort((a, b) => b[`${m}_swing`] - a[`${m}_swing`]);
  const base = rows[0][`${m}_base`] / 1e6;
  const colors = rows.map((_, i) => cssVar(i < 2 ? "--accent" : "--bar"));
  const mk = (dir) => {
    const data = [], labels = [];
    rows.forEach((r) => {
      const ends = ["low", "high"].map((s) => ({ v: r[`${m}_${s}`] / 1e6, lab: r[`${s}_value`] }))
        .filter((e) => (dir === "up" ? e.v > base + 1e-12 : e.v < base - 1e-12));
      const far = ends.sort((a, b) => Math.abs(b.v - base) - Math.abs(a.v - base))[0];
      data.push(far ? [base, far.v] : null);
      labels.push(far ? far.lab : "");
    });
    return { label: dir === "up" ? "raises the uplift" : "lowers the uplift", data, endLabels: labels,
             direction: dir, backgroundColor: colors, borderSkipped: false, grouped: false, barPercentage: 0.65 };
  };
  const what = m === "el_uplift" ? "mean EL uplift" : "ES 99% uplift";
  draw(key, canvasId, {
    type: "bar",
    data: { labels: rows.map((r) => r.label), datasets: [mk("down"), mk("up")] },
    options: {
      indexAxis: "y", maintainAspectRatio: false, animation: false,
      layout: { padding: { left: 4, right: 48 } },
      plugins: { legend: { display: false },
        tooltip: { ...tooltipStyle(), callbacks: { label: (c) => (c.raw
          ? `${c.dataset.endLabels[c.dataIndex]} → ${money(c.raw[1] * 1e6)} (base ${money(base * 1e6)})` : "") } } },
      scales: { x: { ...axis(`${shortLabel(STORY_SCENARIO)} ${what}, $M (base ${millions(base * 1e6)})`), grace: "14%" },
                y: { ...axis(), grid: { display: false }, ticks: { color: cssVar("--ink"), autoSkip: false } } },
    },
    plugins: [barEndLabels],
  });
}

/* ---------- table ---------- */
const COLS = [
  { key: "cd_name", label: "Division", fmt: esc },
  { key: "n_loans", label: "Loans", fmt: int },
  { key: "ead_total", label: "Exposure", fmt: money },
  { key: "hazard_score", label: "Hazard", fmt: num(4) },
  { key: "el_mean", label: "Mean EL", fmt: money },
  { key: "el_uplift", label: "EL uplift", fmt: money },
  { key: "el_uplift_pct", label: "Uplift %", fmt: pct },
  { key: "loss_p99_uplift", label: "p99 uplift", fmt: money },
  { key: "es_99_uplift", label: "ES 99% uplift", fmt: money },
];

function tableRows() {
  return divisionIds().map((id) => ({ cduid: id, ...row(state.scenario, id),
    cd_name: D.props[id].cd_name, hazard_score: D.props[id].hazard_score }));
}

function renderTable() {
  const { key, dir } = state.sort;
  const rows = tableRows().sort((a, b) => {
    const x = a[key], y = b[key];
    if (isNil(x)) return 1;
    if (isNil(y)) return -1;
    return (typeof x === "string" ? x.localeCompare(y) : x - y) * dir;
  });
  const bc = { ...row(state.scenario, BC), cd_name: "BC portfolio", hazard_score: null };
  $("table-title").textContent = `Census divisions, ${shortLabel(state.scenario)}`;
  $("table").querySelector("thead").innerHTML = "<tr>" + COLS.map((c) =>
    `<th scope="col"${c.key === key ? ` aria-sort="${dir > 0 ? "ascending" : "descending"}"` : ""}>` +
    `<button type="button" data-key="${c.key}">${esc(c.label)}</button></th>`).join("") + "</tr>";
  const tr = (r, cls, id) => `<tr class="${cls}"${id ? ` data-id="${id}"` : ""}>` +
    COLS.map((c) => `<td>${c.fmt(r[c.key])}</td>`).join("") + "</tr>";
  $("table").querySelector("tbody").innerHTML =
    rows.map((r) => tr(r, r.cduid === state.selected ? "selected" : "", r.cduid)).join("") + tr(bc, "total");
}

function wireTable() {
  $("table").querySelector("thead").addEventListener("click", (e) => {
    const k = e.target.closest("button")?.dataset.key;
    if (!k) return;
    state.sort = { key: k, dir: state.sort.key === k ? -state.sort.dir : (k === "cd_name" ? 1 : -1) };
    renderTable();
    $("table").querySelector(`button[data-key="${k}"]`).focus();
  });
  $("table").querySelector("tbody").addEventListener("click", (e) => {
    const id = e.target.closest("tr")?.dataset.id;
    if (id) select(id);
  });
}

/* ---------- explore ---------- */
function renderExplore() {
  renderExploreMap();
  renderDetail();
  renderTable();
  renderScenarioChart();
  renderTornado("chart-tornado", state.tornado, "tornado");
}

function wireExplore() {
  $("scenario").innerHTML = D.scenarios.map((s) =>
    `<option value="${esc(s.id)}"${s.id === state.scenario ? " selected" : ""}>${esc(s.label)} (m = ${s.multiplier})</option>`).join("");
  $("metric").innerHTML = Object.entries(METRICS).map(([k, m]) =>
    `<option value="${k}"${k === state.metric ? " selected" : ""}>${esc(m.label)}</option>`).join("");
  $("scenario").addEventListener("change", (e) => {
    state.scenario = e.target.value;
    renderExploreMap(); renderDetail(); renderTable(); renderScenarioChart();
  });
  $("metric").addEventListener("change", (e) => { state.metric = e.target.value; renderExploreMap(); });
  $("tornado-metric").addEventListener("change", (e) => {
    state.tornado = e.target.value;
    renderTornado("chart-tornado", state.tornado, "tornado");
  });
  wireTable();
}

/* ---------- method ---------- */
function fmtValue(v) {
  if (isNil(v)) return "";
  if (Array.isArray(v)) return v.join("–");
  if (typeof v === "object") return Object.entries(v).map(([k, x]) => `${k}: ${fmtValue(x)}`).join(", ");
  return String(v);
}

function renderMethod() {
  const groups = Object.entries(D.assumptions).map(([section, params]) =>
    `<tr class="group"><td colspan="3">${esc(section)}</td></tr>` + Object.entries(params).map(([name, p]) => {
      const status = [p.placeholder && "placeholder", p.proxy && "proxy"].filter(Boolean);
      const src = [p.label, p.source, p.note && `Note: ${p.note}`].filter(Boolean).map(esc).join(" ");
      return `<tr><td>${esc(name)}${status.length ? ` <em class="muted">(${status.join(", ")})</em>` : ""}</td>` +
        `<td>${esc(fmtValue(p.value))}${p.unit ? ` <span class="muted">${esc(p.unit)}</span>` : ""}</td>` +
        `<td class="src">${src}${p.url ? ` <a href="${esc(p.url)}" target="_blank" rel="noopener">${esc(p.url)}</a>` : ""}</td></tr>`;
    }).join("")).join("");
  $("assumptions").innerHTML = `<div class="table-scroll"><table>
    <thead><tr><th>Input</th><th>Value</th><th>Source</th></tr></thead><tbody>${groups}</tbody></table></div>`;
}

/* ---------- tabs ---------- */
function fitVisibleMaps() {
  Object.values(maps).forEach((m) => {
    if (!m.map.getContainer().offsetParent) return;  // hidden panel: zero size, nothing to fit
    m.map.invalidateSize({ pan: false });
    fitMap(m);
  });
}

const RENDER = { story: renderStory, explore: renderExplore, method: renderMethod };

async function showTab(name, { focus = false, scroll = false } = {}) {
  if (!TABS.includes(name)) name = "story";
  state.tab = name;
  TABS.forEach((t) => {
    const tab = $(`tab-${t}`), on = t === name;
    tab.setAttribute("aria-selected", String(on));
    tab.tabIndex = on ? 0 : -1;
    $(`panel-${t}`).hidden = !on;
  });
  if (focus) $(`tab-${name}`).focus();
  if (location.hash !== `#${name}`) history.replaceState(null, "", `#${name}`);
  if (scroll) window.scrollTo(0, 0);
  if (!D.rows) return;
  fitVisibleMaps();
  if (!rendered.has(name)) {
    rendered.add(name);
    await RENDER[name]();
  }
}

function wireTabs() {
  TABS.forEach((t) => $(`tab-${t}`).addEventListener("click", () => showTab(t)));
  document.querySelector('[role="tablist"]').addEventListener("keydown", (e) => {
    const i = TABS.indexOf(state.tab);
    const next = { ArrowRight: i + 1, ArrowLeft: i - 1, Home: 0, End: TABS.length - 1 }[e.key];
    if (next === undefined) return;
    e.preventDefault();
    showTab(TABS[(next + TABS.length) % TABS.length], { focus: true });
  });
  let resizeTimer;
  window.addEventListener("resize", () => { clearTimeout(resizeTimer); resizeTimer = setTimeout(fitVisibleMaps, 150); });
  window.addEventListener("hashchange", () => showTab(location.hash.slice(1), { scroll: true }));
}

/* ---------- boot ---------- */
async function init() {
  wireTabs();
  showTab(location.hash.slice(1));  // panels switch before data arrives

  try {
    const [sc, regions, risk, sens, assumptions] = await Promise.all([
      getJSON("/api/scenarios"), getJSON("/api/regions"), getJSON("/api/risk"),
      getJSON("/api/sensitivity"), getJSON("/api/assumptions")]);
    D.scenarios = sc.scenarios;
    D.ref = sc.reference_scenario;
    D.regions = regions;
    D.props = Object.fromEntries(regions.features.map((f) => [f.properties.cduid, f.properties]));
    D.rows = risk.rows;
    D.byKey = new Map(risk.rows.map((r) => [`${r.scenario}|${r.cduid}`, r]));
    D.sensitivity = sens.rows;
    D.assumptions = assumptions;
  } catch (err) {
    $("status").textContent = `Could not load data from the API at ${API || "(no API_BASE set)"}: ${err.message}. ` +
      "If the backend is on a free tier it may be waking up; retry in about 30 seconds.";
    $("status").classList.add("error");
    return;
  }

  if (!D.scenarios.some((s) => s.id === state.scenario)) state.scenario = D.ref;
  // Open Explore on the division with the largest dollar uplift in the Story scenario.
  state.selected = divisionIds().reduce((best, id) =>
    (row(STORY_SCENARIO, id).el_uplift > row(STORY_SCENARIO, best).el_uplift ? id : best));
  chartDefaults();
  wireExplore();
  await showTab(state.tab);
  $("status").textContent = "";

  // Charts and maps read colours from CSS tokens; redraw when the OS theme flips.
  matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => {
    chartDefaults();
    rendered.forEach((t) => { if (t !== "method") RENDER[t](); });
  });
}

init();
