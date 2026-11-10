/* BC Wildfire Mortgage Stress Test - frontend (M7).
 * Plain JS. Reads everything from the FastAPI backend; computes nothing but
 * histogram bins and map class breaks. */
"use strict";

const API = (new URLSearchParams(location.search).get("api") || window.API_BASE || "").replace(/\/$/, "");
const BC = "BC";
const N_CLASSES = 5;

const state = { scenario: "hot_house", metric: "el_uplift_pct", selected: null,
                sort: { key: "el_uplift", dir: -1 }, tornado: "el_uplift" };
const D = {};          // data from the API
const charts = {};     // Chart.js instances
const lossCache = {};  // scenario -> runs
let map, geoLayer;

/* ---------- helpers ---------- */
const $ = (id) => document.getElementById(id);
const cssVar = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
const esc = (s) => String(s).replace(/[&<>"']/g, (c) =>
  ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

function money(x) {
  if (x === null || x === undefined || Number.isNaN(x)) return "—";
  const a = Math.abs(x), s = x < 0 ? "-" : "";
  if (a >= 1e9) return `${s}$${(a / 1e9).toFixed(2)}B`;
  if (a >= 1e6) return `${s}$${(a / 1e6).toFixed(2)}M`;
  if (a >= 1e3) return `${s}$${(a / 1e3).toFixed(1)}K`;
  return `${s}$${a.toFixed(0)}`;
}
function pct(x) {
  if (x === null || x === undefined || Number.isNaN(x)) return "—";
  return `${x.toFixed(Math.abs(x) < 10 ? 2 : 1)}%`;
}
const num = (d) => (x) => (x === null || x === undefined ? "—" : x.toFixed(d));
const int = (x) => (x === null || x === undefined ? "—" : Math.round(x).toLocaleString());

// Static snapshot (scripts/build_static.py): "/api/losses?scenario=x" -> "data/losses_x.json".
const staticPath = (path) => `data/${path.replace(/^\/api\//, "").replace("?scenario=", "_")}.json`;

async function getJSON(path) {
  const r = await fetch(window.API_STATIC ? staticPath(path) : API + path);
  if (!r.ok) throw new Error(`${path}: HTTP ${r.status}`);
  return r.json();
}

const METRICS = {
  el_uplift_pct: { label: "Mean EL uplift vs historical (%)", fmt: pct },
  el_uplift: { label: "Mean EL uplift vs historical ($)", fmt: money },
  es_99_uplift: { label: "ES 99% uplift vs historical ($)", fmt: money, tail: true },
  hazard_score: { label: "Wildfire hazard score (historical, scenario-independent)", fmt: num(4), static: true },
};

const row = (scenario, cduid) => D.byKey.get(`${scenario}|${cduid}`);
const scenarioLabel = (id) => (D.scenarios.find((s) => s.id === id) || { label: id }).label;
const shortLabel = (id) => scenarioLabel(id).split(" (")[0];  // "Orderly (Net Zero 2050-like)" -> "Orderly"
const isRef = (id) => id === D.ref;

function metricValue(cduid) {
  if (state.metric === "hazard_score") return D.props[cduid].hazard_score;
  const r = row(state.scenario, cduid);
  return r ? r[state.metric] : null;
}

/* ---------- theme ---------- */
const THEMES = ["system", "light", "dark"];
function loadTheme() {
  try { return localStorage.getItem("theme") || "system"; } catch { return "system"; }
}
function applyTheme(t) {
  if (t === "system") delete document.documentElement.dataset.theme;
  else document.documentElement.dataset.theme = t;
  $("theme-toggle").textContent = `Theme: ${t}`;
  try { localStorage.setItem("theme", t); } catch { /* storage unavailable */ }
}

/* ---------- map ---------- */
// No basemap: division outlines on the page surface. Keyless tile services either need
// an API key (CARTO) or discourage app use (OSM), and labels compete with the choropleth.

function quantileBreaks(values) {
  const v = values.filter((x) => x !== null && x !== undefined).sort((a, b) => a - b);
  if (!v.length || v[0] === v[v.length - 1]) return [];
  const breaks = [];
  for (let i = 1; i < N_CLASSES; i++) {
    const q = v[Math.min(v.length - 1, Math.floor((i / N_CLASSES) * v.length))];
    if (!breaks.includes(q) && q > v[0]) breaks.push(q);
  }
  return breaks;
}
const classOf = (v, breaks) => breaks.filter((b) => v >= b).length;
const seqColor = (k) => cssVar(`--seq-${Math.min(k + 1, N_CLASSES)}`);

function styleFeature(f) {
  const id = f.properties.cduid;
  const v = metricValue(id);
  const sel = id === state.selected;
  return {
    fillColor: v === null || v === undefined ? cssVar("--no-data") : seqColor(classOf(v, D.breaks)),
    fillOpacity: 0.85,
    color: sel ? cssVar("--ink") : cssVar("--surface"),
    weight: sel ? 2.5 : 1,
  };
}

function tooltipHtml(id) {
  const p = D.props[id];
  const v = metricValue(id);
  const m = METRICS[state.metric];
  const val = v === null || v === undefined
    ? "not reported (fewer than 100 loans)" : m.fmt(v);
  return `<strong>${esc(p.cd_name)}</strong><br>${esc(m.label)}: ${esc(val)}<br>` +
         `${int(p.n_loans)} loans &middot; ${money(p.ead_total)} exposure`;
}

function buildMap() {
  map = L.map("map", { scrollWheelZoom: false, zoomSnap: 0.25, attributionControl: true });
  map.attributionControl.addAttribution("Boundaries: Statistics Canada 2021");
  D.breaks = quantileBreaks(Object.keys(D.props).map(metricValue));  // styleFeature needs these
  geoLayer = L.geoJSON(D.regions, {
    style: styleFeature,
    onEachFeature: (f, layer) => {
      layer.bindTooltip(() => tooltipHtml(f.properties.cduid), { sticky: true });
      layer.on("click", () => select(f.properties.cduid));
    },
  }).addTo(map);
  map.fitBounds(geoLayer.getBounds(), { padding: [8, 8] });
}

function renderMap() {
  const ids = Object.keys(D.props);
  D.breaks = quantileBreaks(ids.map(metricValue));
  geoLayer.setStyle(styleFeature);
  const sel = state.selected && geoLayer.getLayers().find((l) => l.feature.properties.cduid === state.selected);
  if (sel) sel.bringToFront();
  $("map-title").textContent = state.metric === "hazard_score"
    ? METRICS.hazard_score.label
    : `${METRICS[state.metric].label} — ${scenarioLabel(state.scenario)}`;
  renderLegend(ids.map(metricValue));
}

function renderLegend(values) {
  const m = METRICS[state.metric];
  const v = values.filter((x) => x !== null && x !== undefined);
  const keys = [];
  if (!D.breaks.length) {
    keys.push([seqColor(0), isRef(state.scenario) && !m.static
      ? `${m.fmt(v[0] ?? 0)} (reference scenario: uplift is zero by definition)` : m.fmt(v[0] ?? 0)]);
  } else {
    const edges = [Math.min(...v), ...D.breaks, Math.max(...v)];
    for (let k = 0; k < edges.length - 1; k++) {
      keys.push([seqColor(k), `${m.fmt(edges[k])} – ${m.fmt(edges[k + 1])}`]);
    }
  }
  if (values.some((x) => x === null || x === undefined)) {
    keys.push([cssVar("--no-data"), "Not reported (fewer than 100 loans)"]);
  }
  $("legend").innerHTML = keys.map(([c, t]) =>
    `<span class="key"><span class="swatch" style="background:${c}"></span>${esc(t)}</span>`).join("");
}

/* ---------- tiles ---------- */
function renderTiles() {
  const r = row(state.scenario, BC);
  const ref = isRef(state.scenario);
  const tile = (label, value, sub) =>
    `<div class="tile"><div class="label">${esc(label)}</div><div class="value">${esc(value)}</div>` +
    `<div class="sub">${esc(sub)}</div></div>`;
  $("tiles").innerHTML = [
    tile("Mean expected loss (BC)", money(r.el_mean), `${r.el_rate_bps.toFixed(2)} bps of ${money(r.ead_total)} exposure`),
    tile("Mean EL uplift vs historical", money(r.el_uplift),
         ref ? "reference scenario" : `${pct(r.el_uplift_pct)} ± ${money(1.96 * r.el_uplift_se)} (95% MC)`),
    tile("99th percentile loss uplift", money(r.loss_p99_uplift), `p99 loss ${money(r.loss_p99)}`),
    tile("ES 99% uplift", money(r.es_99_uplift), `ES 99% ${money(r.es_99)}`),
  ].join("");
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
  $("detail").innerHTML = `<h2>${esc(p.cd_name)}</h2>
    <dl class="kv">
      <dt>Loans</dt><dd>${int(p.n_loans)}</dd>
      <dt>Exposure</dt><dd>${money(p.ead_total)}</dd>
      <dt>Hazard score</dt><dd>${num(4)(p.hazard_score)}</dd>
      <dt>Mean annual share burned</dt><dd>${pct(100 * p.annual_burn_share)}</dd>
    </dl>
    <div class="table-scroll"><table>
      <thead><tr><th>Scenario</th><th>Mean EL</th><th>Uplift</th><th>ES 99% uplift</th></tr></thead>
      <tbody>${lines}</tbody></table></div>
    ${p.tail_reportable ? "" : '<p class="note">Fewer than 100 loans: tail measures are not reported; with so few loans they reflect individual loan sizes, not risk. Mean EL is still shown, and these loans count in the BC totals.</p>'}`;
}

function select(id) {
  state.selected = id;
  renderMap();
  renderDetail();
  renderTable();
}

/* ---------- charts ---------- */
function chartDefaults() {
  Chart.defaults.font.family = 'system-ui, -apple-system, "Segoe UI", sans-serif';
  Chart.defaults.color = cssVar("--ink-2");
  Chart.defaults.borderColor = cssVar("--grid");
}
const axis = (title) => ({
  grid: { color: cssVar("--grid") }, border: { color: cssVar("--axis") },
  ticks: { color: cssVar("--ink-2") }, title: title ? { display: true, text: title, color: cssVar("--ink-2") } : undefined,
});

function scenarioBar(canvasId, field, seField) {
  const scen = D.scenarios.filter((s) => !isRef(s.id));
  const vals = scen.map((s) => row(s.id, BC)[field] / 1e6);
  const colors = scen.map((s) => (s.id === state.scenario ? cssVar("--series-1") : cssVar("--series-1-soft")));
  const cfg = {
    type: "bar",
    data: { labels: scen.map((s) => shortLabel(s.id)), datasets: [{ data: vals, backgroundColor: colors,
      borderRadius: 4, borderSkipped: "start", maxBarThickness: 48 }] },
    options: {
      maintainAspectRatio: false, animation: false,
      plugins: { legend: { display: false }, tooltip: { callbacks: {
        label: (c) => {
          const r = row(scen[c.dataIndex].id, BC);
          const se = seField ? ` (±${money(1.96 * r[seField])} 95% MC)` : "";
          return `${money(r[field])}${se}`;
        } } } },
      scales: { x: { ...axis(), grid: { display: false } }, y: { ...axis("$M"), beginAtZero: true } },
    },
  };
  if (charts[canvasId]) charts[canvasId].destroy();
  charts[canvasId] = new Chart($(canvasId), cfg);
}

async function losses(scenario) {
  if (!lossCache[scenario]) {
    lossCache[scenario] = (await getJSON(`/api/losses?scenario=${encodeURIComponent(scenario)}`)).runs;
  }
  return lossCache[scenario];
}

async function renderHist() {
  const ids = isRef(state.scenario) ? [state.scenario] : [D.ref, state.scenario];
  const sets = await Promise.all(ids.map(losses));
  const all = sets.flat().map((r) => r.loss / 1e6);
  const lo = Math.min(...all), hi = Math.max(...all), nb = 40, w = (hi - lo) / nb || 1;
  const centers = Array.from({ length: nb }, (_, i) => lo + (i + 0.5) * w);
  const count = (runs) => {
    const c = new Array(nb).fill(0);
    runs.forEach((r) => { c[Math.min(nb - 1, Math.floor((r.loss / 1e6 - lo) / w))] += 1; });
    return c;
  };
  const colors = { [D.ref]: cssVar("--series-2"), [state.scenario]: cssVar("--series-1") };
  const datasets = ids.map((id, i) => ({
    label: shortLabel(id), data: count(sets[i]), borderColor: colors[id], backgroundColor: colors[id],
    borderWidth: 2, pointRadius: 0, stepped: "middle", fill: false,
  }));
  $("hist-title").textContent = `Simulated annual loss distribution — ${scenarioLabel(state.scenario)}`;
  const cfg = {
    type: "line",
    data: { labels: centers.map((x) => x.toFixed(1)), datasets },
    options: {
      maintainAspectRatio: false, animation: false, interaction: { mode: "index", intersect: false },
      plugins: { legend: { display: true, position: "bottom", labels: { boxWidth: 12, boxHeight: 2 } },
        tooltip: { callbacks: { title: (c) => `Loss ≈ $${c[0].label}M`, label: (c) => `${c.dataset.label}: ${c.raw} runs` } } },
      scales: { x: { ...axis("Annual portfolio loss ($M)"), ticks: { maxTicksLimit: 8, color: cssVar("--ink-2") } },
                y: { ...axis("Runs"), beginAtZero: true } },
    },
  };
  if (charts.hist) charts.hist.destroy();
  charts.hist = new Chart($("chart-hist"), cfg);
}

const barEndLabels = {
  id: "barEndLabels",
  afterDatasetsDraw(chart) {
    const { ctx } = chart;
    ctx.save();
    ctx.font = '11px system-ui, -apple-system, "Segoe UI", sans-serif';
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

function renderTornado() {
  const m = state.tornado;
  const rows = [...D.sensitivity].sort((a, b) => b[`${m}_swing`] - a[`${m}_swing`]);
  const base = rows[0][`${m}_base`] / 1e6;
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
             direction: dir, backgroundColor: cssVar(dir === "up" ? "--above" : "--below"),
             borderRadius: 3, borderSkipped: false, grouped: false, barPercentage: 0.7 };
  };
  const cfg = {
    type: "bar",
    data: { labels: rows.map((r) => r.label), datasets: [mk("down"), mk("up")] },
    options: {
      indexAxis: "y", maintainAspectRatio: false, animation: false,
      layout: { padding: { left: 8, right: 56 } },
      plugins: { legend: { position: "bottom", labels: { boxWidth: 12 } },
        tooltip: { callbacks: { label: (c) => c.raw ? `${c.dataset.endLabels[c.dataIndex]} → ${money(c.raw[1] * 1e6)} (base ${money(base * 1e6)})` : "" } } },
      scales: { x: { ...axis(`$M (base ${money(base * 1e6)})`), grace: "12%" },
                y: { ...axis(), grid: { display: false } } },
    },
    plugins: [barEndLabels],
  };
  if (charts.tornado) charts.tornado.destroy();
  charts.tornado = new Chart($("chart-tornado"), cfg);
}

function renderCharts() {
  chartDefaults();
  scenarioBar("chart-el", "el_uplift", "el_uplift_se");
  scenarioBar("chart-es", "es_99_uplift", null);
  renderTornado();
  return renderHist();
}

/* ---------- table ---------- */
const COLS = [
  { key: "cd_name", label: "Division", fmt: esc, text: true },
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
  return Object.keys(D.props).map((id) => ({ cduid: id, ...row(state.scenario, id),
    cd_name: D.props[id].cd_name, hazard_score: D.props[id].hazard_score }));
}

function renderTable() {
  const { key, dir } = state.sort;
  const rows = tableRows().sort((a, b) => {
    const x = a[key], y = b[key];
    if (x === null || x === undefined) return 1;
    if (y === null || y === undefined) return -1;
    return (typeof x === "string" ? x.localeCompare(y) : x - y) * dir;
  });
  const bc = { ...row(state.scenario, BC), cd_name: "BC portfolio", hazard_score: null };
  $("table-title").textContent = `Census divisions — ${scenarioLabel(state.scenario)}`;
  $("table").querySelector("thead").innerHTML = "<tr>" + COLS.map((c) =>
    `<th scope="col" data-key="${c.key}"${c.key === key ? ` aria-sort="${dir > 0 ? "ascending" : "descending"}"` : ""}>${esc(c.label)}</th>`).join("") + "</tr>";
  const tr = (r, cls, id) => `<tr class="${cls}"${id ? ` data-id="${id}"` : ""}>` +
    COLS.map((c) => `<td>${c.fmt(r[c.key])}</td>`).join("") + "</tr>";
  $("table").querySelector("tbody").innerHTML =
    rows.map((r) => tr(r, r.cduid === state.selected ? "selected" : "", r.cduid)).join("") + tr(bc, "total");
}

function wireTable() {
  $("table").querySelector("thead").addEventListener("click", (e) => {
    const k = e.target.closest("th")?.dataset.key;
    if (!k) return;
    state.sort = { key: k, dir: state.sort.key === k ? -state.sort.dir : (k === "cd_name" ? 1 : -1) };
    renderTable();
  });
  $("table").querySelector("tbody").addEventListener("click", (e) => {
    const id = e.target.closest("tr")?.dataset.id;
    if (id) select(id);
  });
}

/* ---------- assumptions ---------- */
function fmtValue(v) {
  if (v === null || v === undefined) return "";
  if (Array.isArray(v)) return v.join("–");
  if (typeof v === "object") return Object.entries(v).map(([k, x]) => `${k}: ${fmtValue(x)}`).join(", ");
  return String(v);
}

function renderAssumptions() {
  $("assumptions").innerHTML = Object.entries(D.assumptions).map(([section, params]) => `
    <div class="assumption-section"><h3>${esc(section)}</h3>${Object.entries(params).map(([name, p]) => `
      <div class="assumption">
        <span class="name">${esc(name)}</span>:
        <span class="val">${esc(fmtValue(p.value))}${p.unit ? ` <span class="muted">(${esc(p.unit)})</span>` : ""}</span>
        ${p.placeholder ? '<span class="badge">placeholder</span>' : ""}
        ${p.proxy ? '<span class="badge badge-proxy">proxy</span>' : ""}
        ${p.label ? `<div class="src">${esc(p.label)}</div>` : ""}
        ${p.source ? `<div class="src">${esc(p.source)}</div>` : ""}
        ${p.note ? `<div class="src">Note: ${esc(p.note)}</div>` : ""}
        ${p.url ? `<div class="src"><a href="${esc(p.url)}" target="_blank" rel="noopener">${esc(p.url)}</a></div>` : ""}
      </div>`).join("")}</div>`).join("");
}

/* ---------- wiring ---------- */
async function renderScenarioDependent() {
  renderTiles();
  renderMap();
  renderDetail();
  renderTable();
  await renderCharts();
}

async function init() {
  applyTheme(loadTheme());
  $("theme-toggle").addEventListener("click", async () => {
    const cur = loadTheme();
    applyTheme(THEMES[(THEMES.indexOf(cur) + 1) % THEMES.length]);
    if (D.rows) await renderScenarioDependent();
  });

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
      "If the backend is on a free tier it may be waking up; retry in ~30 s.";
    $("status").classList.add("error");
    return;
  }

  if (!D.scenarios.some((s) => s.id === state.scenario)) state.scenario = D.ref;
  $("scenario").innerHTML = D.scenarios.map((s) =>
    `<option value="${esc(s.id)}"${s.id === state.scenario ? " selected" : ""}>${esc(s.label)} (m = ${s.multiplier})</option>`).join("");
  $("metric").innerHTML = Object.entries(METRICS).map(([k, m]) =>
    `<option value="${k}"${k === state.metric ? " selected" : ""}>${esc(m.label)}</option>`).join("");
  $("scenario").addEventListener("change", (e) => { state.scenario = e.target.value; renderScenarioDependent(); });
  $("metric").addEventListener("change", (e) => { state.metric = e.target.value; renderMap(); });
  $("tornado-metric").addEventListener("change", (e) => { state.tornado = e.target.value; renderTornado(); });

  buildMap();
  wireTable();
  renderAssumptions();
  state.selected = "5935";  // Central Okanagan: largest dollar uplift, a useful first view
  await renderScenarioDependent();
  $("status").textContent = "";
}

init();
