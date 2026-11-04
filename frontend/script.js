/* GridMind dashboard — vanilla JS, talks to FastAPI via JSON REST. */
const API = "/api";
// theme colours come from the CSS variables in style.css so the palette lives in one place
const cssVar = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
const THEME = {
  text: cssVar("--text"), muted: cssVar("--muted"), panel: cssVar("--panel"), border: cssVar("--border"),
  accent: cssVar("--accent"), accent2: cssVar("--accent2"), orange: cssVar("--orange"), red: cssVar("--red"),
  green: cssVar("--green"), grey: cssVar("--grey"), edgeOpen: cssVar("--edge-open"), edgeDead: cssVar("--edge-dead"),
};
const COLORS = { connected: THEME.green, faulty: THEME.red, isolated: THEME.orange, disconnected: THEME.grey, source: THEME.accent };
const ACTION_SHORT = ["MAINTAIN", "ISOLATE", "RECONNECT"];
const state = { sim: null, network: null, restoration: null, results: null, charts: {} };

/* ---------------- utilities ---------------- */
const $ = (id) => document.getElementById(id);
const pct = (x, d = 1) => (x == null ? "—" : (100 * x).toFixed(d) + "%");
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

async function api(path, method = "GET", body) {
  const opt = { method, headers: { "Content-Type": "application/json" } };
  if (body !== undefined) opt.body = JSON.stringify(body);
  const res = await fetch(API + path, opt);
  let data;
  try { data = await res.json(); } catch { data = {}; }
  if (!res.ok) throw new Error(data.detail ? (typeof data.detail === "string" ? data.detail : JSON.stringify(data.detail)) : `HTTP ${res.status}`);
  return data;
}
function toast(msg, err = false) {
  const t = $("toast"); t.textContent = msg; t.className = "toast show" + (err ? " err" : "");
  clearTimeout(t._h); t._h = setTimeout(() => (t.className = "toast"), 4000);
}
function log(msg) {
  const el = $("event-log"); const d = document.createElement("div");
  d.textContent = `[${new Date().toLocaleTimeString()}] ${msg}`; el.prepend(d);
}
function setLive(msg) { $("live-status").textContent = msg; }
async function busy(btn, fn) {
  const btns = btn ? [btn] : [];
  btns.forEach((b) => (b.disabled = true));
  try { return await fn(); } catch (e) { toast(e.message, true); log("ERROR: " + e.message); } finally { btns.forEach((b) => (b.disabled = false)); }
}

/* ---------------- navigation ---------------- */
document.querySelectorAll(".nav-item").forEach((n) => n.addEventListener("click", () => {
  document.querySelectorAll(".nav-item").forEach((x) => x.classList.remove("active"));
  document.querySelectorAll(".view").forEach((x) => x.classList.remove("active"));
  n.classList.add("active");
  $("view-" + n.dataset.view).classList.add("active");
  $("view-title").textContent = n.textContent.replace(/^\S+\s/, "");
  Object.values(netViews).forEach((v) => v.resize());
}));

/* ---------------- Chart.js defaults ---------------- */
Chart.defaults.color = THEME.muted;
Chart.defaults.borderColor = THEME.border;
Chart.defaults.font.family = "Segoe UI, system-ui, sans-serif";
Chart.defaults.animation.duration = 500;

function lineChart(id, label, color, opts = {}) {
  return new Chart($(id), {
    type: "line",
    data: { labels: [], datasets: [{ label, data: [], borderColor: color, borderWidth: 1.6, pointRadius: 0, tension: 0.15, fill: false }] },
    options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { display: false } },
      scales: { x: { ticks: { maxTicksLimit: 8 }, title: { display: !!opts.xTitle, text: opts.xTitle } },
                y: { title: { display: !!opts.yTitle, text: opts.yTitle } } } },
  });
}
function barChart(id, labels, colors, opts = {}) {
  return new Chart($(id), {
    type: "bar",
    data: { labels, datasets: [{ data: labels.map(() => 0), backgroundColor: colors, borderRadius: 6 }] },
    options: { responsive: true, maintainAspectRatio: false, indexAxis: opts.horizontal ? "y" : "x",
      plugins: { legend: { display: false } }, scales: { y: { beginAtZero: true, max: opts.max } } },
  });
}

function initCharts() {
  const c = state.charts;
  c.voltage = lineChart("ch-voltage", "Voltage", THEME.accent2, { xTitle: "Time (s)", yTitle: "V" });
  c.current = lineChart("ch-current", "Current", THEME.red, { xTitle: "Time (s)", yTitle: "A" });
  c.frequency = lineChart("ch-frequency", "Frequency", THEME.green, { xTitle: "Time (s)", yTitle: "Hz" });
  c.probs = barChart("ch-probs", ["Normal", "Voltage Sag", "Overload", "Fault"], [THEME.green, THEME.accent2, THEME.orange, THEME.red], { horizontal: true });
  c.probs.options.scales = { x: { beginAtZero: true, max: 1 } };
  c.q = barChart("ch-q", ["Maintain", "Isolate", "Reconnect"], [THEME.grey, THEME.orange, THEME.accent]);
  c.dqnReward = new Chart($("ch-dqn-reward"), {
    type: "line",
    data: { labels: [], datasets: [
      { label: "Episode reward", data: [], borderColor: THEME.accent2 + "66", pointRadius: 0, borderWidth: 1 },
      { label: "10-episode mean", data: [], borderColor: THEME.accent, pointRadius: 0, borderWidth: 2 } ] },
    options: { responsive: true, maintainAspectRatio: false, scales: { x: { ticks: { maxTicksLimit: 10 }, title: { display: true, text: "Episode" } } } },
  });
  const two = (id, a, b) => new Chart($(id), {
    type: "line",
    data: { labels: [], datasets: [
      { label: a, data: [], borderColor: THEME.accent2, pointRadius: 2, borderWidth: 2 },
      { label: b, data: [], borderColor: THEME.accent, pointRadius: 2, borderWidth: 2 } ] },
    options: { responsive: true, maintainAspectRatio: false, scales: { x: { title: { display: true, text: "Epoch" } } } },
  });
  c.cnnLoss = two("ch-cnn-loss", "Train loss", "Val loss");
  c.cnnAcc = two("ch-cnn-acc", "Train acc", "Val acc");
  const cmp = ["DQN", "Rule-Based"], cc = [THEME.accent, THEME.orange];
  c.cmpSteps = barChart("cmp-steps", cmp, cc);
  c.cmpLoad = barChart("cmp-load", cmp, cc, { max: 100 });
  c.cmpInvalid = barChart("cmp-invalid", cmp, cc);
  c.cmpReward = barChart("cmp-reward", cmp, cc);
}

/* ---------------- network canvas (animated) ---------------- */
class NetView {
  constructor(canvas) {
    this.c = canvas; this.ctx = canvas.getContext("2d"); this.net = null;
    this.colors = {}; this.pulses = {}; this.t = 0; this.baseH = canvas.height;
    this.resize(); requestAnimationFrame(this.loop.bind(this));
  }
  resize() {
    const w = this.c.clientWidth || 600, dpr = window.devicePixelRatio || 1;
    this.w = w; this.h = this.baseH;
    this.c.width = w * dpr; this.c.height = this.h * dpr; this.c.style.height = this.h + "px";
    this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }
  set(net) {
    if (this.net) net.sections.forEach((s) => {
      const old = this.net.sections.find((o) => o.id === s.id);
      if (old && old.status !== s.status) this.pulses[s.id] = 1;
    });
    this.net = net;
  }
  pos(id) {
    const pad = Math.min(70, this.w * 0.09), step = (this.w - 2 * pad) / 4;
    return { x: pad + (id - 1) * step, y: this.h * 0.42 };
  }
  radius() {   // node size shrinks with canvas width so nodes never cover the switches between them
    const pad = Math.min(70, this.w * 0.09);
    return Math.min(26, ((this.w - 2 * pad) / 4) * 0.28);
  }
  target(s) { return s.type === "source" ? COLORS.source : COLORS[s.status]; }
  mix(a, b, k) {
    const p = (h) => [1, 3, 5].map((i) => parseInt(h.slice(i, i + 2), 16));
    const A = p(a), B = p(b);
    return "#" + A.map((v, i) => Math.round(v + (B[i] - v) * k).toString(16).padStart(2, "0")).join("");
  }
  loop() {
    this.t += 1; this.draw(); requestAnimationFrame(this.loop.bind(this));
  }
  draw() {
    const ctx = this.ctx; ctx.clearRect(0, 0, this.w, this.h);
    if (!this.net) {
      ctx.fillStyle = THEME.muted; ctx.font = "14px Segoe UI"; ctx.textAlign = "center";
      ctx.fillText("Run a simulation to load the network", this.w / 2, this.h / 2); return;
    }
    const secs = this.net.sections, en = (id) => secs[id - 1].connected, R = this.radius();
    // edges
    this.net.edges.forEach((e) => {
      const a = this.pos(e.from), b = this.pos(e.to), live = e.closed && en(e.from) && en(e.to);
      ctx.save(); ctx.lineWidth = live ? 4 : 2.5;
      ctx.strokeStyle = live ? THEME.accent : e.closed ? THEME.edgeDead : THEME.edgeOpen;
      ctx.setLineDash(e.closed ? (live ? [10, 8] : []) : [4, 6]);
      if (live) ctx.lineDashOffset = -this.t * 0.6;
      ctx.beginPath();
      let mid;
      if (e.tie) {
        const cy = this.h * 0.92;
        ctx.moveTo(a.x, a.y + R); ctx.bezierCurveTo(a.x, cy, b.x, cy, b.x, b.y + R);
        mid = { x: (a.x + b.x) / 2, y: cy - 16 };
      } else { ctx.moveTo(a.x + R + 2, a.y); ctx.lineTo(b.x - R - 2, b.y); mid = { x: (a.x + b.x) / 2, y: a.y }; }
      ctx.stroke(); ctx.restore();
      // switch marker
      ctx.fillStyle = e.closed ? THEME.accent : THEME.panel; ctx.strokeStyle = e.closed ? THEME.accent : THEME.muted;
      ctx.lineWidth = 2; ctx.fillRect(mid.x - 6, mid.y - 6, 12, 12); ctx.strokeRect(mid.x - 6, mid.y - 6, 12, 12);
      ctx.fillStyle = THEME.muted; ctx.font = "10px Segoe UI"; ctx.textAlign = "center";
      if (e.tie || R >= 20) ctx.fillText((e.tie ? "TIE " : "") + (e.closed ? "closed" : "open"), mid.x, mid.y + (e.tie ? 22 : -12));
    });
    // nodes
    secs.forEach((s) => {
      const p = this.pos(s.id), tgt = this.target(s);
      this.colors[s.id] = this.colors[s.id] ? this.mix(this.colors[s.id], tgt, 0.08) : tgt;
      const col = this.colors[s.id];
      if (this.pulses[s.id] > 0) {
        const k = this.pulses[s.id]; ctx.beginPath(); ctx.arc(p.x, p.y, R + 2 + (1 - k) * R, 0, 7);
        ctx.strokeStyle = tgt + Math.round(k * 255).toString(16).padStart(2, "0"); ctx.lineWidth = 3; ctx.stroke();
        this.pulses[s.id] -= 0.015;
      }
      if (s.status === "faulty" && Math.floor(this.t / 20) % 2) { ctx.beginPath(); ctx.arc(p.x, p.y, R + 8, 0, 7); ctx.strokeStyle = THEME.red + "80"; ctx.lineWidth = 2; ctx.stroke(); }
      ctx.save(); ctx.shadowColor = col; ctx.shadowBlur = s.connected || s.type === "source" ? 8 : 0;
      ctx.beginPath(); ctx.arc(p.x, p.y, R, 0, 7); ctx.fillStyle = THEME.panel; ctx.fill();
      ctx.lineWidth = R < 20 ? 3 : 4; ctx.strokeStyle = col; ctx.stroke(); ctx.restore();
      ctx.fillStyle = col; ctx.font = `bold ${R < 20 ? 11 : 14}px Segoe UI`; ctx.textAlign = "center"; ctx.textBaseline = "middle";
      ctx.fillText("S" + s.id, p.x, p.y);
      // stagger labels on narrow canvases so neighbouring sections don't overlap
      const narrow = this.w < 700, lift = narrow && s.id % 2 === 0 ? 16 : 0;
      ctx.textBaseline = "alphabetic"; ctx.fillStyle = THEME.text; ctx.font = (narrow ? "10px" : "12px") + " Segoe UI";
      ctx.fillText(narrow ? s.name.replace(" Load", "") : s.name, p.x, p.y - R - 12 - lift);
      ctx.fillStyle = THEME.muted; ctx.font = (narrow ? "9px" : "11px") + " Segoe UI";
      ctx.fillText(s.type === "source" ? "SOURCE" : narrow ? s.status : `${s.served_load}/${s.demand} kW · ${s.status}`, p.x, p.y + R + 16 + lift);
    });
  }
}
const netViews = {};
function updateNetwork(net) {
  state.network = net;
  Object.values(netViews).forEach((v) => v.set(net));
  const rows = net.sections.map((s) => `<tr><td>S${s.id}</td><td>${s.name}</td><td><span class="tag ${s.status}">${s.status.toUpperCase()}</span></td>
    <td>${s.connected ? "Yes" : "No"}</td><td>${s.fault ? "Yes" : "No"}</td><td>${s.isolated ? "Yes" : "No"}</td><td>${s.demand} kW</td><td>${s.served_load} kW</td></tr>`).join("");
  $("net-table").innerHTML = `<tr><th>Section</th><th>Name</th><th>Status</th><th>Connected</th><th>Fault</th><th>Isolated</th><th>Demand</th><th>Served</th></tr>${rows}`;
  $("ov-load").textContent = pct(net.served_load_ratio);
  $("ov-load-bar").style.width = pct(net.served_load_ratio);
  const g = $("ov-grid");
  const faulty = net.sections.some((s) => s.status === "faulty");
  const down = net.sections.some((s) => s.status === "disconnected");
  g.textContent = faulty ? "FAULTED" : down ? "PARTIAL OUTAGE" : net.complete ? "HEALTHY" : "RESTORING";
  g.className = "value " + (faulty ? "bad" : down ? "warn" : "ok");
  $("ov-stage").textContent = net.complete ? `Restoration complete (${net.steps} steps)` :
    faulty ? "Fault detected – isolation pending" : down ? "Fault isolated – reconnecting loads" : "Monitoring";
}

/* ---------------- simulation ---------------- */
async function simulate(ft) {
  setLive("Simulating " + ft);
  const section = $("sim-section").value ? parseInt($("sim-section").value) : null;
  const r = await api("/simulate", "POST", { fault_type: ft, section });
  state.sim = r; state.restoration = null;
  const w = r.waveform, labels = w.t.map((t) => t.toFixed(3));
  [["voltage", w.voltage], ["current", w.current], ["frequency", w.frequency]].forEach(([k, d]) => {
    state.charts[k].data.labels = labels; state.charts[k].data.datasets[0].data = d; state.charts[k].update();
  });
  const cl = r.classification;
  state.charts.probs.data.datasets[0].data = Object.values(cl.probabilities); state.charts.probs.update();
  $("sim-true").textContent = r.fault_type + (r.section ? ` @ S${r.section}` : "");
  $("sim-pred").textContent = cl.fault_class;
  $("sim-pred").className = "value " + (cl.fault_class === r.fault_type ? "ok" : "bad");
  $("sim-conf").textContent = pct(cl.confidence);
  $("sim-status").innerHTML = `V: ${r.stats.voltage_status}<br>I: ${r.stats.current_status}<br>f: ${r.stats.frequency_status}`;
  $("spec-img").src = r.spectrogram_image;
  $("ov-fault").textContent = cl.fault_class; $("ov-conf").textContent = "CNN confidence " + pct(cl.confidence);
  $("ov-reward").textContent = "—";
  $("ai-fault").textContent = `${cl.fault_class} (${pct(cl.confidence)})`;
  $("ai-dqn").textContent = "Run restoration"; $("ai-safety").textContent = "—";
  resetDqnPanel();
  updateNetwork(r.network);
  log(`Simulated ${r.fault_type}${r.section ? " at S" + r.section : ""} → CNN: ${cl.fault_class} (${pct(cl.confidence)})`);
  setLive(`${r.fault_type} simulated`);
}
document.querySelectorAll("[data-sim]").forEach((b) => b.addEventListener("click", () => busy(b, () => simulate(b.dataset.sim))));

/* ---------------- restoration ---------------- */
function resetDqnPanel() {
  ["dqn-step", "dqn-action", "dqn-reward", "dqn-safety"].forEach((id) => ($(id).textContent = "—"));
  $("dqn-cum").textContent = ""; $("dqn-progress").style.width = "0"; $("dqn-progress-txt").textContent = "";
  $("traj-table").innerHTML = ""; state.charts.q.data.datasets[0].data = [0, 0, 0]; state.charts.q.update();
}
async function animateTrajectory(r, label) {
  updateNetwork(r.initial_network);
  const rows = [];
  const head = `<tr><th>Step</th><th>Action</th><th>Q-values [M, I, R]</th><th>Safety</th><th>Detail</th><th>Reward</th><th>Served</th></tr>`;
  $("traj-table").innerHTML = head;
  await sleep(500);
  for (const st of r.trajectory) {
    const v = st.validation;
    $("dqn-step").textContent = `${label} · step ${st.step}`;
    $("dqn-action").textContent = st.action_name;
    $("dqn-reward").textContent = st.reward.toFixed(2);
    $("dqn-cum").textContent = "Cumulative: " + st.cumulative_reward.toFixed(2);
    $("dqn-safety").innerHTML = `<span class="tag ${v.valid ? "ok" : "bad"}">${v.valid ? "APPROVED" : "REJECTED"}</span><br><span class="muted" style="font-size:12px">${v.reason}</span>`;
    $("dqn-progress").style.width = pct(st.progress); $("dqn-progress-txt").textContent = `Restoration progress ${pct(st.progress)} · served load ${pct(st.served_load_ratio)}`;
    if (st.q_values) { state.charts.q.data.datasets[0].data = st.q_values; state.charts.q.update(); }
    rows.push(`<tr><td>${st.step}</td><td>${ACTION_SHORT[st.action]}</td><td>${st.q_values ? st.q_values.map((q) => q.toFixed(2)).join(", ") : "n/a (rule)"}</td>
      <td><span class="tag ${v.valid ? "ok" : "bad"}">${v.valid ? "APPROVED" : "REJECTED"}</span></td><td>${st.detail}</td><td>${st.reward.toFixed(2)}</td><td>${pct(st.served_load_ratio)}</td></tr>`);
    $("traj-table").innerHTML = head + rows.join("");
    updateNetwork(st.network);
    log(`${label} step ${st.step}: ${st.action_name} → ${v.valid ? "approved" : "REJECTED"} (${st.detail})`);
    await sleep(1200);
  }
  const s = r.summary;
  $("ov-reward").textContent = s.cumulative_reward.toFixed(2);
  toast(`${label}: ${s.success ? "restoration complete" : "restoration incomplete"} in ${s.steps} steps, served ${s.served_load_pct.toFixed(1)}%`);
}
async function runRestore(btn) {
  return busy(btn, async () => {
    if (!state.sim) await simulate("Fault");
    setLive("DQN restoring…");
    const r = await api("/restore", "POST");
    state.restoration = r;
    $("ai-dqn").textContent = r.trajectory.map((s) => ACTION_SHORT[s.action]).join(" → ");
    const rej = r.trajectory.filter((s) => !s.validation.valid).length;
    $("ai-safety").textContent = rej ? `${rej} action(s) rejected` : `All ${r.trajectory.length} actions approved`;
    await animateTrajectory(r, "DQN");
    setLive("Restoration finished");
  });
}
$("btn-restore").addEventListener("click", (e) => runRestore(e.target));
$("btn-pipeline").addEventListener("click", (e) => busy(e.target, async () => {
  const types = ["Voltage Sag", "Overload", "Fault"];
  await simulate(types[Math.floor(Math.random() * types.length)]);
  await sleep(600);
  await runRestore(null);
}));
$("btn-baseline").addEventListener("click", (e) => busy(e.target, async () => {
  if (!state.sim) await simulate("Fault");
  setLive("Rule-based restoring…");
  const r = await api("/baseline", "POST");
  await animateTrajectory(r, "Rule-Based");
  setLive("Baseline finished");
}));
$("btn-reset").addEventListener("click", (e) => busy(e.target, async () => {
  const r = await api("/reset", "POST"); resetDqnPanel(); updateNetwork(r.data); log("Grid reset to post-disturbance state");
}));
$("btn-train-dqn").addEventListener("click", (e) => busy(e.target, async () => {
  setLive("Training DQN…"); toast("Training DQN on CPU…");
  const r = await api("/train/dqn", "POST", { episodes: parseInt($("dqn-episodes").value) || 300, evaluate: true });
  renderDqnTraining(r.data.training); if (r.data.evaluation) renderEvaluation(r.data.evaluation);
  toast(`DQN trained: ${r.data.training.episodes} episodes in ${r.data.training.training_time_s}s`);
  log(`DQN trained (${r.data.training.episodes} episodes)`); setLive("DQN trained");
}));
$("btn-train-cnn").addEventListener("click", (e) => busy(e.target, async () => {
  setLive("Training CNN…"); toast("Training CNN on CPU…");
  const r = await api("/train/cnn", "POST", { epochs: parseInt($("cnn-epochs").value) || 30 });
  renderCnn(r.data); toast(`CNN trained: val acc ${pct(r.data.final_val_accuracy)}`); setLive("CNN trained");
  log("CNN retrained — retrain the DQN so it learns the new feature space");
}));
$("btn-evaluate").addEventListener("click", (e) => busy(e.target, async () => {
  setLive("Evaluating…"); const r = await api("/evaluate", "POST", { episodes: 100 }); renderEvaluation(r.data); setLive("Evaluation done");
}));

/* ---------------- AI assistant ---------------- */
const asList = (v) => (Array.isArray(v) ? v : v ? [v] : []);
const esc = (s) => String(s).replace(/[&<>]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" }[c]));
$("btn-ai").addEventListener("click", (e) => busy(e.target, async () => {
  $("ai-error").classList.add("hidden"); $("ai-report").innerHTML = ""; $("ai-loading").classList.remove("hidden");
  setLive("Llama 3.2 generating report…");
  try {
    if (!state.sim) await simulate("Fault");
    const r = await api("/llm/explain", "POST", { include_baseline: false });
    const rep = r.report;
    $("ai-report").innerHTML = `
      <div class="meta">Model: ${esc(r.model)} · latency ${r.latency_s}s · JSON parsed: ${r.json_parsed} · saved: ${esc(r.report_path || "")}</div>
      <h4>Fault explanation</h4><p>${esc(rep.fault_explanation)}</p>
      <h4>Possible causes</h4><ul>${asList(rep.possible_causes).map((c) => `<li>${esc(typeof c === "object" ? JSON.stringify(c) : c)}</li>`).join("")}</ul>
      <h4>Why the DQN action is appropriate</h4><p>${esc(rep.dqn_action_justification)}</p>
      <h4>Safety considerations</h4><ul>${asList(rep.safety_considerations).map((c) => `<li>${esc(typeof c === "object" ? JSON.stringify(c) : c)}</li>`).join("")}</ul>
      <h4>Recommended restoration sequence</h4><ol>${asList(rep.recommended_restoration_sequence).map((c) => `<li>${esc(typeof c === "object" ? JSON.stringify(c) : c)}</li>`).join("")}</ol>
      <h4>Grid condition summary</h4><p>${esc(rep.grid_condition_summary)}</p>`;
    const ctx = r.context;
    $("ai-dqn").textContent = ctx.dqn_actions.map((a) => a.split(": ")[1]).join(" → ");
    $("ai-safety").textContent = ctx.safety_results.every((s) => s.includes("APPROVED")) ? "All actions approved" : "Some actions rejected";
    log(`Llama 3.2 report generated in ${r.latency_s}s`); setLive("AI report ready");
  } catch (err) {
    $("ai-error").textContent = "Ollama error: " + err.message; $("ai-error").classList.remove("hidden"); setLive("AI report failed");
    throw err;
  } finally { $("ai-loading").classList.add("hidden"); }
}));

/* ---------------- results rendering ---------------- */
function renderCnn(m) {
  if (!m) return;
  const h = m.history, ep = h.train_loss.map((_, i) => i + 1);
  state.charts.cnnLoss.data.labels = ep; state.charts.cnnLoss.data.datasets[0].data = h.train_loss; state.charts.cnnLoss.data.datasets[1].data = h.val_loss; state.charts.cnnLoss.update();
  state.charts.cnnAcc.data.labels = ep; state.charts.cnnAcc.data.datasets[0].data = h.train_acc; state.charts.cnnAcc.data.datasets[1].data = h.val_acc; state.charts.cnnAcc.update();
  $("cnn-train-acc").textContent = pct(m.final_train_accuracy); $("cnn-val-acc").textContent = pct(m.final_val_accuracy);
  $("cnn-loss").textContent = h.train_loss[h.train_loss.length - 1].toFixed(4);
  $("cnn-f1").textContent = m.classification_report["macro avg"]["f1-score"].toFixed(3);
  $("ov-acc").textContent = pct(m.final_val_accuracy);
  $("cnn-info").textContent = `${m.train_samples} train / ${m.val_samples} val samples · ${m.epochs} epochs · ${m.training_time_s}s · ${m.trained_at}`;
  const cm = m.confusion_matrix, mx = Math.max(...cm.flat());
  let html = `<table><tr><th>True \\ Pred</th>${m.classes.map((c) => `<th>${c}</th>`).join("")}</tr>`;
  cm.forEach((row, i) => {
    html += `<tr><th>${m.classes[i]}</th>${row.map((v, j) => {
      const a = mx ? v / mx : 0, col = i === j ? `rgba(${cssVar("--accent-rgb")},${0.12 + 0.6 * a})` : `rgba(220,38,38,${v ? 0.15 + 0.6 * a : 0.04})`;
      return `<td style="background:${col}">${v}</td>`; }).join("")}</tr>`;
  });
  $("cm").innerHTML = html + "</table>";
  const rep = m.classification_report;
  $("cnn-report").innerHTML = `<tr><th>Class</th><th>Precision</th><th>Recall</th><th>F1</th><th>Support</th></tr>` +
    m.classes.map((c) => `<tr><td>${c}</td><td>${rep[c].precision.toFixed(3)}</td><td>${rep[c].recall.toFixed(3)}</td><td>${rep[c]["f1-score"].toFixed(3)}</td><td>${rep[c].support}</td></tr>`).join("") +
    `<tr><td><b>Accuracy</b></td><td colspan="4">${rep.accuracy.toFixed(3)}</td></tr>`;
}
function renderDqnTraining(t) {
  if (!t) return;
  const r = t.rewards, ma = r.map((_, i) => { const s = r.slice(Math.max(0, i - 9), i + 1); return s.reduce((a, b) => a + b, 0) / s.length; });
  const ch = state.charts.dqnReward;
  ch.data.labels = r.map((_, i) => i + 1); ch.data.datasets[0].data = r; ch.data.datasets[1].data = ma; ch.update();
}
function renderEvaluation(ev) {
  if (!ev) return;
  const d = ev.dqn, b = ev.rule_based, set = (k, a, bb) => { state.charts[k].data.datasets[0].data = [a, bb]; state.charts[k].update(); };
  set("cmpSteps", d.avg_restoration_steps, b.avg_restoration_steps);
  set("cmpLoad", d.avg_served_load_pct, b.avg_served_load_pct);
  set("cmpInvalid", d.invalid_actions_total, b.invalid_actions_total);
  set("cmpReward", d.avg_cumulative_reward, b.avg_cumulative_reward);
  const rows = [["Episodes", "episodes", (x) => x], ["Avg restoration steps", "avg_restoration_steps", (x) => x.toFixed(2)],
    ["Avg served load", "avg_served_load_pct", (x) => x.toFixed(2) + "%"], ["Avg restoration progress", "avg_progress_pct", (x) => x.toFixed(2) + "%"],
    ["Invalid actions (total)", "invalid_actions_total", (x) => x], ["Invalid action rate", "invalid_action_rate", (x) => (100 * x).toFixed(2) + "%"],
    ["Avg cumulative reward", "avg_cumulative_reward", (x) => x.toFixed(2)], ["Successful restorations", "successful_restorations", (x) => x],
    ["Success rate", "success_rate", (x) => (100 * x).toFixed(1) + "%"]];
  $("cmp-table").innerHTML = `<tr><th>Metric</th><th>DQN</th><th>Rule-Based</th></tr>` + rows.map(([n, k, f]) => `<tr><td>${n}</td><td>${f(d[k])}</td><td>${f(b[k])}</td></tr>`).join("");
  $("cmp-type").innerHTML = `<tr><th>Fault type</th><th>Episodes</th><th>DQN</th><th>Rule-Based</th></tr>` + Object.entries(ev.by_fault_type).map(([ft, v]) =>
    `<tr><td>${ft}</td><td>${v.dqn.episodes}</td><td>${(100 * v.dqn.success_rate).toFixed(0)}% · ${v.dqn.avg_cumulative_reward.toFixed(2)}</td><td>${(100 * v.rule_based.success_rate).toFixed(0)}% · ${v.rule_based.avg_cumulative_reward.toFixed(2)}</td></tr>`).join("");
  $("eval-info").textContent = `${ev.n_episodes} identical scenarios · evaluated ${ev.evaluated_at}`;
}
function renderGallery(plots) {
  const specs = plots.filter((p) => p.startsWith("spectrogram_"));
  $("spec-gallery").innerHTML = specs.map((p) => `<figure><img src="/results/plots/${p}?v=${Date.now()}" /><figcaption>${p.replace("spectrogram_", "").replace(".png", "").replace("_", " ")}</figcaption></figure>`).join("");
}

/* ---------------- boot ---------------- */
async function health() {
  try {
    const h = await api("/health");
    $("dot-backend").className = "dot ok";
    $("dot-cnn").className = "dot " + (h.cnn_model ? "ok" : "bad");
    $("dot-dqn").className = "dot " + (h.dqn_model ? "ok" : "bad");
    $("dot-ollama").className = "dot " + (h.ollama.available && h.ollama.model_available ? "ok" : "bad");
    $("dot-ollama").parentElement.title = h.ollama.error || "llama3.2 ready";
    return h;
  } catch { $("dot-backend").className = "dot bad"; }
}
async function boot() {
  initCharts();
  netViews.overview = new NetView($("net-overview"));
  netViews.main = new NetView($("net-main"));
  netViews.dqn = new NetView($("net-dqn"));
  window.addEventListener("resize", () => Object.values(netViews).forEach((v) => v.resize()));
  const h = await health();
  try {
    const r = (await api("/results")).data;
    state.results = r; renderCnn(r.cnn); renderDqnTraining(r.dqn_training); renderEvaluation(r.evaluation); renderGallery(r.plots);
  } catch (e) { log("Could not load stored results: " + e.message); }
  if (h && h.cnn_model) await busy(null, () => simulate("Fault"));
  log("Dashboard ready" + (h && h.ollama.error ? " — " + h.ollama.error : ""));
  setInterval(health, 15000);
}
boot();
