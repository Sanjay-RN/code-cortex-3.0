const API = "";

// ---------------- Navigation ----------------
document.querySelectorAll(".nav-item").forEach(btn => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".nav-item").forEach(b => b.classList.remove("active"));
    document.querySelectorAll(".view").forEach(v => v.classList.remove("active"));
    btn.classList.add("active");
    document.getElementById("view-" + btn.dataset.view).classList.add("active");
    if (btn.dataset.view === "overview") loadOverview();
    if (btn.dataset.view === "live") loadLiveFeed();
    if (btn.dataset.view === "model") loadModelPerformance();
    if (btn.dataset.view === "personas") loadPersonas();
    if (btn.dataset.view === "simulate") loadSimulateRanges();
  });
});

// ---------------- Overview ----------------
let PERSONAS = [];

async function loadOverview() {
  const res = await fetch(`${API}/api/overview`);
  const data = await res.json();
  PERSONAS = data.personas;
  document.getElementById("stat-precision").textContent = (data.model_metrics.precision * 100).toFixed(1) + "%";
  document.getElementById("stat-recall").textContent = (data.model_metrics.recall * 100).toFixed(1) + "%";
  updateLiveStats(data.live_stats);
}

function updateLiveStats(stats) {
  if (!stats) return;
  const total = stats.total || 0;
  const flagged = stats.flagged || 0;
  const totalStr = total.toLocaleString();
  const flaggedStr = flagged.toLocaleString();
  const pctStr = total ? ((flagged / total) * 100).toFixed(1) + "% of traffic" : "—";
  const badgePct = total ? ((flagged / total) * 100).toFixed(1) + "%" : "0%";

  const totalEl = document.getElementById("stat-total");
  const flaggedEl = document.getElementById("stat-flagged");
  const pctEl = document.getElementById("stat-flagged-pct");
  const liveCounterEl = document.getElementById("feed-stat-counter");

  if (totalEl) totalEl.textContent = totalStr;
  if (flaggedEl) flaggedEl.textContent = flaggedStr;
  if (pctEl) pctEl.textContent = pctStr;
  if (liveCounterEl) {
    liveCounterEl.innerHTML = `<strong>${totalStr}</strong> processed · <strong>${flaggedStr}</strong> flagged (${badgePct})`;
  }
}

// ---------------- Live Feed ----------------
let streaming = false;
let streamInterval = null;

document.getElementById("btn-toggle-stream").addEventListener("click", (e) => {
  streaming = !streaming;
  if (streaming) {
    e.target.textContent = "⏸ Pause stream";
    streamInterval = setInterval(pullNext, 900);
    pullNext();
  } else {
    e.target.textContent = "▶ Start stream";
    clearInterval(streamInterval);
  }
});

document.getElementById("btn-reset").addEventListener("click", async () => {
  await fetch(`${API}/api/stream/reset`, { method: "POST" });
  document.getElementById("feed-body").innerHTML = '<div class="empty-state">Press "Start stream" to begin simulating live transactions.</div>';
  document.getElementById("detail-empty").style.display = "block";
  document.getElementById("detail-content").style.display = "none";
  updateLiveStats({ total: 0, flagged: 0 });
});

async function loadLiveFeed() {
  try {
    const res = await fetch(`${API}/api/overview`);
    const data = await res.json();
    updateLiveStats(data.live_stats);

    const body = document.getElementById("feed-body");
    if (body.querySelector(".empty-state")) {
      const recentRes = await fetch(`${API}/api/transactions/recent?limit=25`);
      const recent = await recentRes.json();
      if (recent && recent.length > 0) {
        body.innerHTML = "";
        recent.forEach(prependFeedRow);
      }
    }
  } catch (err) {
    console.error("Error loading live feed data:", err);
  }
}

async function pullNext() {
  try {
    const res = await fetch(`${API}/api/stream/next?n=1`);
    const data = await res.json();
    updateLiveStats(data.stats);
    data.transactions.forEach(prependFeedRow);
  } catch (err) {
    console.error("Stream fetch error:", err);
  }
}

function badgeClass(band) {
  return { low: "badge-low", watch: "badge-watch", high: "badge-high", critical: "badge-critical" }[band] || "badge-low";
}

function prependFeedRow(txn) {
  const body = document.getElementById("feed-body");
  if (body.querySelector(".empty-state")) body.innerHTML = "";
  const row = document.createElement("div");
  row.className = "feed-row";
  row.innerHTML = `
    <span class="txn-id">${txn.txn_id}<br><span class="persona-tag">${txn.timestamp}</span></span>
    <span class="amount">₹${txn.amount.toFixed(2)}</span>
    <span class="persona-tag" style="color:${txn.persona_color}">${txn.persona}</span>
    <span>${(txn.fraud_probability * 100).toFixed(1)}%</span>
    <span class="badge ${badgeClass(txn.risk_band)}">${txn.risk_band}</span>
  `;
  row.addEventListener("click", () => showDetail(txn));
  body.prepend(row);
  while (body.children.length > 60) body.removeChild(body.lastChild);
}

function showDetail(txn) {
  document.getElementById("detail-empty").style.display = "none";
  document.getElementById("detail-content").style.display = "block";
  document.getElementById("d-txn").textContent = txn.txn_id;
  document.getElementById("d-amount").textContent = "₹" + txn.amount.toFixed(2);
  document.getElementById("d-proba").textContent = (txn.fraud_probability * 100).toFixed(2) + "%";
  document.getElementById("d-persona").textContent = txn.persona;
  document.getElementById("d-threshold").textContent = (txn.effective_threshold * 100).toFixed(1) + "%";
  document.getElementById("d-verdict").textContent = txn.flagged ? "Flagged for review" : "Passed";
  renderExplanation("d-explanation", txn.explanation);
}

function renderExplanation(elId, explanation) {
  const el = document.getElementById(elId);
  el.innerHTML = explanation.map(f => {
    const isRisk = f.direction.includes("increases risk");
    const dirColor = isRisk ? "color: var(--danger);" : "color: var(--safe);";
    return `
      <div class="explanation-item">
        <span class="ef-name">${f.feature}</span>
        <span>${f.value}</span>
        <span class="ef-dir" style="${dirColor}">${f.direction}</span>
      </div>
    `;
  }).join("");
}

// ---------------- Personas ----------------
let personasLoaded = false;
async function loadPersonas() {
  if (personasLoaded) return;
  const res = await fetch(`${API}/api/personas`);
  const personas = await res.json();
  const grid = document.getElementById("persona-grid");
  grid.innerHTML = personas.map(p => `
    <div class="persona-card" style="--pc:${p.color}">
      <h3>${p.name}</h3>
      <div class="persona-meta">${p.pct_of_customers}% of customers · risk sensitivity ×${p.risk_multiplier}</div>
      <p class="persona-desc">${p.description}</p>
      <div class="persona-stats">
        <div class="persona-stat"><span class="v">₹${Math.round(p.avg_income).toLocaleString()}</span><span class="l">avg income</span></div>
        <div class="persona-stat"><span class="v">${Math.round(p.avg_age)}</span><span class="l">avg age</span></div>
        <div class="persona-stat"><span class="v">₹${Math.round(p.avg_total_spend)}</span><span class="l">avg spend</span></div>
      </div>
    </div>
  `).join("");
  personasLoaded = true;
}

// ---------------- Model Performance ----------------
let modelLoaded = false;
let rocChartInstance = null;
let importanceChartInstance = null;

async function loadModelPerformance() {
  if (modelLoaded) return;
  const res = await fetch(`${API}/api/model/performance`);
  const data = await res.json();
  const m = data.metrics;
  document.getElementById("m-roc").textContent = m.roc_auc.toFixed(3);
  document.getElementById("m-pr").textContent = m.pr_auc.toFixed(3);
  document.getElementById("m-f1").textContent = m.f1.toFixed(3);
  document.getElementById("m-threshold").textContent = (m.threshold * 100).toFixed(0) + "%";

  const [[tn, fp], [fn, tp]] = m.confusion_matrix;
  document.getElementById("confusion-matrix").innerHTML = `
    <div class="cm-cell cm-tn"><div class="cm-val">${tn.toLocaleString()}</div><div class="cm-label">True negative (legit passed)</div></div>
    <div class="cm-cell cm-fp"><div class="cm-val">${fp}</div><div class="cm-label">False positive (legit flagged)</div></div>
    <div class="cm-cell cm-fn"><div class="cm-val">${fn}</div><div class="cm-label">False negative (fraud missed)</div></div>
    <div class="cm-cell cm-tp"><div class="cm-val">${tp}</div><div class="cm-label">True positive (fraud caught)</div></div>
  `;

  if (rocChartInstance) rocChartInstance.destroy();
  if (importanceChartInstance) importanceChartInstance.destroy();

  rocChartInstance = new Chart(document.getElementById("roc-chart"), {
    type: "line",
    data: {
      labels: m.roc_curve.fpr.map(x => x.toFixed(2)),
      datasets: [{
        label: "ROC curve",
        data: m.roc_curve.tpr,
        borderColor: "#5b9dfa",
        backgroundColor: "rgba(91,157,250,0.1)",
        fill: true,
        tension: 0.2,
        pointRadius: 0,
      }]
    },
    options: {
      plugins: { legend: { display: false } },
      scales: {
        x: { title: { display: true, text: "False positive rate", color: "#8a97a6" }, ticks: { color: "#5c6b7a" }, grid: { color: "#1c2530" } },
        y: { title: { display: true, text: "True positive rate", color: "#8a97a6" }, ticks: { color: "#5c6b7a" }, grid: { color: "#1c2530" } },
      }
    }
  });

  importanceChartInstance = new Chart(document.getElementById("importance-chart"), {
    type: "bar",
    data: {
      labels: data.feature_importance.map(f => f.feature),
      datasets: [{
        label: "Importance",
        data: data.feature_importance.map(f => f.importance),
        backgroundColor: "#5b9dfa",
        borderRadius: 4,
      }]
    },
    options: {
      indexAxis: "y",
      plugins: { legend: { display: false } },
      scales: {
        x: { ticks: { color: "#5c6b7a" }, grid: { color: "#1c2530" } },
        y: { ticks: { color: "#e8edf2" }, grid: { display: false } },
      }
    }
  });

  modelLoaded = true;
}

// ---------------- Simulate ----------------
let simRanges = null;

document.getElementById("sim-amount").addEventListener("input", (e) => {
  document.getElementById("sim-amount-val").textContent = "₹" + e.target.value;
});
document.getElementById("sim-hour").addEventListener("input", (e) => {
  document.getElementById("sim-hour-val").textContent = String(e.target.value).padStart(2, "0") + ":00";
});

async function loadSimulateRanges() {
  if (simRanges) return;
  if (!PERSONAS || PERSONAS.length === 0) {
    const resPersonas = await fetch(`${API}/api/personas`);
    PERSONAS = await resPersonas.json();
  }

  const res = await fetch(`${API}/api/simulate/ranges`);
  simRanges = await res.json();

  const container = document.getElementById("sim-feature-sliders");
  container.innerHTML = simRanges.order.map(feat => {
    const r = simRanges.ranges[feat];
    return `
      <div class="form-row">
        <label>${feat} <span style="color:#5c6b7a">(behavioral signal)</span></label>
        <input type="range" class="sim-feat" data-feat="${feat}" min="${r.min}" max="${r.max}" step="0.01" value="${r.median}">
        <span class="range-value" id="val-${feat}">${r.median}</span>
      </div>
    `;
  }).join("");

  container.querySelectorAll(".sim-feat").forEach(input => {
    input.addEventListener("input", (e) => {
      document.getElementById("val-" + e.target.dataset.feat).textContent = parseFloat(e.target.value).toFixed(2);
    });
  });

  const personaSelect = document.getElementById("sim-persona");
  personaSelect.innerHTML = PERSONAS.map(p => `<option value="${p.cluster_id}">${p.name}</option>`).join("");
}

document.getElementById("preset-normal").addEventListener("click", () => {
  document.querySelectorAll(".sim-feat").forEach(input => {
    const r = simRanges.ranges[input.dataset.feat];
    input.value = r.median;
    document.getElementById("val-" + input.dataset.feat).textContent = r.median;
  });
  document.getElementById("sim-amount").value = 85;
  document.getElementById("sim-amount-val").textContent = "₹85";
  document.getElementById("btn-simulate").click();
});

document.getElementById("preset-fraud").addEventListener("click", () => {
  // Push toward extreme values in the "increases risk" direction (negative for most top features)
  document.querySelectorAll(".sim-feat").forEach(input => {
    const r = simRanges.ranges[input.dataset.feat];
    const extreme = r.min + (r.max - r.min) * 0.08; // near the low tail, where fraud clusters for these features
    input.value = extreme.toFixed(2);
    document.getElementById("val-" + input.dataset.feat).textContent = extreme.toFixed(2);
  });
  document.getElementById("sim-amount").value = 1;
  document.getElementById("sim-amount-val").textContent = "₹1";
  document.getElementById("btn-simulate").click();
});

document.getElementById("btn-simulate").addEventListener("click", async () => {
  const amount = parseFloat(document.getElementById("sim-amount").value);
  const hour = parseInt(document.getElementById("sim-hour").value);
  const persona_cluster = parseInt(document.getElementById("sim-persona").value);
  const features = {};
  document.querySelectorAll(".sim-feat").forEach(input => {
    features[input.dataset.feat] = parseFloat(input.value);
  });

  const res = await fetch(`${API}/api/simulate/predict`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ amount, hour, persona_cluster, features }),
  });
  const data = await res.json();

  document.getElementById("sim-empty").style.display = "none";
  document.getElementById("sim-result").style.display = "block";

  const pct = (data.fraud_probability * 100).toFixed(1);
  const gauge = document.getElementById("sim-gauge");
  const color = data.risk_band === "low" ? "#3ecf8e" : data.risk_band === "watch" ? "#f5b942" : "#f2596b";
  gauge.style.background = `conic-gradient(${color} ${pct * 3.6}deg, #17202b 0deg)`;
  document.getElementById("sim-gauge-value").textContent = pct + "%";

  const badge = document.getElementById("sim-verdict-badge");
  badge.textContent = data.flagged ? "Flagged for review" : "Approved";
  badge.style.color = color;

  document.getElementById("sim-threshold").textContent = (data.effective_threshold * 100).toFixed(1) + "%";
  renderExplanation("sim-explanation", data.explanation);
});

// ---------------- Init ----------------
loadOverview();
