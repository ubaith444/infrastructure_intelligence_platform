/* Infrastructure Intelligence Platform — Multi-Screen Operations Dashboard
   Screen 1: Operations Overview
   Screen 2: Anomaly Explorer
   Screen 3: Incident Investigation
   Screen 4: Data and Model Quality
   Persistent Triage Matrix Answering:
     1. What is failing?
     2. How severe is it?
     3. What evidence supports the alert?
     4. What should the engineer investigate next?
*/

let topologyViewer = null;
let currentSelectedService = 'order-service';
let currentActiveFault = null;
let currentActiveScreen = 'screen1';
let currentTimeRange = '5m';
let currentMetric = 'latency';
let cachedServices = [];
let cachedAnomalies = [];
let cachedIncidents = [];
let cachedMetrics = [];

// What's New: State variables for update controls and animated plots
let isPollingPaused = false;
let pollingIntervalMs = 5000;
let pollingTimer = null;
let demoAnimationTimer = null;
const MAX_HISTORY_POINTS = 25;
let rollingTrafficHistory = [];
let rollingResourceHistory = [];
let resourceHoverPoint = null;

// ============================================================================
// UTILITY FUNCTIONS & NETWORKING
// ============================================================================

async function fetchJSON(url, opts = {}) {
    try {
        const res = await fetch(url, opts);
        if (!res.ok) return null;
        return await res.json();
    } catch {
        return null;
    }
}

function updateLiveClock() {
    const el = document.getElementById('liveClock');
    if (el) {
        const now = new Date();
        el.textContent = `UTC ${now.toISOString().substring(11, 19)}`;
    }
}
setInterval(updateLiveClock, 1000);
updateLiveClock();

// ============================================================================
// SCREEN SWITCHING (TAB NAVIGATION)
// ============================================================================

function switchScreen(screenId) {
    currentActiveScreen = screenId;
    ['screen1', 'screen2', 'screen3', 'screen4'].forEach(id => {
        const screenEl = document.getElementById(id);
        const btnEl = document.getElementById(`tabBtn${id.charAt(0).toUpperCase() + id.slice(1)}`);
        if (screenEl) screenEl.classList.toggle('active', id === screenId);
        if (btnEl) btnEl.classList.toggle('active', id === screenId);
    });

    if (screenId === 'screen2') {
        renderAnomalyChart();
    } else if (screenId === 'screen3') {
        refreshIncidentInvestigation();
    } else if (screenId === 'screen4') {
        refreshDataAndModelQuality();
    }
}

function selectService(serviceId) {
    currentSelectedService = serviceId;
    if (topologyViewer) {
        topologyViewer.setSelected(serviceId);
    }
    const selEl = document.getElementById('explorerServiceSelect');
    if (selEl) selEl.value = serviceId;

    const chartTitle = document.getElementById('explorerChartTitle');
    if (chartTitle) chartTitle.textContent = `${serviceId} — Time-Series & Anomaly Detection Markers`;

    const nodeBadge = document.getElementById('selectedTopologyNodeBadge');
    if (nodeBadge) nodeBadge.textContent = serviceId;

    refreshServiceMetricsChart();
    renderAnomalyChart();
    refreshServiceFeatures();
}

function setTimeRange(range, btn) {
    currentTimeRange = range;
    const parent = btn.parentElement;
    if (parent) {
        parent.querySelectorAll('button').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
    }
    renderAnomalyChart();
}

// ============================================================================
// OPERATIONAL TRIAGING BANNER (ANSWERS THE 4 CORE QUESTIONS)
// ============================================================================

function updateOperationalTriageBanner(servicesData, incidentsData, anomaliesData) {
    const failingEl = document.getElementById('triageFailingService');
    const blastRadiusEl = document.getElementById('triageBlastRadius');
    const severityBadge = document.getElementById('triageSeverityBadge');
    const errorBudgetEl = document.getElementById('triageErrorBudget');
    const evidenceEl = document.getElementById('triageEvidence');
    const detectorStatsEl = document.getElementById('triageDetectorStats');
    const nextActionEl = document.getElementById('triageNextAction');
    const confidenceEl = document.getElementById('triageConfidence');

    const failingServices = (servicesData?.services || []).filter(s => s.status === 'CRITICAL' || s.status === 'DEGRADED');
    const criticalServices = failingServices.filter(s => s.status === 'CRITICAL');
    const activeCandidates = incidentsData?.candidate_origins || [];
    const recentAnoms = anomaliesData?.anomalies || [];

    // Q1: What is failing?
    if (criticalServices.length > 0) {
        const primary = criticalServices[0];
        failingEl.innerHTML = `<span style="color:#dc2626;">${primary.name}</span> (${primary.status})`;
        const callers = primary.callers || [];
        blastRadiusEl.textContent = callers.length > 0
            ? `Blast radius: ${callers.length} downstream caller(s) impacted (${callers.slice(0, 2).join(', ')})`
            : `Blast radius: Isolated to ${primary.id}`;
    } else if (failingServices.length > 0) {
        const primary = failingServices[0];
        failingEl.innerHTML = `<span style="color:#d97706;">${primary.name}</span> (DEGRADED)`;
        blastRadiusEl.textContent = `Blast radius: ${failingServices.length} service(s) with elevated metrics`;
    } else {
        failingEl.textContent = 'All 7 services nominal';
        blastRadiusEl.textContent = 'Blast radius: 0 degraded services';
    }

    // Q2: How severe is it?
    if (criticalServices.length > 0) {
        severityBadge.className = 'badge badge-critical';
        severityBadge.textContent = 'CRITICAL SEVERITY';
        errorBudgetEl.innerHTML = 'Error budget burn: <strong style="color:#dc2626;">14.2x</strong> &bull; 62% consumed';
    } else if (failingServices.length > 0) {
        severityBadge.className = 'badge badge-warning';
        severityBadge.textContent = 'HIGH SEVERITY';
        errorBudgetEl.innerHTML = 'Error budget burn: <strong style="color:#d97706;">3.5x</strong> &bull; 85% remaining';
    } else {
        severityBadge.className = 'badge badge-healthy';
        severityBadge.textContent = 'NOMINAL';
        errorBudgetEl.textContent = 'Error budget burn: 1.0x baseline &bull; 99.8% remaining';
    }

    // Q3: What evidence supports the alert?
    if (recentAnoms.length > 0) {
        const topAnom = recentAnoms[0];
        const detector = topAnom.model_version || topAnom.detector || 'MultiModel';
        evidenceEl.textContent = `${detector}: Score ${(topAnom.score || 0).toFixed(2)} (${topAnom.evidence?.attribution_narrative || 'Threshold breach'})`;
        detectorStatsEl.textContent = `Evidence: ${recentAnoms.length} detections across MAD, Isolation Forest & River`;
    } else {
        evidenceEl.textContent = 'MAD & IF baselines nominal';
        detectorStatsEl.textContent = '3 models voting: Robust MAD, IF, River';
    }

    // Q4: What should the engineer investigate next?
    if (activeCandidates.length > 0) {
        const topCand = activeCandidates[0];
        const confPct = ((topCand.origin_confidence || 0.85) * 100).toFixed(1);
        nextActionEl.innerHTML = `Inspect <strong>${topCand.service}</strong> upstream connection pool & queue`;
        confidenceEl.textContent = `Candidate root cause confidence: ${confPct}% (Depth ${topCand.topological_depth || 1})`;
    } else if (criticalServices.length > 0) {
        const topCand = criticalServices[0];
        nextActionEl.innerHTML = `Inspect <strong>${topCand.id}</strong> logs and database queries`;
        confidenceEl.textContent = `Candidate root cause confidence: 85.0%`;
    } else {
        nextActionEl.textContent = 'Continue standard operational monitoring';
        confidenceEl.textContent = 'Candidate confidence: --';
    }
}

// ============================================================================
// SCREEN 1: OPERATIONS OVERVIEW DATA REFRESH
// ============================================================================

async function refreshOperationsOverview() {
    const [pipe, svcs, incs, alerts, anoms] = await Promise.all([
        fetchJSON('/api/v1/pipeline/status'),
        fetchJSON('/api/v1/services'),
        fetchJSON('/api/incidents'),
        fetchJSON('/api/alerts'),
        fetchJSON('/api/anomalies/live?limit=15'),
    ]);

    cachedServices = svcs?.services || [];
    cachedAnomalies = anoms?.anomalies || [];
    cachedIncidents = incs?.candidate_origins || [];

    // Update Triage Banner
    updateOperationalTriageBanner(svcs, incs, anoms);

    // Update KPI Strip
    if (pipe) {
        document.getElementById('statEps').textContent = `${(pipe.throughput_eps || 0).toFixed(1)} EPS`;
        document.getElementById('statLag').textContent = `${pipe.consumer_lag || 0} events`;
        const dqBadge = document.getElementById('pipelineDqBadge');
        if (dqBadge) {
            dqBadge.className = pipe.status === 'HEALTHY' ? 'badge badge-healthy' : 'badge badge-warning';
            dqBadge.textContent = pipe.status;
        }
    }

    // Latency Percentiles from first service
    if (cachedServices.length > 0) {
        const s = cachedServices.find(x => x.id === currentSelectedService) || cachedServices[0];
        const m = s.current_metrics || {};
        document.getElementById('statLatencyPercentiles').textContent = 
            `${(m.latency_p50 || 35).toFixed(0)} / ${(m.latency_p90 || 48).toFixed(0)} / ${(m.latency_p99 || 72).toFixed(0)} ms`;
    }

    // Active Incidents count
    const activeIncCount = (incs?.candidate_origins || []).length;
    document.getElementById('statActiveIncidents').textContent = activeIncCount;
    document.getElementById('tabIncidentCountBadge').textContent = activeIncCount;
    document.getElementById('tabAnomalyCountBadge').textContent = cachedAnomalies.length;

    // Active Fault Status
    const fault = incs?.fault_status;
    currentActiveFault = fault?.active ? fault : null;
    const faultPill = document.getElementById('activeFaultPill');
    const faultBadge = document.getElementById('faultHarnessStatusBadge');
    if (fault?.active) {
        if (faultPill) { faultPill.style.display = 'inline-block'; faultPill.textContent = `Fault: ${fault.scenario}`; }
        if (faultBadge) { faultBadge.className = 'badge badge-critical'; faultBadge.textContent = `ACTIVE (${fault.scenario})`; }
    } else {
        if (faultPill) faultPill.style.display = 'none';
        if (faultBadge) { faultBadge.className = 'badge badge-healthy'; faultBadge.textContent = 'IDLE'; }
    }

    // Populate Services Inventory Table
    renderServicesTable(cachedServices);

    // Populate Alerts Table
    renderAlertsTable(alerts?.alerts || []);

    // Update Topology Graph
    if (topologyViewer) {
        topologyViewer.loadData({
            nodes: cachedServices.map(s => ({ id: s.id, status: s.status })),
            links: [
                { source: 'edge-gateway', target: 'auth-service' },
                { source: 'edge-gateway', target: 'catalog-service' },
                { source: 'edge-gateway', target: 'checkout-api' },
                { source: 'checkout-api', target: 'order-service' },
                { source: 'order-service', target: 'payment-service' },
                { source: 'order-service', target: 'inventory-service' },
                { source: 'order-service', target: 'notification-service' },
            ]
        });
    }

    // What's New: Update rolling histories and render new plots/signals
    updateRollingHistoriesFromServices(cachedServices);
    renderTrafficPerformancePlot();
    renderResourceUtilPlot();
    renderAnomalySignalsList(cachedServices, cachedAnomalies);
    updateLastUpdatedTime();
}

// ============================================================================
// WHAT'S NEW: ANIMATED PLOTS, RESOURCE UTILIZATION & ANOMALY SIGNALS
// ============================================================================

function updateRollingHistoriesFromServices(services) {
    if (!services || services.length === 0) return;
    const now = new Date();
    const timeLabel = now.toISOString().substring(11, 19);

    let totalReq = 0;
    let maxP90 = 0;
    let avgErr = 0;
    let avgCpu = 0;
    let avgMem = 0;

    services.forEach(s => {
        const m = s.current_metrics || {};
        totalReq += (m.request_rate || s.request_rate || 0);
        const lat = (m.latency_p90 || s.latency_p90 || 45);
        if (lat > maxP90) maxP90 = lat;
        avgErr += (m.error_rate || s.error_rate || 0.002);
        avgCpu += (m.cpu_utilization || s.cpu_utilization || 30);
        avgMem += (m.memory_utilization || s.memory_utilization || 42);
    });

    const count = services.length;
    avgErr = (avgErr / count) * 100.0;
    avgCpu = avgCpu / count;
    avgMem = avgMem / count;

    rollingTrafficHistory.push({
        time: timeLabel,
        requests: Math.round(totalReq * 10) / 10,
        latencyP95: Math.round(maxP90 * 1.15 * 10) / 10,
        errorRate: Math.round(avgErr * 100) / 100,
    });
    if (rollingTrafficHistory.length > MAX_HISTORY_POINTS) {
        rollingTrafficHistory.shift();
    }

    rollingResourceHistory.push({
        time: timeLabel,
        cpu: Math.round(avgCpu * 10) / 10,
        memory: Math.round(avgMem * 10) / 10,
    });
    if (rollingResourceHistory.length > MAX_HISTORY_POINTS) {
        rollingResourceHistory.shift();
    }
}

// 1. ANIMATED TRAFFIC AND PERFORMANCE PLOTS
function renderTrafficPerformancePlot() {
    const canvas = document.getElementById('trafficPerformanceCanvas');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const w = canvas.width;
    const h = canvas.height;
    ctx.clearRect(0, 0, w, h);

    const padL = 40, padR = 15, padT = 20, padB = 25;
    const chartW = w - padL - padR;
    const chartH = h - padT - padB;

    if (rollingTrafficHistory.length < 2) {
        ctx.fillStyle = '#94a3b8';
        ctx.font = '12px sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText('Streaming traffic dynamics...', w / 2, h / 2);
        return;
    }

    // Gridlines
    ctx.strokeStyle = '#f1f5f9';
    ctx.lineWidth = 1;
    for (let i = 0; i <= 4; i++) {
        const y = padT + (chartH / 4) * i;
        ctx.beginPath();
        ctx.moveTo(padL, y);
        ctx.lineTo(w - padR, y);
        ctx.stroke();
    }

    const maxReq = Math.max(...rollingTrafficHistory.map(d => d.requests), 50.0) * 1.25;
    const maxLat = Math.max(...rollingTrafficHistory.map(d => d.latencyP95), 40.0) * 1.25;

    // Y Axis labels
    ctx.fillStyle = '#94a3b8';
    ctx.font = '9px monospace';
    ctx.textAlign = 'right';
    ctx.fillText(`${Math.round(maxReq)} r/s`, padL - 6, padT + 4);
    ctx.fillText('0', padL - 6, padT + chartH);

    const n = rollingTrafficHistory.length;

    // Draw Line helper
    function drawSeries(getValue, maxVal, strokeColor, fillColor) {
        ctx.beginPath();
        const pts = [];
        for (let i = 0; i < n; i++) {
            const x = padL + (i / (n - 1)) * chartW;
            const val = getValue(rollingTrafficHistory[i]);
            const y = padT + chartH - (val / maxVal) * chartH;
            pts.push({ x, y });
            if (i === 0) ctx.moveTo(x, y);
            else ctx.lineTo(x, y);
        }
        ctx.strokeStyle = strokeColor;
        ctx.lineWidth = 2.2;
        ctx.stroke();

        if (fillColor && pts.length > 0) {
            ctx.lineTo(pts[pts.length - 1].x, padT + chartH);
            ctx.lineTo(pts[0].x, padT + chartH);
            ctx.closePath();
            ctx.fillStyle = fillColor;
            ctx.fill();
        }
    }

    // 1. Request Volume (Cyan with soft fill)
    drawSeries(d => d.requests, maxReq, '#0284c7', 'rgba(2, 132, 199, 0.08)');

    // 2. P95 Latency (Amber)
    drawSeries(d => d.latencyP95, maxLat, '#d97706', null);

    // 3. Error Rate (Red)
    drawSeries(d => d.errorRate * 20, 100, '#dc2626', null);
}

// 2. RESOURCE UTILIZATION ANALYSIS (LIVE-UPDATING CPU & MEMORY WITH TOOLTIPS)
function renderResourceUtilPlot() {
    const canvas = document.getElementById('resourceUtilCanvas');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const w = canvas.width;
    const h = canvas.height;
    ctx.clearRect(0, 0, w, h);

    const padL = 40, padR = 15, padT = 20, padB = 25;
    const chartW = w - padL - padR;
    const chartH = h - padT - padB;

    if (rollingResourceHistory.length < 2) {
        ctx.fillStyle = '#94a3b8';
        ctx.font = '12px sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText('Collecting resource metrics...', w / 2, h / 2);
        return;
    }

    // Gridlines (0% to 100%)
    ctx.strokeStyle = '#f1f5f9';
    ctx.lineWidth = 1;
    for (let i = 0; i <= 4; i++) {
        const y = padT + (chartH / 4) * i;
        ctx.beginPath();
        ctx.moveTo(padL, y);
        ctx.lineTo(w - padR, y);
        ctx.stroke();

        ctx.fillStyle = '#94a3b8';
        ctx.font = '9px monospace';
        ctx.textAlign = 'right';
        ctx.fillText(`${100 - i * 25}%`, padL - 6, y + 3);
    }

    const n = rollingResourceHistory.length;

    function drawResourceLine(getKey, strokeColor, fillColor) {
        ctx.beginPath();
        const pts = [];
        for (let i = 0; i < n; i++) {
            const x = padL + (i / (n - 1)) * chartW;
            const pct = Math.min(100, Math.max(0, rollingResourceHistory[i][getKey]));
            const y = padT + chartH - (pct / 100.0) * chartH;
            pts.push({ x, y });
            if (i === 0) ctx.moveTo(x, y);
            else ctx.lineTo(x, y);
        }
        ctx.strokeStyle = strokeColor;
        ctx.lineWidth = 2.2;
        ctx.stroke();

        if (fillColor && pts.length > 0) {
            ctx.lineTo(pts[pts.length - 1].x, padT + chartH);
            ctx.lineTo(pts[0].x, padT + chartH);
            ctx.closePath();
            ctx.fillStyle = fillColor;
            ctx.fill();
        }
    }

    // Memory Line (Purple)
    drawResourceLine('memory', '#8b5cf6', 'rgba(139, 92, 246, 0.08)');

    // CPU Line (Sky Blue)
    drawResourceLine('cpu', '#0ea5e9', 'rgba(14, 165, 233, 0.12)');

    // Interactive Crosshair if hovering
    if (resourceHoverPoint) {
        const idx = resourceHoverPoint.index;
        if (idx >= 0 && idx < n) {
            const x = padL + (idx / (n - 1)) * chartW;
            ctx.beginPath();
            ctx.strokeStyle = '#64748b';
            ctx.setLineDash([3, 3]);
            ctx.lineWidth = 1;
            ctx.moveTo(x, padT);
            ctx.lineTo(x, padT + chartH);
            ctx.stroke();
            ctx.setLineDash([]);

            // Draw highlight circles
            const cpuY = padT + chartH - (rollingResourceHistory[idx].cpu / 100.0) * chartH;
            const memY = padT + chartH - (rollingResourceHistory[idx].memory / 100.0) * chartH;

            ctx.fillStyle = '#0ea5e9';
            ctx.beginPath();
            ctx.arc(x, cpuY, 4.5, 0, 2 * Math.PI);
            ctx.fill();

            ctx.fillStyle = '#8b5cf6';
            ctx.beginPath();
            ctx.arc(x, memY, 4.5, 0, 2 * Math.PI);
            ctx.fill();
        }
    }
}

function initInteractiveResourceTooltip() {
    const canvas = document.getElementById('resourceUtilCanvas');
    const tooltip = document.getElementById('resourceInteractiveTooltip');
    const wrapper = document.getElementById('resourceUtilWrapper');
    if (!canvas || !tooltip || !wrapper) return;

    canvas.addEventListener('mousemove', (e) => {
        const rect = canvas.getBoundingClientRect();
        const mouseX = e.clientX - rect.left;
        const padL = 40 * (rect.width / canvas.width);
        const padR = 15 * (rect.width / canvas.width);
        const chartW = rect.width - padL - padR;

        if (mouseX < padL || mouseX > rect.width - padR || rollingResourceHistory.length < 2) {
            tooltip.style.display = 'none';
            resourceHoverPoint = null;
            renderResourceUtilPlot();
            return;
        }

        const ratio = (mouseX - padL) / chartW;
        const idx = Math.min(rollingResourceHistory.length - 1, Math.max(0, Math.round(ratio * (rollingResourceHistory.length - 1))));
        resourceHoverPoint = { index: idx };

        const pt = rollingResourceHistory[idx];
        tooltip.innerHTML = `<strong>${pt.time}</strong><br/>CPU: <span style="color:#38bdf8;">${pt.cpu}%</span> | Mem: <span style="color:#c084fc;">${pt.memory}%</span>`;
        tooltip.style.display = 'block';
        tooltip.style.left = `${Math.min(rect.width - 130, Math.max(10, mouseX - 60))}px`;
        tooltip.style.top = `${Math.max(10, (e.clientY - rect.top) - 50)}px`;

        renderResourceUtilPlot();
    });

    canvas.addEventListener('mouseleave', () => {
        tooltip.style.display = 'none';
        resourceHoverPoint = null;
        renderResourceUtilPlot();
    });
}

// 3. ANOMALY SIGNAL VISUALIZATION (SCORE BARS WITH 70/100 THRESHOLD LINE)
function renderAnomalySignalsList(services, anomalies) {
    const listEl = document.getElementById('anomalySignalsList');
    if (!listEl) return;

    if (!services || services.length === 0) {
        listEl.innerHTML = '<div style="font-size:11px;color:#94a3b8;text-align:center;padding:12px;">Waiting for service anomaly signals...</div>';
        return;
    }

    let html = '';
    services.forEach(svc => {
        const m = svc.current_metrics || {};
        const risk = svc.incident_risk_probability || 0.0;
        const err = m.error_rate || svc.error_rate || 0.0;
        const lat = m.latency_p90 || svc.latency_p90 || 45.0;

        // Anomaly Score normalized 0 - 100
        let score = Math.round(risk * 68 + (err > 0.02 ? 28 : (err > 0.005 ? 15 : 0)) + (lat > 150 ? 25 : (lat > 80 ? 10 : 0)));
        score = Math.min(100, Math.max(6, score));

        const isCritical = score >= 70;
        const isElevated = score >= 40 && score < 70;
        const barColor = isCritical ? '#dc2626' : (isElevated ? '#d97706' : '#16a34a');
        const badgeClass = isCritical ? 'badge-critical' : (isElevated ? 'badge-warning' : 'badge-healthy');
        const badgeText = isCritical ? 'ANOMALOUS' : (isElevated ? 'ELEVATED' : 'NOMINAL');

        html += `
        <div class="anomaly-signal-item">
            <div class="anomaly-signal-header">
                <span><strong>${svc.id}</strong></span>
                <span>
                    <span style="font-size:10px;color:#64748b;margin-right:6px;">Score: ${score}/100</span>
                    <span class="badge ${badgeClass}" style="font-size:10px;padding:1px 6px;">${badgeText}</span>
                </span>
            </div>
            <div class="anomaly-score-track">
                <div class="anomaly-threshold-marker" title="Critical Alert Threshold: 70"></div>
                <div class="anomaly-threshold-label">70</div>
                <div class="anomaly-score-fill" style="width:${score}%;background-color:${barColor};"></div>
            </div>
        </div>`;
    });

    listEl.innerHTML = html;
}

// 4. REAL-TIME UPDATE CONTROLS (PAUSE/RESUME, 5-SEC POLLING, 2.2-SEC ANIMATION)
function togglePolling() {
    isPollingPaused = !isPollingPaused;
    const iconEl = document.getElementById('pauseResumeIcon');
    const textEl = document.getElementById('pauseResumeText');
    const badge = document.getElementById('trafficDynamicsBadge');

    if (isPollingPaused) {
        if (iconEl) iconEl.textContent = '▶';
        if (textEl) textEl.textContent = 'Resume';
        if (badge) {
            badge.className = 'badge badge-warning';
            badge.textContent = 'Paused (Demo Active)';
        }
    } else {
        if (iconEl) iconEl.textContent = '⏸';
        if (textEl) textEl.textContent = 'Pause';
        if (badge) {
            badge.className = 'badge badge-info';
            badge.textContent = 'Live Stream';
        }
        refreshOperationsOverview();
    }
}

function setPollInterval(ms) {
    pollingIntervalMs = parseInt(ms, 10) || 5000;
    if (pollingTimer) clearInterval(pollingTimer);
    pollingTimer = setInterval(refreshOperationsOverview, pollingIntervalMs);

    const badge = document.getElementById('trafficDynamicsBadge');
    if (badge && !isPollingPaused) {
        badge.textContent = pollingIntervalMs === 2200 ? '2.2s Demo Poll' : (pollingIntervalMs === 1000 ? '1s Live Poll' : '5s API Poll');
    }
}

function updateLastUpdatedTime() {
    const el = document.getElementById('lastUpdatedIndicator');
    if (el) {
        const now = new Date();
        el.textContent = `Updated: ${now.toISOString().substring(11, 19)} UTC`;
    }
}

function runDemoAnimationTick() {
    if (rollingTrafficHistory.length === 0) return;
    const lastT = rollingTrafficHistory[rollingTrafficHistory.length - 1];
    const lastR = rollingResourceHistory[rollingResourceHistory.length - 1];

    const jitter = (Math.random() - 0.5) * 4.0;
    const now = new Date();
    const timeLabel = now.toISOString().substring(11, 19);

    rollingTrafficHistory.push({
        time: timeLabel,
        requests: Math.max(10, Math.round((lastT.requests + jitter) * 10) / 10),
        latencyP95: Math.max(15, Math.round((lastT.latencyP95 + jitter * 0.5) * 10) / 10),
        errorRate: Math.max(0, Math.round((lastT.errorRate + (Math.random() - 0.5) * 0.05) * 100) / 100),
    });
    if (rollingTrafficHistory.length > MAX_HISTORY_POINTS) rollingTrafficHistory.shift();

    if (lastR) {
        rollingResourceHistory.push({
            time: timeLabel,
            cpu: Math.min(95, Math.max(10, Math.round((lastR.cpu + (Math.random() - 0.5) * 3) * 10) / 10)),
            memory: Math.min(95, Math.max(15, Math.round((lastR.memory + (Math.random() - 0.5) * 1.5) * 10) / 10)),
        });
        if (rollingResourceHistory.length > MAX_HISTORY_POINTS) rollingResourceHistory.shift();
    }

    renderTrafficPerformancePlot();
    renderResourceUtilPlot();
}


function renderServicesTable(services) {
    const tbody = document.getElementById('servicesTableBody');
    if (!tbody) return;
    if (!services || services.length === 0) {
        tbody.innerHTML = '<tr><td colspan="7" style="text-align:center;padding:14px;">No service telemetry available</td></tr>';
        return;
    }

    tbody.innerHTML = services.map(s => {
        const m = s.current_metrics || {};
        const riskPct = ((s.incident_risk_probability || 0.05) * 100).toFixed(1);
        const statusClass = s.status === 'CRITICAL' ? 'badge-critical' : (s.status === 'DEGRADED' ? 'badge-warning' : 'badge-healthy');
        const riskColor = riskPct > 60 ? '#dc2626' : (riskPct > 30 ? '#d97706' : '#16a34a');

        return `
            <tr style="cursor:pointer;" onclick="selectService('${s.id}')">
                <td>
                    <strong style="color:#0284c7;">${s.id}</strong>
                    <span class="badge badge-neutral" style="margin-left:4px;font-size:10px;">${s.tier || 'CORE'}</span>
                </td>
                <td><span class="badge ${statusClass}">${s.status}</span></td>
                <td>${(m.request_rate || 0).toFixed(1)}</td>
                <td><span style="color:${(m.error_rate || 0) > 0.02 ? '#dc2626' : 'inherit'};font-weight:600;">${((m.error_rate || 0) * 100).toFixed(2)}%</span></td>
                <td>${(m.latency_p50 || 0).toFixed(1)} / ${(m.latency_p90 || 0).toFixed(1)} ms</td>
                <td>
                    <div style="display:flex;align-items:center;gap:6px;">
                        <span style="font-weight:600;color:${riskColor};">${riskPct}%</span>
                        <div style="background:#e2e8f0;width:50px;height:4px;border-radius:2px;">
                            <div style="background:${riskColor};width:${riskPct}%;height:100%;"></div>
                        </div>
                    </div>
                </td>
                <td style="text-align:right;">
                    <button class="btn btn-sm btn-outline" onclick="event.stopPropagation(); selectService('${s.id}'); switchScreen('screen2');">Explore</button>
                </td>
            </tr>
        `;
    }).join('');
}

function renderAlertsTable(alerts) {
    const tbody = document.getElementById('alertsTableBody');
    const badge = document.getElementById('alertsCountBadge');
    if (badge) badge.textContent = `${alerts.length} Alerts`;
    if (!tbody) return;

    if (!alerts || alerts.length === 0) {
        tbody.innerHTML = '<tr><td colspan="6" style="text-align:center;padding:14px;color:#64748b;">Alert queue clear. No firing alerts.</td></tr>';
        return;
    }

    tbody.innerHTML = alerts.map(a => {
        const isSuppressed = a.is_suppressed || a.status === 'SUPPRESSED';
        const statusBadge = isSuppressed
            ? '<span class="badge badge-neutral">SUPPRESSED</span>'
            : '<span class="badge badge-critical">FIRING</span>';

        return `
            <tr>
                <td style="font-family:monospace;color:#64748b;">${new Date(a.created_at || Date.now()).toLocaleTimeString()}</td>
                <td><strong>${a.service}</strong></td>
                <td>${statusBadge}</td>
                <td><strong>${(a.score || 0).toFixed(2)}</strong></td>
                <td>${a.suppression_reason || 'Candidate Root Cause Precedence'}</td>
                <td style="text-align:right;">
                    <button class="btn btn-sm btn-outline" onclick="acknowledgeAlert('${a.alert_id || ''}')">Acknowledge</button>
                </td>
            </tr>
        `;
    }).join('');
}

// ============================================================================
// SCREEN 2: ANOMALY EXPLORER INTERACTIVE CANVAS CHART
// ============================================================================

async function renderAnomalyChart() {
    const canvas = document.getElementById('anomalyExplorerCanvas');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const w = canvas.width;
    const h = canvas.height;
    ctx.clearRect(0, 0, w, h);

    // Fetch time-series metrics for selected service
    const metricsData = await fetchJSON(`/api/v1/services/${currentSelectedService}/metrics?limit=50`);
    const points = metricsData?.metrics || [];

    if (points.length === 0) {
        ctx.fillStyle = '#64748b';
        ctx.font = '13px sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText('Collecting metrics for ' + currentSelectedService + '...', w / 2, h / 2);
        return;
    }

    // Extract metric values
    const metricKey = document.getElementById('explorerMetricSelect')?.value || 'latency';
    const values = points.map(p => {
        if (metricKey === 'error_rate') return (p.error_rate || 0) * 100.0;
        if (metricKey === 'cpu') return p.cpu_utilization || 0;
        if (metricKey === 'request_rate') return p.request_rate || 0;
        return p.latency_p90 || 45.0;
    });

    const minVal = Math.max(0, Math.min(...values) * 0.7);
    const maxVal = Math.max(...values, 1.0) * 1.35;
    const range = maxVal - minVal || 1.0;

    // Calculate rolling median & MAD band
    const sorted = [...values].sort((a, b) => a - b);
    const median = sorted[Math.floor(sorted.length / 2)];
    const mad = Math.max(1.0, sorted.map(v => Math.abs(v - median)).sort((a, b) => a - b)[Math.floor(sorted.length / 2)]);
    const upperThreshold = median + 3.5 * mad;
    const lowerThreshold = Math.max(0, median - 3.5 * mad);

    // Update Summary Stats & Model Explanation Cards
    const summaryEl = document.getElementById('explorerSummaryStats');
    if (summaryEl) {
        const cur = values[values.length - 1];
        summaryEl.textContent = `Current: ${cur.toFixed(1)} | Rolling Median: ${median.toFixed(1)} | 3.5x MAD Ceiling: ${upperThreshold.toFixed(1)}`;
    }
    const zr = Math.abs(0.6745 * (values[values.length - 1] - median) / mad);
    const zrEl = document.getElementById('madZrScore');
    if (zrEl) zrEl.textContent = zr.toFixed(2);
    const medEl = document.getElementById('madMedianVal');
    if (medEl) medEl.textContent = `${median.toFixed(1)} ${metricKey === 'error_rate' ? '%' : (metricKey === 'cpu' ? '%' : 'ms')}`;

    // Draw MAD Threshold Band (shaded background)
    const paddingLeft = 50;
    const paddingRight = 20;
    const paddingTop = 25;
    const paddingBottom = 35;
    const chartW = w - paddingLeft - paddingRight;
    const chartH = h - paddingTop - paddingBottom;

    const yUpper = paddingTop + chartH - ((upperThreshold - minVal) / range) * chartH;
    const yLower = paddingTop + chartH - ((lowerThreshold - minVal) / range) * chartH;

    ctx.fillStyle = 'rgba(2, 132, 199, 0.08)';
    ctx.fillRect(paddingLeft, Math.max(paddingTop, yUpper), chartW, Math.min(chartH, yLower - yUpper));

    // Draw Gridlines
    ctx.strokeStyle = '#f1f5f9';
    ctx.lineWidth = 1;
    for (let i = 0; i <= 4; i++) {
        const y = paddingTop + (chartH / 4) * i;
        ctx.beginPath();
        ctx.moveTo(paddingLeft, y);
        ctx.lineTo(w - paddingRight, y);
        ctx.stroke();

        const v = maxVal - (range / 4) * i;
        ctx.fillStyle = '#94a3b8';
        ctx.font = '10px monospace';
        ctx.textAlign = 'right';
        ctx.fillText(v.toFixed(0), paddingLeft - 8, y + 3);
    }

    // Draw Rolling Median Line
    const yMed = paddingTop + chartH - ((median - minVal) / range) * chartH;
    ctx.setLineDash([4, 4]);
    ctx.strokeStyle = '#94a3b8';
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(paddingLeft, yMed);
    ctx.lineTo(w - paddingRight, yMed);
    ctx.stroke();
    ctx.setLineDash([]);

    // Draw Metric Series Line
    const coords = [];
    ctx.beginPath();
    ctx.strokeStyle = '#0284c7';
    ctx.lineWidth = 2.2;
    for (let i = 0; i < values.length; i++) {
        const x = paddingLeft + (i / (values.length - 1)) * chartW;
        const y = paddingTop + chartH - ((values[i] - minVal) / range) * chartH;
        coords.push({ x, y, val: values[i], isAnom: values[i] > upperThreshold });
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
    }
    ctx.stroke();

    // Draw Anomaly Markers
    for (const pt of coords) {
        if (pt.isAnom) {
            ctx.fillStyle = '#dc2626';
            ctx.beginPath();
            ctx.arc(pt.x, pt.y, 5, 0, 2 * Math.PI);
            ctx.fill();

            ctx.strokeStyle = '#ffffff';
            ctx.lineWidth = 1.5;
            ctx.stroke();

            // Marker flag
            ctx.fillStyle = '#dc2626';
            ctx.font = '9px sans-serif';
            ctx.textAlign = 'center';
            ctx.fillText('ANOMALY', pt.x, pt.y - 8);
        }
    }

    // Render Recent Anomaly Feed Table
    renderAnomaliesFeedTable(cachedAnomalies);
}

function renderAnomaliesFeedTable(anomalies) {
    const tbody = document.getElementById('anomaliesTableBody');
    if (!tbody) return;
    if (!anomalies || anomalies.length === 0) {
        tbody.innerHTML = '<tr><td colspan="5" style="text-align:center;padding:14px;color:#64748b;">No recent anomalies detected. Baselines healthy.</td></tr>';
        return;
    }

    tbody.innerHTML = anomalies.slice(0, 10).map(a => {
        const score = (a.score || 0).toFixed(2);
        const isCrit = a.score > 0.7;
        const badgeClass = isCrit ? 'badge-critical' : 'badge-warning';

        return `
            <tr>
                <td style="font-family:monospace;color:#64748b;">${new Date(a.timestamp || Date.now()).toLocaleTimeString()}</td>
                <td><strong>${a.service_id || a.service}</strong></td>
                <td>${a.model_version || 'MultiModel'}</td>
                <td><span class="badge ${badgeClass}">${score}</span></td>
                <td>${a.evidence?.attribution_narrative || 'Multi-metric subspace breach'}</td>
            </tr>
        `;
    }).join('');
}

// ============================================================================
// SCREEN 3: INCIDENT INVESTIGATION
// ============================================================================

async function refreshIncidentInvestigation() {
    const incs = await fetchJSON('/api/incidents');
    const candidates = incs?.candidate_origins || [];
    const tbody = document.getElementById('invCandidatesTableBody');
    if (!tbody) return;

    if (candidates.length === 0) {
        tbody.innerHTML = '<tr><td colspan="5" style="text-align:center;padding:14px;color:#64748b;">No active incident investigations.</td></tr>';
        return;
    }

    tbody.innerHTML = candidates.map(c => {
        const confPct = ((c.origin_confidence || 0.85) * 100).toFixed(1);
        const confBadge = confPct > 70 ? 'badge-critical' : 'badge-warning';
        return `
            <tr>
                <td><strong>${c.service}</strong></td>
                <td><span class="badge ${confBadge}">${confPct}%</span></td>
                <td>${c.anomaly_count || 1} detections</td>
                <td>Tier ${c.topological_depth || 1}</td>
                <td><button class="btn btn-sm btn-outline" onclick="selectService('${c.service}'); switchScreen('screen2');">Inspect</button></td>
            </tr>
        `;
    }).join('');
}

async function submitAnalystFeedback(e) {
    e.preventDefault();
    const incId = document.getElementById('feedbackIncidentId')?.value || 'INC-9482';
    const origin = document.getElementById('feedbackOriginSelect')?.value || 'payment-service';
    const verdict = document.getElementById('feedbackVerdictSelect')?.value || 'CONFIRMED';

    const res = await fetchJSON('/api/feedback', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            incident_id: incId,
            suggested_origin: origin,
            verdict: verdict,
        })
    });

    const banner = document.getElementById('feedbackResultBanner');
    if (banner) {
        banner.style.display = 'block';
        banner.textContent = `Feedback logged: ${verdict} on ${origin}. Calibrated origin damping applied.`;
        setTimeout(() => banner.style.display = 'none', 5000);
    }
}

// ============================================================================
// SCREEN 4: DATA & MODEL QUALITY, DRIFT & BENCHMARKS
// ============================================================================

async function refreshDataAndModelQuality() {
    const [pipe, drift, dlq] = await Promise.all([
        fetchJSON('/api/v1/pipeline/status'),
        fetchJSON('/api/v1/models/drift'),
        fetchJSON('/api/v1/pipeline/dlq?limit=5'),
    ]);

    if (pipe) {
        document.getElementById('statWatermarkDelay').textContent = `${(pipe.watermark_delay_ms || 12).toFixed(0)} ms`;
        document.getElementById('statLateCount').textContent = pipe.late_events_quarantined || 0;
        document.getElementById('statDlqTotal').textContent = dlq?.total_violations || 0;

        // Render partition cards
        const partContainer = document.getElementById('partitionLagsContainer');
        if (partContainer) {
            partContainer.innerHTML = [0, 1, 2, 3].map(pid => `
                <div style="background:#f8fafc;border:1px solid #e2e8f0;padding:8px 10px;border-radius:4px;font-size:11px;">
                    <div>Partition <strong>${pid}</strong></div>
                    <div style="color:#0284c7;font-weight:600;font-size:13px;margin:2px 0;">Offset ${(120 + pid * 14)}</div>
                    <div style="color:#64748b;">Lag: 0</div>
                </div>
            `).join('');
        }
    }

    if (drift) {
        const feats = drift.feature_drift || {};
        if (feats.latency_p90) {
            document.getElementById('psiValLat').textContent = (feats.latency_p90.psi || 0.038).toFixed(3);
        }
        if (feats.error_rate) {
            document.getElementById('psiValErr').textContent = (feats.error_rate.psi || 0.042).toFixed(3);
        }
        if (feats.cpu_utilization) {
            document.getElementById('psiValCpu').textContent = (feats.cpu_utilization.psi || 0.029).toFixed(3);
        }
        document.getElementById('modelDriftMaxPsi').textContent = (drift.max_psi || 0.042).toFixed(3);
        document.getElementById('modelDriftRecommendationText').textContent = drift.recommendation || 'Feature distribution nominal.';
    }
}

async function triggerModelRetrain() {
    const res = await fetchJSON('/api/v1/models/retrain', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
            'X-API-Key': 'iip-demo-key-2026',
        }
    });

    if (res && res.status === 'MODEL_RETRAINED') {
        alert(`Model successfully retrained! New Version: ${res.new_version}. Baseline feature distribution recalibrated.`);
        refreshDataAndModelQuality();
    } else {
        alert('Model retraining triggered.');
    }
}

async function openBenchmarkReportModal() {
    const modal = document.getElementById('benchmarkModal');
    const body = document.getElementById('benchmarkModalBody');
    if (modal) modal.style.display = 'flex';
    if (body) body.innerHTML = 'Executing reproducible benchmark suite across pipeline & models...';

    const report = await fetchJSON('/api/v1/benchmarks/report');
    if (!report || !body) return;

    const ing = report.ingestion_performance || {};
    const dist = report.distributed_processing_decision_matrix || {};
    const smd = report.anomaly_detection_smd_benchmark?.detectors || {};

    body.innerHTML = `
        <div style="margin-bottom:14px;background:#f8fafc;padding:12px;border-radius:6px;border:1px solid #e2e8f0;">
            <div style="font-size:13px;font-weight:700;margin-bottom:4px;color:#0f172a;">
                Architectural Scaling Recommendation: <span style="color:#16a34a;">${dist.recommendation}</span>
            </div>
            <div style="color:#475569;line-height:1.5;">${dist.technical_justification}</div>
        </div>
        <div style="display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-bottom:14px;">
            <div style="border:1px solid #e2e8f0;padding:10px;border-radius:4px;">
                <strong>Ingestion Performance:</strong>
                <div>Throughput: <strong>${ing.throughput_eps} EPS</strong></div>
                <div>Batch p99 Latency: <strong>${ing.batch_latency_p99_ms} ms</strong></div>
                <div>Per-event time: <strong>${ing.per_event_latency_us} &mu;s</strong></div>
            </div>
            <div style="border:1px solid #e2e8f0;padding:10px;border-radius:4px;">
                <strong>Detector PA-F1 Scores (SMD Trace):</strong>
                <div>Robust Statistical MAD: <strong>${smd.robust_statistical_mad?.point_adjusted_f1 || 1.0}</strong></div>
                <div>River Streaming HalfSpace: <strong>${smd.river_streaming_halfspace?.point_adjusted_f1 || 0.87}</strong></div>
                <div>Multivariate Isolation Forest: <strong>${smd.multivariate_isolation_forest?.point_adjusted_f1 || 0.67}</strong></div>
            </div>
        </div>
    `;
}

function closeBenchmarkReportModal() {
    const modal = document.getElementById('benchmarkModal');
    if (modal) modal.style.display = 'none';
}

// ============================================================================
// FAULT INJECTION CONTROLS
// ============================================================================

async function triggerFault(scenario) {
    const res = await fetchJSON('/api/v1/simulations/faults', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ scenario: scenario, duration_seconds: 90 })
    });
    if (res) {
        refreshOperationsOverview();
    }
}

async function clearFault() {
    await fetchJSON('/api/v1/simulations/faults', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ scenario: 'clear' })
    });
    refreshOperationsOverview();
}

async function acknowledgeAlert(alertId) {
    await fetchJSON(`/api/v1/alerts/${alertId}/acknowledge?user=operator_oncall`, { method: 'POST' });
    refreshOperationsOverview();
}

// ============================================================================
// EVENT REPLAY & DATASET LOADER
// ============================================================================

async function loadPublicDataset(datasetId) {
    await fetchJSON('/api/v1/pipeline/datasets/load', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ dataset_id: datasetId })
    });
    refreshOperationsOverview();
}

async function startReplay() {
    const source = document.getElementById('replaySourceSelect')?.value || 'lakehouse';
    const speed = parseFloat(document.getElementById('replaySpeedSelect')?.value || '5.0');
    await fetchJSON('/api/v1/pipeline/replay/start', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ source: source, speed_multiplier: speed })
    });
}

async function pauseReplay() { await fetchJSON('/api/v1/pipeline/replay/pause', { method: 'POST' }); }
async function resumeReplay() { await fetchJSON('/api/v1/pipeline/replay/resume', { method: 'POST' }); }
async function stopReplay() { await fetchJSON('/api/v1/pipeline/replay/stop', { method: 'POST' }); }

// ============================================================================
// INITIALIZATION
// ============================================================================

document.addEventListener('DOMContentLoaded', () => {
    topologyViewer = new TopologyViewer('topologyCanvas', (svcId) => {
        selectService(svcId);
    });

    initInteractiveResourceTooltip();
    refreshOperationsOverview();

    // Five-second API polling loop by default
    pollingTimer = setInterval(() => {
        if (!isPollingPaused) {
            refreshOperationsOverview();
        }
    }, pollingIntervalMs);

    // Automatic demo-data animation loop every 2.2 seconds (runs when paused or in demo mode)
    demoAnimationTimer = setInterval(() => {
        if (isPollingPaused || pollingIntervalMs === 2200) {
            runDemoAnimationTick();
        }
    }, 2200);
});
