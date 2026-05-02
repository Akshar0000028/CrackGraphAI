/* =========================================================================
   CrackGraphAI — Dashboard JS
   Connects to FastAPI backend at API_BASE, renders all results.
   ========================================================================= */

const API_BASE = 'http://127.0.0.1:8000';

/* ── State ──────────────────────────────────────────────────────────────── */
let currentFile   = null;   // File object
let currentResult = null;   // Last API response
let imageDataMap  = {};     // { original, segmentation, skeleton, keypoints } → data-URL

/* =========================================================================
   API health check
   ========================================================================= */
async function checkApiHealth() {
  const dot  = document.getElementById('status-dot');
  const ring = document.getElementById('status-ring');
  const txt  = document.getElementById('status-text');
  try {
    const res = await fetch(`${API_BASE}/health`, { signal: AbortSignal.timeout(4000) });
    if (res.ok) {
      dot.className  = 'relative inline-flex rounded-full h-2.5 w-2.5 bg-green-400';
      ring.className = 'animate-ping absolute inline-flex h-full w-full rounded-full bg-green-400 opacity-75';
      txt.textContent = 'API online';
      txt.className   = 'text-green-400';
    } else {
      throw new Error(`HTTP ${res.status}`);
    }
  } catch {
    dot.className  = 'relative inline-flex rounded-full h-2.5 w-2.5 bg-red-400';
    ring.className = 'animate-ping absolute inline-flex h-full w-full rounded-full bg-red-400 opacity-75';
    txt.textContent = 'API offline';
    txt.className   = 'text-red-400';
  }
}

checkApiHealth();
setInterval(checkApiHealth, 30_000);

/* =========================================================================
   File handling
   ========================================================================= */
function handleDragOver(e) {
  e.preventDefault();
  document.getElementById('drop-zone').classList.add('drag-over');
}
function handleDragLeave(e) {
  document.getElementById('drop-zone').classList.remove('drag-over');
}
function handleDrop(e) {
  e.preventDefault();
  document.getElementById('drop-zone').classList.remove('drag-over');
  const file = e.dataTransfer.files[0];
  if (file) setFile(file);
}
function handleFileSelect(e) {
  const file = e.target.files[0];
  if (file) setFile(file);
}

function setFile(file) {
  const allowed = ['image/png', 'image/jpeg', 'image/jpg'];
  if (!allowed.includes(file.type)) {
    showError('Unsupported file type. Please upload PNG or JPG.');
    return;
  }
  if (file.size > 10 * 1024 * 1024) {
    showError('File too large. Maximum size is 10 MB.');
    return;
  }
  currentFile = file;

  // Show preview
  const reader = new FileReader();
  reader.onload = (e) => {
    const img = document.getElementById('preview-img');
    img.src = e.target.result;
    img.classList.remove('hidden');
    document.getElementById('drop-content').classList.add('hidden');
    imageDataMap.original = e.target.result;
  };
  reader.readAsDataURL(file);

  // Show file info
  document.getElementById('file-name').textContent = file.name;
  document.getElementById('file-size').textContent = `(${(file.size / 1024).toFixed(1)} KB)`;
  document.getElementById('file-info').classList.remove('hidden');

  hideError();
  hideResults();
}

function clearFile() {
  currentFile = null;
  currentResult = null;
  imageDataMap = {};
  document.getElementById('file-input').value = '';
  document.getElementById('preview-img').classList.add('hidden');
  document.getElementById('drop-content').classList.remove('hidden');
  document.getElementById('file-info').classList.add('hidden');
  hideResults();
  hideError();
}

/* =========================================================================
   Analysis
   ========================================================================= */
async function runAnalysis() {
  if (!currentFile) return;

  const btn = document.getElementById('analyse-btn');
  btn.disabled = true;
  btn.innerHTML = '<span class="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin inline-block"></span> Analysing…';

  hideError();
  hideResults();
  showProgress();

  const formData = new FormData();
  formData.append('image', currentFile, currentFile.name);

  try {
    const res = await fetch(`${API_BASE}/predict`, {
      method: 'POST',
      body: formData,
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: `HTTP ${res.status}` }));
      throw new Error(err.detail || `HTTP ${res.status}`);
    }

    currentResult = await res.json();
    renderResults(currentResult);

  } catch (err) {
    showError(err.message || 'Unknown error. Is the API running?');
  } finally {
    hideProgress();
    btn.disabled = false;
    btn.innerHTML = '<span>🔍</span> Analyse';
  }
}

/* =========================================================================
   Render results
   ========================================================================= */
function renderResults(r) {
  /* ── SI gauge ─────────────────────────────────────────────────────── */
  const si = r.si_score ?? 0;
  document.getElementById('si-value').textContent = si.toFixed(3);

  const circumference = 2 * Math.PI * 50; // 314.16
  const offset = circumference * (1 - si);
  const gaugeFill = document.getElementById('gauge-fill');
  gaugeFill.style.strokeDashoffset = offset;
  gaugeFill.style.stroke = siColor(si);

  /* ── Risk badge ───────────────────────────────────────────────────── */
  const risk = r.damage_metrics?.risk_level ?? 'Unknown';
  const badge = document.getElementById('risk-badge');
  badge.textContent = risk;
  badge.className = `mt-4 px-4 py-1.5 rounded-full text-xs font-600 uppercase tracking-wide ${riskBadgeClass(risk)}`;

  /* ── Damage bars ──────────────────────────────────────────────────── */
  const dm = r.damage_metrics ?? {};
  const bars = [
    { label: 'Crack Density',  key: 'density_damage',    color: 'bg-blue-500' },
    { label: 'Network',        key: 'network_damage',     color: 'bg-violet-500' },
    { label: 'Complexity',     key: 'complexity_damage',  color: 'bg-amber-500' },
    { label: 'Width',          key: 'width_damage',       color: 'bg-rose-500' },
  ];
  document.getElementById('damage-bars').innerHTML = bars.map(b => {
    const val = dm[b.key] ?? 0;
    const pct = (val * 100).toFixed(1);
    return `
      <div>
        <div class="flex justify-between text-xs text-slate-600 mb-1">
          <span>${b.label}</span><span class="font-600">${pct}%</span>
        </div>
        <div class="h-2 bg-slate-100 rounded-full overflow-hidden">
          <div class="${b.color} h-full rounded-full transition-all duration-700"
               style="width:${pct}%"></div>
        </div>
      </div>`;
  }).join('');

  const totalDmg = dm.total_damage ?? 0;
  document.getElementById('total-damage-val').textContent =
    `${(totalDmg * 100).toFixed(1)}% damage`;
  document.getElementById('total-damage-val').style.color = totalDmg > 0.6 ? '#dc2626' : totalDmg > 0.3 ? '#d97706' : '#16a34a';

  /* ── Metrics grid ─────────────────────────────────────────────────── */
  const gf = r.graph_features ?? {};
  const metrics = [
    { label: 'Connectivity',    value: (r.connectivity_score ?? 0).toFixed(3), unit: '/ 1.0' },
    { label: 'Crack Length',    value: Math.round(gf.total_crack_length ?? 0), unit: 'px' },
    { label: 'Branches',        value: Math.round(gf.num_branches ?? 0),       unit: '' },
    { label: 'Endpoints',       value: Math.round(gf.endpoints ?? 0),          unit: '' },
    { label: 'Junctions',       value: Math.round(gf.junctions ?? 0),          unit: '' },
    { label: 'Longest Path',    value: Math.round(gf.longest_path ?? 0),       unit: 'px' },
    { label: 'Graph Diameter',  value: Math.round(gf.graph_diameter ?? 0),     unit: 'px' },
    { label: 'Mean Degree',     value: (gf.mean_node_degree ?? 0).toFixed(2),  unit: '' },
  ];
  document.getElementById('metrics-grid').innerHTML = metrics.map(m => `
    <div class="bg-slate-50 rounded-xl p-4 border border-slate-100">
      <p class="text-xs text-slate-500 mb-1">${m.label}</p>
      <p class="text-xl font-700 text-slate-800">${m.value}
        ${m.unit ? `<span class="text-xs font-400 text-slate-400">${m.unit}</span>` : ''}
      </p>
    </div>`).join('');

  /* ── Visualisations ───────────────────────────────────────────────── */
  imageDataMap.segmentation = r.segmentation_mask_png_b64
    ? `data:image/png;base64,${r.segmentation_mask_png_b64}` : null;
  imageDataMap.skeleton = r.skeleton_png_b64
    ? `data:image/png;base64,${r.skeleton_png_b64}` : null;
  imageDataMap.keypoints = r.keypoints_overlay_png_b64
    ? `data:image/png;base64,${r.keypoints_overlay_png_b64}` : null;

  const visDefs = [
    { id: 'vis-original',      key: 'original',      label: 'Original Image',         icon: '🖼️' },
    { id: 'vis-segmentation',  key: 'segmentation',  label: 'Crack Segmentation',      icon: '🔍' },
    { id: 'vis-skeleton',      key: 'skeleton',       label: 'Structural Skeleton',     icon: '🦴' },
    { id: 'vis-keypoints',     key: 'keypoints',      label: 'Endpoints & Junctions',   icon: '📍' },
  ];
  visDefs.forEach(v => {
    const src = imageDataMap[v.key];
    document.getElementById(v.id).innerHTML = src
      ? `<div class="rounded-xl overflow-hidden border border-slate-100 bg-slate-900">
           <div class="px-3 py-2 bg-slate-800 flex items-center gap-2">
             <span class="text-sm">${v.icon}</span>
             <span class="text-xs font-500 text-slate-300">${v.label}</span>
           </div>
           <img src="${src}" alt="${v.label}"
                class="w-full object-contain img-zoom"
                onclick="openLightbox('${src}')" />
         </div>`
      : `<div class="rounded-xl border border-slate-100 bg-slate-50 flex items-center justify-center h-40 text-slate-400 text-xs">
           No data
         </div>`;
  });

  // Keypoints legend
  if (imageDataMap.keypoints) {
    document.getElementById('vis-keypoints').insertAdjacentHTML('beforeend', `
      <div class="flex gap-4 justify-center mt-2 text-xs text-slate-500">
        <span class="flex items-center gap-1"><span class="w-3 h-3 rounded-full bg-red-500 inline-block"></span> Endpoints</span>
        <span class="flex items-center gap-1"><span class="w-3 h-3 rounded-full bg-yellow-400 inline-block"></span> Junctions</span>
      </div>`);
  }

  updateCompare();

  /* ── Assessment ───────────────────────────────────────────────────── */
  const { text: assessText, border, bg } = riskAssessment(risk, si);
  document.getElementById('assessment-box').className =
    `rounded-xl p-4 border-l-4 text-sm leading-relaxed ${bg} ${border}`;
  document.getElementById('assessment-box').innerHTML = assessText;

  document.getElementById('req-id').textContent  = r.request_id ?? '—';
  document.getElementById('latency').textContent = r.latency_seconds != null
    ? `${r.latency_seconds.toFixed(2)} s` : '—';

  showResults();
}

/* =========================================================================
   Compare tab
   ========================================================================= */
function switchTab(tab, btn) {
  document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
  document.getElementById('tab-grid').classList.toggle('hidden', tab !== 'grid');
  document.getElementById('tab-compare').classList.toggle('hidden', tab !== 'compare');
  if (tab === 'compare') updateCompare();
}

function updateCompare() {
  const lKey = document.getElementById('compare-left')?.value;
  const rKey = document.getElementById('compare-right')?.value;
  renderComparePanel('compare-left-panel',  lKey);
  renderComparePanel('compare-right-panel', rKey);
}

function renderComparePanel(panelId, key) {
  const labels = { original:'Original', segmentation:'Segmentation', skeleton:'Skeleton', keypoints:'Keypoints' };
  const src = imageDataMap[key];
  document.getElementById(panelId).innerHTML = src
    ? `<div class="rounded-xl overflow-hidden border border-slate-200">
         <p class="text-xs font-500 text-slate-500 px-3 py-2 bg-slate-50 border-b border-slate-100">${labels[key]}</p>
         <img src="${src}" class="w-full object-contain img-zoom" onclick="openLightbox('${src}')" />
       </div>`
    : `<div class="rounded-xl border border-slate-100 bg-slate-50 flex items-center justify-center h-48 text-slate-400 text-xs">No data</div>`;
}

/* =========================================================================
   Lightbox
   ========================================================================= */
function openLightbox(src) {
  document.getElementById('lightbox-img').src = src;
  document.getElementById('lightbox').classList.remove('hidden');
  document.body.style.overflow = 'hidden';
}
function closeLightbox() {
  document.getElementById('lightbox').classList.add('hidden');
  document.body.style.overflow = '';
}
document.addEventListener('keydown', e => { if (e.key === 'Escape') closeLightbox(); });

/* =========================================================================
   Downloads
   ========================================================================= */
function downloadJSON() {
  if (!currentResult) return;
  const blob = new Blob([JSON.stringify(currentResult, null, 2)], { type: 'application/json' });
  triggerDownload(URL.createObjectURL(blob), `crack_analysis_${currentResult.request_id}.json`);
}

function downloadImage(key) {
  const src = imageDataMap[key];
  if (!src) return;
  triggerDownload(src, `crack_${key}_${currentResult?.request_id ?? 'export'}.png`);
}

function triggerDownload(href, filename) {
  const a = document.createElement('a');
  a.href = href; a.download = filename;
  document.body.appendChild(a); a.click();
  document.body.removeChild(a);
}

/* =========================================================================
   UI helpers
   ========================================================================= */
function showProgress()  { document.getElementById('progress-section').classList.remove('hidden'); }
function hideProgress()  { document.getElementById('progress-section').classList.add('hidden'); }
function showResults()   { document.getElementById('results').classList.remove('hidden'); }
function hideResults()   { document.getElementById('results').classList.add('hidden'); }
function showError(msg)  {
  document.getElementById('error-msg').textContent = msg;
  document.getElementById('error-banner').classList.remove('hidden');
}
function hideError()     { document.getElementById('error-banner').classList.add('hidden'); }

/* =========================================================================
   Colour / style helpers
   ========================================================================= */
function siColor(si) {
  if (si >= 0.85) return '#16a34a';
  if (si >= 0.70) return '#22c55e';
  if (si >= 0.50) return '#eab308';
  if (si >= 0.30) return '#f97316';
  return '#dc2626';
}

function riskBadgeClass(risk) {
  const map = {
    'Low':              'bg-green-100 text-green-700',
    'Moderate':         'bg-yellow-100 text-yellow-700',
    'High':             'bg-orange-100 text-orange-700',
    'Critical':         'bg-red-100 text-red-700',
    'Failure Imminent': 'bg-red-200 text-red-800',
  };
  return map[risk] ?? 'bg-slate-100 text-slate-600';
}

function riskAssessment(risk, si) {
  const map = {
    'Low': {
      text: '<strong>Structure is in good condition.</strong> Routine monitoring recommended. No immediate action required.',
      border: 'border-green-500', bg: 'bg-green-50',
    },
    'Moderate': {
      text: '<strong>Minor structural concerns detected.</strong> Schedule a professional inspection within 6 months.',
      border: 'border-yellow-500', bg: 'bg-yellow-50',
    },
    'High': {
      text: '<strong>Significant structural concerns.</strong> Professional assessment required within 1 month.',
      border: 'border-orange-500', bg: 'bg-orange-50',
    },
    'Critical': {
      text: '<strong>Severe structural damage.</strong> Immediate intervention advised. Restrict access if necessary.',
      border: 'border-red-500', bg: 'bg-red-50',
    },
    'Failure Imminent': {
      text: '<strong>⚠ Structural failure risk.</strong> Immediate evacuation and emergency repairs required.',
      border: 'border-red-700', bg: 'bg-red-100',
    },
  };
  return map[risk] ?? { text: 'Assessment unavailable.', border: 'border-slate-300', bg: 'bg-slate-50' };
}
