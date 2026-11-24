/**
 * api.js
 * ───────
 * All HTTP calls to the FastAPI backend.
 * Base URL is auto-detected from window.location so the same JS works
 * whether the page is served on port 8000 (FastAPI static files) or
 * opened from a different port during development.
 */

const API_BASE = `${window.location.protocol}//${window.location.host}/api/v1`;

// ── Generic fetch helper ──────────────────────────────────────────────────────

async function apiFetch(path, options = {}) {
  const url = `${API_BASE}${path}`;
  const defaults = {
    headers: { 'Content-Type': 'application/json' },
  };
  const merged = { ...defaults, ...options };
  if (merged.body && typeof merged.body === 'object') {
    merged.body = JSON.stringify(merged.body);
  }

  const resp = await fetch(url, merged);

  if (!resp.ok) {
    let detail = `HTTP ${resp.status}`;
    try {
      const err = await resp.json();
      detail = err.detail || err.message || detail;
    } catch (_) {}
    throw new Error(detail);
  }

  // Handle binary responses (download endpoints)
  const ct = resp.headers.get('content-type') || '';
  if (ct.includes('application/octet-stream') ||
      ct.includes('application/vnd.openxmlformats') ||
      ct.includes('text/plain') && resp.headers.get('content-disposition')) {
    return resp;  // caller handles blob
  }

  return resp.json();
}


// ── Health ────────────────────────────────────────────────────────────────────

async function apiHealth() {
  return apiFetch('/health'.replace('/api/v1', ''), { method: 'GET',
    headers: { 'Content-Type': 'application/json' } });
}
// Health is at root, not under /api/v1
async function fetchHealth() {
  const resp = await fetch('/health');
  return resp.json();
}


// ── Providers ─────────────────────────────────────────────────────────────────

async function fetchProviders() {
  return apiFetch('/providers');
}

async function switchProvider(providerName) {
  return apiFetch('/providers/switch', {
    method: 'POST',
    body: { provider: providerName },
  });
}

async function pingProvider(providerName) {
  return apiFetch(`/providers/${encodeURIComponent(providerName)}/ping`);
}


// ── Analyse ───────────────────────────────────────────────────────────────────

async function analyzeText(text) {
  return apiFetch('/analyze', {
    method: 'POST',
    body: { text },
  });
}


// ── Humanize ──────────────────────────────────────────────────────────────────

/**
 * @param {Object} params
 * @param {string} params.text
 * @param {string} params.mode
 * @param {string} params.intensity
 * @param {string} params.granularity
 * @param {boolean} params.preserve_citations
 * @param {boolean} params.preserve_technical_terms
 * @param {boolean} params.preserve_original_voice
 * @param {string} params.english_variant
 * @returns {Promise<Object>} HumanizeResponse
 */
async function humanizeText(params) {
  return apiFetch('/humanize', {
    method: 'POST',
    body: params,
  });
}


// ── File Upload ───────────────────────────────────────────────────────────────

async function uploadFile(file) {
  const formData = new FormData();
  formData.append('file', file);

  const resp = await fetch(`${API_BASE}/upload`, {
    method: 'POST',
    body: formData,  // no Content-Type — let browser set multipart boundary
  });

  if (!resp.ok) {
    let detail = `HTTP ${resp.status}`;
    try { const e = await resp.json(); detail = e.detail || detail; } catch (_) {}
    throw new Error(detail);
  }

  return resp.json();
}


// ── Download ──────────────────────────────────────────────────────────────────

async function downloadAsTxt(text, filename = 'humanized_output') {
  const resp = await fetch(`${API_BASE}/download/txt`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ text, filename }),
  });
  if (!resp.ok) throw new Error('Download failed');
  const blob = await resp.blob();
  _triggerDownload(blob, `${filename}.txt`);
}

async function downloadAsDocx(text, filename = 'humanized_output', title = 'Humanized Output') {
  const resp = await fetch(`${API_BASE}/download/docx`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ text, filename, title }),
  });
  if (!resp.ok) throw new Error('Download failed');
  const blob = await resp.blob();
  _triggerDownload(blob, `${filename}.docx`);
}

function _triggerDownload(blob, filename) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  setTimeout(() => URL.revokeObjectURL(url), 5000);
}


// ── History ───────────────────────────────────────────────────────────────────

async function fetchHistory(page = 1, pageSize = 30) {
  return apiFetch(`/history?page=${page}&page_size=${pageSize}`);
}

async function fetchHistoryItem(id) {
  return apiFetch(`/history/${id}`);
}

async function deleteHistoryItem(id) {
  return apiFetch(`/history/${id}`, { method: 'DELETE' });
}

async function clearHistory() {
  return apiFetch('/history', { method: 'DELETE' });
}
