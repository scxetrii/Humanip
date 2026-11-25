/**
 * ui.js
 * ──────
 * Pure UI rendering functions — no business logic.
 * All functions take data and mutate the DOM.
 */

// ── Toast notifications ───────────────────────────────────────────────────────

const toastContainer = () => document.getElementById('toastContainer');

function showToast(message, type = 'info', duration = 3500) {
  const toast = document.createElement('div');
  toast.className = `toast ${type}`;

  const icon = { success: '✓', error: '✕', warning: '⚠', info: 'ℹ' }[type] || 'ℹ';
  toast.innerHTML = `<span>${icon}</span><span>${escHtml(message)}</span>`;

  toastContainer().appendChild(toast);
  setTimeout(() => {
    toast.style.opacity = '0';
    toast.style.transition = 'opacity 0.3s';
    setTimeout(() => toast.remove(), 300);
  }, duration);
}


// ── Loading overlay ───────────────────────────────────────────────────────────

function showLoading(label = 'Processing…', sub = '') {
  document.getElementById('loadingLabel').textContent = label;
  document.getElementById('loadingSub').textContent = sub;
  document.getElementById('loadingOverlay').hidden = false;
}

function hideLoading() {
  document.getElementById('loadingOverlay').hidden = true;
}


// ── In-page humanize progress bar ────────────────────────────────────────────

const _humanizeProgress = {
  current: 0,
  target: 0,
  active: false,
  done: false,
  timer: null,
};

function _hpRender() {
  const valueEl = document.getElementById('humanizeProgressValue');
  const fillEl = document.getElementById('humanizeProgressFill');
  if (!valueEl || !fillEl) return;
  const pct = Math.max(0, Math.min(100, Math.round(_humanizeProgress.current)));
  valueEl.textContent = pct;
  fillEl.style.width = `${pct}%`;
}

function _hpTick() {
  const s = _humanizeProgress;

  if (!s.active) {
    clearInterval(s.timer);
    s.timer = null;
    return;
  }

  if (s.current < s.target) {
    const step = s.done ? Math.max(1, Math.ceil((100 - s.current) / 5)) : 1;
    s.current = Math.min(s.target, s.current + step);
    _hpRender();

    if (s.done && s.current >= 100) {
      clearInterval(s.timer);
      s.timer = null;
      const wrap = document.getElementById('humanizeProgress');
      if (wrap && wrap.dataset.state !== 'error') wrap.dataset.state = 'complete';
    }
  } else if (!s.done && s.target < 90) {
    s.target += 1;
  }
}

function showHumanizeProgress(label) {
  const wrap = document.getElementById('humanizeProgress');
  if (!wrap) return;

  const s = _humanizeProgress;
  s.active = true;
  s.done = false;
  s.current = 0;
  s.target = 2;

  wrap.hidden = false;
  wrap.dataset.state = 'running';
  const labelEl = document.getElementById('humanizeProgressLabel');
  if (labelEl) labelEl.textContent = label || 'Humanizing…';
  _hpRender();

  if (!s.timer) s.timer = setInterval(_hpTick, 55);
}

function setHumanizeProgress(fraction, label) {
  const s = _humanizeProgress;
  if (!s.active) return;

  const pct = Math.max(0, Math.min(99, Math.round(fraction * 100)));
  if (pct > s.target) s.target = pct;

  if (label) {
    const labelEl = document.getElementById('humanizeProgressLabel');
    if (labelEl) labelEl.textContent = label;
  }
}

function finishHumanizeProgress(label) {
  const s = _humanizeProgress;
  if (!s.active) return;

  s.done = true;
  s.target = 100;

  if (label) {
    const labelEl = document.getElementById('humanizeProgressLabel');
    if (labelEl) labelEl.textContent = label;
  }
}

function failHumanizeProgress(label) {
  const s = _humanizeProgress;
  if (!s.active) return;

  const wrap = document.getElementById('humanizeProgress');
  if (wrap) wrap.dataset.state = 'error';
  const labelEl = document.getElementById('humanizeProgressLabel');
  if (labelEl) labelEl.textContent = label || 'Failed';

  s.done = true;
  s.target = 100;
}

function hideHumanizeProgress() {
  const wrap = document.getElementById('humanizeProgress');
  const s = _humanizeProgress;

  clearInterval(s.timer);
  s.timer = null;
  s.active = false;
  s.done = false;
  s.current = 0;
  s.target = 0;

  if (wrap) {
    wrap.hidden = true;
    wrap.dataset.state = '';
  }
  _hpRender();
}


// ── Stats bar ─────────────────────────────────────────────────────────────────

function updateInputStats(text) {
  const words = text.trim() ? text.trim().split(/\s+/).length : 0;
  const chars = text.length;
  const sentences = (text.match(/[.!?]+/g) || []).length;

  document.getElementById('inputWords').textContent = words.toLocaleString();
  document.getElementById('inputChars').textContent = chars.toLocaleString();
  document.getElementById('inputSentences').textContent = sentences.toLocaleString();
}

function updateOutputStats(response) {
  const words = response.humanized_text.trim().split(/\s+/).length;
  const chars = response.humanized_text.length;
  const ms = response.processing_time_ms;
  const time = ms >= 1000 ? `${(ms / 1000).toFixed(1)}s` : `${ms}ms`;

  document.getElementById('outputWords').textContent = words.toLocaleString();
  document.getElementById('outputChars').textContent = chars.toLocaleString();
  document.getElementById('processingTime').textContent = time;
}


// ── Human-likeness meter ──────────────────────────────────────────────────────

function renderHumanMeter(score) {
  const meter = document.getElementById('humanMeter');
  if (score === null || score === undefined || Number.isNaN(Number(score))) {
    meter.hidden = true;
    return;
  }

  const pct = Math.max(0, Math.min(100, Math.round(Number(score))));
  const band = pct >= 75 ? 'high' : pct >= 50 ? 'mid' : 'low';
  const verdict = band === 'high' ? 'Human-like' : band === 'mid' ? 'Mixed' : 'AI-like';

  const h = pct / 100;
  const mixedFrac = 0.5 * Math.max(0, 1 - 2 * Math.abs(h - 0.5));
  let humanPct = Math.round((h - mixedFrac / 2) * 100);
  let mixedPct = Math.round(mixedFrac * 100);
  let aiPct = 100 - humanPct - mixedPct;
  if (aiPct < 0) {
    mixedPct += aiPct;
    aiPct = 0;
  }

  meter.hidden = false;
  meter.dataset.band = band;
  meter.setAttribute('aria-valuenow', pct);
  meter.setAttribute(
    'aria-valuetext',
    `${pct}% human-likeness, ${verdict}. Estimated split: human-written ${humanPct}%, mixed ${mixedPct}%, AI-written ${aiPct}%.`
  );
  document.getElementById('humanMeterValue').textContent = pct;

  const verdictEl = document.getElementById('humanMeterVerdict');
  if (verdictEl) verdictEl.textContent = verdict;

  const humanEl = document.getElementById('humanMeterHuman');
  if (humanEl) humanEl.textContent = humanPct;
  const mixedEl = document.getElementById('humanMeterMixed');
  if (mixedEl) mixedEl.textContent = mixedPct;
  const aiEl = document.getElementById('humanMeterAi');
  if (aiEl) aiEl.textContent = aiPct;

  const needle = document.getElementById('humanMeterNeedle');
  if (needle) {
    const angle = -90 + pct * 1.8;
    needle.style.setProperty('--angle', `${angle.toFixed(1)}deg`);
  }
}


// ── Output text ───────────────────────────────────────────────────────────────

function renderOutput(text) {
  const el = document.getElementById('outputText');
  el.textContent = text;
  el.hidden = false;
  document.getElementById('outputEmpty').hidden = true;
}

function resetOutput() {
  document.getElementById('outputText').hidden = true;
  document.getElementById('outputEmpty').hidden = false;
  document.getElementById('outputText').textContent = '';
  document.getElementById('outputWords').textContent = '—';
  document.getElementById('outputChars').textContent = '—';
  document.getElementById('processingTime').textContent = '—';
  document.getElementById('copyOutput').disabled = true;
  document.getElementById('downloadBtn').disabled = true;
  renderHumanMeter(null);
  renderDiff([]);
  renderQuality(null);
  renderIssues([]);
  hideHumanizeProgress();
}


// ── Diff rendering ────────────────────────────────────────────────────────────

function renderDiff(segments) {
  const container = document.getElementById('diffContent');
  if (!segments || segments.length === 0) {
    container.innerHTML = '<p class="hint-center">Run humanization first to see the word-level comparison.</p>';
    return;
  }

  container.innerHTML = '';
  segments.forEach(seg => {
    const span = document.createElement('span');
    span.textContent = seg.text;
    if (seg.change_type === 'added') span.className = 'diff-added';
    else if (seg.change_type === 'removed') span.className = 'diff-removed';
    container.appendChild(span);
  });
}


// ── Quality gauge & metrics ───────────────────────────────────────────────────

const METRIC_LABELS = {
  semantic_similarity:     'Meaning Preserved',
  factual_preservation:    'Factual Accuracy',
  grammar_quality:         'Grammar Quality',
  readability_improvement: 'Readability Change',
  repetition_reduction:    'Repetition Reduced',
  citation_integrity:      'Citations Intact',
  sentence_variation:      'Sentence Variety',
  paragraph_coherence:     'Paragraph Flow',
};

const METRIC_COLORS = {
  semantic_similarity:     '#3b5bdb',
  factual_preservation:    '#12b76a',
  grammar_quality:         '#3b5bdb',
  readability_improvement: '#f79009',
  repetition_reduction:    '#12b76a',
  citation_integrity:      '#12b76a',
  sentence_variation:      '#3b5bdb',
  paragraph_coherence:     '#3b5bdb',
};

function renderQuality(quality) {
  const emptyEl = document.getElementById('qualityEmpty');
  const panelEl = document.getElementById('qualityPanel');

  if (!quality) {
    emptyEl.hidden = false;
    panelEl.hidden = true;
    return;
  }

  emptyEl.hidden = true;
  panelEl.hidden = false;

  // Gauge
  const score = quality.overall_score;
  document.getElementById('overallScore').textContent = Math.round(score);
  const gaugeFill = document.getElementById('gaugeFill');
  const totalLength = 157;
  const offset = totalLength - (totalLength * score / 100);
  gaugeFill.setAttribute('stroke-dashoffset', offset.toFixed(1));
  gaugeFill.setAttribute('stroke', score >= 75 ? '#12b76a' : score >= 50 ? '#f79009' : '#f04438');

  // Verdict
  const verdict =
    score >= 85 ? '✨ Excellent — natural, accurate, and well-written.' :
    score >= 70 ? '✅ Good — solid humanization with minor rough edges.' :
    score >= 55 ? '⚠️ Fair — some AI patterns may remain.' :
    '❌ Poor — review factual content carefully.';
  document.getElementById('qualityVerdict').textContent = verdict;

  // Metric bars
  const metricsEl = document.getElementById('qualityMetrics');
  metricsEl.innerHTML = '';

  for (const [key, label] of Object.entries(METRIC_LABELS)) {
    let raw = quality[key] ?? 0;
    // readability_improvement is signed (-0.5 to +0.5) → normalize
    let pct;
    if (key === 'readability_improvement') {
      pct = Math.round((raw + 0.5) * 100);
    } else {
      pct = Math.round(raw * 100);
    }
    pct = Math.max(0, Math.min(100, pct));

    const displayVal = key === 'readability_improvement'
      ? (raw > 0 ? `+${(raw * 100).toFixed(0)}%` : `${(raw * 100).toFixed(0)}%`)
      : `${pct}%`;

    const color = METRIC_COLORS[key] || '#3b5bdb';

    metricsEl.insertAdjacentHTML('beforeend', `
      <div class="metric-row">
        <div class="metric-header">
          <span class="metric-label">${label}</span>
          <span class="metric-value">${displayVal}</span>
        </div>
        <div class="metric-bar">
          <div class="metric-fill" style="width:${pct}%;background:${color};"></div>
        </div>
      </div>
    `);
  }

  // Warnings
  const warningsEl = document.getElementById('qualityWarnings');
  if (quality.warnings && quality.warnings.length > 0) {
    warningsEl.hidden = false;
    warningsEl.innerHTML = '<strong>⚠ Warnings:</strong><ul>' +
      quality.warnings.map(w => `<li>${escHtml(w)}</li>`).join('') + '</ul>';
  } else {
    warningsEl.hidden = true;
  }
}


// ── Issues panel ──────────────────────────────────────────────────────────────

function renderIssues(analysis) {
  const panel = document.getElementById('issuesPanel');

  // Collect all issues from the analysis object
  const issues = [];

  if (analysis) {
    const addIssues = (arr, type, severity) => {
      (arr || []).forEach(item => {
        const text = typeof item === 'string' ? item : item.phrase || JSON.stringify(item);
        issues.push({ type, text, severity });
      });
    };

    addIssues(analysis.excessive_transitions,     'Excessive Transition',     'medium');
    addIssues(analysis.unnatural_phrases,          'Unnatural Phrase',         'high');
    addIssues(analysis.generic_phrases,            'Generic AI Phrase',        'high');
    addIssues(analysis.overly_formal_expressions,  'Overly Formal',            'medium');
    addIssues(analysis.repetitive_openers,         'Repetitive Opener',        'medium');
    addIssues(analysis.weak_connections,           'Weak Connection',          'low');
    addIssues(analysis.redundant_explanations,     'Redundant Explanation',    'low');
    (analysis.repetitive_structures || []).forEach(r => {
      issues.push({
        type: 'Repetitive Structure',
        text: `"${r.phrase}" repeated ${r.count}× `,
        severity: r.count >= 3 ? 'high' : 'medium',
      });
    });
  }

  if (issues.length === 0) {
    panel.innerHTML = analysis
      ? '<p class="hint-center">✅ No significant AI-writing issues detected.</p>'
      : '<p class="hint-center">Run analysis or humanization to see detected writing issues.</p>';
    return;
  }

  panel.innerHTML = issues.map(issue => `
    <div class="issue-item severity-${issue.severity}">
      <div class="issue-type">${escHtml(issue.type)}</div>
      <div class="issue-desc">${escHtml(issue.text)}</div>
    </div>
  `).join('');
}


// ── Provider list (settings modal) ───────────────────────────────────────────

function renderProviders(data, onSwitch) {
  const list = document.getElementById('providerList');
  list.innerHTML = '';

  data.providers.forEach(p => {
    const isActive = p.name === data.active_provider;
    const statusLabel = p.is_configured ? 'Ready' : 'No API key';
    const statusClass = p.is_configured ? 'ok' : 'unconfigured';

    const item = document.createElement('div');
    item.className = `provider-item ${isActive ? 'active' : ''}`;
    item.innerHTML = `
      <div class="provider-info">
        <span class="provider-name">${escHtml(p.name)} ${p.is_local ? '🏠' : ''}</span>
        <span class="provider-model">${escHtml(p.model)}</span>
      </div>
      <span class="provider-status ${statusClass}">${statusLabel}</span>
    `;

    if (p.is_configured && !isActive) {
      item.addEventListener('click', () => onSwitch(p.name, item));
    }

    list.appendChild(item);
  });
}


// ── History list (drawer) ─────────────────────────────────────────────────────

function renderHistory(data, onSelect) {
  const listEl = document.getElementById('historyList');
  listEl.innerHTML = '';

  if (!data.items || data.items.length === 0) {
    listEl.innerHTML = '<p class="hint-center">No history yet.</p>';
    return;
  }

  data.items.forEach(item => {
    const div = document.createElement('div');
    div.className = 'history-item';
    const date = new Date(item.created_at).toLocaleString();
    div.innerHTML = `
      <div class="history-snippet">${escHtml(item.original_snippet || '(no preview)')}</div>
      <div class="history-meta">
        <span class="history-badge">${escHtml(item.mode)}</span>
        <span class="history-badge">${Math.round(item.overall_score || 0)}/100</span>
        <span>${date}</span>
      </div>
    `;
    div.addEventListener('click', () => onSelect(item.id));
    listEl.appendChild(div);
  });
}


// ── AI-pattern score badge ────────────────────────────────────────────────────

function updateAiScore(analysis) {
  const badge = document.getElementById('analysisBadge');
  const score = document.getElementById('aiScore');

  if (!analysis) {
    badge.hidden = true;
    return;
  }

  badge.hidden = false;
  const pct = Math.round((analysis.ai_pattern_score || 0) * 100);
  score.textContent = `${pct}%`;
  score.style.color = pct >= 60 ? '#f04438' : pct >= 30 ? '#f79009' : '#12b76a';
}


// ── Tab switching ─────────────────────────────────────────────────────────────

function activateTab(tabId) {
  document.querySelectorAll('.tab-btn').forEach(btn => {
    btn.classList.toggle('active', btn.dataset.tab === tabId);
  });
  document.querySelectorAll('.tab-content').forEach(content => {
    content.classList.toggle('active', content.id === `tab-${tabId}`);
  });
}


// ── Helpers ───────────────────────────────────────────────────────────────────

function escHtml(str) {
  if (!str) return '';
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

function countWords(text) {
  return text.trim() ? text.trim().split(/\s+/).length : 0;
}
