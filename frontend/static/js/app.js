/**
 * app.js
 * ───────
 * Main application controller.
 * Wires DOM events to API calls and UI rendering.
 * Manages undo/redo stack and application state.
 */

'use strict';

// ── Application State ─────────────────────────────────────────────────────────

const state = {
  mode:        'natural',
  intensity:   'balanced',
  granularity: 'document',
  variant:     'american',
  preserveCitations:  true,
  preserveTerms:      true,
  preserveVoice:      false,

  currentOutput: null,       // last HumanizeResponse
  currentAnalysis: null,     // last TextAnalysis
  activeProvider: null,      // currently active provider name

  // Undo / Redo stacks (each entry = { input, output, analysis })
  undoStack: [],
  redoStack: [],
};


// ── Boot ──────────────────────────────────────────────────────────────────────

document.addEventListener('DOMContentLoaded', async () => {
  setupControls();
  setupEditor();
  setupOutput();
  setupModals();
  setupDragDrop();
  await loadProviderStatus();
});


// ── Provider status ───────────────────────────────────────────────────────────

async function loadProviderStatus() {
  try {
    const data = await fetchProviders();
    state.activeProvider = data.active_provider;
    document.getElementById('providerName').textContent =
      `${data.active_provider} · ${(data.providers.find(p => p.name === data.active_provider) || {}).model || ''}`;
    document.querySelector('.provider-dot').style.background = '#12b76a';
  } catch (e) {
    document.getElementById('providerName').textContent = 'API offline';
    document.querySelector('.provider-dot').style.background = '#f04438';
  }
}


// ── Controls ──────────────────────────────────────────────────────────────────

function setupControls() {
  // Mode buttons
  document.querySelectorAll('.mode-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.mode-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      state.mode = btn.dataset.mode;
    });
  });

  // Intensity buttons
  const intensityHints = {
    light:    'Minor polish — fixes obvious AI patterns with minimal restructuring.',
    balanced: 'Moderate rewrite — fixes AI patterns while keeping structure.',
    strong:   'Full rewrite — maximum humanization, varied rhythm and structure.',
  };
  document.querySelectorAll('.intensity-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.intensity-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      state.intensity = btn.dataset.intensity;
      document.getElementById('intensityHint').textContent = intensityHints[state.intensity];
    });
  });

  // Granularity
  document.getElementById('granularity').addEventListener('change', e => {
    state.granularity = e.target.value;
  });

  // Toggles
  document.getElementById('toggleCitations').addEventListener('change', e => {
    state.preserveCitations = e.target.checked;
  });
  document.getElementById('toggleTerms').addEventListener('change', e => {
    state.preserveTerms = e.target.checked;
  });
  document.getElementById('toggleVoice').addEventListener('change', e => {
    state.preserveVoice = e.target.checked;
  });

  // Variant buttons
  document.querySelectorAll('.variant-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.variant-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      state.variant = btn.dataset.variant;
    });
  });

  // Humanize button
  document.getElementById('humanizeBtn').addEventListener('click', runHumanize);

  // Analyse only
  document.getElementById('analyzeBtn').addEventListener('click', runAnalyse);

  // Undo / Redo
  document.getElementById('undoBtn').addEventListener('click', doUndo);
  document.getElementById('redoBtn').addEventListener('click', doRedo);

  // Tabs
  document.querySelectorAll('.tab-btn').forEach(btn => {
    btn.addEventListener('click', () => activateTab(btn.dataset.tab));
  });
}


// ── Editor ────────────────────────────────────────────────────────────────────

function setupEditor() {
  const textarea = document.getElementById('inputText');
  const humanizeBtn = document.getElementById('humanizeBtn');
  const clearBtn = document.getElementById('clearInput');
  const fileInput = document.getElementById('fileInput');

  // Real-time stats
  textarea.addEventListener('input', () => {
    updateInputStats(textarea.value);
    humanizeBtn.disabled = textarea.value.trim().length < 10;

    // Clear analysis badge when user edits
    state.currentAnalysis = null;
    updateAiScore(null);
  });

  // Clear button
  clearBtn.addEventListener('click', () => {
    textarea.value = '';
    updateInputStats('');
    humanizeBtn.disabled = true;
    state.currentAnalysis = null;
    updateAiScore(null);
  });

  // File upload
  fileInput.addEventListener('change', async e => {
    const file = e.target.files[0];
    if (!file) return;
    await handleFileUpload(file);
    e.target.value = '';  // reset so same file can be re-uploaded
  });
}


// ── File Upload ───────────────────────────────────────────────────────────────

async function handleFileUpload(file) {
  showLoading(`Reading ${file.name}…`, 'Extracting text from file.');
  try {
    const result = await uploadFile(file);
    document.getElementById('inputText').value = result.extracted_text;
    updateInputStats(result.extracted_text);
    document.getElementById('humanizeBtn').disabled = result.extracted_text.length < 10;
    showToast(`Extracted ${result.word_count.toLocaleString()} words from ${result.file_type}`, 'success');
  } catch (err) {
    showToast(`Upload failed: ${err.message}`, 'error');
  } finally {
    hideLoading();
  }
}


// ── Drag & Drop ───────────────────────────────────────────────────────────────

function setupDragDrop() {
  const overlay = document.getElementById('uploadOverlay');
  const editorWrapper = document.querySelector('.editor-wrapper');

  editorWrapper.addEventListener('dragover', e => {
    e.preventDefault();
    overlay.hidden = false;
  });
  editorWrapper.addEventListener('dragleave', e => {
    if (!editorWrapper.contains(e.relatedTarget)) overlay.hidden = true;
  });
  editorWrapper.addEventListener('drop', async e => {
    e.preventDefault();
    overlay.hidden = true;
    const file = e.dataTransfer.files[0];
    if (file) await handleFileUpload(file);
  });
  overlay.addEventListener('click', () => {
    document.getElementById('fileInput').click();
  });
}


// ── Humanize ──────────────────────────────────────────────────────────────────

async function runHumanize() {
  const text = document.getElementById('inputText').value.trim();
  if (!text || text.length < 10) {
    showToast('Please enter at least 10 characters of text.', 'warning');
    return;
  }

  // Snapshot for undo
  _pushUndo(text, state.currentOutput);

  showLoading('Humanizing text…', getModeDescription());

  const btn = document.getElementById('humanizeBtn');
  btn.disabled = true;
  btn.querySelector('.btn-label').textContent = 'Working…';
  btn.querySelector('.btn-spinner').classList.remove('hidden');

  try {
    const response = await humanizeText({
      text,
      mode:                       state.mode,
      intensity:                  state.intensity,
      granularity:                state.granularity,
      english_variant:            state.variant,
      preserve_citations:         state.preserveCitations,
      preserve_technical_terms:   state.preserveTerms,
      preserve_original_voice:    state.preserveVoice,
    });

    state.currentOutput = response;
    state.currentAnalysis = response.analysis;

    // Render output
    renderOutput(response.humanized_text);
    updateOutputStats(response);
    updateAiScore({ ai_pattern_score: 0 });  // post-humanize — score implied lower

    // Render diff
    renderDiff(response.diff);

    // Render quality
    renderQuality(response.quality);

    // Render issues
    renderIssues(response.analysis);

    // Enable copy & download
    document.getElementById('copyOutput').disabled = false;
    document.getElementById('downloadBtn').disabled = false;

    // Auto-switch to output tab
    activateTab('output');

    const score = Math.round(response.quality.overall_score);
    showToast(`Done! Quality score: ${score}/100`, score >= 70 ? 'success' : 'warning');

  } catch (err) {
    showToast(`Humanization failed: ${err.message}`, 'error');
  } finally {
    hideLoading();
    btn.disabled = false;
    btn.querySelector('.btn-label').textContent = 'Humanize Text';
    btn.querySelector('.btn-spinner').classList.add('hidden');
  }
}


// ── Analyse only ──────────────────────────────────────────────────────────────

async function runAnalyse() {
  const text = document.getElementById('inputText').value.trim();
  if (!text || text.length < 10) {
    showToast('Please enter at least 10 characters of text.', 'warning');
    return;
  }

  showLoading('Analysing text…', 'No LLM call — just NLP analysis.');
  try {
    const analysis = await analyzeText(text);
    state.currentAnalysis = analysis;
    updateAiScore({ ai_pattern_score: analysis.ai_pattern_score || 0 });
    renderIssues(analysis);
    activateTab('issues');

    const issueCount =
      (analysis.unnatural_phrases?.length || 0) +
      (analysis.generic_phrases?.length || 0) +
      (analysis.excessive_transitions?.length || 0) +
      (analysis.repetitive_structures?.length || 0);

    showToast(`Found ${issueCount} issues. See the Issues tab.`, issueCount > 5 ? 'warning' : 'success');
  } catch (err) {
    showToast(`Analysis failed: ${err.message}`, 'error');
  } finally {
    hideLoading();
  }
}


// ── Undo / Redo ───────────────────────────────────────────────────────────────

function _pushUndo(inputText, currentOutput) {
  state.undoStack.push({ inputText, currentOutput });
  if (state.undoStack.length > 20) state.undoStack.shift();
  state.redoStack = [];
  _syncUndoButtons();
}

function doUndo() {
  if (state.undoStack.length === 0) return;
  const { inputText, currentOutput } = state.undoStack.pop();

  state.redoStack.push({
    inputText: document.getElementById('inputText').value,
    currentOutput: state.currentOutput,
  });

  document.getElementById('inputText').value = inputText;
  updateInputStats(inputText);

  if (currentOutput) {
    state.currentOutput = currentOutput;
    renderOutput(currentOutput.humanized_text);
    renderDiff(currentOutput.diff);
    renderQuality(currentOutput.quality);
  } else {
    resetOutput();
  }
  _syncUndoButtons();
}

function doRedo() {
  if (state.redoStack.length === 0) return;
  const { inputText, currentOutput } = state.redoStack.pop();

  state.undoStack.push({
    inputText: document.getElementById('inputText').value,
    currentOutput: state.currentOutput,
  });

  document.getElementById('inputText').value = inputText;
  updateInputStats(inputText);

  if (currentOutput) {
    state.currentOutput = currentOutput;
    renderOutput(currentOutput.humanized_text);
    renderDiff(currentOutput.diff);
    renderQuality(currentOutput.quality);
  }
  _syncUndoButtons();
}

function _syncUndoButtons() {
  document.getElementById('undoBtn').disabled = state.undoStack.length === 0;
  document.getElementById('redoBtn').disabled = state.redoStack.length === 0;
}


// ── Output actions ────────────────────────────────────────────────────────────

function setupOutput() {
  // Copy
  document.getElementById('copyOutput').addEventListener('click', async () => {
    const text = document.getElementById('outputText').textContent;
    if (!text) return;
    try {
      await navigator.clipboard.writeText(text);
      showToast('Copied to clipboard!', 'success', 2000);
    } catch {
      showToast('Copy failed — please select and copy manually.', 'error');
    }
  });

  // Download dropdown
  const downloadBtn = document.getElementById('downloadBtn');
  const downloadMenu = document.getElementById('downloadMenu');
  downloadBtn.addEventListener('click', e => {
    e.stopPropagation();
    downloadMenu.hidden = !downloadMenu.hidden;
  });
  document.addEventListener('click', () => { downloadMenu.hidden = true; });

  document.getElementById('downloadTxt').addEventListener('click', async () => {
    const text = document.getElementById('outputText').textContent;
    if (!text) return;
    try {
      await downloadAsTxt(text);
      showToast('Downloaded as .txt', 'success', 2000);
    } catch (e) {
      showToast(`Download failed: ${e.message}`, 'error');
    }
    downloadMenu.hidden = true;
  });

  document.getElementById('downloadDocx').addEventListener('click', async () => {
    const text = document.getElementById('outputText').textContent;
    if (!text) return;
    try {
      await downloadAsDocx(text);
      showToast('Downloaded as .docx', 'success', 2000);
    } catch (e) {
      showToast(`Download failed: ${e.message}`, 'error');
    }
    downloadMenu.hidden = true;
  });

  // Reset
  document.getElementById('resetOutput').addEventListener('click', resetOutput);
}


// ── Modals & Drawers ──────────────────────────────────────────────────────────

function setupModals() {
  // Settings modal
  const settingsModal = document.getElementById('settingsModal');
  document.getElementById('settingsToggle').addEventListener('click', async () => {
    settingsModal.hidden = false;
    // Load provider list
    try {
      const data = await fetchProviders();
      renderProviders(data, async (providerName, itemEl) => {
        try {
          await switchProvider(providerName);
          state.activeProvider = providerName;
          await loadProviderStatus();
          document.querySelectorAll('.provider-item').forEach(el => el.classList.remove('active'));
          itemEl.classList.add('active');
          showToast(`Switched to ${providerName}`, 'success');
        } catch (e) {
          showToast(`Switch failed: ${e.message}`, 'error');
        }
      });
    } catch (e) {
      document.getElementById('providerList').textContent = `Could not load providers: ${e.message}`;
    }
  });
  document.getElementById('settingsClose').addEventListener('click', () => {
    settingsModal.hidden = true;
  });
  settingsModal.addEventListener('click', e => {
    if (e.target === settingsModal) settingsModal.hidden = true;
  });

  // History drawer
  const historyDrawer = document.getElementById('historyDrawer');
  document.getElementById('historyToggle').addEventListener('click', async () => {
    historyDrawer.hidden = false;
    try {
      const data = await fetchHistory();
      renderHistory(data, async (id) => {
        try {
          const item = await fetchHistoryItem(id);
          document.getElementById('inputText').value = item.original_text || '';
          updateInputStats(item.original_text || '');

          // Reconstruct a minimal response to display
          if (item.humanized_text) {
            renderOutput(item.humanized_text);
            document.getElementById('copyOutput').disabled = false;
            document.getElementById('downloadBtn').disabled = false;
          }
          historyDrawer.hidden = true;
          showToast('History item loaded', 'success', 2000);
        } catch (e) {
          showToast(`Could not load item: ${e.message}`, 'error');
        }
      });
    } catch (e) {
      document.getElementById('historyList').textContent = `Error: ${e.message}`;
    }
  });
  document.getElementById('historyClose').addEventListener('click', () => {
    historyDrawer.hidden = true;
  });
  historyDrawer.addEventListener('click', e => {
    if (e.target === historyDrawer) historyDrawer.hidden = true;
  });
  document.getElementById('clearHistory').addEventListener('click', async () => {
    if (!confirm('Clear all history? This cannot be undone.')) return;
    try {
      await clearHistory();
      document.getElementById('historyList').innerHTML = '<p class="hint-center">History cleared.</p>';
      showToast('History cleared', 'success');
    } catch (e) {
      showToast(`Error: ${e.message}`, 'error');
    }
  });
}


// ── Helpers ───────────────────────────────────────────────────────────────────

function getModeDescription() {
  const descriptions = {
    natural:       'Rewriting as natural everyday prose…',
    academic:      'Rewriting in scholarly academic style…',
    professional:  'Rewriting for professional communication…',
    simple:        'Rewriting in plain English…',
    research:      'Rewriting for research writing…',
    conversational:'Rewriting in conversational style…',
    personal:      'Rewriting while preserving your voice…',
  };
  return descriptions[state.mode] || 'Processing…';
}
