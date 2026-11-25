/**
 * api_stream_patch.js
 * ────────────────────
 * Streaming client for POST /api/v1/humanize/stream (Server-Sent Events).
 *
 * It overrides the original humanizeText() from api.js so the existing
 * app.js controller works unchanged, while chunks appear in the output
 * pane as soon as each one is rewritten. Progress is shown in the
 * in-page percentage bar (#humanizeProgress), not in a popup.
 * If streaming fails, it falls back to the standard /humanize endpoint.
 */

(function () {
  'use strict';

  const STREAM_PATH = '/api/v1/humanize/stream';

  // ── Progress helpers (delegate to ui.js when available) ───────────────────

  function reportProgress(fraction, label) {
    if (typeof setHumanizeProgress === 'function') {
      setHumanizeProgress(fraction, label);
    }
  }

  function reportDone() {
    if (typeof finishHumanizeProgress === 'function') {
      finishHumanizeProgress('Complete');
    }
  }

  // ── Output rendering helpers ──────────────────────────────────────────────

  function prepareOutput() {
    const el = document.getElementById('outputText');
    if (!el) return null;
    el.textContent = '';
    el.dataset.streaming = 'true';
    el.hidden = false;
    const empty = document.getElementById('outputEmpty');
    if (empty) empty.hidden = true;
    return el;
  }

  function appendChunk(el, text) {
    if (!el) return;
    el.textContent += (el.textContent ? '\n\n' : '') + text;
  }

  function resetOutput(el) {
    if (!el) return;
    el.dataset.streaming = 'false';
    el.textContent = '';
  }

  // ── SSE fetch ─────────────────────────────────────────────────────────────

  async function humanizeStream(payload, handlers) {
    const { onChunk, onDone, onError } = handlers;
    let totalChunks = 0;
    let completed = 0;
    let settled = false;

    try {
      const response = await fetch(STREAM_PATH, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });

      if (!response.ok || !response.body) {
        let detail = `HTTP ${response.status}`;
        try {
          const err = await response.json();
          detail = err.detail || detail;
        } catch (_) {}
        throw new Error(detail);
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });

        let sep;
        while ((sep = buffer.indexOf('\n\n')) !== -1) {
          const raw = buffer.slice(0, sep);
          buffer = buffer.slice(sep + 2);

          const dataLines = raw
            .split('\n')
            .filter(line => line.startsWith('data:'))
            .map(line => line.slice(5).trimStart());

          if (dataLines.length === 0) continue;

          let event;
          try {
            event = JSON.parse(dataLines.join('\n'));
          } catch (_) {
            continue;
          }

          if (event.type === 'start') {
            totalChunks = event.total_chunks || 1;
            reportProgress(0.05, 'Preparing sections…');
          } else if (event.type === 'chunk') {
            completed += 1;
            const fraction = totalChunks
              ? 0.05 + 0.85 * (completed / totalChunks)
              : 0.5;
            reportProgress(
              fraction,
              `Rewriting section ${completed} of ${totalChunks || '…'}…`
            );
            if (onChunk) onChunk(event.text, event.index, totalChunks);
          } else if (event.type === 'done') {
            settled = true;
            reportDone();
            if (onDone) onDone(event);
          } else if (event.type === 'error') {
            settled = true;
            throw new Error(event.message || 'Streaming failed');
          }
        }
      }

      if (!settled) {
        throw new Error('The stream ended before the final result arrived.');
      }
    } catch (err) {
      if (!settled) {
        if (onError) onError(err.message || 'Network error');
      }
    }
  }

  // ── Override humanizeText() ───────────────────────────────────────────────

  function install() {
    if (typeof window.humanizeText !== 'function') {
      setTimeout(install, 50);
      return;
    }
    if (window.humanizeText.__streaming) return;

    const originalHumanize = window.humanizeText;

    const streamingHumanize = function (payload) {
      return new Promise((resolve, reject) => {
        const outputEl = prepareOutput();

        humanizeStream(payload, {
          onChunk(text) {
            appendChunk(outputEl, text);
          },

          onDone(event) {
            if (outputEl) outputEl.dataset.streaming = 'false';
            resolve({
              original_text: payload.text,
              mode: payload.mode,
              intensity: payload.intensity,
              humanized_text: event.humanized_text,
              analysis: event.analysis || null,
              quality: event.quality,
              diff: event.diff || [],
              human_score: event.human_score,
              provider_used: event.provider_used || '',
              model_used: event.model_used || '',
              processing_time_ms: event.processing_time_ms,
            });
          },

          onError(message) {
            console.warn(
              '[humanip] Streaming failed (' + message + '), falling back to standard endpoint'
            );
            resetOutput(outputEl);
            reportProgress(0.9, 'Retrying without streaming…');
            originalHumanize(payload).then(resolve, reject);
          },
        });
      });
    };

    streamingHumanize.__streaming = true;
    window.humanizeText = streamingHumanize;
    console.log('[humanip] Streaming mode activated');
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', install);
  } else {
    install();
  }
})();
