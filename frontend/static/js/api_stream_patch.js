/**
 * api_stream_patch.js
 * ────────────────────
 * Streaming client for POST /api/v1/humanize/stream (Server-Sent Events).
 *
 * It overrides the original humanizeText() from api.js so the existing
 * app.js controller works unchanged, while chunks appear in the output
 * pane as soon as each one is rewritten. If streaming fails for any
 * reason, it automatically falls back to the standard /humanize endpoint.
 */

(function () {
  'use strict';

  const STREAM_PATH = '/api/v1/humanize/stream';

  // ── Progress bar ──────────────────────────────────────────────────────────

  function getOrCreateProgressBar() {
    let bar = document.getElementById('humanip-stream-progress');
    if (!bar) {
      bar = document.createElement('div');
      bar.id = 'humanip-stream-progress';
      bar.style.cssText = `
        position: fixed;
        top: 0; left: 0; right: 0;
        height: 3px;
        background: linear-gradient(90deg, #6366f1, #8b5cf6);
        transform-origin: left;
        transform: scaleX(0);
        transition: transform 0.3s ease, opacity 0.4s ease;
        z-index: 9999;
        border-radius: 0 2px 2px 0;
      `;
      document.body.appendChild(bar);
    }
    return bar;
  }

  function setProgress(fraction) {
    const bar = getOrCreateProgressBar();
    bar.style.opacity = '1';
    bar.style.transform = `scaleX(${Math.min(1, Math.max(0, fraction))})`;
  }

  function hideProgress() {
    const bar = document.getElementById('humanip-stream-progress');
    if (!bar) return;
    bar.style.transform = 'scaleX(1)';
    setTimeout(() => {
      bar.style.opacity = '0';
      setTimeout(() => {
        bar.style.transform = 'scaleX(0)';
      }, 400);
    }, 200);
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
    const { onStart, onChunk, onDone, onError } = handlers;
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
            setProgress(0.05);
            if (onStart) onStart(event);
          } else if (event.type === 'chunk') {
            completed += 1;
            setProgress(totalChunks ? 0.05 + 0.85 * (completed / totalChunks) : 0.5);
            if (onChunk) onChunk(event.text, event.index, totalChunks);
          } else if (event.type === 'done') {
            settled = true;
            setProgress(1);
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
        hideProgress();
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
        let firstChunkShown = false;

        humanizeStream(payload, {
          onChunk(text) {
            if (!firstChunkShown) {
              firstChunkShown = true;
              if (typeof hideLoading === 'function') hideLoading();
            }
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
            if (typeof showLoading === 'function') {
              showLoading('Reconnecting…', 'Streaming unavailable — retrying normally.');
            }
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
