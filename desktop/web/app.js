/* =========================================================================
   Local Chat — frontend bridge + rendering (spec §10, §7).
   Talks to Python via window.pywebview.api.*; receives streamed tokens via the
   global onThinking/onAnswer/onDone/onError/onModelStatus callbacks.
   ========================================================================= */
'use strict';

/* ---- Markdown / math / code pipeline (all vendored, offline) ------------ */
const md = window.markdownit({
  html: false,            // never inject raw model text as HTML (spec §7.5)
  linkify: false,
  breaks: false,
  typographer: false,
  highlight(str, lang) {
    if (lang && window.hljs.getLanguage(lang)) {
      try { return window.hljs.highlight(str, { language: lang, ignoreIllegals: true }).value; }
      catch (_) { /* fall through to escaped */ }
    }
    return '';            // empty -> markdown-it escapes the code itself
  },
}).use(window.texmath, {
  engine: window.katex,
  delimiters: 'dollars',  // currency-safe $…$ heuristics (spec §7.5 / §15.12)
  katexOptions: { throwOnError: false },
});

/* ---- DOM refs ----------------------------------------------------------- */
const $ = (id) => document.getElementById(id);
const root = $('root');
const transcript = $('transcript');
const emptyState = $('empty-state');
const modelCards = $('model-cards');
const modelButton = $('model-button');
const modelButtonLabel = $('model-button-label');
const modelMenu = $('model-menu');
const input = $('input');
const sendBtn = $('btn-send');
const stopBtn = $('btn-stop');
const statusLine = $('status-line');
const settingsDrawer = $('settings');
const settingsBody = $('settings-body');
const scrim = $('scrim');

/* ---- State -------------------------------------------------------------- */
let models = [];
let currentModelId = null;
let generating = false;
let turn = null;            // { id, root, thinkEl, thinkBody, answerEl, answer, think, answered }
let settingsScope = 'model';
let settingsSnap = null;    // { defaults, params, global_overrides, model_overrides }
const labelFor = (id) => (models.find((m) => m.id === id) || {}).label || id;

/* ===========================================================================
   Bridge bootstrap
   ========================================================================= */
function ready() {
  if (window.pywebview && window.pywebview.api) init();
  else window.addEventListener('pywebviewready', init, { once: true });
}

async function init() {
  models = await window.pywebview.api.list_models();
  renderModelMenu();
  renderModelCards();
  const cur = await window.pywebview.api.current_model();
  currentModelId = cur && cur.id;
  syncView();
  wireEvents();
}

/* ===========================================================================
   Model selection
   ========================================================================= */
function renderModelMenu() {
  modelMenu.innerHTML = '';
  if (!models.length) {
    const none = document.createElement('div');
    none.className = 'model-menu__empty';
    none.textContent = 'No models found. Add one with the add command, then rescan.';
    modelMenu.appendChild(none);
  }
  models.forEach((m) => {
    const item = document.createElement('button');
    item.className = 'model-menu__item';
    item.role = 'option';
    const kind = m.kind || 'mlx';
    item.innerHTML =
      `<span class="model-menu__check">${m.id === currentModelId ? '✓' : ''}</span>` +
      `<span class="model-menu__name">${escapeText(m.label)}` +
      `<span class="model-menu__id">${escapeText(m.id)}</span></span>` +
      `<span class="model-kind model-kind--${kind}">${kind.toUpperCase()}</span>`;
    item.addEventListener('click', () => { closeModelMenu(); selectModel(m.id); });
    modelMenu.appendChild(item);
  });
  const sep = document.createElement('div');
  sep.className = 'model-menu__sep';
  modelMenu.appendChild(sep);
  const rescan = document.createElement('button');
  rescan.className = 'model-menu__rescan';
  rescan.type = 'button';
  rescan.innerHTML = '<span class="model-menu__rescan-icon" aria-hidden="true">↻</span> Rescan models';
  rescan.addEventListener('click', (e) => { e.stopPropagation(); rescanModels(); });
  modelMenu.appendChild(rescan);
}

async function rescanModels() {
  const before = models.length;
  try { models = await window.pywebview.api.list_models(); }
  catch (_) { return; }
  renderModelMenu();           // re-render in place; keep the menu open
  renderModelCards();
  const delta = models.length - before;
  setStatus(delta > 0 ? `found ${delta} new model${delta > 1 ? 's' : ''}`
            : (models.length ? 'model list up to date' : 'no models found'));
}

function renderModelCards() {
  modelCards.innerHTML = '';
  models.forEach((m) => {
    const card = document.createElement('button');
    card.className = 'model-card';
    const kind = m.kind || 'mlx';
    card.innerHTML =
      `<div class="model-card__name">${escapeText(m.label)}` +
      ` <span class="model-kind model-kind--${kind}">${kind.toUpperCase()}</span></div>` +
      `<div class="model-card__id">${escapeText(m.id)}</div>`;
    card.addEventListener('click', () => selectModel(m.id));
    modelCards.appendChild(card);
  });
  if (!models.length) {
    modelCards.innerHTML =
      '<p class="empty-state__sub">No models in <code>models/hub</code> yet.</p>';
  }
}

function selectModel(id) {
  if (generating) return;
  modelButton.classList.add('is-loading');
  modelButton.classList.remove('is-ready');
  modelButtonLabel.textContent = labelFor(id);
  setStatus(`loading ${labelFor(id)} …`);
  input.disabled = true;
  window.pywebview.api.load_model(id);
}

/* Python -> JS: model lifecycle */
window.onModelStatus = (s) => {
  if (s.state === 'loading') {
    modelButton.classList.add('is-loading');
    setStatus(`loading ${labelFor(s.id)} …`);
  } else if (s.state === 'ready') {
    currentModelId = s.id;
    modelButton.classList.remove('is-loading');
    modelButton.classList.add('is-ready');
    modelButtonLabel.textContent = labelFor(s.id);
    transcript.innerHTML = '';          // model switch = fresh conversation
    renderModelMenu();
    setStatus('');
    syncView();
    input.focus();
  } else if (s.state === 'error') {
    modelButton.classList.remove('is-loading', 'is-ready');
    setStatus(s.message || 'Could not load that model.', true);
    syncView();
  }
};

/* ===========================================================================
   Conversation view
   ========================================================================= */
function syncView() {
  const hasModel = !!currentModelId;
  emptyState.hidden = hasModel;
  transcript.hidden = !hasModel;
  input.disabled = !hasModel || generating;
  sendBtn.disabled = !hasModel || generating || input.value.trim() === '';
  if (!hasModel) modelButtonLabel.textContent = 'No model';
}

function addUserMessage(text) {
  const msg = document.createElement('div');
  msg.className = 'msg msg-user';
  const bubble = document.createElement('div');
  bubble.className = 'bubble';
  bubble.textContent = text;                 // textContent = inert, no injection
  msg.appendChild(bubble);
  transcript.appendChild(msg);
  scrollToBottom(true);
}

function startAssistantTurn(turnId) {
  const msg = document.createElement('div');
  msg.className = 'msg msg-assistant';
  const answerEl = document.createElement('div');
  answerEl.className = 'answer is-streaming';
  msg.appendChild(answerEl);
  transcript.appendChild(msg);
  turn = { id: turnId, root: msg, thinkEl: null, thinkBody: null,
           answerEl, answer: '', think: '', answered: false };
  scrollToBottom(true);
}

function ensureThinkingPanel() {
  if (turn.thinkEl) return;
  const panel = document.createElement('div');
  panel.className = 'thinking is-live';
  panel.innerHTML =
    '<button class="thinking__head" type="button">' +
      '<span class="thinking__pulse"></span>' +
      '<span class="thinking__chev">▾</span>' +
      '<span class="thinking__title">Thinking</span>' +
    '</button>' +
    '<div class="thinking__body"></div>';
  panel.querySelector('.thinking__head').addEventListener('click', () => {
    panel.classList.toggle('is-collapsed');
  });
  turn.root.insertBefore(panel, turn.answerEl);
  turn.thinkEl = panel;
  turn.thinkBody = panel.querySelector('.thinking__body');
}

/* Python -> JS: streaming */
window.onThinking = (turnId, chunk) => {
  if (!turn || turn.id !== turnId) return;
  ensureThinkingPanel();
  turn.think += chunk;
  turn.thinkBody.textContent = turn.think;
  scrollToBottom();
};

window.onAnswer = (turnId, chunk) => {
  if (!turn || turn.id !== turnId) return;
  if (!turn.answered) {
    turn.answered = true;
    if (turn.thinkEl) {                        // auto-collapse on answer start
      turn.thinkEl.classList.add('is-collapsed');
      turn.thinkEl.classList.remove('is-live');
    }
  }
  turn.answer += chunk;
  renderAnswer();
  scrollToBottom();
};

window.onDone = (turnId, info) => {
  if (!turn || turn.id !== turnId) { endGenerating(); return; }
  turn.answerEl.classList.remove('is-streaming');
  if (turn.thinkEl) turn.thinkEl.classList.remove('is-live');
  if (!turn.answer.trim()) {
    const note = document.createElement('div');
    note.className = 'turn-note';
    note.textContent = (info && info.stopped)
      ? 'Stopped.'
      : (turn.think ? 'Stopped while still thinking — try raising max tokens.' : 'No output.');
    turn.root.appendChild(note);
  } else if (info && info.stopped) {
    const note = document.createElement('div');
    note.className = 'turn-note';
    note.textContent = 'Stopped.';
    turn.root.appendChild(note);
  }
  endGenerating();
};

window.onError = (turnId, message) => {
  if (turn && turn.id === turnId) {
    turn.answerEl.classList.remove('is-streaming');
    if (turn.thinkEl) turn.thinkEl.classList.remove('is-live');
    const note = document.createElement('div');
    note.className = 'turn-note is-error';
    note.textContent = message || 'Generation failed.';
    turn.root.appendChild(note);
  }
  endGenerating();
};

function renderAnswer() {
  turn.answerEl.innerHTML = md.render(turn.answer);
  enhanceCodeBlocks(turn.answerEl);
}

function enhanceCodeBlocks(scope) {
  scope.querySelectorAll('pre').forEach((pre) => {
    if (pre.querySelector('.code-copy')) return;
    const btn = document.createElement('button');
    btn.className = 'code-copy';
    btn.type = 'button';
    btn.textContent = 'Copy';
    btn.addEventListener('click', () => {
      const code = pre.querySelector('code');
      navigator.clipboard.writeText(code ? code.innerText : pre.innerText).then(() => {
        btn.textContent = 'Copied'; btn.classList.add('is-done');
        setTimeout(() => { btn.textContent = 'Copy'; btn.classList.remove('is-done'); }, 1400);
      }).catch(() => {});
    });
    pre.appendChild(btn);
  });
}

/* ===========================================================================
   Send / stop / new chat
   ========================================================================= */
async function send() {
  const text = input.value.trim();
  if (!text || generating || !currentModelId) return;
  addUserMessage(text);
  input.value = '';
  autogrow();
  beginGenerating();
  const res = await window.pywebview.api.send_message(text);
  if (!res || !res.ok) {
    endGenerating();
    setStatus((res && res.error) || 'Could not start generation.', true);
    return;
  }
  startAssistantTurn(res.turn_id);
}

function beginGenerating() {
  generating = true;
  sendBtn.hidden = true;
  stopBtn.hidden = false;
  input.disabled = false;
  setStatus('generating…');
}

function endGenerating() {
  generating = false;
  turn = null;
  stopBtn.hidden = true;
  sendBtn.hidden = false;
  setStatus('');
  syncView();
}

function stop() {
  if (!generating) return;
  window.pywebview.api.stop();
  setStatus('stopping…');
}

async function newChat() {
  if (generating) window.pywebview.api.stop();
  await window.pywebview.api.new_chat();
  transcript.innerHTML = '';
  turn = null;
  setStatus('');
  input.focus();
}

/* ===========================================================================
   Settings drawer (auto-save, This-model / All-models scope)
   ========================================================================= */
const PARAM_FIELDS = [
  { group: 'Sampling' },
  { key: 'temperature', label: 'Temperature', type: 'range', min: 0, max: 2, step: 0.05 },
  { key: 'top_p', label: 'Top-p', type: 'range', min: 0, max: 1, step: 0.01 },
  { key: 'top_k', label: 'Top-k', type: 'number', min: 0, max: 200, step: 1, hint: '0 = off' },
  { key: 'min_p', label: 'Min-p', type: 'range', min: 0, max: 1, step: 0.01, hint: '0 = off' },
  { group: 'Repetition' },
  { key: 'repetition_penalty', label: 'Repetition penalty', type: 'range', min: 1, max: 2, step: 0.01, hint: '1 = off' },
  { key: 'repetition_context_size', label: 'Repetition window', type: 'number', min: 0, max: 4096, step: 1 },
  { group: 'Generation' },
  { key: 'max_tokens', label: 'Max tokens', type: 'number', min: 64, max: 32768, step: 64 },
  { key: 'seed', label: 'Seed', type: 'number', min: 0, max: 999999999, step: 1, hint: 'no-op when temp = 0' },
  { group: 'System prompt' },
  { key: 'system_prompt', label: '', type: 'textarea' },
];

async function openSettings() {
  settingsSnap = await window.pywebview.api.get_settings();
  settingsScope = settingsSnap.scope || 'model';
  $('scope-model').classList.toggle('is-on', settingsScope === 'model');
  $('scope-global').classList.toggle('is-on', settingsScope === 'global');
  $('scope-context').textContent = settingsScope === 'model'
    ? `applies to ${labelFor(currentModelId)}`
    : 'applies to all models';
  buildSettingsFields();
  scrim.hidden = false;
  settingsDrawer.hidden = false;
}

function closeSettings() {
  settingsDrawer.hidden = true;
  scrim.hidden = true;
}

function displayValue(key) {
  if (settingsScope === 'global') {
    const g = settingsSnap.global_overrides || {};
    return key in g ? g[key] : settingsSnap.defaults[key];
  }
  return settingsSnap.params[key];   // model scope: effective value the model uses
}

function overrideSet() {
  return settingsScope === 'global'
    ? (settingsSnap.global_overrides || {})
    : (settingsSnap.model_overrides || {});
}

function buildSettingsFields() {
  settingsBody.innerHTML = '';
  const ov = overrideSet();
  PARAM_FIELDS.forEach((f) => {
    if (f.group) {
      const sep = document.createElement('div');
      sep.className = 'field__group-label';
      sep.textContent = f.group;
      settingsBody.appendChild(sep);
      return;
    }
    const field = document.createElement('div');
    field.className = 'field' + (f.key in ov ? ' is-overridden' : '');
    field.dataset.key = f.key;
    const val = displayValue(f.key);

    if (f.type === 'textarea') {
      const ta = document.createElement('textarea');
      ta.className = 'sys-input';
      ta.placeholder = 'Optional. Prepended as a system message on the next turn.';
      ta.value = val || '';
      ta.addEventListener('input', () => persist(f.key, ta.value));
      field.appendChild(ta);
    } else if (f.type === 'range') {
      const row = rowWith(f, val);
      const valEl = row.querySelector('.field__val');
      const range = document.createElement('input');
      range.type = 'range';
      range.min = f.min; range.max = f.max; range.step = f.step;
      range.value = val;
      range.addEventListener('input', () => {
        valEl.textContent = fmt(range.value);
        persist(f.key, parseFloat(range.value));
      });
      field.appendChild(row);
      field.appendChild(range);
    } else { // number
      const row = rowWith(f, null);
      const num = document.createElement('input');
      num.type = 'number';
      num.className = 'num-input';
      num.min = f.min; num.max = f.max; num.step = f.step;
      num.value = val;
      num.addEventListener('change', () => {
        let v = clampNum(parseInt(num.value, 10) || 0, f.min, f.max);
        num.value = v;
        persist(f.key, v);
      });
      field.appendChild(row);
      field.appendChild(num);
    }
    settingsBody.appendChild(field);
  });
}

function rowWith(f, val) {
  const row = document.createElement('div');
  row.className = 'field__row';
  const label = document.createElement('span');
  label.className = 'field__label';
  label.textContent = f.label;
  row.appendChild(label);
  if (f.hint) {
    const hint = document.createElement('span');
    hint.className = 'field__hint';
    hint.textContent = f.hint;
    row.appendChild(hint);
  }
  if (val !== null) {
    const v = document.createElement('span');
    v.className = 'field__val';
    v.textContent = fmt(val);
    row.appendChild(v);
  }
  return row;
}

let persistTimer = null;
const pendingPartial = {};
function persist(key, value) {
  pendingPartial[key] = value;
  clearTimeout(persistTimer);
  persistTimer = setTimeout(flushPersist, 280);
}

async function flushPersist() {
  const partial = { ...pendingPartial };
  for (const k in pendingPartial) delete pendingPartial[k];
  const res = await window.pywebview.api.update_settings(partial, settingsScope);
  if (res && res.ok) {
    settingsSnap.params = res.params;
    const ov = settingsScope === 'global'
      ? (settingsSnap.global_overrides = settingsSnap.global_overrides || {})
      : (settingsSnap.model_overrides = settingsSnap.model_overrides || {});
    Object.assign(ov, partial);
    markOverrides();
  }
}

function markOverrides() {
  const ov = overrideSet();
  settingsBody.querySelectorAll('.field').forEach((el) => {
    el.classList.toggle('is-overridden', el.dataset.key in ov);
  });
}

async function resetScope() {
  const res = await window.pywebview.api.reset_settings(settingsScope);
  if (res && res.ok) {
    settingsSnap = await window.pywebview.api.get_settings();
    buildSettingsFields();
  }
}

function switchScope(scope) {
  settingsScope = scope;
  $('scope-model').classList.toggle('is-on', scope === 'model');
  $('scope-global').classList.toggle('is-on', scope === 'global');
  $('scope-context').textContent = scope === 'model'
    ? `applies to ${labelFor(currentModelId)}`
    : 'applies to all models';
  window.pywebview.api.set_scope(scope);
  buildSettingsFields();
}

/* ===========================================================================
   Wiring + helpers
   ========================================================================= */
function wireEvents() {
  sendBtn.addEventListener('click', send);
  stopBtn.addEventListener('click', stop);
  $('btn-new').addEventListener('click', newChat);
  $('btn-settings').addEventListener('click', openSettings);
  $('settings-close').addEventListener('click', closeSettings);
  scrim.addEventListener('click', closeSettings);
  $('settings-reset').addEventListener('click', resetScope);
  $('scope-model').addEventListener('click', () => switchScope('model'));
  $('scope-global').addEventListener('click', () => switchScope('global'));

  $('btn-close').addEventListener('click', () => window.pywebview.api.close_window());
  $('btn-min').addEventListener('click', () => window.pywebview.api.minimize_window());

  modelButton.addEventListener('click', toggleModelMenu);
  document.addEventListener('click', (e) => {
    if (!modelMenu.hidden && !modelMenu.contains(e.target) && e.target !== modelButton
        && !modelButton.contains(e.target)) closeModelMenu();
  });

  input.addEventListener('input', () => { autogrow(); syncSendEnabled(); });
  input.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send(); }
  });

  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') {
      if (!settingsDrawer.hidden) closeSettings();
      else if (!modelMenu.hidden) closeModelMenu();
      else stop();
    } else if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === 'n') {
      e.preventDefault(); newChat();
    }
  });

  // Focus affordance — no OS title bar, so dim the frame when blurred (§12.1).
  window.addEventListener('focus', () => root.classList.add('is-active'));
  window.addEventListener('blur', () => root.classList.remove('is-active'));
}

function toggleModelMenu() {
  if (modelMenu.hidden) { renderModelMenu(); modelMenu.hidden = false; modelButton.setAttribute('aria-expanded', 'true'); }
  else closeModelMenu();
}
function closeModelMenu() { modelMenu.hidden = true; modelButton.setAttribute('aria-expanded', 'false'); }

function syncSendEnabled() {
  sendBtn.disabled = generating || !currentModelId || input.value.trim() === '';
}

function autogrow() {
  input.style.height = 'auto';
  input.style.height = Math.min(input.scrollHeight, 200) + 'px';
}

function scrollToBottom(force) {
  const chat = document.getElementById('chat');
  const near = chat.scrollHeight - chat.scrollTop - chat.clientHeight < 120;
  if (force || near) chat.scrollTop = chat.scrollHeight;
}

function setStatus(text, isError) {
  statusLine.textContent = text || '';
  statusLine.classList.toggle('is-error', !!isError);
}

function fmt(v) {
  const n = parseFloat(v);
  if (Number.isInteger(n)) return String(n);
  return n.toFixed(2).replace(/\.?0+$/, '') || '0';
}
function clampNum(v, lo, hi) { return Math.max(lo, Math.min(hi, v)); }
function escapeText(s) {
  return String(s).replace(/[&<>"']/g, (c) =>
    ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#039;' }[c]));
}

ready();
