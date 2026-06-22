# Local Chat — Desktop App Specification

**Status:** Draft v1
**Date:** 2026-06-21
**Owner:** danny
**Scope:** A small, offline macOS desktop app that provides a clean chat UI for
talking to the local MLX models already installed in this project. No internet,
no model file changes — just chat.

---

## 1. Goal

Provide a native desktop window with a modern chat interface where the user can:

1. Pick one of the locally-installed models.
2. Have a multi-turn conversation.
3. See the answer stream in, with the model's hidden "thinking" trace shown
   separately, Markdown rendered, and math rendered readably.

It is the GUI sibling of the existing terminal tool `scripts/chat_pretty.py` and must
reuse the same behavior and the same local `.venv` / `mlx_lm` stack.

---

## 2. Non-Goals

- No fine-tuning, training, LoRA, quantization, or any model editing.
- No model downloading, model browsing, or Hugging Face integration in-app.
- No accounts, sync, telemetry, analytics, crash reporting, or auto-update.
- No remote/cloud inference, no API keys.
- No prompt *templates* / preset library in v1. (A free-form **system prompt**
  and all sampling parameters ARE user-adjustable at runtime, **per model** —
  see §8 — they simply must not be hardcoded in source during the build.)
- Not a server product; single local user, single window.

---

## 3. Hard Constraints

### 3.1 No network calls (strict)
- The app MUST NOT make any outbound connection (internet or LAN).
- Prefer **in-process** model inference (call `mlx_lm` directly in Python, no
  sockets at all). A loopback HTTP server is discouraged; if ever used it MUST
  bind `127.0.0.1` only and is considered a fallback, not the design.
- All frontend assets (JS/CSS/fonts/math/highlighting libraries) MUST be
  **vendored locally** and loaded from disk. No CDNs, no remote URLs.
- Set `HF_HUB_OFFLINE=1` and `HF_HOME` to the local cache so no library ever
  attempts a network fetch.
- Disable any framework "phone-home"/update features.
- **Acceptance:** with Wi‑Fi off (or Little Snitch in "deny" mode) the app
  launches, lists models, and completes a full conversation with zero network
  attempts.

### 3.2 Model files read-only; parameters adjustable at runtime (not hardcoded)
- The app MUST treat model files (weights, `config.json`, tokenizer) as
  **read-only**. Never write to anything under `models/`.
- It MUST NOT re-quantize, convert, or modify model configs.
- **Generation/sampling parameters AND a system prompt ARE user-adjustable from
  the UI at runtime**, and can be saved **per model** after the user changes
  them (temperature, top-p, top-k, min-p, max tokens, repetition penalty,
  system prompt, etc. — see §8). **On first load, every model uses
  `DEFAULT_PARAMS` only** — no baked-in per-model presets.
- **The key rule:** parameters must NOT be hardcoded or edited in the source
  code when building. They live as **runtime settings** with sensible defaults,
  changeable in-app, and persisted to a **local config file** (see §8.2) — not
  baked into Python constants that would require a code change to alter.
- So "no model parameter alterations" means: never modify the model *files* on
  disk, and never tune behavior by hand-editing source. It does **not** mean the
  user can't tune inference — they can, at runtime, through the UI.

### 3.3 Privacy
- Conversations live in memory only in v1; nothing is written to disk. Chat
  persistence (a "Save chat" action writing a local JSON file) is **deferred**
  to Future scope (§17 #1, §18). The only file v1 ever writes is the settings
  config (§8.2).

---

## 4. Environment / Platform

- **OS:** macOS on Apple Silicon (primary target).
- **Runtime:** the existing project virtual environment at
  `/Users/danny/Documents/Gemma/.venv` (Python 3.14.6).
- **Inference:** `mlx_lm` 0.31.3 + `mlx` (Apple GPU), `rich` available.
- **Models:** read from the local Hugging Face cache at
  `/Users/danny/Documents/Gemma/models` (`HF_HOME`). Currently installed:
  - `mlx-community/gemma-4-12B-it-qat-6bit`
  - `mlx-community/gemma-4-31B-it-qat-4bit`
  - `mlx-community/gemma-4-26b-a4b-8bit`
  - `mlx-community/gemma-4-e4b-it-4bit`
- **Critical:** the app MUST run inside this `.venv` so it inherits the existing
  compatibility fixes:
  - `mlx_lm/models/gemma4_unified.py` + `gemma4_unified_text.py` shims
    (map `gemma4_unified` → `gemma4`, strip multimodal weights).
  - the `gemma4_text.py` `sanitize()` patch that drops KV-shared-layer weights
    (`num_kv_shared_layers`) so models like `e4b` load.

---

## 5. Architecture

```
┌──────────────────────────────────────────────┐
│  Native window (pywebview)                     │
│  ┌──────────────────────────────────────────┐ │
│  │  Web UI  (local files, no network)         │ │
│  │  - chat transcript, composer, model picker │ │
│  │  - markdown-it + KaTeX + highlight.js      │ │
│  │    (all vendored locally)                  │ │
│  └───────────────▲───────────────┬────────────┘ │
│        evaluate_js │ (stream)     │ js_api calls  │
│                    │              ▼               │
│  ┌─────────────────┴──────────────────────────┐ │
│  │  Python backend (in-process)                 │ │
│  │  api.py        → JS-exposed methods           │ │
│  │  chatcore.py   → load, generate, thinking/    │ │
│  │                  answer split, clean, mathify │ │
│  │  mlx_lm (load + stream_generate)              │ │
│  └──────────────────────────────────────────────┘ │
└──────────────────────────────────────────────┘
                 reads (read-only)
                        │
                 models/ (local HF cache)
```

- **No sockets.** The frontend talks to Python through the webview bridge
  (`window.pywebview.api.*`), and Python pushes streaming tokens to the
  frontend via `window.evaluate_js(...)`.
- Generation runs on a **background thread**; the UI thread stays responsive.

---

## 6. Recommended Tech Stack

**Primary recommendation: `pywebview` (Python) + vendored HTML/CSS/JS.**

Rationale: directly reuses the existing Python/`mlx_lm`/`chat_pretty` code, is
genuinely offline (no HTTP), ships a small native window, and lets us build a
modern chat UI with HTML/CSS. Markdown/math/code rendering use mature JS
libraries running locally in the webview.

- Window: `pywebview` (`webview.create_window(...)`, `webview.start()`),
  created **frameless** — no OS title bar, no window header (see §12.1):
  `create_window(..., frameless=True, easy_drag=False)`.
- Bridge: a Python `Api` class exposed via `js_api`; streaming via
  `window.evaluate_js`.
- Frontend libs (vendored, no CDN):
  - `markdown-it` — Markdown → HTML.
  - `KaTeX` — real math typesetting (`$…$`, `$$…$$`). This is an upgrade over
    the terminal's Unicode `mathify`; in a browser we can typeset properly.
  - `highlight.js` — code block syntax highlighting.
  - A clean system font stack; bundle any custom fonts locally.

**Acceptable fallback: `PySide6` (Qt).** Fully native widgets, `QTextBrowser`
for rich text. More native packaging, but Markdown/math rendering is less
flexible than a webview. Use only if webview proves problematic.

**Not recommended:** Electron (heavy, needs Node + Python bridge), Gradio/
Streamlit (browser/server model, not a true desktop app, opt-out telemetry).

---

## 7. Functional Requirements

### 7.1 Model selection
- On launch, discover installed models by scanning `models/hub/models--*`
  (same source the CLI uses) and present them in a dropdown/sidebar with
  friendly labels (e.g., "Gemma 4 12B (6bit QAT)").
- Selecting a model loads it on demand (with a visible loading state) and
  applies **`DEFAULT_PARAMS`** for that model unless the user has previously
  saved overrides for it in `settings.json` (§8.2).
- Switching models **unloads** the previous model first to free RAM, then loads
  the new one. Only one model resident at a time. **Dropping Python references
  (`del model, tokenizer`) is not sufficient** — MLX retains a Metal buffer
  cache, so the unload path MUST also drop the `prompt_cache` and explicitly
  clear MLX's buffer cache by calling **`mx.clear_cache()`** — the public API in
  the pinned `mlx` 0.31.x — or resident memory won't return and §15.7 will fail.
  Use **`mx.get_active_memory()`** before/after unload to verify the drop. (These
  are the current public names; confirm against your pinned `mlx` version, but do
  **not** reach for a `mx.metal.*` variant — `mx.clear_cache()` is correct.)
- If a model fails to load (e.g., unsupported architecture), show a clear,
  non-fatal error and let the user pick another (see §11).

### 7.2 Conversation
- Multi-turn chat with a visible transcript (user + assistant bubbles).
- A `prompt_cache` is kept per chat and reused across turns (fast follow-ups).
- "New chat" clears the transcript and resets the prompt cache.
- Switching models resets the conversation (cache is model-specific).

### 7.3 Streaming + thinking trace
- Assistant responses stream live into the UI.
- The model's hidden "thinking" channel is detected at the **token level**
  (reuse the logic in `scripts/chat_pretty.py`: `tokenizer.has_thinking`,
  `think_start_tokens`/`think_end_tokens`, including the two-token start-marker
  stash, plus the `clean()` channel-string scrub).
- Thinking is shown in a separate, **collapsible** "Thinking…" panel:
  - dim/secondary styling, expanded while thinking, auto-collapses when the
    answer begins; user can re-expand.
- The final answer renders as Markdown + math + highlighted code.

### 7.4 Controls
- Composer textarea + Send (Enter to send, Shift+Enter for newline).
- **Stop** button to abort the current generation (cooperative cancel).
- **New chat** button.
- Model picker.
- **Settings panel** (gear icon → side drawer / modal) exposing the adjustable
  parameters in §8: sliders/number inputs, a multi-line **system prompt** box, a
  scope toggle (**This model** vs **All models**), a "Reset to defaults" action,
  and live validation. The panel shows which model the settings apply to and the
  currently effective values. Changes apply to the **next** turn.
  - **Persistence (v1): auto-save on change.** Each edit is validated/clamped and
    immediately merged into the selected scope and written to `settings.json`
    (§8.2) — there is **no separate Save button**. The **scope toggle** picks the
    target: **This model** → `by_model[model_id]`, **All models** → `global`.
  - **Defaults-first.** On a fresh install there is **no config file** and every
    model uses `DEFAULT_PARAMS`; the file is **created only on the first persisted
    user change**, and only the keys actually changed are stored (sparse — §8.2).
    A model the user never customizes keeps pure `DEFAULT_PARAMS`.
  - Collapsed/out of the way by default so the default experience is still
    "just chat."

### 7.5 Rendering rules
- Markdown: headings, bold/italic, lists, tables, blockquotes, inline code,
  fenced code blocks (with language highlighting).
- Math: `$inline$` and `$$display$$` typeset via KaTeX. (If KaTeX is omitted,
  fall back to the backend `mathify()` Unicode pass already in
  `scripts/chat_pretty.py`.)
- **Currency / stray `$` guard:** plain prose like "$5 to $10" must NOT be
  parsed as math. Use a markdown-it KaTeX plugin (e.g. `markdown-it-katex` /
  `texmath`) with its standard heuristics — inline `$` requires a non-space
  immediately inside the delimiters and no digit directly after the closing `$`
  — rather than a naive `/\$(.+?)\$/` regex.
- Code/inline-code must **never** be treated as math.
- Output is model-generated; render through the Markdown/KaTeX pipeline (which
  escapes HTML). Do **not** inject raw model text as HTML without escaping.

---

## 8. Inference Parameters (adjustable at runtime)

Defaults mirror the validated `scripts/chat_pretty.py` profile, but **every value is a
runtime setting** the user can change in the Settings panel (§7.4). **Every
model loads with these defaults** on first use — no per-model presets, no
pre-populated overrides. Defaults live in exactly one place
(`chatcore.DEFAULT_PARAMS`); user-saved settings overlay them only **after** the
user explicitly changes and saves something (§8.2).

| Param | Default | Range / type | UI control | Notes |
|---|---|---|---|---|
| `max_tokens` | 4096 | 64 – 32768 (int) | number | per-turn cap |
| `temperature` | 0.0 | 0.0 – 2.0 (float) | slider | 0 = greedy |
| `top_p` | 1.0 | 0.0 – 1.0 | slider | nucleus |
| `top_k` | 0 | 0 – 200 (int) | number | 0 = off — **new for GUI; not wired in the CLI yet** |
| `min_p` | 0.0 | 0.0 – 1.0 | slider | 0 = off — **new for GUI; not wired in the CLI yet** |
| `repetition_penalty` | 1.1 | 1.0 – 2.0 | slider | 1.0 = off (anti-loop fix) |
| `repetition_context_size` | 64 | 0 – 4096 (int) | number | |
| `seed` | 0 | ≥ 0 (int) | number | reproducibility; **no-op while `temperature` = 0 (greedy)** — only affects stochastic sampling |
| `system_prompt` | "" (none) | string | textarea | prepended as a system message when non-empty |

- Built with `mlx_lm.sample_utils.make_sampler(temp, top_p, min_p=…, top_k=…)`
  and `make_logits_processors(repetition_penalty=…, repetition_context_size=…)`,
  fed to `stream_generate`. This is the same `stream_generate` path as the CLI,
  but **`min_p` and `top_k` are NOT plumbed through `scripts/chat_pretty.py` today** —
  it calls `make_sampler(temp, top_p)` only (`scripts/chat_pretty.py:204`). The
  GUI must **add** these two arguments and confirm `make_sampler` in the pinned
  `mlx_lm` 0.31.3 accepts `min_p=`/`top_k=` keywords; treat them as new work,
  not a verbatim port. `temperature`, `top_p`, `repetition_penalty`, and
  `repetition_context_size` are genuine reuse; the rest are runtime-sourced.
- Always `validate_params()` (clamp to ranges; floor `seed` at 0) before
  building the sampler.
- A parameter change applies to the **next** turn, never mid-generation.
- `system_prompt`: when non-empty, prepended as a `{"role":"system", …}` message
  before the conversation when applying the chat template (empty = none).
- The prompt cache is per chat and reset on New chat / model switch — **and also
  whenever the system prompt changes**, since the cached prefix is now invalid.

### 8.1 Defaults — single source of truth
- One `DEFAULT_PARAMS` dict in `chatcore.py`. This is the *only* place defaults
  are defined; the UI and config layer read from it. This encodes the "don't
  hardcode param changes" rule: tuning happens via settings/config at runtime,
  not by editing constants in source.

### 8.2 Settings persistence + per-model overrides (local, offline)

**Initial state:** no config file. Every model uses `DEFAULT_PARAMS` exactly.
The app does **not** ship with, pre-create, or auto-populate per-model overrides.

- Saved to a local JSON config, e.g.
  `~/Library/Application Support/LocalChat/settings.json` — created **only on
  first user save** (changing a setting in the UI). No network; nothing under
  `models/` or the project is touched.
- File shape is **sparse**: only keys the user explicitly changed are stored.
  Absent file, absent `global`, or absent `by_model[model_id]` entry ⇒ that
  layer contributes nothing and `DEFAULT_PARAMS` applies.

**Fresh install (no file yet):** every model resolves to `DEFAULT_PARAMS`.

```json
{}
```

**After the user has customized settings** (illustrative — not shipped defaults):

```json
{
  "global": { "temperature": 0.3 },
  "by_model": {
    "mlx-community/gemma-4-12B-it-qat-6bit": { "system_prompt": "You are concise." },
    "mlx-community/gemma-4-e4b-it-4bit": { "max_tokens": 2048, "temperature": 0.7 }
  }
}
```

- **Resolution order** for the active model (later overrides earlier):
  1. `DEFAULT_PARAMS` (code) — **always the baseline**
  2. `global` (optional; only if user saved "All models" changes)
  3. `by_model[model_id]` (optional; only if user saved "This model" changes)
  4. live UI edits for the current session (not persisted until saved)

  `resolve_params(model_id)` returns the effective dict. Sparse storage means a
  model with no entry in `by_model` keeps pure defaults (plus any `global`
  overlay). A later change to `DEFAULT_PARAMS` in code still propagates to any
  model/key the user hasn't overridden.
- The Settings panel writes to **`by_model[current]`** when scope = "This model"
  and to **`global`** when scope = "All models". Opening the panel for a model
  that has no saved overrides shows **`DEFAULT_PARAMS`** values.
- "Reset to defaults" clears the chosen scope's saved overrides (values fall
  back to `DEFAULT_PARAMS`, or `DEFAULT_PARAMS` + `global` when resetting a
  per-model scope).

---

## 9. Backend Module: `chatcore.py` (shared logic)

Extract the reusable pieces (today they live in `scripts/chat_pretty.py`) into a small
module the GUI imports. Optionally refactor `scripts/chat_pretty.py` to import from it
too (DRY); if that's deemed risky, treat `scripts/chat_pretty.py` as the reference and
port the algorithms verbatim.

Required functions/classes:

- `list_models() -> list[ModelInfo]` — scan local HF cache; return
  `{id, label, path}`.
- `load(model_id) -> (model, tokenizer)` — wraps `mlx_lm.utils.load` with
  `HF_HOME` set and `HF_HUB_OFFLINE=1`; raises a typed error on failure.
- `ThinkingSplitter` — token-level state machine producing a stream of
  `("thinking"|"answer", text_chunk)` events. Port from `scripts/chat_pretty.py`:
  `has_thinking`, `think_start_tokens` (ts0/ts1 two-token stash), single-token
  `think_end`, and `clean()`.
- `generate(model, tokenizer, messages, cache, params, stop_flag) -> iterator` —
  builds the message list (prepending `params["system_prompt"]` as a system
  message when non-empty), applies the chat template, runs `stream_generate`
  using `params` (validated runtime settings; see §8), yields events; checks
  `stop_flag` each step for cooperative cancellation.
- `DEFAULT_PARAMS: dict` — the **single source of truth** for default inference
  parameters incl. `system_prompt` (§8.1). `validate_params(partial) -> dict`
  clamps inputs to allowed ranges. `resolve_params(model_id) -> dict` merges
  `DEFAULT_PARAMS` → `global` → per-model overrides (§8.2).
- `clean(text)` — strip channel control strings. **Note:** in `scripts/chat_pretty.py`
  this is currently a nested closure inside `main()` (`scripts/chat_pretty.py:194`)
  that captures the `think_start`/`think_end` marker strings. Extracting it means
  refactoring those markers into explicit parameters (or a small `ThinkingSplitter`
  field), not a verbatim copy.
- `mathify(text)` — LaTeX → Unicode (already implemented; used only if KaTeX is
  not the chosen renderer).

---

## 10. Frontend ↔ Backend Contract

**JS → Python (via `window.pywebview.api`):**

- `list_models() -> [{id, label}]`
- `load_model(id) -> {ok, error?}`  (runs in background; UI shows spinner)
- `send_message(text) -> {turn_id}`  (starts generation thread using current params)
- `stop() -> {ok}`  (sets cancel flag)
- `new_chat() -> {ok}`  (clears cache + transcript)
- `current_model() -> {id|null}`
- `get_settings() -> {model_id, params, defaults, scope}`  (effective params for
  the current model + defaults)
- `update_settings(partial, scope="model"|"global") -> {ok, params}`  (validate/
  clamp; write to `by_model[current]` or `global`; persist)
- `reset_settings(scope="model"|"global") -> {ok, params}`  (clear that scope's
  overrides)

**Python → JS (via `window.evaluate_js`), per active turn (throttled):**

- `onThinking(turn_id, textChunk)`
- `onAnswer(turn_id, textChunk)`
- `onDone(turn_id, {stopped: bool})`
- `onError(turn_id, message)`
- `onModelStatus({state: "loading"|"ready"|"error", id, message?})`

**Bridge-encoding rule (correctness, not just safety):** every value pushed
through `evaluate_js` MUST be serialized with `json.dumps(...)` and interpolated
as a single JS argument — e.g.
`window.evaluate_js(f"onAnswer({turn_id}, {json.dumps(chunk)})")`. Model output
routinely contains quotes, backticks, backslashes, and newlines that will
otherwise terminate the JS call string and break streaming (and constitute a
script-injection surface). Do **not** build these calls with bare f-string
interpolation of raw text. This is separate from, and in addition to, the
HTML-escaping done by the Markdown/KaTeX render pipeline (§7.5).

**Performance rule (important):** do **not** push every token across the
bridge / re-render Markdown per token. Coalesce chunks and update at ~10–15 Hz
(reuse the throttling approach that fixed the CLI "freeze": a small time-based
buffer flush). The browser DOM handles incremental Markdown cheaply, but bridge
calls and KaTeX re-typesetting should still be batched.

---

## 11. Error Handling

- **Model load failure** (unsupported arch / corrupt cache): catch, surface a
  readable message ("This model's architecture isn't supported by the current
  mlx_lm build"), keep the app usable, allow choosing another model.
- **Out of memory:** detect load/inference failure, message the user to try a
  smaller model (e.g., `e4b` or `12B`), unload cleanly.
- **Generation exception:** end the turn, show `onError`, keep the session.
- **No models found:** show an empty-state with guidance to add models via the
  existing `add-model.sh` (outside the app — the app does not download).

---

## 12. UX / UI Design

- **Layout:** left sidebar (model picker + New chat + chat list if persistence
  added later); main pane = transcript; bottom = composer with Send/Stop.
- **Messages:** user bubble (right/neutral), assistant bubble (left), generous
  spacing, readable measure (~70–80ch), good typographic defaults.
- **Thinking panel:** a subtle, collapsible block above each assistant answer
  labeled "Thinking" with a chevron; dim text; auto-collapse on answer start.
- **Streaming affordance:** caret/typing indicator while generating; Stop button
  enabled only during generation.
- **States:** model loading (spinner + name), ready, generating, error.
- **Theme:** clean light/dark following the system appearance; high-contrast,
  modern, minimal chrome.
- **Keyboard:** Enter = send, Shift+Enter = newline, Esc = stop, Cmd+N = new
  chat. **Cmd+W / Cmd+Q / Cmd+M** (close/quit/minimize) may still be honored
  depending on pywebview/macOS behavior, but are **best-effort** — since there is
  no OS chrome, the in-app window controls (§12.1) are the reliable path; do not
  rely on OS shortcuts alone.
- **Accessibility:** sufficient contrast, focus states, selectable/copyable
  message text, copy button on code blocks.

### 12.1 Frameless window (no title bar, no header)

The app ships as a **chrome-less window**: no OS title bar, no app header strip
— just the chat surface, for a clean, modern look. Created with pywebview's
**`frameless=True, easy_drag=False`** (§6). `easy_drag=False` is deliberate: with
`easy_drag=True` the *entire* window becomes a drag handle, which hijacks text
selection and clicks — so the app provides its own scoped drag region instead.

With `frameless=True` there is **no OS title bar to drag and no traffic-light
buttons**, and macOS **rounded corners** are also dropped, so the app MUST
re-provide window chrome in the web UI:

- **Move the window:** a slim (~28–36px) **custom drag region / header bar** along
  the top of the UI. Implement by marking that strip draggable and calling
  pywebview's drag API on mousedown. The drag region MUST be a *specific* strip,
  **not** the whole window. Interactive controls inside the strip (e.g. the window
  buttons below) must opt out of dragging.
- **Close / minimize (optionally maximize):** because there are no traffic lights,
  the web UI MUST render its own **window control buttons** in the header bar
  (close + minimize, optionally maximize/zoom), wired to pywebview's
  `window.destroy()` / `minimize()` (/ `toggle_fullscreen()`) APIs. These are the
  reliable controls. OS keyboard shortcuts (Cmd+W/Cmd+Q/Cmd+M) **may** also work
  but are best-effort and must not be assumed — do not depend on OS chrome
  existing. Style the buttons subtly (e.g. emphasized on hover) so they stay clean
  but always reachable. Never trap the user in an unclosable window.
- **Rounded corners + shadow:** a frameless WKWebView is a hard rectangle. To
  keep corners rounded, apply CSS `border-radius` on the root container; the
  native window stays rectangular, so either (i) accept square *window* corners
  with rounded *content*, or (ii) use a transparent window (`transparent=True`
  where supported on macOS) so the rounded CSS edge is what shows. Pick (i) for
  v1 (more reliable); treat (ii) as a polish stretch — verify transparency
  actually works on the target macOS before relying on it.
- **Focus / drag affordance:** since there's no title bar to indicate focus,
  give the window a subtle active/inactive visual state (e.g. shadow or border
  intensity) so the user can tell when it's focused.

---

## 13. Proposed Project Structure

```
desktop/
  app.py            # entry point: build window, wire Api, webview.start()
  api.py            # Api class exposed to JS (js_api); owns the session
  chatcore.py       # shared: list_models, load, ThinkingSplitter, generate, clean, mathify
  session.py        # in-memory chat state: messages, prompt_cache, model handle, stop flag
  settings.py       # load/validate/save runtime params to local JSON (§8.2); reads DEFAULT_PARAMS
  web/
    index.html
    styles.css
    app.js          # bridge calls, transcript rendering, streaming, throttle
    vendor/         # markdown-it, katex (+fonts), highlight.js — all local, no CDN
  assets/
    icon.icns
  packaging/
    setup.py        # py2app config (or pyinstaller spec)
  README.md         # how to run from .venv and how to build the .app
```

- Must be launched with the project `.venv` interpreter so the shims/fixes load.
- Dev run: `.venv/bin/python desktop/app.py`.

---

## 14. Packaging

> **Risk note — this is the least-proven part of the plan.** py2app/PyInstaller
> on **Python 3.14.6** (very new) packaging a WKWebView app whose `mlx_lm` is
> **monkeypatched in site-packages** (the `gemma4_unified*`/`gemma4_text`
> shims, §4) is the highest-risk path in this spec. The patched shim files MUST
> survive bundling intact or every model fails to load. Do not assume `.app`
> parity until proven.

- **v1 supported path: run from the project `.venv`.** `.venv/bin/python
  desktop/app.py` is the officially supported, must-work launch for v1. All
  acceptance criteria (§15) target this path.
- **Standalone `.app` is a stretch goal**, not a v1 gate. Build with **py2app**
  (preferred on macOS) or PyInstaller, but only after it's verified to preserve
  the patched `mlx_lm` and launch offline. Until then, ship a documented
  "run from the venv" flow rather than implying a bundled app exists.
- **Do not bundle models** — they live in the external `models/` cache. Make
  `HF_HOME` configurable (default to the project `models/`, fall back to
  `~/.cache/huggingface` if absent).
- Bundle the frontend `web/` assets inside the app.
- Verify any built app still finds and uses the venv's patched `mlx_lm` (bundle
  the patched site-packages or document running from the venv).
- App must launch and run fully offline on whichever path ships.

---

## 15. Acceptance Criteria

1. **Offline:** with networking disabled, the app launches, lists the 4 models,
   loads any of them, and completes a multi-turn chat. Zero network attempts
   (verify with Little Snitch / `nettop`).
2. **Read-only models:** checksums of files under `models/` are identical before
   and after a session.
3. **Each model works:** `12B`, `31B`, `26B`, and `e4b` each load and answer.
4. **Thinking split:** the thinking trace appears in its own collapsible panel,
   never leaks channel markers (`<|channel>` etc.) into the answer.
5. **Rendering:** Markdown, fenced code with highlighting, and `$…$` / `$$…$$`
   math all render correctly; code is never math-mangled.
6. **Streaming:** output streams smoothly with no freeze on the fast `e4b`
   model (throttled rendering), and **Stop** aborts promptly.
7. **RAM:** switching models releases the previous model's memory — resident
   memory measured via **`mx.get_active_memory()`** (or Activity Monitor) returns
   to roughly the pre-load baseline, confirming the explicit **`mx.clear_cache()`**
   call on unload (§7.1) ran, not just Python GC.
8. **Adjustable params, not hardcoded:** on first launch every model uses
   `DEFAULT_PARAMS` with no pre-existing overrides; the Settings panel changes
   temperature / top-p / top-k / min-p / repetition penalty / max tokens /
   system prompt and the change visibly affects the next turn; only explicit
   user saves write to `settings.json`; tuning requires **no source edit**;
   "Reset to defaults" restores `DEFAULT_PARAMS`; model files stay unmodified.
9. **System prompt:** default is empty (none); setting a system prompt changes
   behavior on the next turn and resets the prompt cache; clearing it removes
   the system message.
10. **Per-model settings:** a model with no saved entry uses `DEFAULT_PARAMS`
    only; overrides exist only after the user saves with "This model" scope,
    persist and re-apply when that model is reselected, and don't affect other
    models; "All models" scope applies globally unless a per-model value
    overrides it.
11. **Frameless window:** the app launches with **no OS title bar and no header**
    (§12.1); the window can still be **moved** (via the custom drag strip),
    **minimized**, and **closed** (custom in-app controls), and text in the
    transcript remains selectable — i.e. the drag region does not hijack selection.
12. **Currency / `$` guard (§7.5):** plain prose containing `$` — e.g. "it costs
    $5 to $10" — renders as **literal text, not KaTeX math**, while deliberate
    delimiters (`$x$` inline and `$$…$$` display) still typeset correctly in the
    same message.

---

## 16. Reuse Map (from existing code)

| Existing | Reused for |
|---|---|
| `scripts/chat_pretty.py` thinking/answer state machine, `clean()` | `chatcore.ThinkingSplitter` |
| `scripts/chat_pretty.py` `mathify()` + tables | optional Unicode math fallback |
| `scripts/chat_pretty.py` sampler + `make_logits_processors` + `stream_generate` | `chatcore.generate` |
| `scripts/chat_pretty.py` rendering throttle (≈10 Hz) | frontend stream-flush cadence |
| `serve.sh` / model list logic | `chatcore.list_models` |
| venv shims `gemma4_unified*.py` | loading 12B/31B/26B |
| venv `gemma4_text.sanitize` KV-share patch | loading `e4b` |

---

## 17. Open Questions — Resolved for v1

1. **Local chat persistence (save/load JSON):** **Deferred** to Future (§18). It
   pulls in file I/O plus a chat-list UI; v1 stays "just chat," with
   conversations in memory only per §3.3.
2. **KaTeX vs. Unicode `mathify`:** **KaTeX is the v1 default.** We're already in
   a webview, so real typesetting is strictly better; `mathify()` remains only as
   the documented fallback (§7.5) if KaTeX is ever omitted.
3. **Light/dark:** **Follow system appearance only** in v1 (§12). A manual toggle
   is cheap to add later and is deferred.
4. **Read-only "About this model" panel** (arch, quant, context length, no
   editable fields): **Accepted as a stretch goal** — low-cost and aligned with
   the read-only/transparency theme, but **not blocking v1.** If included, it must
   stay strictly read-only (no fields that could imply editing model files).

*(Also resolved into v1: per-model parameter overrides keyed by model id, and a
free-form system-prompt field — see §8.)*

---

## 18. Future (explicitly out of v1 scope)

- Local chat history persistence and search.
- Multiple concurrent chats / tabs.
- Image/audio input for multimodal Gemma variants.
- Per-chat model memory (keep two models resident if RAM allows).
