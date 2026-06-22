# Local Chat — desktop app

A small, **offline** macOS chat UI for the local MLX Gemma models in this project.
It is the GUI sibling of `scripts/chat_pretty.py` and reuses the same `.venv` /
`mlx_lm` stack (thinking/answer split, sampler + `stream_generate`, the throttle),
upgraded to real Markdown + KaTeX in a `pywebview` window.

See `docs/DESKTOP_APP_SPEC.md` for the full specification.

## Run (the supported v1 path)

From the **project root**, using the project venv so the patched `mlx_lm`
gemma4 shims load:

```bash
.venv/bin/python desktop/app.py
```

That's it — no build step, no server, no network. The window is frameless: use
the in-app dots (top-left) to close / minimize, and drag the strip between the
model picker and the right-hand buttons to move the window.

## What it does

- **Pick a model** (top-left picker or the start screen) — discovered by scanning
  `models/hub/models--*`. Loading one unloads the previous model and calls
  `mx.clear_cache()` so RAM returns.
- **Chat** multi-turn with a live-streaming answer. The model's hidden *thinking*
  trace streams into a separate, collapsible panel that auto-collapses when the
  answer begins.
- **Render** Markdown, fenced code (highlight.js), and math (`$…$`, `$$…$$`) via
  KaTeX. Plain prose like "$5 to $10" stays literal — it is not parsed as math.
- **Settings** (gear → drawer): temperature, top-p, top-k, min-p, repetition
  penalty + window, max tokens, seed, and a system prompt. Changes **auto-save**
  (no Save button) and apply to the next turn. Scope toggle: **This model** vs
  **All models**. "Reset to defaults" clears the chosen scope.

## Where things live

| File | Role |
|---|---|
| `app.py` | entry point: builds the frameless window, wires the bridge |
| `api.py` | `js_api` methods exposed to JS; streams tokens back via `evaluate_js` (json-encoded, ~12 Hz) |
| `chatcore.py` | shared backend: `list_models`, `load`, `ThinkingSplitter`, `generate`, `DEFAULT_PARAMS`, `validate_params`, `mathify` |
| `session.py` | in-memory chat state: model handle, prompt cache, transcript, stop flag |
| `settings.py` | sparse runtime settings persistence + `resolve_params` |
| `web/` | `index.html`, `styles.css`, `app.js`, and `vendor/` (markdown-it, KaTeX + fonts, highlight.js, texmath) — all local |

### Inference parameters

Defaults live in exactly one place — `chatcore.DEFAULT_PARAMS`. They are **not**
per-model presets and are not meant to be tuned by editing source: change them at
runtime in the Settings panel. User overrides persist **sparsely** (only the keys
you changed) to:

```
~/Library/Application Support/LocalChat/settings.json
```

That file does not exist until your first change. A model you never customize
always resolves to pure `DEFAULT_PARAMS` (plus any "All models" overlay).

## Settings file

The only file the app ever writes is `settings.json` above. Nothing under
`models/` is touched (model files are read-only), and conversations live in
memory only (no chat history on disk in v1).

## Dev checks (optional)

- `.venv/bin/python desktop/_smoke.py` — headless backend proof against `e4b`
  (load, thinking/answer split, no marker leakage, RAM release).
- `.venv/bin/python desktop/_app_test.py` — drives the real window end-to-end
  (select model → stream a reply) and verifies the bridge + rendering.

## Packaging

Building a standalone `.app` is a **stretch goal, not part of v1** — see
`docs/DESKTOP_APP_SPEC.md` §14 for why (Python 3.14 + a monkeypatched `mlx_lm`
in site-packages is the riskiest path to bundle). For now, run from the venv as
shown above. `packaging/` is a placeholder for that future work.
