# Packaging (deferred — stretch goal, not v1)

Per `docs/DESKTOP_APP_SPEC.md` §14, a standalone `.app` is **not** a v1 gate.
The supported v1 launch is running from the project venv:

```bash
.venv/bin/python desktop/app.py
```

A future py2app/PyInstaller bundle must preserve the patched `mlx_lm` gemma4
shims in site-packages (or it bundles the venv's site-packages wholesale), keep
`HF_HOME` pointed at the external `models/` cache (do **not** bundle models), and
still launch fully offline. None of that is built yet — this directory is a
placeholder so the structure matches the spec.
