#!/usr/bin/env python3
"""Local Chat — desktop entry point (spec §6, §12.1).

Run from the project venv so the patched mlx_lm gemma4 shims load:
    .venv/bin/python desktop/app.py

Builds a single frameless pywebview window over the vendored web UI and wires
the Python ``Api`` bridge. No sockets, no network.
"""
import os
from pathlib import Path

# Configure the offline HF cache before anything imports mlx_lm. Importing
# chatcore also sets these; doing it here keeps app.py self-documenting.
HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
MODELS_DIR = PROJECT_ROOT / "models"
if MODELS_DIR.is_dir():
    os.environ.setdefault("HF_HOME", str(MODELS_DIR))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import webview  # noqa: E402

from api import Api  # noqa: E402

WEB_DIR = HERE / "web"
INDEX = WEB_DIR / "index.html"
ICON = HERE / "assets" / "icon.png"


def _set_dock_icon(*_):
    """Set the macOS dock icon at runtime.

    We run from the venv (not a bundled .app), so the dock would otherwise show
    the generic Python icon. pywebview's Cocoa backend uses the shared
    NSApplication, so we set its icon image directly. Best-effort and silent;
    applied both before start and on `shown` so it survives app launch.
    """
    if not ICON.exists():
        return
    try:
        from AppKit import NSApplication, NSImage
        image = NSImage.alloc().initWithContentsOfFile_(str(ICON))
        if image is not None:
            NSApplication.sharedApplication().setApplicationIconImage_(image)
    except Exception:  # noqa: BLE001 — icon is cosmetic, never fail launch
        pass


def main():
    api = Api()
    window = webview.create_window(
        "Local Chat",
        url=str(INDEX),
        js_api=api,
        width=1080,
        height=760,
        min_size=(760, 520),
        frameless=True,
        easy_drag=False,           # only our scoped drag strip moves the window
        background_color="#101114",
    )
    api.set_window(window)
    _set_dock_icon()
    window.events.shown += _set_dock_icon   # re-apply after app finishes launching
    # gui=None lets pywebview pick the macOS (Cocoa/WKWebView) backend; no http.
    webview.start()


if __name__ == "__main__":
    main()
