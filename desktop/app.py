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
    # gui=None lets pywebview pick the macOS (Cocoa/WKWebView) backend; no http.
    webview.start()


if __name__ == "__main__":
    main()
