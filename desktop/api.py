#!/usr/bin/env python3
"""The js_api bridge exposed to the web UI (spec §10).

JS calls ``window.pywebview.api.<method>`` (these return values); Python pushes
streaming events back with ``window.evaluate_js`` — every payload serialized via
``json.dumps`` so model output containing quotes/backticks/backslashes/newlines
can never break the JS call or inject script (spec §10).

Generation runs on a background thread so the UI thread stays responsive;
streamed chunks are coalesced and flushed at ~12 Hz (spec §10 perf rule).
"""
from __future__ import annotations

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import mlx.core as mx

import chatcore
import settings
from session import Session

FLUSH_INTERVAL = 0.08  # ~12.5 Hz bridge/render cadence


class _Streamer:
    """Coalesces (channel, chunk) events and flushes to JS at FLUSH_INTERVAL."""

    def __init__(self, window, turn_id: int, interval: float = FLUSH_INTERVAL):
        self._window = window
        self._turn_id = turn_id
        self._interval = interval
        self._buf = ""
        self._buf_channel: str | None = None
        self._last = 0.0

    def push(self, channel: str, text: str):
        if self._buf and channel != self._buf_channel:
            self._flush()                       # never mix channels in one call
        self._buf_channel = channel
        self._buf += text
        if time.monotonic() - self._last >= self._interval:
            self._flush()

    def _flush(self):
        if not self._buf:
            return
        fn = "onThinking" if self._buf_channel == "thinking" else "onAnswer"
        self._eval(f"{fn}({self._turn_id}, {json.dumps(self._buf)})")
        self._buf = ""
        self._last = time.monotonic()

    def finish(self):
        self._flush()

    def _eval(self, js: str):
        try:
            self._window.evaluate_js(js)
        except Exception:  # noqa: BLE001 — window may be closing; ignore
            pass


class Api:
    def __init__(self):
        self.session = Session()
        self.window = None
        self._scope = "model"          # current Settings scope (UI toggle)
        self._turn_id = 0
        self._busy = threading.Lock()  # one model/generation op at a time
        # All MLX work (load, generate, unload) runs on ONE dedicated thread:
        # MLX streams are thread-local, so crossing threads between load and
        # generate raises "no Stream(gpu) in current thread".
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mlx")

    def set_window(self, window):
        self.window = window

    def _eval(self, js: str):
        if self.window is None:
            return
        try:
            self.window.evaluate_js(js)
        except Exception:  # noqa: BLE001
            pass

    # --- model -------------------------------------------------------------
    def list_models(self):
        return chatcore.list_models()

    def current_model(self):
        return {"id": self.session.model_id}

    def load_model(self, model_id: str):
        """Start loading ``model_id`` on a background thread (spec §10).

        Returns immediately; progress/errors are reported via ``onModelStatus``.
        """
        if not self._busy.acquire(blocking=False):
            return {"ok": False, "error": "busy"}

        def worker():
            try:
                self._eval(
                    "onModelStatus(%s)"
                    % json.dumps({"state": "loading", "id": model_id})
                )
                before = mx.get_active_memory()
                self.session.load_model(model_id)
                after = mx.get_active_memory()
                print(f"[mem] load {model_id}: "
                      f"{before/1e9:.2f} -> {after/1e9:.2f} GB")
                self._eval(
                    "onModelStatus(%s)"
                    % json.dumps({"state": "ready", "id": model_id})
                )
            except chatcore.ModelLoadError as exc:
                self._eval(
                    "onModelStatus(%s)" % json.dumps(
                        {"state": "error", "id": model_id, "message": str(exc)})
                )
            except Exception as exc:  # noqa: BLE001
                self._eval(
                    "onModelStatus(%s)" % json.dumps(
                        {"state": "error", "id": model_id, "message": str(exc)})
                )
            finally:
                self._busy.release()

        self._pool.submit(worker)
        return {"ok": True}

    # --- conversation ------------------------------------------------------
    def new_chat(self):
        self.session.new_chat()
        return {"ok": True}

    def send_message(self, text: str):
        """Start generating a reply on a background thread; return its turn_id."""
        if self.session.model is None:
            return {"ok": False, "error": "no model loaded"}
        if not self._busy.acquire(blocking=False):
            return {"ok": False, "error": "busy"}

        self._turn_id += 1
        turn_id = self._turn_id
        self.session.arm_stop()
        params = settings.resolve_params(self.session.model_id)
        messages = self.session.begin_turn(text, params.get("system_prompt", ""))

        def worker():
            streamer = _Streamer(self.window, turn_id)
            answer_parts: list[str] = []
            try:
                for channel, chunk in chatcore.generate(
                    self.session.model, self.session.tokenizer, messages,
                    self.session.cache, params, self.session.stop_requested,
                ):
                    if channel == "answer":
                        answer_parts.append(chunk)
                    streamer.push(channel, chunk)
                streamer.finish()
                self.session.end_turn("".join(answer_parts))
                stopped = self.session.stop_requested()
                self._eval(
                    f"onDone({turn_id}, {json.dumps({'stopped': stopped})})")
            except Exception as exc:  # noqa: BLE001
                streamer.finish()
                # Still record whatever we produced so the cache stays coherent.
                self.session.end_turn("".join(answer_parts))
                self._eval(f"onError({turn_id}, {json.dumps(str(exc))})")
            finally:
                self._busy.release()

        self._pool.submit(worker)
        return {"ok": True, "turn_id": turn_id}

    def stop(self):
        self.session.request_stop()
        return {"ok": True}

    # --- settings ----------------------------------------------------------
    def get_settings(self):
        return settings.get_settings(self.session.model_id, self._scope)

    def set_scope(self, scope: str):
        self._scope = "global" if scope == "global" else "model"
        return {"ok": True, "scope": self._scope}

    def update_settings(self, partial: dict, scope: str = "model"):
        self._scope = "global" if scope == "global" else "model"
        params = settings.update_settings(partial, self._scope, self.session.model_id)
        return {"ok": True, "params": params}

    def reset_settings(self, scope: str = "model"):
        self._scope = "global" if scope == "global" else "model"
        params = settings.reset_settings(self._scope, self.session.model_id)
        return {"ok": True, "params": params}

    # --- window chrome (frameless, spec §12.1) -----------------------------
    def minimize_window(self):
        if self.window is not None:
            self.window.minimize()
        return {"ok": True}

    def close_window(self):
        if self.window is not None:
            self.window.destroy()
        return {"ok": True}

    def toggle_maximize(self):
        if self.window is None:
            return {"ok": False}
        try:
            if getattr(self.window, "maximized", False):
                self.window.restore()
            else:
                self.window.maximize()
        except Exception:  # noqa: BLE001 — maximize support varies
            return {"ok": False}
        return {"ok": True}
