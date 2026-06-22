#!/usr/bin/env python3
"""In-memory chat session state (spec §7.2).

Owns the resident model, its tokenizer, the per-chat ``prompt_cache``, the
transcript, and the cooperative stop flag. Conversations live in memory only
(spec §3.3) — nothing here is written to disk.

Multi-turn context is carried by the ``prompt_cache``: once a chat is going we
template only the *new* user turn and let the cache supply the prior KV
(matching scripts/chat_pretty.py). The cache is rebuilt — and the full history
re-fed — on New chat, model switch, or a system-prompt change (which invalidates
the cached prefix, spec §8).
"""
from __future__ import annotations

import threading

import mlx.core as mx
from mlx_lm.models.cache import make_prompt_cache

import chatcore

_UNSET = object()


class Session:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.model_id: str | None = None
        self.cache = None
        self.history: list[dict] = []      # full transcript, for display + re-feed
        self._stop = threading.Event()
        self._cache_started = False        # has the cache been primed this chat?
        self._system_in_cache = _UNSET     # system prompt baked into the cache

    # --- model lifecycle ---------------------------------------------------
    def load_model(self, model_id: str):
        """Load ``model_id``, unloading any previous model first (frees RAM)."""
        if model_id == self.model_id and self.model is not None:
            return
        self.unload()
        self.model, self.tokenizer = chatcore.load(model_id)
        self.model_id = model_id
        self.reset_cache()
        self.history = []

    def unload(self):
        """Drop the resident model + cache and force MLX to release Metal buffers.

        Dropping Python refs alone is not enough — MLX keeps a Metal buffer
        cache, so we must also clear the prompt_cache and call ``mx.clear_cache()``
        (spec §7.1) or resident memory won't return.
        """
        if self.model is None:
            return
        self.cache = None
        self.model = None
        self.tokenizer = None
        self.model_id = None
        self._cache_started = False
        self._system_in_cache = _UNSET
        mx.clear_cache()

    # --- chat lifecycle ----------------------------------------------------
    def reset_cache(self):
        """Build a fresh prompt cache (history is re-fed on the next turn)."""
        self.cache = make_prompt_cache(self.model) if self.model is not None else None
        self._cache_started = False
        self._system_in_cache = _UNSET

    def new_chat(self):
        self.history = []
        self.reset_cache()

    # --- turns -------------------------------------------------------------
    def begin_turn(self, user_text: str, system_prompt: str) -> list:
        """Record the user turn and return the messages to template this turn.

        If the cache is fresh (new chat / model switch / system change) we
        template the *whole* history with the system message; otherwise just the
        new user turn (the cache supplies the rest).
        """
        if system_prompt != self._system_in_cache and self._cache_started:
            # System prompt changed mid-chat -> cached prefix is invalid.
            self.reset_cache()
        self.history.append({"role": "user", "content": user_text})
        if not self._cache_started:
            messages = chatcore.build_messages(self.history, system_prompt)
            self._cache_started = True
            self._system_in_cache = system_prompt
        else:
            messages = [self.history[-1]]
        return messages

    def end_turn(self, answer_text: str):
        self.history.append({"role": "assistant", "content": answer_text})

    # --- stop flag ---------------------------------------------------------
    def arm_stop(self):
        self._stop.clear()

    def request_stop(self):
        self._stop.set()

    def stop_requested(self) -> bool:
        return self._stop.is_set()
