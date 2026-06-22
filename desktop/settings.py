#!/usr/bin/env python3
"""Runtime settings persistence (spec §8.2).

Stores user parameter overrides in a sparse local JSON file, created **only on
the first save**. Resolution order for a model is:

    DEFAULT_PARAMS (code)  ->  global overrides  ->  by_model[model_id] overrides

so a model the user never customizes always resolves to pure ``DEFAULT_PARAMS``
(plus any global overlay). Nothing under ``models/`` or the project is touched.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import chatcore

CONFIG_DIR = Path.home() / "Library" / "Application Support" / "LocalChat"
CONFIG_PATH = CONFIG_DIR / "settings.json"


def _read() -> dict:
    """Return the raw config dict, or ``{}`` if absent/unreadable/corrupt."""
    try:
        with open(CONFIG_PATH, encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}


def _write(data: dict) -> None:
    """Atomically write the config (creating the dir on first save)."""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    tmp = CONFIG_PATH.with_name(CONFIG_PATH.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, CONFIG_PATH)


def resolve_params(model_id: str | None) -> dict:
    """Effective params for ``model_id``: defaults <- global <- per-model."""
    data = _read()
    merged = dict(chatcore.DEFAULT_PARAMS)
    merged.update(chatcore.validate_params(data.get("global")))
    if model_id:
        by_model = data.get("by_model") or {}
        merged.update(chatcore.validate_params(by_model.get(model_id)))
    return merged


def get_settings(model_id: str | None, scope: str = "model") -> dict:
    """Snapshot for the Settings panel (effective params + defaults + overrides)."""
    data = _read()
    by_model = data.get("by_model") or {}
    return {
        "model_id": model_id,
        "params": resolve_params(model_id),
        "defaults": dict(chatcore.DEFAULT_PARAMS),
        "scope": scope,
        "global_overrides": data.get("global") or {},
        "model_overrides": by_model.get(model_id) or {},
    }


def update_settings(partial: dict, scope: str, model_id: str | None) -> dict:
    """Validate ``partial``, sparse-merge into the chosen scope, persist.

    ``scope`` is ``"global"`` (All models) or ``"model"`` (This model). Only the
    keys actually provided are stored. Returns the new resolved params.
    """
    clean = chatcore.validate_params(partial)
    if not clean:
        return resolve_params(model_id)
    data = _read()
    if scope == "global":
        bucket = dict(data.get("global") or {})
        bucket.update(clean)
        data["global"] = bucket
    else:
        if not model_id:
            return resolve_params(model_id)
        by_model = dict(data.get("by_model") or {})
        bucket = dict(by_model.get(model_id) or {})
        bucket.update(clean)
        by_model[model_id] = bucket
        data["by_model"] = by_model
    _write(data)
    return resolve_params(model_id)


def reset_settings(scope: str, model_id: str | None) -> dict:
    """Clear the chosen scope's saved overrides (spec §8.2).

    No-op (no file created) when there is nothing persisted yet.
    """
    if not CONFIG_PATH.exists():
        return resolve_params(model_id)
    data = _read()
    if scope == "global":
        data.pop("global", None)
    else:
        by_model = data.get("by_model") or {}
        by_model.pop(model_id, None)
        if by_model:
            data["by_model"] = by_model
        else:
            data.pop("by_model", None)
    _write(data)
    return resolve_params(model_id)
