#!/usr/bin/env python3
"""Shared inference backend for the Local Chat desktop app.

This is the GUI sibling of ``scripts/chat_pretty.py``: it extracts and reuses the
validated pieces of that terminal tool — the token-level thinking/answer split,
``clean()``, the sampler + ``make_logits_processors`` + ``stream_generate`` path,
and ``mathify()`` — so the desktop UI behaves identically to the CLI.

Nothing here writes to ``models/`` or mutates model files. Inference parameters
are *runtime* values (see ``DEFAULT_PARAMS`` — the single source of truth); they
are never baked into per-model constants.
"""
from __future__ import annotations

import os
import re
import shutil
from pathlib import Path

# --- Offline / cache wiring -------------------------------------------------
# Mirror scripts/chat.sh: HF_HOME -> project models cache, strictly offline.
# Done before importing mlx_lm so no library ever attempts a network fetch.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
if MODELS_DIR.is_dir():
    os.environ["HF_HOME"] = str(MODELS_DIR)
# else: fall back to the inherited / default HF cache (~/.cache/huggingface).
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import mlx.core as mx  # noqa: E402
from mlx_lm.utils import load as _mlx_load  # noqa: E402
from mlx_lm.generate import stream_generate  # noqa: E402
from mlx_lm.sample_utils import make_sampler, make_logits_processors  # noqa: E402
from mlx_lm.models.cache import make_prompt_cache  # noqa: E402


# ===========================================================================
# Default inference parameters — THE single source of truth (spec §8.1).
# Tuning happens at runtime via the Settings panel + settings.json, never by
# editing these constants and never per-model in source.
# ===========================================================================
DEFAULT_PARAMS: dict = {
    "max_tokens": 8192,
    "temperature": 0.7,
    "top_p": 1.0,
    "top_k": 0,
    "min_p": 0.0,
    "repetition_penalty": 1.1,
    "repetition_context_size": 64,
    "seed": 0,
    "system_prompt": "",
}

# (low, high, caster) per numeric param. ``None`` bound = unbounded that side.
_RANGES: dict = {
    "max_tokens": (64, 32768, int),
    "temperature": (0.0, 2.0, float),
    "top_p": (0.0, 1.0, float),
    "top_k": (0, 200, int),
    "min_p": (0.0, 1.0, float),
    "repetition_penalty": (1.0, 2.0, float),
    "repetition_context_size": (0, 4096, int),
    "seed": (0, None, int),
}


def _clamp(value, lo, hi, cast):
    try:
        v = cast(value)
    except (TypeError, ValueError):
        raise ValueError(f"expected {cast.__name__}, got {value!r}")
    if lo is not None and v < lo:
        v = lo
    if hi is not None and v > hi:
        v = hi
    return v


def validate_params(partial: dict | None) -> dict:
    """Clamp/cast a (possibly sparse) param dict to allowed ranges (spec §8).

    Returns only the keys present in ``partial`` (sparse-friendly: used both for
    full effective params and for persisting just-changed keys). Unknown keys
    are dropped; ``seed`` is floored at 0.
    """
    out: dict = {}
    if not partial:
        return out
    for key, value in partial.items():
        if key == "system_prompt":
            out[key] = "" if value is None else str(value)
        elif key in _RANGES:
            lo, hi, cast = _RANGES[key]
            out[key] = _clamp(value, lo, hi, cast)
        # silently ignore unrecognised keys
    return out


def effective_params(params: dict | None) -> dict:
    """DEFAULT_PARAMS overlaid with a validated ``params`` dict (full result)."""
    p = dict(DEFAULT_PARAMS)
    p.update(validate_params(params))
    return p


# ===========================================================================
# Model discovery (spec §7.1) — scan the local HF cache, same source as the CLI.
# ===========================================================================
# Friendly labels are display-only (not tunable params). Unknown ids get a
# derived label so newly added models still show something readable.
_LABELS: dict = {
    "mlx-community/gemma-4-12B-it-qat-6bit": "Gemma 4 12B (6bit QAT)",
    "mlx-community/gemma-4-31B-it-qat-4bit": "Gemma 4 31B (4bit QAT)",
    "mlx-community/gemma-4-26b-a4b-8bit": "Gemma 4 26B-A4B (8bit MoE)",
    "mlx-community/gemma-4-e4b-it-4bit": "Gemma 4 E4B (4bit)",
}


def _derive_label(model_id: str) -> str:
    if model_id in _LABELS:
        return _LABELS[model_id]
    return model_id.split("/")[-1]


def _gguf_label(path: Path) -> str:
    return path.stem


def list_models() -> list[dict]:
    """Return ``[{id, label, path, kind}]`` for every local model.

    Two kinds are discovered:
      - ``mlx``: HF-cache dirs under ``models/hub/models--*``. Directory
        ``models--mlx-community--gemma-4-12B-it-qat-6bit`` maps to id
        ``mlx-community/gemma-4-12B-it-qat-6bit`` (the org/name separator is the
        only ``--`` since model names never contain ``--``). mlx_lm loads these,
        including plain (non-quantized) HF checkpoints of any of its supported
        architectures — not just pre-quantized ``mlx-community`` repos.
      - ``gguf``: ``*.gguf`` files in ``models/gguf/`` or downloaded into the hub
        (``models--*/snapshots/*/*.gguf``). The file path is the id; loading uses
        llama.cpp (see ``GGUFBackend``).
    """
    hub = MODELS_DIR / "hub"
    out: list[dict] = []
    if hub.is_dir():
        for entry in sorted(hub.iterdir()):
            if entry.is_dir() and entry.name.startswith("models--"):
                model_id = entry.name[len("models--"):].replace("--", "/")
                out.append({
                    "id": model_id,
                    "label": _derive_label(model_id),
                    "path": str(entry),
                    "kind": "mlx",
                })

    gguf_paths: list[Path] = []
    gdir = MODELS_DIR / "gguf"
    if gdir.is_dir():
        gguf_paths += sorted(gdir.glob("*.gguf"))
    if hub.is_dir():
        # Bounded glob (snapshots only) so we don't walk the whole 69 GB cache.
        gguf_paths += sorted(hub.glob("models--*/snapshots/*/*.gguf"))
    seen: set[str] = set()
    for p in gguf_paths:
        # Dedupe by the real file (hub entries are symlinks into blobs/), but
        # keep the .gguf-named path as the id: resolving to the blob strips the
        # extension, which would misroute load() to the MLX backend.
        real = str(p.resolve())
        if real in seen:
            continue
        seen.add(real)
        rp = str(p)
        out.append({"id": rp, "label": _gguf_label(p), "path": rp, "kind": "gguf"})
    return out


def delete_model(model_id: str) -> str:
    """Permanently delete a model's files from the local cache and return the path.

    The app is otherwise read-only under ``models/`` (spec §3.2); this is an
    explicit, user-initiated deletion. Guarded to refuse anything that does not
    resolve to a path strictly inside the models cache.
    """
    models_root = MODELS_DIR.resolve()
    hub_root = (MODELS_DIR / "hub").resolve()
    if model_id.endswith(".gguf"):
        p = Path(model_id)
        # A hub GGUF id is a snapshot symlink into blobs/. Resolving it would
        # point at the bare blob, so deleting that leaves the whole models--*
        # repo dir (and a dangling symlink) behind — the model would keep
        # showing up. Delete the entire repo dir instead. A standalone file in
        # models/gguf/ has no models--* ancestor, so we delete the file itself.
        target = next(
            (par for par in p.parents
             if par.parent == hub_root and par.name.startswith("models--")),
            p,
        )
    else:
        target = MODELS_DIR / "hub" / ("models--" + model_id.replace("/", "--"))
    target = target.resolve()
    if target == models_root or models_root not in target.parents:
        raise ValueError(f"refusing to delete outside the models cache: {target}")
    if not target.exists():
        raise FileNotFoundError(f"not found: {target}")
    if target.is_dir():
        shutil.rmtree(target)
    else:
        target.unlink()
    return str(target)


class ModelLoadError(Exception):
    """Raised when a model cannot be loaded (unsupported arch, corrupt cache…)."""


def load(model_id: str) -> "Backend":
    """Load ``model_id`` and return a ``Backend`` (MLX or GGUF).

    Dispatch is by id: a ``*.gguf`` path uses llama.cpp; anything else is an
    mlx_lm model id. Raises ``ModelLoadError`` (a typed error the UI surfaces)
    on failure.
    """
    if model_id.endswith(".gguf"):
        return _load_gguf(model_id)
    try:
        model, tokenizer = _mlx_load(model_id)
    except Exception as exc:  # noqa: BLE001 — surface any load failure uniformly
        raise ModelLoadError(_friendly_load_error(str(exc))) from exc
    _ensure_turn_stops(tokenizer)
    return MLXBackend(model, tokenizer)


def _friendly_load_error(msg: str) -> str:
    """Turn a raw mlx_lm load failure into a clear, actionable message."""
    low = msg.lower()
    if "diffusion" in low:
        return ("This is a diffusion (block/denoising) language model — it doesn't "
                "generate text autoregressively. Local Chat runs autoregressive "
                "models via mlx_lm/llama.cpp, so this architecture can't run here. "
                f"({msg.strip()})")
    if "not supported" in low:
        return (f"{msg.strip()} This architecture isn't implemented in the "
                "installed mlx_lm build — try a different model.")
    return msg


# Turn terminators that should stop generation. Gemma 4's harmony-style format
# ends an assistant turn with ``<turn|>``, and older Gemma builds use
# ``<end_of_turn>`` — but some converted checkpoints declare only ``<eos>`` as
# the eos id, so ``stream_generate`` would run past the answer to max_tokens.
_TURN_STOP_TOKENS = ("<turn|>", "<end_of_turn>")


def _ensure_turn_stops(tokenizer) -> None:
    """Add any present turn terminator to the tokenizer's stop set (in place)."""
    eos = getattr(tokenizer, "eos_token_ids", None)
    if eos is None:
        return
    try:
        vocab = tokenizer.get_vocab()
    except Exception:  # noqa: BLE001
        return
    for tok in _TURN_STOP_TOKENS:
        if tok in vocab:
            eos.add(vocab[tok])


class Backend:
    """Common interface so the rest of the app is engine-agnostic."""
    kind = "?"

    def new_cache(self):
        """Return a fresh per-chat cache object (or None if the engine self-manages)."""
        return None

    def generate(self, messages, cache, params, stop_flag=None):
        """Yield ``(channel, text_chunk)`` events for one assistant turn."""
        raise NotImplementedError

    def close(self):
        """Release the model and any device memory."""


class MLXBackend(Backend):
    kind = "mlx"

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def new_cache(self):
        return make_prompt_cache(self.model)

    def generate(self, messages, cache, params, stop_flag=None):
        yield from generate(self.model, self.tokenizer, messages, cache,
                            params, stop_flag)

    def close(self):
        # Drop refs AND clear MLX's Metal buffer cache (spec §7.1) — GC alone
        # won't return resident memory.
        self.model = None
        self.tokenizer = None
        mx.clear_cache()


class GGUFBackend(Backend):
    kind = "gguf"

    def __init__(self, llm):
        self.llm = llm

    def new_cache(self):
        # llama.cpp keeps its own KV cache; reset it so a new chat starts clean.
        try:
            self.llm.reset()
        except Exception:  # noqa: BLE001
            pass
        return None

    def generate(self, messages, cache, params, stop_flag=None):
        yield from _gguf_generate(self.llm, messages, params, stop_flag)

    def close(self):
        try:
            self.llm.close()
        except Exception:  # noqa: BLE001
            pass
        self.llm = None


def _load_gguf(path: str) -> "GGUFBackend":
    try:
        from llama_cpp import Llama
    except ImportError as exc:
        raise ModelLoadError(
            "GGUF models need llama-cpp-python, which isn't installed. "
            "See desktop/README.md (Models & formats) for the install command — "
            "on this beta SDK it requires the packaging/sdk-shim include path."
        ) from exc
    try:
        llm = Llama(
            model_path=path,
            n_gpu_layers=-1,     # offload all layers to Metal
            n_ctx=8192,
            verbose=False,
        )
    except Exception as exc:  # noqa: BLE001
        raise ModelLoadError(str(exc)) from exc
    return GGUFBackend(llm)


def _gguf_generate(llm, messages, params, stop_flag=None):
    """Stream one turn from a llama.cpp model as ``(channel, chunk)`` events.

    Uses the GGUF's embedded chat template via ``create_chat_completion``. The
    full conversation is passed each turn (llama.cpp reuses its cached prefix);
    thinking is detected from textual ``<think>…</think>`` tags (the GGUF path
    has no token-level thinking markers), so non-reasoning models simply stream a
    plain answer.
    """
    p = effective_params(params)
    kwargs = dict(
        messages=messages,
        max_tokens=p["max_tokens"],
        temperature=p["temperature"],
        top_p=p["top_p"],
        top_k=int(p["top_k"]),
        min_p=p["min_p"],
        repeat_penalty=p["repetition_penalty"],
        stream=True,
    )
    try:
        kwargs["seed"] = int(p["seed"])
        stream = llm.create_chat_completion(**kwargs)
    except TypeError:
        kwargs.pop("seed", None)   # older llama_cpp: seed not a per-call arg
        stream = llm.create_chat_completion(**kwargs)

    splitter = TagThinkingSplitter()
    for chunk in stream:
        if stop_flag is not None and stop_flag():
            break
        try:
            delta = chunk["choices"][0]["delta"]
        except (KeyError, IndexError):
            continue
        text = delta.get("content")
        if not text:
            continue
        for ev in splitter.push(text):
            yield ev
    for ev in splitter.flush():
        yield ev


def _prefix_hold(buf: str, marker: str) -> int:
    """Length of the longest suffix of ``buf`` that is a proper prefix of ``marker``.

    Lets the tag splitter hold back just enough trailing text to detect a marker
    that straddles two streamed chunks, without buffering more than necessary.
    """
    for k in range(min(len(marker) - 1, len(buf)), 0, -1):
        if buf.endswith(marker[:k]):
            return k
    return 0


class TagThinkingSplitter:
    """Split a plain-text stream on ``<think>…</think>`` into thinking/answer.

    Used by the GGUF/HF path (no token-level markers). Holds back a short tail so
    a tag split across chunks is still detected; a model that never emits a
    ``<think>`` tag just streams everything as the answer.
    """
    START = "<think>"
    END = "</think>"

    def __init__(self):
        self.state = "normal"
        self.buf = ""

    def push(self, text: str) -> list:
        self.buf += text
        return self._run(flush=False)

    def flush(self) -> list:
        return self._run(flush=True)

    def _run(self, flush: bool) -> list:
        events: list = []
        while True:
            if self.state == "normal":
                i = self.buf.find(self.START)
                if i != -1:
                    if i > 0:
                        events.append(("answer", self.buf[:i]))
                    self.buf = self.buf[i + len(self.START):]
                    self.state = "reasoning"
                    continue
                hold = 0 if flush else _prefix_hold(self.buf, self.START)
                cut = len(self.buf) - hold
                if cut > 0:
                    events.append(("answer", self.buf[:cut]))
                    self.buf = self.buf[cut:]
                break
            else:  # reasoning
                i = self.buf.find(self.END)
                if i != -1:
                    if i > 0:
                        events.append(("thinking", self.buf[:i]))
                    self.buf = self.buf[i + len(self.END):]
                    self.state = "normal"
                    continue
                hold = 0 if flush else _prefix_hold(self.buf, self.END)
                cut = len(self.buf) - hold
                if cut > 0:
                    events.append(("thinking", self.buf[:cut]))
                    self.buf = self.buf[cut:]
                break
        return events


# ===========================================================================
# Thinking / answer split — token-level state machine (spec §7.3).
# Ported from scripts/chat_pretty.py (the marker stash + clean() scrub), but
# refactored from inline closures into a reusable class that emits
# ("thinking"|"answer", text_chunk) events.
# ===========================================================================
class ThinkingSplitter:
    """Routes streamed tokens into the thinking vs answer channel.

    Gemma 4 builds use OpenAI-style ``<|channel>thought … <channel|>`` channels.
    The tokenizer exposes ``has_thinking`` plus ``think_start_tokens`` (a two
    token marker: ts0 then ts1) and ``think_end_tokens`` (single token). We stash
    ts0 until we see whether ts1 follows, so a lone ts0 that is *not* the start
    marker is emitted rather than swallowed.
    """

    def __init__(self, tokenizer, prompt_tokens):
        self.has_think = bool(getattr(tokenizer, "has_thinking", False))
        ts_tokens = tuple(getattr(tokenizer, "think_start_tokens", ()) or ())
        te_tokens = tuple(getattr(tokenizer, "think_end_tokens", ()) or ())
        self.ts0 = ts_tokens[0] if len(ts_tokens) >= 1 else None
        self.ts1 = ts_tokens[1] if len(ts_tokens) >= 2 else None
        self.te_tokens = te_tokens
        # Marker *strings* used by clean() as a belt-and-suspenders scrub.
        self.ts_str = getattr(tokenizer, "think_start", None)
        self.te_str = getattr(tokenizer, "think_end", None)

        starts_thinking = False
        if self.has_think:
            try:
                starts_thinking = (
                    tokenizer.rfind_think_start(prompt_tokens)
                    > tokenizer.rfind_think_end(prompt_tokens)
                )
            except Exception:  # noqa: BLE001
                starts_thinking = False
        self.state = "reasoning" if (self.has_think and starts_thinking) else "normal"
        self.hold_start = False
        self.ts0_txt = ""

    def clean(self, s: str) -> str:
        """Strip channel control strings that slip past token-level filtering."""
        for marker in (self.ts_str, self.te_str, "<|channel>", "<channel|>"):
            if marker:
                s = s.replace(marker, "")
        return s

    def _emit(self, text: str, events: list):
        text = self.clean(text)
        if not text:
            return
        channel = "thinking" if self.state == "reasoning" else "answer"
        events.append((channel, text))

    def push(self, token_id, text: str) -> list:
        """Feed one streamed token; return 0+ ``(channel, chunk)`` events."""
        events: list = []
        text = text or ""
        if self.has_think:
            if self.hold_start:
                self.hold_start = False
                if self.ts1 is not None and token_id == self.ts1:
                    self.state = "reasoning"
                    return events  # swallow the whole two-token start marker
                self._emit(self.ts0_txt, events)  # lone ts0 — not a marker
            if len(self.te_tokens) == 1 and token_id == self.te_tokens[0]:
                self.state = "normal"
                return events
            if self.ts1 is not None and token_id == self.ts0:
                self.hold_start = True
                self.ts0_txt = text
                return events
            if self.ts1 is None and self.ts0 is not None and token_id == self.ts0:
                self.state = "reasoning"
                return events
        self._emit(text, events)
        return events

    def flush(self) -> list:
        """Flush any token held pending the two-token start-marker decision."""
        events: list = []
        if self.hold_start:
            self.hold_start = False
            self._emit(self.ts0_txt, events)
        return events


def build_messages(history: list, system_prompt: str = "") -> list:
    """Prepend a system message when ``system_prompt`` is non-empty (spec §8)."""
    msgs: list = []
    if system_prompt:
        msgs.append({"role": "system", "content": system_prompt})
    msgs.extend(history)
    return msgs


# Some converted checkpoints ship a tokenizer with no ``chat_template`` baked in
# (notably the gemma-4 base / 26b-a4b MoE conversions, which carry no
# ``chat_template.jinja``). transformers then refuses to template at all
# ("tokenizer.chat_template is not set"). We vendor the canonical Gemma 4 chat
# template (the harmony-style ``<|turn>`` / ``<|channel>`` format these builds
# were trained on — NOT the old ``<start_of_turn>`` Gemma 1-3 format) and supply
# it as the fallback so templating still works fully offline. The template
# handles the system role natively, so no system-merge is needed on this path.
_FALLBACK_TEMPLATE_PATH = Path(__file__).resolve().parent / "gemma4_chat_template.jinja"
_fallback_template_cache: str | None = None


def _fallback_template() -> str:
    global _fallback_template_cache
    if _fallback_template_cache is None:
        _fallback_template_cache = _FALLBACK_TEMPLATE_PATH.read_text(encoding="utf-8")
    return _fallback_template_cache


def _merge_system(messages: list) -> list:
    """Fold system messages into the following user turn (Gemma has no system role)."""
    merged: list = []
    sys_txt = ""
    for m in messages:
        if m.get("role") == "system":
            sys_txt += (m.get("content") or "") + "\n\n"
        else:
            if sys_txt and m.get("role") == "user":
                m = {"role": "user", "content": sys_txt + (m.get("content") or "")}
                sys_txt = ""
            merged.append(m)
    if sys_txt:  # no user turn to attach to — prepend as a lone user message
        merged.insert(0, {"role": "user", "content": sys_txt.strip()})
    return merged


def _has_chat_template(tokenizer) -> bool:
    """True if the tokenizer can template on its own (mlx exposes this directly)."""
    flag = getattr(tokenizer, "has_chat_template", None)
    if flag is not None:
        return bool(flag)
    return bool(getattr(tokenizer, "chat_template", None))


def _apply_template(tokenizer, messages):
    """Apply the chat template, tolerating models without one or without a system role.

    If the checkpoint ships no template at all we supply the vendored Gemma 4
    template (``_fallback_template``). Otherwise we use the tokenizer's own
    template, merging any system text into the first user turn if that template
    rejects a standalone system role (older Gemma 1-3 builds).
    """
    if not _has_chat_template(tokenizer):
        return tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, chat_template=_fallback_template()
        )
    try:
        return tokenizer.apply_chat_template(messages, add_generation_prompt=True)
    except Exception:  # noqa: BLE001 — template present but rejects the system role
        return tokenizer.apply_chat_template(
            _merge_system(messages), add_generation_prompt=True
        )


def generate(model, tokenizer, messages, cache, params, stop_flag=None):
    """Stream a single assistant turn as ``(channel, chunk)`` events (spec §9).

    ``messages`` is the *final* message list to template for this turn (the
    caller decides whether the system message and/or prior history is included —
    when a ``cache`` is reused across turns, pass only the new turn). ``params``
    is overlaid on ``DEFAULT_PARAMS`` and validated here. ``stop_flag`` is a
    zero-arg callable polled each step for cooperative cancellation.
    """
    p = effective_params(params)
    prompt = _apply_template(tokenizer, messages)
    splitter = ThinkingSplitter(tokenizer, prompt)

    sampler = make_sampler(
        p["temperature"], p["top_p"], min_p=p["min_p"], top_k=p["top_k"],
    )
    logits_processors = None
    if p["repetition_penalty"] and p["repetition_penalty"] != 1.0:
        logits_processors = make_logits_processors(
            repetition_penalty=p["repetition_penalty"],
            repetition_context_size=p["repetition_context_size"],
        )
    # Seed is a no-op while greedy (temperature == 0); set it anyway so stochastic
    # runs are reproducible.
    mx.random.seed(int(p["seed"]))

    for resp in stream_generate(
        model, tokenizer, prompt,
        max_tokens=p["max_tokens"], sampler=sampler,
        prompt_cache=cache, logits_processors=logits_processors,
    ):
        if stop_flag is not None and stop_flag():
            break
        for ev in splitter.push(resp.token, resp.text or ""):
            yield ev
    for ev in splitter.flush():
        yield ev


# ===========================================================================
# mathify() — LaTeX → Unicode. Ported verbatim from scripts/chat_pretty.py.
# Used only as the documented fallback when KaTeX is not the renderer (spec
# §7.5, §17 #2); the desktop UI typesets with KaTeX by default.
# ===========================================================================
_MATH_SYMBOLS = {
    "alpha": "α", "beta": "β", "gamma": "γ", "delta": "δ", "epsilon": "ε",
    "varepsilon": "ε", "zeta": "ζ", "eta": "η", "theta": "θ", "vartheta": "ϑ",
    "iota": "ι", "kappa": "κ", "lambda": "λ", "mu": "μ", "nu": "ν", "xi": "ξ",
    "pi": "π", "rho": "ρ", "sigma": "σ", "tau": "τ", "upsilon": "υ", "phi": "φ",
    "varphi": "φ", "chi": "χ", "psi": "ψ", "omega": "ω",
    "Gamma": "Γ", "Delta": "Δ", "Theta": "Θ", "Lambda": "Λ", "Xi": "Ξ",
    "Pi": "Π", "Sigma": "Σ", "Upsilon": "Υ", "Phi": "Φ", "Psi": "Ψ", "Omega": "Ω",
    "times": "×", "cdot": "·", "div": "÷", "pm": "±", "mp": "∓", "ast": "∗",
    "leq": "≤", "le": "≤", "geq": "≥", "ge": "≥", "neq": "≠", "ne": "≠",
    "equiv": "≡", "approx": "≈", "cong": "≅", "sim": "∼", "simeq": "≃",
    "propto": "∝", "ll": "≪", "gg": "≫", "subset": "⊂", "subseteq": "⊆",
    "supset": "⊃", "supseteq": "⊇", "in": "∈", "notin": "∉", "ni": "∋",
    "cup": "∪", "cap": "∩", "setminus": "∖", "forall": "∀", "exists": "∃",
    "nexists": "∄", "emptyset": "∅", "varnothing": "∅", "neg": "¬",
    "land": "∧", "wedge": "∧", "lor": "∨", "vee": "∨", "oplus": "⊕",
    "otimes": "⊗", "rightarrow": "→", "to": "→", "longrightarrow": "⟶",
    "leftarrow": "←", "gets": "←", "Rightarrow": "⇒", "implies": "⇒",
    "Leftarrow": "⇐", "iff": "⇔", "Leftrightarrow": "⇔", "leftrightarrow": "↔",
    "mapsto": "↦", "uparrow": "↑", "downarrow": "↓", "infty": "∞",
    "partial": "∂", "nabla": "∇", "sum": "∑", "prod": "∏", "int": "∫",
    "oint": "∮", "sqrt": "√", "angle": "∠", "perp": "⊥", "parallel": "∥",
    "mid": "∣", "langle": "⟨", "rangle": "⟩", "lceil": "⌈", "rceil": "⌉",
    "lfloor": "⌊", "rfloor": "⌋", "ldots": "…", "dots": "…", "cdots": "⋯",
    "vdots": "⋮", "ddots": "⋱", "circ": "∘", "bullet": "•", "prime": "′",
    "degree": "°", "Re": "ℜ", "Im": "ℑ", "aleph": "ℵ", "hbar": "ℏ", "ell": "ℓ",
}
_MATH_BB = {"R": "ℝ", "N": "ℕ", "Z": "ℤ", "Q": "ℚ", "C": "ℂ", "H": "ℍ", "P": "ℙ"}
_SUP = {"0": "⁰", "1": "¹", "2": "²", "3": "³", "4": "⁴", "5": "⁵", "6": "⁶",
        "7": "⁷", "8": "⁸", "9": "⁹", "+": "⁺", "-": "⁻", "=": "⁼", "(": "⁽",
        ")": "⁾", "n": "ⁿ", "i": "ⁱ", "T": "ᵀ", "a": "ᵃ", "b": "ᵇ", "c": "ᶜ",
        "d": "ᵈ", "e": "ᵉ", "f": "ᶠ", "g": "ᵍ", "h": "ʰ", "j": "ʲ", "k": "ᵏ",
        "l": "ˡ", "m": "ᵐ", "o": "ᵒ", "p": "ᵖ", "r": "ʳ", "s": "ˢ", "t": "ᵗ",
        "u": "ᵘ", "v": "ᵛ", "w": "ʷ", "x": "ˣ", "y": "ʸ", "z": "ᶻ"}
_SUB = {"0": "₀", "1": "₁", "2": "₂", "3": "₃", "4": "₄", "5": "₅", "6": "₆",
        "7": "₇", "8": "₈", "9": "₉", "+": "₊", "-": "₋", "=": "₌", "(": "₍",
        ")": "₎", "a": "ₐ", "e": "ₑ", "h": "ₕ", "i": "ᵢ", "j": "ⱼ", "k": "ₖ",
        "l": "ₗ", "m": "ₘ", "n": "ₙ", "o": "ₒ", "p": "ₚ", "r": "ᵣ", "s": "ₛ",
        "t": "ₜ", "u": "ᵤ", "v": "ᵥ", "x": "ₓ"}


def _to_script(s, table):
    return "".join(table[c] for c in s) if s and all(c in table for c in s) else None


def _convert_expr(e):
    """Convert one LaTeX math expression (no $ delimiters) to Unicode-ish text."""
    e = re.sub(r"\\mathbb\s*\{([A-Z])\}",
               lambda m: _MATH_BB.get(m.group(1), m.group(1)), e)
    e = re.sub(r"\\(?:text|mathrm|mathbf|mathit|mathsf|mathtt|mathcal|"
               r"boldsymbol|operatorname)\s*\{([^{}]*)\}", r"\1", e)
    for _ in range(3):                                   # a few passes: simple nesting
        new = re.sub(r"\\frac\s*\{([^{}]*)\}\s*\{([^{}]*)\}", r"(\1)/(\2)", e)
        if new == e:
            break
        e = new
    e = re.sub(r"\\sqrt\s*\{([^{}]*)\}", r"√(\1)", e)
    e = re.sub(r"\\(?:left|right)\s*", "", e)
    e = (e.replace("\\\\", "\n").replace("\\,", " ").replace("\\;", " ")
          .replace("\\:", " ").replace("\\!", "").replace("\\ ", " "))

    def _sup(m):
        g = m.group(1)
        content = g[1:-1] if g.startswith("{") else g
        r = _to_script(content, _SUP)
        return r if r is not None else "^" + content

    def _sub(m):
        g = m.group(1)
        content = g[1:-1] if g.startswith("{") else g
        r = _to_script(content, _SUB)
        return r if r is not None else "_" + content

    # One pass each (braced or single char) so a fallback like "_new" isn't
    # re-scanned and partially subscripted ("Wₙew").
    e = re.sub(r"\^(\{[^{}]*\}|[A-Za-z0-9])", _sup, e)
    e = re.sub(r"_(\{[^{}]*\}|[A-Za-z0-9])", _sub, e)
    e = re.sub(r"\\([A-Za-z]+)",
               lambda m: _MATH_SYMBOLS.get(m.group(1), m.group(1)), e)
    return e.replace("{", "").replace("}", "")


def mathify(text):
    """Render $…$/$$…$$/\\(…\\)/\\[…\\] LaTeX math to Unicode, skipping code."""
    if "$" not in text and "\\[" not in text and "\\(" not in text:
        return text

    def _inline(m):
        c = m.group(1)
        cs = c.strip()
        if any(t in c for t in "\\^_{") or (0 < len(cs) <= 8 and " " not in cs):
            return _convert_expr(c)
        return m.group(0)                                # leave prose/currency alone

    # Don't touch fenced ``` blocks or `inline code`.
    parts = re.split(r"(```.*?```|`[^`\n]*`)", text, flags=re.DOTALL)
    for i in range(0, len(parts), 2):
        s = parts[i]
        s = re.sub(r"\$\$(.+?)\$\$", lambda m: _convert_expr(m.group(1)), s, flags=re.DOTALL)
        s = re.sub(r"\\\[(.+?)\\\]", lambda m: _convert_expr(m.group(1)), s, flags=re.DOTALL)
        s = re.sub(r"\\\((.+?)\\\)", lambda m: _convert_expr(m.group(1)), s, flags=re.DOTALL)
        s = re.sub(r"(?<!\\)\$(.+?)(?<!\\)\$", _inline, s)
        parts[i] = s
    return "".join(parts)
