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


def list_models() -> list[dict]:
    """Return ``[{id, label, path}]`` for every model in ``models/hub``.

    Directory ``models--mlx-community--gemma-4-12B-it-qat-6bit`` maps to id
    ``mlx-community/gemma-4-12B-it-qat-6bit`` (HF cache naming; the org/name
    separator is the only ``--`` since model names never contain ``--``).
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
                })
    return out


class ModelLoadError(Exception):
    """Raised when a model cannot be loaded (unsupported arch, corrupt cache…)."""


def load(model_id: str):
    """Load ``(model, tokenizer)`` via mlx_lm with the offline cache configured.

    Raises ``ModelLoadError`` (a typed error the UI can surface) on failure.
    """
    try:
        return _mlx_load(model_id)
    except Exception as exc:  # noqa: BLE001 — surface any load failure uniformly
        raise ModelLoadError(str(exc)) from exc


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


def _apply_template(tokenizer, messages):
    """Apply the chat template, tolerating models without a system role (Gemma).

    Gemma's template has no standalone system role; if it rejects one we merge
    the system text into the first user turn instead.
    """
    try:
        return tokenizer.apply_chat_template(messages, add_generation_prompt=True)
    except Exception:  # noqa: BLE001
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
        return tokenizer.apply_chat_template(merged, add_generation_prompt=True)


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
