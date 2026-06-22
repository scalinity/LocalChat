#!/usr/bin/env python3
"""Pretty terminal chat for local MLX models (used by chat.sh).

What it adds over `mlx_lm.chat`:
  1. The model's hidden "thinking" trace streams dimmed and clearly separated
     from the answer (detected at the token level via the tokenizer's
     think_start / think_end markers — these Gemma 4 builds use OpenAI-style
     <|channel>thought … <channel|> channels).
  2. The answer renders as Markdown *live as it streams* — **bold**, *italics*,
     lists, headings, tables, syntax-highlighted ``` code ```, and LaTeX math
     mapped to Unicode (see mathify). Thinking stays dim; the answer uses normal
     styling. Duplication was caused by ``rich.Live(vertical_overflow="visible")``:
     once the answer exceeds the terminal height, every refresh re-printed the
     entire answer below the live region (worse with long pasted prompts).
     Fixed by bounding live previews to the viewport and rendering the full
     answer exactly once after streaming completes. Use --raw for plain tokens.
  3. A repetition penalty (default 1.1) to curb genuine degenerate loops.

REPL commands:  q quit · r reset · h help · t toggle the thinking trace
"""
import argparse
import os
import re
import readline
import select
import sys
import time

import mlx.core as mx
from mlx_lm.utils import load
from mlx_lm.generate import stream_generate
from mlx_lm.models.cache import make_prompt_cache
from mlx_lm.sample_utils import make_sampler, make_logits_processors

from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.rule import Rule

DIM = "\x1b[2m"
RESET = "\x1b[0m"

# Terminals wrap bracketed paste (Cmd+V) in OSC 200/201. Without stripping,
# those bytes become part of the prompt text and get fed to the model.
_BRACKETED_PASTE_START = re.compile(r"\x1b\[200~")
_BRACKETED_PASTE_END = re.compile(r"\x1b\[201~")
_READAHEAD: list[str] = []  # bytes consumed while probing for bracketed paste


def _strip_bracketed_paste(text: str) -> str:
    text = _BRACKETED_PASTE_START.sub("", text)
    return _BRACKETED_PASTE_END.sub("", text)


def _emit_prompt():
    sys.stdout.write("\x1b[1m›\x1b[0m ")
    sys.stdout.flush()


def _read_line(prefill: str | None = None) -> str:
    """Read one line; the › prompt must already be on screen."""
    if prefill is not None:
        buf = prefill

        def _prefill_hook():
            nonlocal buf
            if buf is not None:
                readline.insert_text(buf)
                buf = None
            if _READAHEAD:
                readline.insert_text("".join(_READAHEAD))
                _READAHEAD.clear()
            readline.set_pre_input_hook()

        readline.set_pre_input_hook(_prefill_hook)
    elif _READAHEAD:
        pending = "".join(_READAHEAD)

        def _readahead_hook():
            nonlocal pending
            if pending:
                readline.insert_text(pending)
                pending = ""
            readline.set_pre_input_hook()

        readline.set_pre_input_hook(_readahead_hook)
    try:
        line = sys.stdin.readline()
    finally:
        readline.set_pre_input_hook()
    if line == "":
        raise EOFError
    return _strip_bracketed_paste(line.rstrip("\n"))


def _prompt_line(console, prefill: str | None = None) -> str:
    """Show the › prompt and read one line (optionally pre-filled for editing)."""
    console.show_cursor(True)
    _emit_prompt()
    return _read_line(prefill)


def _try_consume_bracketed_paste() -> str | None:
    """If stdin holds a bracketed paste (Cmd+V), consume it and return the body."""
    fileno = sys.stdin.fileno()
    b0 = os.read(fileno, 1)
    if not b0:
        raise EOFError
    if b0 != b"\x1b":
        _READAHEAD.append(b0.decode("utf-8", errors="replace"))
        return None

    buf = bytearray(b0)
    decoded = buf.decode("utf-8", errors="replace")
    while "\x1b[201~" not in decoded:
        chunk = os.read(fileno, 4096)
        if not chunk:
            break
        buf.extend(chunk)
        decoded = buf.decode("utf-8", errors="replace")
        if len(buf) > 12 and "[200~" not in decoded:
            _READAHEAD.append(decoded)
            return None
    if "\x1b[200~" not in decoded or "\x1b[201~" not in decoded:
        _READAHEAD.append(decoded)
        return None
    start = decoded.index("\x1b[200~") + len("\x1b[200~")
    end = decoded.index("\x1b[201~")
    body = decoded[start:end]
    tail = decoded[end + len("\x1b[201~") :]
    while tail.startswith("\r") or tail.startswith("\n"):
        tail = tail[1:]
    if tail:
        _READAHEAD.append(tail)
    return body


def read_query(console) -> str:
    """Read one user message from stdin.

    Terminal paste (Cmd+V) is bracketed and often ends with ``\\n``, which makes
    plain ``readline()`` look like an immediate Enter — the message auto-sends.
    When we detect paste we pre-fill the prompt so you can edit, then press Enter
    to send deliberately.

    Multiline paste used to also split across loop iterations (looked like a crash);
    we slurp buffered follow-on lines into one blob before confirming.
    """
    console.show_cursor(True)
    _emit_prompt()
    select.select([sys.stdin], [], [], None)

    pasted = _try_consume_bracketed_paste()
    if pasted is not None:
        if "\n" in pasted:
            n = len(pasted.splitlines())
            console.print(
                f"[dim]({n} lines pasted — Enter to send, or type to replace)[/]"
            )
            _emit_prompt()
            replacement = _read_line()
            return pasted if replacement == "" else replacement
        return _read_line(prefill=pasted)

    first = _read_line()
    lines = [first]
    while True:
        ready, _, _ = select.select([sys.stdin], [], [], 0)
        if not ready:
            break
        line = sys.stdin.readline()
        if line == "":
            break
        lines.append(line.rstrip("\n"))

    raw = "\n".join(lines)
    text = _strip_bracketed_paste(raw)

    # Buffered extra lines = multiline paste without bracketed-paste mode.
    if len(lines) > 1:
        n = len(text.splitlines())
        console.print(
            f"[dim]({n} lines pasted — Enter to send, or type to replace)[/]"
        )
        _emit_prompt()
        replacement = _read_line()
        return text if replacement == "" else replacement

    return text


# ---------------------------------------------------------------------------
# Lightweight LaTeX-math → Unicode rendering.
#
# rich's Markdown prints $…$ / $$…$$ verbatim, so model output full of LaTeX
# (\Delta, x^2, W_{new}, d \times r, …) is painful to read in a terminal. We
# can't typeset real math, but mapping the common commands to Unicode makes it
# read naturally:  $\Delta W$ → ΔW,  $d \times r$ → d × r,  $x^2$ → x².
# ---------------------------------------------------------------------------
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


def _live_markdown(console, answer: str) -> Markdown:
    """Markdown preview sized to fit the terminal (tail of a long answer).

    Feeding an unbounded growing string into ``rich.Live`` makes the renderable
    taller than the viewport; with ``vertical_overflow="visible"`` (Rich's
    stop() path included) that re-prints the whole answer every refresh.
    Keep the live widget small — show the latest lines only — and print the
    full answer once after streaming.
    """
    text = mathify(answer)
    budget = max(8, console.size.height - 8)
    lines = text.splitlines()
    if len(lines) > budget:
        text = "…\n\n" + "\n".join(lines[-budget:])
    return Markdown(text)


def parse_args():
    p = argparse.ArgumentParser(description="Pretty chat with a local MLX model")
    p.add_argument("--model", required=True)
    p.add_argument("--max-tokens", "-m", type=int, default=4096)
    p.add_argument("--temp", type=float, default=0.0)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-kv-size", type=int, default=None)
    p.add_argument("--system-prompt", default=None)
    p.add_argument("--repetition-penalty", type=float, default=1.1,
                   help="Penalize recently-used tokens (1.0 = off). Curbs loops.")
    p.add_argument("--repetition-context-size", type=int, default=64)
    p.add_argument("--hide-thinking", action="store_true",
                   help="Do not display the thinking trace at all")
    p.add_argument("--refresh", type=float, default=10.0,
                   help="Live Markdown refreshes per second")
    p.add_argument("--raw", action="store_true",
                   help="Stream plain tokens with no Markdown rendering")
    return p.parse_args()


def main():
    args = parse_args()
    mx.random.seed(args.seed)
    console = Console()
    # Live Markdown needs a real terminal; when piped/redirected we fall back to
    # rendering the answer once at the end (no live preview).
    live_capable = console.is_terminal and not args.raw

    with console.status(f"[dim]loading {args.model} …[/]", spinner="dots"):
        model, tokenizer = load(args.model)

    has_think = bool(getattr(tokenizer, "has_thinking", False))
    ts_tokens = tuple(getattr(tokenizer, "think_start_tokens", ()) or ())
    te_tokens = tuple(getattr(tokenizer, "think_end_tokens", ()) or ())
    ts_str = getattr(tokenizer, "think_start", None)
    te_str = getattr(tokenizer, "think_end", None)
    ts0 = ts_tokens[0] if len(ts_tokens) >= 1 else None
    ts1 = ts_tokens[1] if len(ts_tokens) >= 2 else None

    def clean(s):
        # Belt-and-suspenders: strip channel control strings that slip through
        # token-level filtering (the answer/thinking never contain these).
        for marker in (ts_str, te_str, "<|channel>", "<channel|>"):
            if marker:
                s = s.replace(marker, "")
        return s

    show_thinking = not args.hide_thinking

    sampler = make_sampler(args.temp, args.top_p)
    logits_processors = None
    if args.repetition_penalty and args.repetition_penalty != 1.0:
        logits_processors = make_logits_processors(
            repetition_penalty=args.repetition_penalty,
            repetition_context_size=args.repetition_context_size,
        )

    # Rebuilding/parsing the whole Markdown on every token is O(n²) and, with a
    # fast model, starves rich's refresh thread (the UI "freezes" until the end).
    # Throttle the live re-render to the refresh cadence instead.
    live_interval = 1.0 / max(args.refresh, 1.0)

    console.print(f"[bold green]chat[/]  [cyan]{args.model}[/]")
    console.print("[dim]q quit · r reset · h help · t toggle thinking[/]\n")

    cache = make_prompt_cache(model, args.max_kv_size)

    while True:
        try:
            query = read_query(console)
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if query == "":
            continue
        if query == "q":
            break
        if query == "r":
            cache = make_prompt_cache(model, args.max_kv_size)
            console.print("[dim]— conversation reset —[/]\n")
            continue
        if query == "h":
            console.print("[dim]q quit · r reset · h help · t toggle thinking[/]")
            continue
        if query == "t":
            show_thinking = not show_thinking
            console.print(f"[dim]thinking trace: {'shown' if show_thinking else 'hidden'}[/]")
            continue

        messages = []
        if args.system_prompt:
            messages.append({"role": "system", "content": args.system_prompt})
        messages.append({"role": "user", "content": query})
        prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True)

        starts_thinking = False
        if has_think:
            try:
                starts_thinking = (
                    tokenizer.rfind_think_start(prompt)
                    > tokenizer.rfind_think_end(prompt)
                )
            except Exception:
                starts_thinking = False

        state = "reasoning" if (has_think and starts_thinking) else "normal"
        thinking = ""
        answer = ""
        think_header_shown = False
        answer_started = False
        live = None
        live_used = False
        last_live = 0.0
        hold_start = False
        ts0_txt = ""

        def emit(text):
            nonlocal thinking, answer, think_header_shown, answer_started, live
            nonlocal last_live, live_used
            text = clean(text)
            if not text:
                return
            if state == "reasoning":
                thinking += text
                if show_thinking:
                    if not think_header_shown:
                        console.print(Rule("thinking", style="grey42", characters="·"))
                        think_header_shown = True
                    sys.stdout.write(DIM + text + RESET)
                    sys.stdout.flush()
            else:
                if not answer_started:
                    answer_started = True
                    if show_thinking and think_header_shown:
                        sys.stdout.write(RESET + "\n")
                        sys.stdout.flush()
                        console.print(Rule("answer", style="green"))
                answer += text
                if args.raw:
                    sys.stdout.write(text)
                    sys.stdout.flush()
                elif live_capable:
                    if live is None:
                        live = Live(
                            console=console,
                            refresh_per_second=args.refresh,
                            transient=True,
                            vertical_overflow="crop",
                        )
                        live.start()
                        live_used = True
                    now = time.monotonic()
                    if now - last_live >= live_interval:
                        last_live = now
                        live.update(_live_markdown(console, answer))

        interrupted = False
        try:
            for resp in stream_generate(
                model, tokenizer, prompt,
                max_tokens=args.max_tokens, sampler=sampler,
                prompt_cache=cache, logits_processors=logits_processors,
            ):
                tid = resp.token
                txt = resp.text or ""

                if has_think:
                    if hold_start:
                        hold_start = False
                        if ts1 is not None and tid == ts1:
                            state = "reasoning"
                            continue                  # swallow whole start marker
                        emit(ts0_txt)                 # not a marker — emit it
                    if len(te_tokens) == 1 and tid == te_tokens[0]:
                        state = "normal"
                        continue
                    if ts1 is not None and tid == ts0:
                        hold_start = True
                        ts0_txt = txt
                        continue
                    if ts1 is None and ts0 is not None and tid == ts0:
                        state = "reasoning"
                        continue

                emit(txt)
        except KeyboardInterrupt:
            interrupted = True
            if live is not None:
                live.stop()
                live = None
            console.print("\n[yellow dim]— interrupted —[/]")
        finally:
            if live is not None:
                live.update(_live_markdown(console, answer))
                live.stop()
                live = None
            if hold_start:
                emit(ts0_txt)
            console.show_cursor(True)

        if interrupted:
            console.print()
            continue

        if args.raw:
            if answer_started:
                sys.stdout.write("\n")
                sys.stdout.flush()
        else:
            final = clean(answer).strip()
            if final:
                # Live preview is transient (viewport-sized tail only); print the
                # full answer once so nothing is truncated and nothing duplicates.
                console.print(Markdown(mathify(final)))
            elif has_think:
                console.print("[yellow dim]…stopped while still thinking — try a "
                              "higher --max-tokens, or 'r' to reset.[/]")
        console.print()


if __name__ == "__main__":
    main()
