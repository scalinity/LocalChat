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
  4. A prompt_toolkit input box: a hairline-framed editor with a live status
     footer (ttfs · tok/s · tokens in/out · context window). The box erases
     itself on Enter, leaving only the typed message in the scrollback.

REPL commands:  q quit · r reset · h help · t toggle the thinking trace
"""
import argparse
import re
import sys
import time

import mlx.core as mx
from mlx_lm.utils import load
from mlx_lm.generate import stream_generate
from mlx_lm.models.cache import make_prompt_cache
from mlx_lm.sample_utils import make_sampler, make_logits_processors

from rich.console import Console, ConsoleOptions, RenderResult
from rich.live import Live
from rich.markdown import Markdown
from rich.markup import escape
from rich.rule import Rule
from rich.segment import Segment
from rich.text import Text

from prompt_toolkit.application import Application, get_app
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.formatted_text import ANSI
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.key_binding import KeyBindings, merge_key_bindings
from prompt_toolkit.key_binding.defaults import load_key_bindings
from prompt_toolkit.layout import Layout
from prompt_toolkit.layout.containers import HSplit, Window
from prompt_toolkit.layout.controls import BufferControl, FormattedTextControl
from prompt_toolkit.layout.dimension import Dimension
from prompt_toolkit.styles import Style

DIM = "\x1b[2m"
RESET = "\x1b[0m"


# ---------------------------------------------------------------------------
# The input box.
#
# A self-contained prompt_toolkit screen drawn inline: a hairline top rule, the
# ``›`` editor (which grows as you type or paste), a hairline bottom rule, and
# the status footer beneath it. The whole box is erased the instant you press
# Enter (``erase_when_done``) — the caller then echoes just the message, so the
# scrollback stays clean (no border lines trailing each turn). prompt_toolkit
# handles editing, wrapping, history and bracketed paste natively, which is why
# the old hand-rolled stdin/readline machinery is gone.
# ---------------------------------------------------------------------------
class InputBox:
    def __init__(self):
        self.history = InMemoryHistory()
        self._status = ""
        self._style = Style.from_dict({"border": "fg:#444444", "prompt": "bold"})

    def read(self, status: str) -> str:
        self._status = status

        def _accept(buff):
            get_app().exit(result=buff.text)
            return True

        # multiline=False ⇒ a typed Enter sends; pasted newlines are inserted as
        # text (bracketed paste) rather than submitting line-by-line.
        buf = Buffer(history=self.history, multiline=False, accept_handler=_accept)

        def line_prefix(lineno, wrap_count):
            return [("class:prompt", "› ")] if lineno == 0 else "  "

        editor = Window(
            BufferControl(buffer=buf),
            wrap_lines=True,
            height=Dimension(min=1, max=12),
            # Without this the editor expands to its max height whenever there's
            # spare vertical space below the cursor (which the HSplit happily
            # hands out), leaving a tall empty box. Clamp it to the content's
            # own height so it sits at one line and grows only as you type.
            dont_extend_height=True,
            get_line_prefix=line_prefix,
        )
        rule = lambda: Window(height=1, char="─", style="class:border")
        footer = Window(
            FormattedTextControl(lambda: ANSI(self._status)), height=1,
        )
        root = HSplit([rule(), editor, rule(), footer])

        kb = KeyBindings()

        @kb.add("c-c")
        def _(event):
            event.app.exit(exception=KeyboardInterrupt)

        @kb.add("c-d")
        def _(event):
            if not buf.text:
                event.app.exit(exception=EOFError)

        @kb.add("escape", "enter")          # Alt/Option+Enter inserts a newline
        def _(event):
            buf.insert_text("\n")

        app = Application(
            layout=Layout(root, focused_element=editor),
            key_bindings=merge_key_bindings([load_key_bindings(), kb]),
            style=self._style,
            erase_when_done=True,
            full_screen=False,
            mouse_support=False,
        )
        return app.run()


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


class _SegmentLines:
    """A renderable wrapping pre-rendered, width-bounded segment lines.

    The live preview must never be *taller than the viewport* or Rich's
    ``transient`` cleanup can't erase it: the overflow has already scrolled off
    the top of the screen by stop() time, so Rich strands it in the scrollback —
    this is the stray ``…`` frame seen at the top of a long answer. We therefore
    render the Markdown ourselves, keep only the last N *visual* lines, and hand
    Rich exactly those, so the live region's height is known and bounded.
    """

    def __init__(self, lines: list[list[Segment]]):
        self._lines = lines

    def __rich_console__(self, console: Console, options: ConsoleOptions) -> RenderResult:
        for line in self._lines:
            yield from line
            yield Segment.line()


def _live_markdown(console, answer: str) -> Markdown | _SegmentLines:
    """Markdown preview bounded to the terminal viewport (tail of a long answer).

    Budgeting by ``splitlines()`` (logical lines) is wrong: one logical line
    wraps to several *visual* lines, so a "16-line" preview can render to 25+
    rows and overflow the viewport, breaking the transient erase. We measure the
    real rendered height and crop to the last ``budget`` visual lines instead.
    The full, untruncated answer is printed once after streaming completes.
    """
    md = Markdown(mathify(answer))
    budget = max(4, console.size.height - 3)
    # Coarse logical-line cap first so we don't re-render a huge answer every
    # frame; ×3 leaves plenty of lines to fill `budget` even when all of them
    # wrap. The exact crop below is what actually bounds the height.
    lines = answer.splitlines()
    if len(lines) > budget * 3:
        md = Markdown(mathify("\n".join(lines[-budget * 3:])))
    rendered = console.render_lines(md, console.options, pad=False)
    if len(rendered) <= budget:
        return md
    ellipsis = console.render_lines(Text("…", style="dim"), console.options, pad=False)
    return _SegmentLines(ellipsis + rendered[-(budget - len(ellipsis)):])


# ---------------------------------------------------------------------------
# Minimalist status footer (the bottom line of the input box).
#
# Time-to-first-token, generation speed, the tokens consumed/produced last turn,
# and how full the context window is. Kept deliberately quiet — dim labels, a
# single accent colour for the numbers, hair separators — so it reads as chrome.
# Rendered as a raw ANSI string because it lives inside the prompt_toolkit box.
# ---------------------------------------------------------------------------
_S_LABEL = "\x1b[38;5;245m"             # dim grey labels (ttfs, tok/s, ctx, ↑↓)
_S_VALUE = "\x1b[38;5;80m"              # cyan accent for the numbers
_S_FAINT = "\x1b[38;5;240m"            # faintest grey — separators and the gauge
_S_RESET = "\x1b[0m"


def _kfmt(n: int) -> str:
    """Compact token count: 587 · 1.0k · 131k."""
    n = int(n)
    if n < 1000:
        return str(n)
    if n < 10_000:
        return f"{n / 1000:.1f}k"
    return f"{round(n / 1000)}k"


def _ctx_bar(frac: float, width: int = 10) -> str:
    """A hair-thin fill gauge for context usage: ▓▓░░░░░░░░."""
    frac = max(0.0, min(1.0, frac))
    fill = int(frac * width + 0.5)
    if frac > 0 and fill == 0:          # never show a totally empty bar mid-chat
        fill = 1
    return "▓" * fill + "░" * (width - fill)


def _context_window(model) -> int | None:
    """The model's max context length, if discoverable from its config."""
    args = getattr(model, "args", None)
    val = getattr(args, "max_position_embeddings", None)
    return val if isinstance(val, int) and val > 0 else None


def status_ansi(last, ctx_used, ctx_window) -> str:
    """The footer text (ANSI) that sits just under the input editor.

    ``last`` carries the most recent turn's metrics (or None before the first
    reply); context usage persists across the whole conversation and so always
    shows. Speed metrics only appear once there is a turn to describe.
    """
    L, V, F, R = _S_LABEL, _S_VALUE, _S_FAINT, _S_RESET
    parts = []
    if last is not None:
        if last.get("ttfs") is not None:
            parts.append(f"{L}ttfs {V}{last['ttfs']:.2f}s{R}")
        if last.get("tps"):
            parts.append(f"{V}{last['tps']:.1f}{L} tok/s{R}")
        parts.append(
            f"{L}↑{V}{_kfmt(last['tok_in'])} {L}↓{V}{_kfmt(last['tok_out'])}{R}"
        )
    if ctx_window:
        frac = ctx_used / ctx_window
        pct = max(1, round(frac * 100)) if ctx_used else 0
        parts.append(
            f"{L}ctx {F}{_ctx_bar(frac)} "
            f"{V}{_kfmt(ctx_used)}{L}/{_kfmt(ctx_window)}  {pct}%{R}"
        )
    else:
        parts.append(f"{L}ctx {V}{_kfmt(ctx_used)}{R}")
    return "  " + f" {F}·{R} ".join(parts)


def parse_args():
    p = argparse.ArgumentParser(description="Pretty chat with a local MLX model")
    p.add_argument("--model", required=True)
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

    ctx_window = _context_window(model)
    ctx_used = 0                       # running conversation length (mirrors the KV cache)

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
    last = None                          # metrics from the most recent completed turn
    box = InputBox()

    while True:
        # The input is a self-erasing box (status footer carries the last turn's
        # metrics). On submit the box vanishes; we echo only the message so the
        # scrollback shows the conversation, never the border chrome.
        try:
            query = box.read(status_ansi(last, ctx_used, ctx_window))
        except (EOFError, KeyboardInterrupt):
            console.print()
            break

        cmd = query.strip()
        if cmd == "":
            continue
        console.print(f"[bold]›[/] {escape(query)}")

        if cmd == "q":
            break
        if cmd == "r":
            cache = make_prompt_cache(model, args.max_kv_size)
            ctx_used = 0
            last = None
            console.print("[dim]— conversation reset —[/]\n")
            continue
        if cmd == "h":
            console.print("[dim]q quit · r reset · h help · t toggle thinking[/]\n")
            continue
        if cmd == "t":
            show_thinking = not show_thinking
            console.print(f"[dim]thinking trace: {'shown' if show_thinking else 'hidden'}[/]\n")
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
        t_start = time.monotonic()
        ttfs = None
        last_resp = None
        try:
            for resp in stream_generate(
                model, tokenizer, prompt,
                max_tokens=-1, sampler=sampler,       # generate until the model stops
                prompt_cache=cache, logits_processors=logits_processors,
            ):
                if ttfs is None:
                    ttfs = time.monotonic() - t_start
                last_resp = resp
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

        # Tokens generated this turn live on in the KV cache, so they count
        # toward context whether the turn finished cleanly or was interrupted.
        # Stash this turn's metrics; they surface in the next box's footer.
        if last_resp is not None:
            ctx_used += last_resp.prompt_tokens + last_resp.generation_tokens
            last = {
                "ttfs": ttfs,
                "tps": last_resp.generation_tps,
                "tok_in": last_resp.prompt_tokens,
                "tok_out": last_resp.generation_tokens,
            }

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
                console.print("[yellow dim]…stopped while still thinking — "
                              "press 'r' to reset.[/]")
        console.print()


if __name__ == "__main__":
    main()
