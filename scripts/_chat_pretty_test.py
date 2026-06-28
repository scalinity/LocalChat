#!/usr/bin/env python3
"""Fast unit tests for chat_pretty's backend seam — no model load, no network.

Run from the project venv:
    .venv/bin/python scripts/_chat_pretty_test.py

Covers the parts of the MLX/GGUF refactor that are new and not exercised by just
running the REPL: make_backend's dispatch (.gguf -> GGUFChat via chatcore, plain
id -> MLXChat via mlx_lm) and GGUFChat.start_turn's translation of a chatcore
(thinking|answer|stats) event stream into (reasoning|normal) channels plus the
per-turn metrics. Exits non-zero on the first failure.
"""
import contextlib
import importlib.util
import os
import sys
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent

# chat.sh puts LocalChat on the path at launch (for chatcore); mirror that here so
# the GGUF-dispatch test can import + monkeypatch chatcore.load.
_LC = os.environ.get("LOCALCHAT_HOME", os.path.expanduser("~/Documents/Apps/LocalChat"))
if _LC not in sys.path:
    sys.path.insert(0, _LC)

# Load chat_pretty.py as a module without requiring scripts/ to be a package.
_spec = importlib.util.spec_from_file_location("chat_pretty", HERE / "chat_pretty.py")
cp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cp)

PASS = 0
def check(name, cond, extra=""):
    global PASS
    mark = "ok " if cond else "FAIL"
    print(f"  [{mark}] {name}" + (f"  — {extra}" if extra and not cond else ""))
    if not cond:
        raise AssertionError(name + (": " + extra if extra else ""))
    PASS += 1


class _FakeConsole:
    def status(self, *a, **k):
        return contextlib.nullcontext()
    def print(self, *a, **k):
        pass


def _args(model):
    return types.SimpleNamespace(
        model=model, temp=0.0, top_p=1.0, seed=0,
        repetition_penalty=1.1, repetition_context_size=64, max_kv_size=None)


class _FakeGGUFBackend:
    """Duck-typed stand-in for chatcore.GGUFBackend."""
    context_length = 4096
    def count_tokens(self, messages):
        return 7
    def generate(self, messages, cache, params):
        yield ("thinking", "let me think")
        yield ("answer", "the answer")
        yield ("stats", {"tps": 12.5, "tokens": 5})


def main():
    print("== make_backend: dispatch by model id ==")
    # .gguf -> GGUFChat, loaded via chatcore.load (stub the load + the wrapper).
    import chatcore  # importable in the LocalAI venv (LocalChat is on sys.path via chat.sh)
    made = {}
    orig_load, orig_gguf, orig_mlx = chatcore.load, cp.GGUFChat, cp.MLXChat
    chatcore.load = lambda mid: _FakeGGUFBackend()
    cp.GGUFChat = lambda be, args: (made.__setitem__("gguf", (be, args)), "GGUF")[1]
    cp.MLXChat = lambda *a, **k: (made.__setitem__("mlx", a), "MLX")[1]
    try:
        b = cp.make_backend(_args("/models/gguf/foo.gguf"), _FakeConsole())
        check("'*.gguf' routes to GGUFChat", b == "GGUF" and "gguf" in made)
        check("GGUFChat is handed the chatcore backend", isinstance(made["gguf"][0], _FakeGGUFBackend))
        # a plain hf id -> MLXChat via mlx_lm load (stub mlx load so no weights are touched)
        made.clear()
        cp.load = lambda mid: ("MODEL", "TOK")
        b2 = cp.make_backend(_args("mlx-community/gemma-4-E4B-it-qat-4bit"), _FakeConsole())
        check("a plain repo id routes to MLXChat", b2 == "MLX" and "mlx" in made)
    finally:
        chatcore.load, cp.GGUFChat, cp.MLXChat = orig_load, orig_gguf, orig_mlx

    print("== GGUFChat.start_turn: channel mapping + metrics ==")
    g = cp.GGUFChat(_FakeGGUFBackend(), _args("x.gguf"))
    check("ctx_window comes from the backend", g.ctx_window == 4096)
    evs = list(g.start_turn([{"role": "user", "content": "hi"}]))
    check("thinking->reasoning, answer->normal; stats not yielded",
          evs == [("reasoning", "let me think"), ("normal", "the answer")], repr(evs))
    m = g.last_metrics
    check("tok_in from backend.count_tokens", m["tok_in"] == 7)
    check("tok_out is the authoritative stats count (not the event count)", m["tok_out"] == 5)
    check("tps from the stats event", m["tps"] == 12.5)
    check("ctx_delta = tok_in + tok_out", m["ctx_delta"] == 12)
    check("ttfs was measured", isinstance(m["ttfs"], float) and m["ttfs"] >= 0)

    print("== GGUFChat.start_turn: interrupted turn still leaves metrics ==")
    class _Boom(_FakeGGUFBackend):
        def generate(self, messages, cache, params):
            yield ("answer", "partial")
            raise KeyboardInterrupt
    g2 = cp.GGUFChat(_Boom(), _args("x.gguf"))
    got = []
    try:
        for ev in g2.start_turn([{"role": "user", "content": "hi"}]):
            got.append(ev)
    except KeyboardInterrupt:
        pass
    check("partial event delivered before the interrupt", got == [("normal", "partial")], repr(got))
    check("interrupted turn still recorded metrics (live estimate)",
          g2.last_metrics is not None and g2.last_metrics["tok_out"] >= 1)

    print(f"\nALL {PASS} CHECKS PASSED")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as e:
        print(f"\nTEST FAILED: {e}")
        sys.exit(1)
