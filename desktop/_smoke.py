#!/usr/bin/env python3
"""Headless proof that chatcore works before any UI exists (plan M1).

Run from the project venv:
    .venv/bin/python desktop/_smoke.py [model_id]

Defaults to the fast e4b model. Loads it, runs one prompt that should exercise
the thinking channel, and asserts:
  - the model loads and answers,
  - thinking and answer arrive as separate channels,
  - no channel markers (<|channel> etc.) leak into either stream,
  - RAM is released after an explicit unload (mx.clear_cache).
"""
import sys

import mlx.core as mx
from mlx_lm.models.cache import make_prompt_cache

import chatcore


def main():
    model_id = sys.argv[1] if len(sys.argv) > 1 else "mlx-community/gemma-4-e4b-it-4bit"

    models = chatcore.list_models()
    print(f"discovered {len(models)} models:")
    for m in models:
        print(f"  - {m['id']}  ->  {m['label']}")
    assert any(m["id"] == model_id for m in models), f"{model_id} not discovered"

    print(f"\nDEFAULT_PARAMS: {chatcore.DEFAULT_PARAMS}")
    # validate_params should clamp out-of-range values and floor seed.
    clamped = chatcore.validate_params(
        {"temperature": 5.0, "top_k": -3, "seed": -1, "bogus": 1}
    )
    print(f"validate_params(out-of-range) -> {clamped}")
    assert clamped["temperature"] == 2.0 and clamped["top_k"] == 0
    assert clamped["seed"] == 0 and "bogus" not in clamped

    base = mx.get_active_memory()
    print(f"\nactive memory before load: {base/1e9:.2f} GB")
    print(f"loading {model_id} …")
    model, tokenizer = chatcore.load(model_id)
    after_load = mx.get_active_memory()
    print(f"active memory after load:  {after_load/1e9:.2f} GB")

    cache = make_prompt_cache(model)
    history = [{"role": "user",
                "content": "In one short sentence, what is 6 times 7? Think first."}]
    messages = chatcore.build_messages(history, chatcore.DEFAULT_PARAMS["system_prompt"])

    thinking, answer = "", ""
    markers = ("<|channel>", "<channel|>")
    print("\n--- streaming ---")
    for channel, chunk in chatcore.generate(model, tokenizer, messages, cache,
                                             chatcore.DEFAULT_PARAMS):
        if channel == "thinking":
            thinking += chunk
        else:
            answer += chunk
    print(f"[thinking] {thinking!r}")
    print(f"[answer]   {answer!r}")

    leaked = [mk for mk in markers if mk in thinking or mk in answer]
    assert not leaked, f"channel markers leaked: {leaked}"
    assert answer.strip(), "no answer produced"
    has_think = bool(getattr(tokenizer, "has_thinking", False))
    if has_think:
        assert thinking.strip(), "thinking-capable model produced no thinking trace"

    # --- RAM release on unload (spec §7.1 / §15.7) ---
    del cache
    del model, tokenizer
    mx.clear_cache()
    after_unload = mx.get_active_memory()
    print(f"\nactive memory after unload+clear_cache: {after_unload/1e9:.2f} GB")
    print(f"released {(after_load - after_unload)/1e9:.2f} GB back toward baseline "
          f"{base/1e9:.2f} GB")

    print("\nSMOKE PASS")


if __name__ == "__main__":
    main()
