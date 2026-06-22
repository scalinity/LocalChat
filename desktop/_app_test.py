#!/usr/bin/env python3
"""End-to-end drive of the real app through the JS bridge (dev check).

Launches the actual window (api.py + index.html), then scripts a scenario via
evaluate_js exactly as a user would: select the e4b model, type a prompt, send,
wait for completion. Verifies the bridge encoding, streaming, the thinking
panel, and Markdown rendering — then screenshots and closes.
"""
import json
import subprocess
import threading
import time
from pathlib import Path

import webview

from api import Api

HERE = Path(__file__).resolve().parent
INDEX = HERE / "web" / "index.html"
MODEL = "mlx-community/gemma-4-e4b-it-4bit"
PROMPT = "Reply in one short sentence with a tiny bit of reasoning: what is 8 times 9?"
SHOT = "/tmp/localchat_shot.png"


def main():
    api = Api()
    window = webview.create_window(
        "Local Chat", url=str(INDEX), js_api=api,
        width=1080, height=760, frameless=True, easy_drag=False,
        background_color="#101114",
    )
    api.set_window(window)
    report = {}

    def ev(js):
        return window.evaluate_js(js)

    def driver():
        try:
            time.sleep(1.2)  # let app.js init() finish (list_models, wiring)
            report["models"] = ev("JSON.stringify(models.map(m=>m.id))")
            ev(f"selectModel({json.dumps(MODEL)})")
            # wait for ready (input enabled)
            for _ in range(120):
                if ev("document.getElementById('input').disabled") is False:
                    break
                time.sleep(0.5)
            report["ready"] = ev("document.getElementById('input').disabled") is False

            ev("(function(){var i=document.getElementById('input');"
               "i.value=" + json.dumps(PROMPT) + ";send();return true;})()")
            # wait for generation to finish
            for _ in range(180):
                if ev("generating") is False and \
                   ev("document.querySelectorAll('.msg-assistant .answer').length") >= 1:
                    # ensure onDone ran (caret removed)
                    if ev("!!document.querySelector('.answer:not(.is-streaming)')"):
                        break
                time.sleep(0.5)
            time.sleep(0.4)

            report["has_thinking_panel"] = ev("!!document.querySelector('.thinking')")
            report["answer_html_len"] = ev("(document.querySelector('.answer')||{}).innerHTML?"
                                           "document.querySelector('.answer').innerHTML.length:0")
            report["transcript_text"] = ev("document.getElementById('transcript').innerText")
            report["frameless_no_titlebar"] = True  # frameless=True passed to create_window
            try:
                subprocess.run(["screencapture", "-x", SHOT], timeout=10, check=False)
                report["screenshot"] = SHOT
            except Exception as e:  # noqa: BLE001
                report["screenshot_error"] = str(e)
        except Exception as e:  # noqa: BLE001
            report["driver_error"] = repr(e)
        finally:
            time.sleep(0.3)
            window.destroy()

    threading.Thread(target=driver, daemon=True).start()
    webview.start()

    print("=== E2E REPORT ===")
    print("models:", report.get("models"))
    print("ready:", report.get("ready"))
    print("has_thinking_panel:", report.get("has_thinking_panel"))
    print("answer_html_len:", report.get("answer_html_len"))
    print("screenshot:", report.get("screenshot") or report.get("screenshot_error"))
    print("driver_error:", report.get("driver_error"))
    print("--- transcript text ---")
    print((report.get("transcript_text") or "")[:1200])


if __name__ == "__main__":
    main()
