#!/usr/bin/env python3
"""Does reasoning_effort actually reach the chat template?

The launcher sets the DEFAULT to "off". If per-request override works, "high"
must produce visibly more completion tokens (and/or a reasoning_content field)
than "off" on the same prompt. If both are identical, the kwarg is being
swallowed and DSH's reasoningEfforts map is decorative.
"""
import json, urllib.request

URL = "http://10.100.128.10:8888/v1/chat/completions"
MODEL = "glm-5.3-uncensored"
Q = "A farmer has 17 sheep. All but 9 run away. How many are left? Think it through."

def ask(label, extra):
    body = {"model": MODEL, "messages": [{"role": "user", "content": Q}],
            "max_tokens": 1200, "temperature": 0}
    body.update(extra)
    req = urllib.request.Request(URL, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    d = json.load(urllib.request.urlopen(req, timeout=600))
    m = d["choices"][0]["message"]
    u = d.get("usage", {})
    rc = m.get("reasoning_content") or ""
    print(f"--- {label}")
    print(f"    completion_tokens = {u.get('completion_tokens')}")
    print(f"    reasoning_content = {len(rc)} chars")
    print(f"    content[:150]     = {(m.get('content') or '')[:150]!r}")
    return u.get("completion_tokens"), len(rc)

a = ask("default (launcher: reasoning_effort=off)", {})
b = ask("chat_template_kwargs reasoning_effort=high",
        {"chat_template_kwargs": {"reasoning_effort": "high"}})
c = ask("top-level reasoning_effort=high (OpenAI style)",
        {"reasoning_effort": "high"})
print()
print("VERDICT:")
print(f"  chat_template_kwargs works : {b != a}   (off={a} vs high={b})")
print(f"  top-level field works      : {c != a}   (off={a} vs high={c})")
