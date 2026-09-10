#!/usr/bin/env python3
"""Prove the 1M window is real on the node: send a prompt LONGER than the old
524,288 limit and check it is accepted, prefilled, and answered (needle recall).
Plain-text output, no unicode."""
import json, time, urllib.request, urllib.error, sys

BASE = "http://NODE_PREFIX_PLACEHOLDER.10:8888"
MODEL = "glm-5.3-uncensored"
TARGET_TOKENS = int(sys.argv[1]) if len(sys.argv) > 1 else 560_000
NEEDLE = "The access code is KESTREL-7731."

def post(path, body, timeout):
    r = urllib.request.Request(BASE + path, data=json.dumps(body).encode(),
                               headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(r, timeout=timeout))

# 0. advertised limit
m = json.load(urllib.request.urlopen(BASE + "/v1/models", timeout=10))["data"][0]
print(f"advertised max_model_len: {m.get('max_model_len')}", flush=True)

# 1. filler calibrated with the server's own tokenizer
para = ("The scheduler drains the ready queue in priority order, re-checking the deadline "
        "of each task before dispatch. A task whose deadline has passed is moved to the "
        "expired list and its owner is notified through the completion channel. Metrics "
        "are sampled every 250 milliseconds and flushed to the ring buffer. ")
sample = para * 200
t = post("/tokenize", {"model": MODEL, "prompt": sample}, 60)
ratio = t["count"] / len(sample)
reps = int(TARGET_TOKENS / (len(para) * ratio))
filler = para * reps
prompt = (NEEDLE + "\n\n" + filler +
          "\n\nWhat is the access code stated at the very beginning of this message? "
          "Reply with just the code.")
est = int(len(prompt) * ratio)
print(f"tokens/char {ratio:.4f}; built prompt ~{est:,} tokens "
      f"(old limit 524,288; new 1,048,576)", flush=True)

# 2. send it
body = {"model": MODEL, "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 64, "temperature": 0,
        "chat_template_kwargs": {"reasoning_effort": "low"}}
t0 = time.time()
try:
    d = post("/v1/chat/completions", body, 3600)
except urllib.error.HTTPError as e:
    print(f"REJECTED HTTP {e.code}: {e.read().decode()[:300]}", flush=True)
    sys.exit(1)
el = time.time() - t0
u = d.get("usage", {})
c = d["choices"][0]
ans = (c["message"].get("content") or "").strip()
pt = u.get("prompt_tokens", 0)
print(f"ACCEPTED: prompt_tokens={pt:,} completion_tokens={u.get('completion_tokens')} "
      f"finish={c.get('finish_reason')} wall={el:.0f}s "
      f"(~{pt/max(el,1):,.0f} prompt tok/s incl. decode)", flush=True)
print(f"answer: {ans[:120]!r}", flush=True)
print("needle recall:", "FOUND" if "KESTREL-7731" in ans else "MISSED", flush=True)
print("1M WINDOW CONFIRMED" if pt > 524_288 else "prompt did not exceed old limit; inconclusive",
      flush=True)
