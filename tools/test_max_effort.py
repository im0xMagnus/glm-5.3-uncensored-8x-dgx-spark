#!/usr/bin/env python3
"""Does reasoning_effort=max stay inside DSH's 16,384-token per-turn budget on
NVFP4 (B-prime, DCP8) and still return an answer?  Same prompt as the FP8
measurements in docs/solutions/2026-09-07-glm53-reasoning-effort-runaway-empty-answers.md
so the numbers are comparable.  Order: max (the question), high, low (controls).
Each run waits for an idle engine.  Plain-text output, no unicode."""
import json, re, time, urllib.request

BASE = "http://<NODE_PREFIX>.10:8888"
URL = BASE + "/v1/chat/completions"
MT = 16384  # DSH maxTokens
P = ("Write an exhaustive technical deep-dive on memory-mapped hardware register "
     "interfaces: address decoding, side-effect-on-read registers, and debugging a "
     "register that returns shifting values across reads. Be extremely thorough.")

def running():
    try:
        m = urllib.request.urlopen(BASE + "/metrics", timeout=8).read().decode()
        return float(re.search(r"^vllm:num_requests_running\S*\s+([0-9.]+)", m, re.M).group(1))
    except Exception:
        return -1

def rep(s):
    w = 0
    for n in (2, 3, 4, 6):
        for mm in re.finditer(r"(.{%d}?)\1{3,}" % n, s):
            w = max(w, len(mm.group(0)) // max(1, len(mm.group(1))))
    return w

def go(eff):
    for _ in range(60):
        if running() == 0: break
        time.sleep(10)
    b = {"model": "glm-5.3-uncensored", "messages": [{"role": "user", "content": P}],
         "max_tokens": MT, "temperature": 0,
         "chat_template_kwargs": {"reasoning_effort": eff}}
    r = urllib.request.Request(URL, data=json.dumps(b).encode(),
                               headers={"Content-Type": "application/json"})
    t0 = time.time()
    d = json.load(urllib.request.urlopen(r, timeout=3600))
    el = time.time() - t0
    c = d["choices"][0]; m = c["message"]; u = d.get("usage", {})
    reas = m.get("reasoning") or m.get("reasoning_content") or ""
    cont = m.get("content") or ""
    ct = u.get("completion_tokens", 0)
    print(f"effort={eff:<4} max_tokens={MT} finish={str(c.get('finish_reason')):<6} "
          f"reasoning={len(reas):>6}ch answer={len(cont):>6}ch rep={rep(cont):>3}x "
          f"tokens={ct:>5} wall={el:>5.0f}s decode={ct/max(el,1):>5.1f} tok/s"
          f"{'  <== NO ANSWER' if not cont.strip() else ''}", flush=True)
    return bool(cont.strip())

print(f"{time.strftime('%H:%M:%S')} start (prompt: register deep-dive; budget {MT})", flush=True)
go("max")
go("high")
go("low")
print(f"{time.strftime('%H:%M:%S')} done", flush=True)
