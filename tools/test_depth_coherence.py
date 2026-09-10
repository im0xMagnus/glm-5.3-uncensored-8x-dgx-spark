#!/usr/bin/env python3
"""Does output coherence degrade with CONTEXT DEPTH?

The earlier repetition test asked for long OUTPUT from a SHORT prompt and found
nothing. The reported failure is a conversation that starts fine and degrades --
that is depth-dependent, so grow real multi-turn context and score each reply.

Scores per turn:
  ascii_ratio   - fraction of chars that are ordinary text (symbols spike on corruption)
  uniq_ratio    - unique/total whitespace tokens (repetition collapses this)
  worst_repeat  - longest immediately-repeated substring run
"""
import json, re, time, urllib.request

URL = "http://<NODE_PREFIX>.10:8888/v1/chat/completions"
MODEL = "glm-5.3-uncensored"

FILLER = ("Consider the following technical background material. "
          "Distributed inference partitions a model across accelerators; tensor "
          "parallelism splits individual weight matrices while pipeline parallelism "
          "splits layers. Collective operations synchronise partial results. ")

QUESTIONS = [
    "Summarise the tradeoff between tensor and pipeline parallelism in three sentences.",
    "Now explain how KV cache size scales with context length.",
    "What happens to decode throughput when the model is memory-bandwidth bound?",
    "Describe one way speculative decoding can fail silently.",
    "Explain why collective latency matters more for decode than for prefill.",
]


def score(t):
    if not t:
        return 0.0, 0.0, 999
    ascii_ratio = sum(1 for c in t if c.isascii() and (c.isalnum() or c.isspace() or c in ".,;:'\"()-—/*`#[]{}=+<>!?$%&_|\\~^")) / len(t)
    toks = t.split()
    uniq = len(set(toks)) / len(toks) if toks else 0.0
    worst = 0
    for n in (2, 3, 4, 6):
        for m in re.finditer(r"(.{%d}?)\1{3,}" % n, t):
            worst = max(worst, len(m.group(0)) // max(1, len(m.group(1))))
    return ascii_ratio, uniq, worst


msgs = []
print(f"  {'turn':<6}{'ctx_tok':>9}{'out':>6}{'sec':>8}{'ascii':>8}{'uniq':>7}{'rep':>6}  verdict")
print("  " + "-" * 66)

for i, q in enumerate(QUESTIONS):
    # grow context hard between turns
    pad = FILLER * (1200 * (i + 1) // len(FILLER.split()) + 1)
    msgs.append({"role": "user", "content": (pad + "\n\n" + q) if i else q})
    body = {"model": MODEL, "messages": msgs, "max_tokens": 400, "temperature": 0}
    req = urllib.request.Request(URL, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    try:
        d = json.load(urllib.request.urlopen(req, timeout=1200))
    except Exception as e:
        print(f"  {i+1:<6}{'-':>9}{'-':>6}{time.time()-t0:>8.1f}  FAILED {type(e).__name__}")
        break
    dt = time.time() - t0
    txt = d["choices"][0]["message"].get("content") or ""
    u = d["usage"]
    a, uq, rp = score(txt)
    bad = (a < 0.95) or (uq < 0.30) or (rp >= 8)
    print(f"  {i+1:<6}{u['prompt_tokens']:>9}{u['completion_tokens']:>6}{dt:>8.1f}"
          f"{a:>8.3f}{uq:>7.2f}{rp:>6}  {'*** DEGRADED' if bad else 'ok'}", flush=True)
    if bad:
        print(f"      sample: {txt[:220]!r}", flush=True)
    msgs.append({"role": "assistant", "content": txt})
