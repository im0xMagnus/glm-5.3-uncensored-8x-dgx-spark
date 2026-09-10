#!/usr/bin/env python3
"""Clock-lock A/B for GLM-5.3-FP8 @ TP=8 on 8x DGX Spark GB10.

Separates the two regimes the -lgc tip hinges on:
  PREFILL - compute-bound  -> should degrade when the clock is capped
  DECODE  - bandwidth-bound -> should NOT degrade

Unique filler per prefill run so prefix caching cannot flatter the number.
"""
import json, time, urllib.request, random, string, sys, statistics

URL   = "http://10.100.128.10:8888/v1/chat/completions"
MODEL = "glm-5.3-uncensored"
LABEL = sys.argv[1] if len(sys.argv) > 1 else "run"
REPS  = int(sys.argv[2]) if len(sys.argv) > 2 else 3


def post(body, timeout=1800):
    req = urllib.request.Request(URL, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    d = json.load(urllib.request.urlopen(req, timeout=timeout))
    return d, time.time() - t0


def prefill(approx_tokens):
    uniq = "".join(random.choices(string.ascii_lowercase, k=14))
    filler = " ".join(f"{uniq}{i} the quick brown fox jumps over the lazy dog"
                      for i in range(approx_tokens // 11))
    d, dt = post({"model": MODEL,
                  "messages": [{"role": "user", "content": filler + "\n\nReply with just: OK"}],
                  "max_tokens": 1, "temperature": 0})
    pt = d.get("usage", {}).get("prompt_tokens", 0)
    return pt, dt, (pt / dt if dt else 0)


def decode(n=600):
    # short prompt => time is dominated by token generation, not prefill
    d, dt = post({"model": MODEL,
                  "messages": [{"role": "user",
                                "content": "Count from 1 to 300, one number per line."}],
                  "max_tokens": n, "temperature": 0})
    c = d.get("usage", {}).get("completion_tokens", 0)
    return c, dt, (c / dt if dt else 0)


res = {"label": LABEL, "prefill": {}, "decode": {}}
print(f"===== {LABEL} =====", flush=True)

for size in (2048, 16384):
    rates, toks = [], 0
    for _ in range(REPS):
        pt, dt, r = prefill(size)
        rates.append(r); toks = pt
        print(f"  prefill ~{size:>5} : {pt:>6} tok in {dt:6.2f}s = {r:8.1f} tok/s", flush=True)
    med = statistics.median(rates)
    res["prefill"][size] = {"tokens": toks, "median_tok_s": round(med, 1),
                            "all": [round(x, 1) for x in rates]}
    print(f"  -> prefill ~{size} MEDIAN {med:.1f} tok/s", flush=True)

rates = []
for _ in range(REPS):
    c, dt, r = decode()
    rates.append(r)
    print(f"  decode          : {c:>6} tok in {dt:6.2f}s = {r:8.2f} tok/s", flush=True)
med = statistics.median(rates)
res["decode"] = {"median_tok_s": round(med, 2), "all": [round(x, 2) for x in rates]}
print(f"  -> decode MEDIAN {med:.2f} tok/s", flush=True)

with open(f"/tmp/bench-{LABEL}.json", "w") as f:
    json.dump(res, f, indent=2)
print(f"  saved /tmp/bench-{LABEL}.json", flush=True)
