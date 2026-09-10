#!/usr/bin/env python3
"""Hunt the degenerate-repetition bug ("TheTheTheThe...").

Prime suspect: MTP. dealignai's card says "MTP speculative decoding is
non-functional on GLM-5.3 regular in vLLM" -- we overrode that on the strength
of a measured speedup, but a speculative decoder that accepts wrong drafts
produces FAST GARBAGE, which is exactly this signature.

Detect: longest run of an immediately-repeated short substring, plus the ratio
of unique to total whitespace tokens. Degenerate output scores near-zero unique.
"""
import json, re, time, urllib.request

URL = "http://NODE_PREFIX_PLACEHOLDER.10:8888/v1/chat/completions"
MODEL = "glm-5.3-uncensored"

PROMPTS = [
    ("long-form prose", "Write a detailed 800-word essay on the history of the transistor."),
    ("code, long output", "Write a complete Python implementation of a red-black tree with "
                          "insert, delete, search, and rotations. Include full docstrings."),
    ("list/enumeration", "List 60 distinct programming languages, one per line, each with a "
                         "one-sentence description."),
    ("repetitive-prone", "Write the word 'The' followed by a sentence, 40 times, varying each sentence."),
]


def degeneracy(text):
    """Return (worst_repeat_run, unique_ratio)."""
    worst = 0
    for n in (2, 3, 4, 5, 8):
        for m in re.finditer(r"(.{%d}?)\1{3,}" % n, text):
            worst = max(worst, len(m.group(0)) // max(1, len(m.group(1))))
    toks = text.split()
    uniq = len(set(toks)) / len(toks) if toks else 1.0
    return worst, uniq


def run(label, prompt, max_tokens=1200):
    body = {"model": MODEL, "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens, "temperature": 0}
    req = urllib.request.Request(URL, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    d = json.load(urllib.request.urlopen(req, timeout=900))
    dt = time.time() - t0
    txt = d["choices"][0]["message"].get("content") or ""
    n = d["usage"]["completion_tokens"]
    rep, uniq = degeneracy(txt)
    flag = "  <-- DEGENERATE" if (rep >= 8 or uniq < 0.25) else ""
    print(f"  {label:<20} out={n:>5} {dt:>6.1f}s  worst_repeat={rep:>4}x  uniq={uniq:.2f}{flag}", flush=True)
    if rep >= 8 or uniq < 0.25:
        print(f"      tail: {txt[-160:]!r}", flush=True)
    return rep, uniq


print("  MTP is currently ON (k=5, adaptive depths 2,4,5)")
print()
bad = 0
for label, p in PROMPTS:
    try:
        rep, uniq = run(label, p)
        if rep >= 8 or uniq < 0.25:
            bad += 1
    except Exception as e:
        print(f"  {label:<20} FAILED {type(e).__name__}: {str(e)[:120]}", flush=True)
print()
print(f"  degenerate outputs: {bad}/{len(PROMPTS)}")
