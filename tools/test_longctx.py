#!/usr/bin/env python3
"""Does decode survive past ~24K context?

The claim (from the Flash/SM121 recipes): without the SM121 indexer patch the
engine boots, answers short prompts, then dies on every decode past ~24K --
the DSA persistent TopK kernel asserting total_ctas > num_sms*occupancy.

We serve the 744B DSA model with --attention-backend B12X_MLA_SPARSE and
VLLM_USE_B12X_SPARSE_INDEXER=1, so this is exactly the kernel in question.
Every benchmark so far used <=16K prompts, so this is untested on our stack.

Walk the ladder and make the model ACTUALLY DECODE (not max_tokens=1) at each
depth, with a needle so we can also see whether retrieval survives.
"""
import json, time, urllib.request, urllib.error, random, string

URL = "http://<NODE_PREFIX>.10:8888/v1/chat/completions"
MODEL = "glm-5.3-uncensored"

def build(approx_tokens, needle_frac=0.5):
    uniq = "".join(random.choices(string.ascii_lowercase, k=8))
    code = int(random.random() * 900000) + 100000
    filler = [f"{uniq}{i} the quick brown fox jumps over the lazy dog"
              for i in range(approx_tokens // 11)]
    pos = int(len(filler) * needle_frac)
    filler.insert(pos, f"IMPORTANT: the secret access code is {code}. Remember it.")
    return " ".join(filler), code

def run(approx):
    prompt, code = build(approx)
    body = {"model": MODEL,
            "messages": [{"role": "user", "content":
                          prompt + "\n\nWhat is the secret access code? Answer with the number, then briefly say how you found it."}],
            "max_tokens": 200, "temperature": 0}
    req = urllib.request.Request(URL, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    try:
        d = json.load(urllib.request.urlopen(req, timeout=1800))
    except urllib.error.HTTPError as e:
        print(f"  {approx:>7} -> HTTP {e.code}: {e.read().decode()[:200]}")
        return False
    except Exception as ex:
        print(f"  {approx:>7} -> {type(ex).__name__}: {str(ex)[:160]}")
        return False
    dt = time.time() - t0
    u = d["usage"]
    txt = (d["choices"][0]["message"].get("content") or "")
    ok = str(code) in txt.replace(",", "")
    print(f"  {approx:>7} -> prompt={u['prompt_tokens']:>7} out={u['completion_tokens']:>4} "
          f"{dt:>6.1f}s  needle={'FOUND' if ok else 'MISSED'}")
    return True

for n in (8000, 24000, 32000, 64000, 100000, 125000):
    if not run(n):
        print(f"  ** FAILED at ~{n} -- stopping ladder")
        break
