import json, urllib.request, re
BASE="http://<NODE_PREFIX>.10:8888"
P=("Write an exhaustive technical deep-dive on memory-mapped hardware register "
   "interfaces: address decoding, side-effect-on-read registers, and debugging a "
   "register that returns shifting values across reads. Be extremely thorough.")

def post(path, body, t=1200):
    r=urllib.request.Request(BASE+path, data=json.dumps(body).encode(),
                             headers={"Content-Type":"application/json"})
    return json.load(urllib.request.urlopen(r, timeout=t))

print("=== A. /v1/chat/completions, max_tokens=3000 : inspect EVERY field ===")
d=post("/v1/chat/completions", {"model":"glm-5.3-uncensored",
        "messages":[{"role":"user","content":P}],"max_tokens":3000,"temperature":0})
c=d["choices"][0]; m=c["message"]
print(f"  finish_reason      : {c.get('finish_reason')}")
print(f"  usage              : {d.get('usage')}")
for k,v in m.items():
    if isinstance(v,str): print(f"  message.{k:<16} len={len(v)}")
    else: print(f"  message.{k:<16} {type(v).__name__} {str(v)[:60]}")
txt=m.get('content') or ''
print(f"  content[:200]      : {txt[:200]!r}")

print()
print("=== B. /v1/completions (RAW - no chat template, no reasoning parser) ===")
try:
    d2=post("/v1/completions", {"model":"glm-5.3-uncensored",
            "prompt":"[gMASK]<sop><|user|>"+P+"<|assistant|><think>",
            "max_tokens":3000,"temperature":0})
    t2=d2["choices"][0]
    raw=t2.get("text") or ""
    print(f"  finish_reason : {t2.get('finish_reason')}")
    print(f"  usage         : {d2.get('usage')}")
    print(f"  raw len       : {len(raw)}")
    print(f"  has </think>  : {'</think>' in raw}")
    w=0
    for n in (2,3,4,6):
        for mm in re.finditer(r'(.{%d}?)\1{3,}'%n, raw): w=max(w,len(mm.group(0))//max(1,len(mm.group(1))))
    print(f"  worst_repeat  : {w}x")
    print(f"  head          : {raw[:200]!r}")
    print(f"  tail          : {raw[-200:]!r}")
except Exception as e:
    print(f"  /v1/completions failed: {type(e).__name__} {str(e)[:150]}")
