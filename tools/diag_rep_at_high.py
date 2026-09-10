import json, re, urllib.request
URL="http://NODE_PREFIX_PLACEHOLDER.10:8888/v1/chat/completions"
P=("Write an exhaustive technical deep-dive on memory-mapped hardware register "
   "interfaces: address decoding, side-effect-on-read registers, and debugging a "
   "register that returns shifting values across reads. Be extremely thorough.")
def go(label, extra):
    b={"model":"glm-5.3-uncensored","messages":[{"role":"user","content":P}],
       "max_tokens":8000,"temperature":0,
       "chat_template_kwargs":{"reasoning_effort":"high"}}
    b.update(extra)
    r=urllib.request.Request(URL,data=json.dumps(b).encode(),headers={"Content-Type":"application/json"})
    d=json.load(urllib.request.urlopen(r,timeout=1500))
    m=d["choices"][0]["message"]; c=m.get("content") or ""
    w=0
    for n in (2,3,4,6):
        for mm in re.finditer(r'(.{%d}?)\1{3,}'%n,c): w=max(w,len(mm.group(0))//max(1,len(mm.group(1))))
    print(f"  {label:<38} answer={len(c):>6}ch  worst_repeat={w:>4}x"
          f"{'  <== COLLAPSE' if w>=50 else ''}", flush=True)
go("effort=high, temp 0 (baseline)",        {})
go("effort=high + repetition_penalty 1.05", {"repetition_penalty":1.05})
go("effort=high + temp 0.7 top_p 0.95",     {"temperature":0.7,"top_p":0.95})
