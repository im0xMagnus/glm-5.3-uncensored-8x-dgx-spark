import json, urllib.request, re
URL="http://NODE_PREFIX_PLACEHOLDER.10:8888/v1/chat/completions"
P=("Write an exhaustive technical deep-dive on memory-mapped hardware register "
   "interfaces: address decoding, side-effect-on-read registers, and debugging a "
   "register that returns shifting values across reads. Be extremely thorough.")
def go(label, eff, mt):
    b={"model":"glm-5.3-uncensored","messages":[{"role":"user","content":P}],
       "max_tokens":mt,"temperature":0}
    if eff: b["chat_template_kwargs"]={"reasoning_effort":eff}
    r=urllib.request.Request(URL,data=json.dumps(b).encode(),headers={"Content-Type":"application/json"})
    d=json.load(urllib.request.urlopen(r,timeout=1500))
    c=d["choices"][0]; m=c["message"]
    reas=m.get("reasoning") or ""; cont=m.get("content") or ""
    w=0
    for n in (2,3,4,6):
        for mm in re.finditer(r'(.{%d}?)\1{3,}'%n,cont): w=max(w,len(mm.group(0))//max(1,len(mm.group(1))))
    print(f"  {label:<30} finish={str(c.get('finish_reason')):<6} "
          f"reasoning={len(reas):>6}ch answer={len(cont):>6}ch rep={w:>3}x"
          f"{'  <== NO ANSWER' if not cont else ''}", flush=True)
go("effort=low,  max_tokens 3000",  "low", 3000)
go("effort=high, max_tokens 3000",  "high", 3000)
go("effort=low,  max_tokens 8000",  "low", 8000)
