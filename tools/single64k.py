import json, time, urllib.request, random, string, sys
URL="http://NODE_PREFIX_PLACEHOLDER.10:8888/v1/chat/completions"
uniq="".join(random.choices(string.ascii_lowercase,k=8)); code=734519
f=[f"{uniq}{i} the quick brown fox jumps over the lazy dog" for i in range(64000//11)]
f.insert(len(f)//2, f"IMPORTANT: the secret access code is {code}. Remember it.")
body={"model":"glm-5.3-uncensored","messages":[{"role":"user","content":" ".join(f)+"\n\nWhat is the secret access code?"}],
      "max_tokens":120,"temperature":0}
r=urllib.request.Request(URL,data=json.dumps(body).encode(),headers={"Content-Type":"application/json"})
t0=time.time()
try:
    d=json.load(urllib.request.urlopen(r,timeout=2400)); dt=time.time()-t0
    u=d["usage"]; txt=d["choices"][0]["message"].get("content") or ""
    print(f"  SOLO 64K: prompt={u['prompt_tokens']} out={u['completion_tokens']} {dt:.1f}s "
          f"prefill={u['prompt_tokens']/dt:.0f} tok/s needle={'FOUND' if str(code) in txt.replace(',','') else 'MISSED'}", flush=True)
except Exception as e:
    print(f"  SOLO 64K FAILED after {time.time()-t0:.0f}s: {type(e).__name__} {str(e)[:120]}", flush=True)
