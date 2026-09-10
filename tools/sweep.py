#!/usr/bin/env python3
"""Unattended overnight recipe sweep for the 8-node GB10 TP=8 cluster.

DESIGN CONSTRAINTS learned the hard way on 2026-09-06:
  * Run-to-run spread on prefill is 33.7% -- LARGER than most config effects.
    So a linear sweep is worthless: it confounds config with boot variance and
    thermal drift. This harness therefore (a) re-measures a fixed CONTROL config
    every N experiments, and (b) shuffles the queue order per pass.
  * Each boot is ~14 min and can fail on memory variance (~10% of the time).
    A failed boot must be RECORDED and skipped, never left hanging.
  * The cluster must be LEFT SERVING a known-good config in the morning.

Usage:
    python3 sweep.py --plan            # show what would run, no changes
    python3 sweep.py --run --hours 8   # execute until the budget is spent
"""
import argparse, json, os, random, subprocess, sys, time, datetime
# sudo password: from ~/.config/spark-ops/env (SPARK_SUDO_PASS), never inline
os.environ.setdefault("SPARK_SUDO_PASS", "")
for _l in open(os.path.expanduser("~/.config/spark-ops/env")) if os.path.exists(os.path.expanduser("~/.config/spark-ops/env")) else []:
    if _l.startswith("SPARK_SUDO_PASS="): os.environ["SPARK_SUDO_PASS"]=_l.split("=",1)[1].strip().strip('"\'')


HERE = os.path.dirname(os.path.abspath(__file__))
IPS = [10, 11, 12, 13, 14, 15, 16, 17]
HEAD = "NODE_PREFIX_PLACEHOLDER.10"
RESULTS = os.path.join(HERE, "sweep-results.jsonl")

# The config we KNOW works and that the cluster is left on. Also the control.
# 2026-09-07 rebuild: MTP is OFF (card: non-functional; measured: irrelevant to the
# repetition bug), reasoning_effort=low (card + measured: high/max never emit an
# answer), enforce-eager (card: required under concurrency). spec_k=0 everywhere.
CONTROL = {
    "name": "control-eager-low",
    "gmu": "0.86", "max_model_len": "131072", "spec_k": 0,
    "env": {},
}

QUEUE = [
    # --- THE one remaining FP8 lever: CUDA graphs back on, MTP still off.
    # --- Deviates from the card ("enforce-eager required under concurrency"), so
    # --- the repetition harness MUST run against it, not be assumed safe.
    {"name": "cudagraphs-on",   "gmu": "0.86", "max_model_len": "131072", "spec_k": 0,
     "env": {}, "cudagraphs": True},
    # --- NCCL small-message latency: decode is hundreds of tiny all-reduces/token
    {"name": "nccl-proto-LL",   "gmu": "0.86", "max_model_len": "131072", "spec_k": 0,
     "env": {"NCCL_PROTO": "LL"}},
    {"name": "nccl-proto-LL128","gmu": "0.86", "max_model_len": "131072", "spec_k": 0,
     "env": {"NCCL_PROTO": "LL128"}},
    {"name": "nccl-nch-2",      "gmu": "0.86", "max_model_len": "131072", "spec_k": 0,
     "env": {"NCCL_MAX_NCHANNELS": "2", "NCCL_MIN_NCHANNELS": "2"}},
    {"name": "nccl-nch-8",      "gmu": "0.86", "max_model_len": "131072", "spec_k": 0,
     "env": {"NCCL_MAX_NCHANNELS": "8", "NCCL_MIN_NCHANNELS": "8"}},
    {"name": "nccl-tc-106",     "gmu": "0.86", "max_model_len": "131072", "spec_k": 0,
     "env": {"NCCL_IB_TC": "106", "NCCL_IB_PCI_RELAXED_ORDERING": "1"}},
    # --- vLLM's own warning about batched-tokens
    {"name": "batched-8192",    "gmu": "0.86", "max_model_len": "131072", "spec_k": 0,
     "env": {}, "max_num_batched_tokens": "8192"},
    # --- from the Flash recipes: block-size, plausibly relevant to long-context stability
    {"name": "block-2304",      "gmu": "0.86", "max_model_len": "131072", "spec_k": 0,
     "env": {}, "extra_flag": "--block-size 2304"},
]
# DROPPED vs the 2026-09-06 queue: spec-k3/k7/depths (MTP is non-functional per the
# card), seqs-8/16 (seqs=24 measured: aggregate ceiling unchanged at ~37 tok/s).

CONTROL_EVERY = 3          # re-measure the control every N experiments


def sh(cmd, timeout=300):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True,
                          timeout=timeout).stdout.strip()


def ssh(ip, cmd, timeout=120):
    return sh(f"ssh -o BatchMode=yes -o ConnectTimeout=10 $USER@NODE_PREFIX_PLACEHOLDER.{ip} "
              f"{json.dumps(cmd)}", timeout=timeout)


def apply_config(cfg):
    """Patch the launcher on every rank for this config."""
    seds = [
        f"s/--gpu-memory-utilization [0-9.]*/--gpu-memory-utilization {cfg['gmu']}/",
        f"s/--max-model-len [0-9]*/--max-model-len {cfg['max_model_len']}/",
        f"s/\\\"num_speculative_tokens\\\":[0-9]*/\\\"num_speculative_tokens\\\":{cfg['spec_k']}/",
    ]
    if cfg["spec_k"] == 0:          # 0 = drop MTP entirely, not k=0
        seds = [x for x in seds if "num_speculative_tokens" not in x]
        seds.append("/--speculative-config/d")
    if cfg.get("cudagraphs"):
        # swap enforce-eager for FULL cudagraphs (mirrors ciprianveg v19)
        seds.append("s|^    --enforce-eager \\\\$|    --compilation-config '{\"cudagraph_mode\":\"FULL\",\"max_cudagraph_capture_size\":30}' \\\\|")
    if "extra_flag" in cfg:
        seds.append(f"s|^    --seed 42|    {cfg['extra_flag']} \\\\\n    --seed 42|")
    if "max_num_seqs" in cfg:
        seds.append(f"s/--max-num-seqs [0-9]*/--max-num-seqs {cfg['max_num_seqs']}/")
    if "max_num_batched_tokens" in cfg:
        seds.append(f"s/--max-num-batched-tokens [0-9]*/--max-num-batched-tokens {cfg['max_num_batched_tokens']}/")
    sed = "; ".join(seds)

    # env overrides are injected as extra -e flags on the docker run line
    env_flags = " ".join(f"-e {k}={v}" for k, v in cfg.get("env", {}).items())
    for ip in IPS:
        ssh(ip, f"cp ~/launch-fp8-tp8.sh.sweepbase ~/launch-fp8-tp8.sh 2>/dev/null || "
                f"cp ~/launch-fp8-tp8.sh ~/launch-fp8-tp8.sh.sweepbase; "
                f"sed -i '{sed}' ~/launch-fp8-tp8.sh")
        if env_flags:
            ssh(ip, f"sed -i 's|-e NCCL_DEBUG=WARN|{env_flags} -e NCCL_DEBUG=WARN|' "
                    f"~/launch-fp8-tp8.sh")


def boot():
    """Stop, drop caches, launch workers then head. Returns True if READY."""
    for ip in IPS:
        ssh(ip, "docker rm -f vllm_fp8_tp8 >/dev/null 2>&1; true")
    for ip in IPS:
        pw = os.environ.get("SPARK_SUDO_PASS", "")
        ssh(ip, f"printf '%s\\n' '{pw}' | sudo -S -p '' sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches' 2>/dev/null; true")
    for r in [7, 6, 5, 4, 3, 2, 1]:
        ssh(10 + r, f"cd ~ && ./launch-fp8-tp8.sh {r}", timeout=180)
        time.sleep(4)
    ssh(10, "cd ~ && ./launch-fp8-tp8.sh 0", timeout=180)

    deadline = time.time() + 32 * 60
    while time.time() < deadline:
        log = ssh(10, "docker logs vllm_fp8_tp8 2>&1 | grep -aE "
                      "'Application startup complete|ValueError|OutOfMemoryError|EngineCore failed' | tail -3")
        if "Application startup complete" in log:
            return True, ""
        if any(x in log for x in ("ValueError", "OutOfMemoryError", "EngineCore failed")):
            return False, log[:400]
        time.sleep(20)
    return False, "boot timeout"


def measure():
    """Single-stream decode AND concurrency scaling. Both are needed: MTP helps
    latency but may cost aggregate throughput under batch, so a config can win
    on one metric and lose on the other."""
    sh(f"cd {HERE} && python3 bench_decode_real.py", timeout=2400)
    sh(f"cd {HERE} && python3 bench_concurrency.py 1 4 8", timeout=2400)
    kv = ssh(10, "docker logs vllm_fp8_tp8 2>&1 | grep -a 'GPU KV cache size' | tail -1")
    acc = ssh(10, "docker logs vllm_fp8_tp8 2>&1 | grep -a 'Mean acceptance length' | tail -1")
    def load(p):
        try:
            return json.load(open(p))
        except Exception:
            return None
    return {"decode": load("/tmp/bench-decode-real.json"),
            "concurrency": load("/tmp/bench-concurrency.json"),
            "kv_line": kv[-120:], "acceptance": acc[-120:]}


def record(entry):
    with open(RESULTS, "a") as f:
        f.write(json.dumps(entry) + "\n")


def run_one(cfg):
    t0 = time.time()
    print(f"\n=== {cfg['name']}  {datetime.datetime.now():%H:%M:%S}", flush=True)
    apply_config(cfg)
    ok, err = boot()
    entry = {"ts": datetime.datetime.now().isoformat(), "config": cfg,
             "booted": ok, "error": err}
    if ok:
        entry.update(measure())
        print(f"    decode: {entry.get('decode')}", flush=True)
    else:
        print(f"    BOOT FAILED: {err[:160]}", flush=True)
    entry["minutes"] = round((time.time() - t0) / 60, 1)
    record(entry)
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--hours", type=float, default=8.0)
    a = ap.parse_args()

    q = QUEUE[:]
    random.shuffle(q)                      # order must not correlate with drift
    sched = []
    for i, cfg in enumerate(q):
        if i % CONTROL_EVERY == 0:
            sched.append(CONTROL)
        sched.append(cfg)
    sched.append(CONTROL)                  # cluster is LEFT on the control

    if a.plan or not a.run:
        print(f"{len(sched)} experiments, ~26 min each = ~{len(sched)*26/60:.1f}h")
        for s in sched:
            print(f"  {s['name']:<22} k={s['spec_k']} gmu={s['gmu']} env={s.get('env', {})}")
        print(f"\nresults -> {RESULTS}")
        return

    deadline = time.time() + a.hours * 3600
    for cfg in sched:
        if time.time() > deadline and cfg is not sched[-1]:
            print("budget spent; jumping to final control boot", flush=True)
            continue
        run_one(cfg)
    print("\nsweep complete; cluster left on control config", flush=True)


if __name__ == "__main__":
    main()
