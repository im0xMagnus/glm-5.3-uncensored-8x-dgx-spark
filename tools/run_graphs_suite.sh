#!/bin/bash
# Chain the three open FP8 items against the cudagraphs-on / MTP-off / low config,
# in the order that keeps each test clean:
#   1. decode bench      (short; quantifies the graphs-vs-eager delta)
#   2. solo 88K          (needs an IDLE engine - verified before start)
#   3. repetition @ low  (three 8K generations; longest, runs last)
set -u
cd "$(dirname "$0")"
OUT=/tmp/graphs-suite.txt
: > "$OUT"
log(){ echo "$(date +%H:%M:%S) $*" | tee -a "$OUT"; }
M="http://10.100.128.10:8888/metrics"
running(){ curl -sf -m 8 "$M" 2>/dev/null | grep -E '^vllm:num_requests_running' | grep -oE '[0-9.]+$'; }

log "=== 0. wait for boot ==="
for i in $(seq 1 200); do
  o=$(ssh -o BatchMode=yes -o ConnectTimeout=10 spark-00 'docker logs vllm_fp8_tp8 2>&1 | grep -aE "Available KV cache memory|GPU KV cache size|Maximum concurrency|Application startup complete|ValueError|OutOfMemoryError|EngineCore failed" | tail -5' 2>/dev/null)
  if echo "$o" | grep -q "Application startup complete"; then echo "$o" | cut -c1-200 | tee -a "$OUT"; break; fi
  if echo "$o" | grep -qE "ValueError|OutOfMemoryError|EngineCore failed"; then log "BOOT FAILED"; echo "$o" | tee -a "$OUT"; exit 1; fi
  sleep 15
done
curl -sf -m 10 http://10.100.128.10:8888/v1/models >/dev/null || { log "API not up after boot loop"; exit 1; }
log "cudagraph capture in log: $(ssh -o BatchMode=yes spark-00 'docker logs vllm_fp8_tp8 2>&1 | grep -aciE "cudagraph|capturing"' 2>/dev/null)"

log "=== 1. decode bench (effort=low default, graphs on) ==="
python3 -u bench_decode_real.py 2>&1 | tee -a "$OUT"

log "=== 2. solo 88K long-context (engine must be idle) ==="
for i in $(seq 1 30); do r=$(running); [ "${r%.*}" = "0" ] && break; log "  waiting, running=$r"; sleep 10; done
log "  running=$(running) -> starting"
python3 -u single64k.py 2>&1 | tee -a "$OUT"
log "  preemptions now: $(curl -sf -m 8 $M 2>/dev/null | grep -E '^vllm:num_preemptions_total' | grep -oE '[0-9.]+$')"

log "=== 3. repetition harness @ low (3 x 8000 tok) ==="
for i in $(seq 1 30); do r=$(running); [ "${r%.*}" = "0" ] && break; sleep 10; done
python3 -u diag_rep_at_low.py 2>&1 | tee -a "$OUT"

log "=== DONE ==="
