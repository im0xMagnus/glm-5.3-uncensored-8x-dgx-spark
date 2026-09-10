#!/bin/bash
# Runs ON spark-02 (detached, immune to the Mac). Same measurements we trust from FP8,
# plus repetition at HIGH (the card's "0 degeneration in every mode" claim).
# Order keeps each test clean: short benches first, solo 96K on a verified-idle engine, long gens last.
set -u; cd ~/bench
OUT=/tmp/nvfp4-suite.txt; : > "$OUT"
log(){ echo "$(date +%H:%M:%S) $*" | tee -a "$OUT"; }
M=http://10.100.128.10:8888/metrics
running(){ curl -sf -m 8 "$M" 2>/dev/null | grep -E '^vllm:num_requests_running' | grep -oE '[0-9.]+$'; }
idle(){ for i in $(seq 1 30); do r=$(running); [ "${r%.*}" = "0" ] && return 0; sleep 10; done; return 1; }
log "=== 0. smoke ==="
curl -sf -m 90 http://10.100.128.10:8888/v1/chat/completions -H 'Content-Type: application/json' \
 -d '{"model":"glm-5.3-uncensored","messages":[{"role":"user","content":"Reply with exactly: ALIVE"}],"max_tokens":8,"temperature":0}' \
 | python3 -c "import sys,json;d=json.load(sys.stdin);print('  smoke:',repr(d['choices'][0]['message'].get('content')))" 2>&1 | tee -a "$OUT"
log "=== 1. decode bench (MTP k=1, low) ==="; python3 -u bench_decode_real.py 2>&1 | tee -a "$OUT"
log "  acceptance: $(ssh -o BatchMode=yes -o ConnectTimeout=10 $USER@10.100.128.10 'docker logs vllm_nvfp4_tp8 2>&1 | grep -a "Mean acceptance length" | tail -1 | grep -oE "Mean acceptance length: [0-9.]+"' 2>/dev/null)"
log "=== 2. concurrency 1/4/8 ==="; python3 -u bench_concurrency.py 1 4 8 2>&1 | tee -a "$OUT"
log "=== 3. solo 96K (engine must be idle) ==="; idle || log "  WARN: engine not idle after 5 min"; python3 -u single64k.py 2>&1 | tee -a "$OUT"
log "=== 4. repetition @ low (3 x 8K) ==="; idle; python3 -u diag_rep_at_low.py 2>&1 | tee -a "$OUT"
log "=== 5. repetition @ HIGH (card claims 0 degeneration) ==="; idle; python3 -u diag_rep_at_high.py 2>&1 | tee -a "$OUT"
log "=== DONE ==="; touch /tmp/nvfp4-suite-DONE
