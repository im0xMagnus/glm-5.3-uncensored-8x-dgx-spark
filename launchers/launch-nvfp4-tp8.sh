#!/bin/bash
# GLM-5.3-UNCENSORED-NVFP4 · TP=8 · 8x DGX Spark GB10 — VALIDATED 2026-09-08 (2nd boot)
#
# Based on the draft launcher from the nvfp4-cutover-tooling branch, with three measured changes:
#   --gpu-memory-utilization 0.80  (0.86 -> global OOM on the head during FP4 autotune; 0.80 boots
#                                   and serves; steady-state avail 4.4 GB head / 6-8 GB workers — thin)
#   --max-model-len 524288          (KV at 0.80 = 694,784 tokens -> 1.33x concurrency at 512K; the
#                                   1M the card claims needs the NVFP4 *KV cache* format, separate experiment)
#   VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR  (autotune results are in-process by default; without this every
#                                   boot re-profiles ~10-15 min and repeats the memory spike)
# Measured: 21.6 tok/s realistic decode (FP8: 15.9), MTP acceptance 1.70 @ depth 1, `high` produces
# answers on this checkpoint (FP8 did not). NOTE: image ships FlashInfer 0.6.15, not >=0.6.18 — the
# DSA path works regardless (same backend FP8 ran on all night).
set -u

NODE_RANK="${1:-}"
case "$NODE_RANK" in 0|1|2|3|4|5|6|7) ;; *) echo "rank must be 0-7" >&2; exit 2 ;; esac

# Pulled by digest sha256:7ac0031cfefc... on the head, then fanned over the fabric.
# `docker load` drops the digest ref, so workers address it by tag; byte-identity
# is established by every rank resolving the same image id ba79eedf36272341...
IMAGE="ghcr.io/ciprianveg/gb10-glm-5.2:v19-vision"   # same image: carries the SM121 patches + FlashInfer >= 0.6.18 the NVFP4 card requires
NAME="vllm_nvfp4_tp8"
MODEL_HOST=/data/models/GLM-5.3-UNCENSORED-NVFP4      # local copy on every node
MODEL_PATH=/models/glm-5.3-nvfp4
CACHE_HOST=/var/tmp/nvfp4-vllm-cache
HEAD_IP=NODE_PREFIX_PLACEHOLDER.10
MPORT=29531
PORT=8888

# rank -> fabric IP  (.12 is NODE_HOST_2_PLACEHOLDER, kept in-cluster as rank 2)
FAB=(NODE_PREFIX_PLACEHOLDER.10 NODE_PREFIX_PLACEHOLDER.11 NODE_PREFIX_PLACEHOLDER.12 NODE_PREFIX_PLACEHOLDER.13 \
     NODE_PREFIX_PLACEHOLDER.14 NODE_PREFIX_PLACEHOLDER.15 NODE_PREFIX_PLACEHOLDER.16 NODE_PREFIX_PLACEHOLDER.17)
HOST_IP="${FAB[$NODE_RANK]}"
[ "$NODE_RANK" = "0" ] && HEADLESS="" || HEADLESS="--headless"

# RoCEv2 GID index is per-NIC and moves across firmware updates — probe it.
GIDX=3
for i in 0 1 2 3 4 5 6 7; do
  t=$(cat /sys/class/infiniband/rocep1s0f0/ports/1/gid_attrs/types/$i 2>/dev/null)
  g=$(cat /sys/class/infiniband/rocep1s0f0/ports/1/gids/$i 2>/dev/null)
  case "$t" in *"RoCE v2"*) case "$g" in *ffff*) GIDX=$i; break ;; esac ;; esac
done
echo "rank $NODE_RANK  ip=$HOST_IP  NCCL_IB_GID_INDEX=$GIDX"

# Fail loudly rather than let Docker invent an empty dir over a missing mount.
test -f "$MODEL_HOST/config.json" || { echo "MISSING: $MODEL_HOST/config.json" >&2; exit 3; }
n=$(ls "$MODEL_HOST"/*.safetensors 2>/dev/null | wc -l)
[ "$n" -eq 282 ] || { echo "MISSING: only $n/282 safetensors in $MODEL_HOST" >&2; exit 3; }
mkdir -p "$CACHE_HOST"
docker rm -f "$NAME" >/dev/null 2>&1 || true

docker run --gpus all -d --name "$NAME" --restart no \
  --entrypoint /opt/venv/bin/vllm \
  --network host --ipc host --shm-size 32g \
  --ulimit memlock=-1:-1 --cap-add IPC_LOCK \
  --device /dev/infiniband:/dev/infiniband \
  -v "$MODEL_HOST:$MODEL_PATH:ro" \
  -v "$CACHE_HOST:/cache" \
  -e VLLM_HOST_IP=$HOST_IP \
  -e HF_HOME=/cache/huggingface -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  -e VLLM_ENGINE_READY_TIMEOUT_S=7200 \
  -e VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1800 \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e TORCH_CUDA_ARCH_LIST=12.1a -e CUTE_DSL_ARCH=sm_121a \
  -e VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR=/cache/flashinfer-autotune \
  -e VLLM_USE_V2_MODEL_RUNNER=1 -e VLLM_USE_B12X_SPARSE_INDEXER=1 \
  -e VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
  -e VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0 \
  -e DETERMINISTIC_MOE_ALIGN=1 -e VLLM_MARLIN_USE_ATOMIC_ADD=1 \
  -e VLLM_DCP_QUERY_SPLIT=1 -e VLLM_B12X_MLA_CKG_GATHER=1 \
  -e OMP_NUM_THREADS=8 \
  -e VLLM_ADAPTIVE_SPEC_DEPTHS=2,4,5 -e VLLM_MTP_INSTRUMENT=1 -e VLLM_MTP_INSTRUMENT_WINDOW=32 \
  -e VLLM_B12X_MLA_SPEC_EXTEND_AS_DECODE=1 -e VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=256 \
  -e GLM52_BIND_HOST_TRITON=0 \
  `# compile caches into the persistent /cache mount so a reboot does not re-JIT` \
  -e TORCH_EXTENSIONS_DIR=/cache/torch_extensions \
  -e TORCHINDUCTOR_CACHE_DIR=/cache/torchinductor-cache \
  -e TRITON_CACHE_DIR=/cache/triton-cache -e XDG_CACHE_HOME=/cache \
  -e B12X_CUTE_COMPILE_CACHE_DIR=/cache/b12x/cute_compile \
  -e B12X_LOG_CUTE_COMPILES_AFTER_ENGINE_START=0 \
  `# DELTA: this cluster's measured fabric — both logical halves of QSFP port 0.` \
  `# Listing one half caps at ~100G; NCCL confirmed both at 23.6 GB/s busbw.` \
  -e NCCL_NET=IB -e NCCL_IB_DISABLE=0 \
  -e NCCL_IB_HCA=rocep1s0f0,roceP2p1s0f0 -e NCCL_IB_GID_INDEX=$GIDX \
  -e NCCL_IB_ROCE_VERSION_NUM=2 -e NCCL_IB_ADDR_FAMILY=AF_INET \
  -e NCCL_IB_ADDR_RANGE=NODE_PREFIX_PLACEHOLDER.0/24 \
  -e NCCL_SOCKET_IFNAME=enp1s0f0np0 -e GLOO_SOCKET_IFNAME=enp1s0f0np0 \
  -e TP_SOCKET_IFNAME=enp1s0f0np0 -e MN_IF_NAME=enp1s0f0np0 \
  -e NCCL_CROSS_NIC=1 -e NCCL_NVLS_ENABLE=0 -e NCCL_CUMEM_ENABLE=0 \
  -e NCCL_IGNORE_CPU_AFFINITY=1 -e NCCL_BUFFSIZE=16777216 \
  -e NCCL_MAX_NCHANNELS=4 -e NCCL_MIN_NCHANNELS=4 \
  -e NCCL_DEBUG=WARN -e TORCH_NCCL_ASYNC_ERROR_HANDLING=1 \
  "$IMAGE" \
    serve "$MODEL_PATH" \
    --served-model-name glm-5.3-uncensored \
    --host 0.0.0.0 --port "$PORT" \
    --trust-remote-code \
    --tensor-parallel-size 8 --pipeline-parallel-size 1 \
    --decode-context-parallel-size 1 --dcp-kv-cache-interleave-size 1 \
    --attention-backend B12X_MLA_SPARSE \
    --hf-overrides '{"index_topk_pattern":"FFFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSS"}' \
    `# MEASURED 2026-09-06: at 0.80 (97.35 GiB budget) the model itself took` \
    `# 90.96 GiB/rank (incl. the MTP drafter) leaving -0.42 GiB -> "No available` \
    `# memory for the cache blocks". 0.85 = 103.4 GiB budget -> ~12.4 GiB/rank KV.` \
    `# This is ABOVE ciprianveg's 0.69 because our FP8 checkpoint is 1.87x theirs.` \
    `# MEASURED: launch-time free is 105.81 GiB (not the 111-115 GiB idle`  \
    `# reading) -> usable gmu ceiling ~0.869. 0.88 needs 107.09 and fails`  \
    `# the startup check. 0.85 (103.4 GiB) is proven to pass.`              \
    --gpu-memory-utilization 0.80 \
    `# DELTA: 410000 -> 1000000 (model-native 1M). vLLM sizes the KV pool from gmu,` \
    `# NOT from max-model-len, and VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 (their setting)` \
    `# turns "exceeds KV capacity" into a warning rather than a fatal. So asking for` \
    `# 1M is safe: if the pool cannot back it we see the real number in` \
    `# "GPU KV cache size" and lower it, without wasting a 30-45 min boot proving it.` \
    `# MEASURED: KV ~53.4 KB/token. gmu 0.85 gave 6.25 GiB/rank -> engine`  \
    `# reported max len 122,560. 1M needs 50.97 GiB/rank KV; model is 90.96,` \
    `# so 141.9 GiB vs a 121.69 GiB node -> 1M IMPOSSIBLE at any gmu.`       \
    `# 0.88 -> ~16 GiB/rank -> ~300K ceiling. 262144 leaves margin.`         \
    `# Engine's own estimate at this gmu: max len 122,560. 120000 sits just` \
    `# under it. dealignai document 131072 - close, on larger-memory GPUs.`  \
    `# MEASURED across boots: KV pool 5.62-6.25 GiB/rank at gmu 0.85 (~10%`  \
    `# boot-to-boot variance). Engine estimates 110,208 max at 5.62 GiB.`    \
    `# 98304 needs ~5.0 GiB -> ~11% margin against the low reading.`         \
    --max-model-len 524288 \
    --max-num-seqs 4 --max-num-batched-tokens 4096 \
    --enable-chunked-prefill --enable-prefix-caching --async-scheduling \
    --long-prefill-token-threshold 2048 \
    `# KEPT FROM ciprianveg glm-5.3/v19: decode-aware prefill scheduling.` \
    --enable-decode-aware-prefill \
    --decode-prefill-token-budget 1024 \
    --idle-prefill-token-budget 16384 \
    --max-long-prefills-per-step 1 \
    --kv-cache-dtype fp8_ds_mla \
    `# KEPT FROM ciprianveg: they run k=5 + adaptive depths on THIS architecture` \
    `# (glm_moe_dsa, 78 layers) on a real 8-node GB10 cluster. Our k=2 finding was` \
    `# on GLM-5.3-Flash (glm5_next, 45 layers) — a different MTP head. Their number` \
    `# wins. Only DELTA: quantization fp8 (our checkpoint) not compressed-tensors.` \
    `# EXPERIMENT 1 (2026-09-06): MTP DISABLED per the dealignai model card:` \
    `# "MTP speculative decoding is non-functional on GLM-5.3 regular in vLLM".` \
    `# Costs ~3 GiB/rank (model+drafter 90.96 vs 88.0 alone) = ~56K tokens of` \
    `# context. Baseline WITH MTP k=5: 20.3 tok/s decode, 109,888-token KV.` \
    `# NVFP4 card: MTP head is also CRACK'd, ~87% draft acceptance; their k=1. Do NOT force` \
    `# --moe-backend with MTP on (the bf16 MTP head is unquantized) - let vLLM auto-select.` \
    --speculative-config '{"model":"/models/glm-5.3-nvfp4","draft_attention_backend":"B12X_MLA_SPARSE","method":"mtp","num_speculative_tokens":1,"draft_tensor_parallel_size":1}' \
    --compilation-config '{"cudagraph_mode":"FULL","max_cudagraph_capture_size":30}' \
    --disable-custom-all-reduce \
    --reasoning-parser glm45 --tool-call-parser glm47 --enable-auto-tool-choice \
    --generation-config vllm \
    `# FIX: this template accepts reasoning_effort + clear_thinking, NOT`      \
    `# enable_thinking (that is the GLM-5.3-Flash convention and is silently`  \
    `# ignored here - which is why reasoning leaked into content). dealignai`  \
    `# document reasoning_effort in {off, low, high, max}.`                    \
    --default-chat-template-kwargs '{"reasoning_effort":"low"}' \
    --distributed-executor-backend mp \
    --nnodes 8 --node-rank "$NODE_RANK" \
    --master-addr "$HEAD_IP" --master-port "$MPORT" \
    --seed 42 \
    $HEADLESS

sleep 3
docker ps --format '{{.Names}} {{.Status}}' | grep -q "$NAME" \
  && echo "launched $NAME rank=$NODE_RANK" \
  || { echo "$NAME exited immediately — docker logs $NAME" >&2; exit 1; }
