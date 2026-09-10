#!/bin/bash
# 8-node NCCL all-reduce over RoCE. Rank 0 = HEAD_HOST_PLACEHOLDER @ NODE_PREFIX_PLACEHOLDER.10
IMG="mia/glm53-flash-spark:mm-ray-v1"
LOGDIR="$(cd "$(dirname "$0")" && pwd)/nccl-logs8"
mkdir -p "$LOGDIR"; rm -f "$LOGDIR"/*.log

DC() {
cat <<EOF
docker run --rm --network host --gpus all --ipc=host \
  --device=/dev/infiniband --cap-add=IPC_LOCK --ulimit memlock=-1:-1 \
  -v /tmp/nccl_allreduce.py:/tmp/nccl_allreduce.py \
  -e MASTER_ADDR=NODE_PREFIX_PLACEHOLDER.10 -e MASTER_PORT=29500 \
  -e RANK=$1 -e WORLD_SIZE=8 \
  -e NCCL_IB_HCA=rocep1s0f0,roceP2p1s0f0 \
  -e NCCL_IB_GID_INDEX=3 \
  -e NCCL_SOCKET_IFNAME=enp1s0f0np0 \
  -e NCCL_IB_DISABLE=0 \
  -e NCCL_DEBUG=INFO -e NCCL_DEBUG_SUBSYS=INIT,NET \
  --entrypoint python3 $IMG /tmp/nccl_allreduce.py
EOF
}

echo "  clearing stale containers..."
for n in 0 1 2 3 4 5 6 7; do
  ssh -o ConnectTimeout=8 -o BatchMode=yes spark-0$n \
    "docker ps -q --filter ancestor=$IMG | xargs -r docker kill >/dev/null 2>&1" 2>/dev/null &
done
wait

echo "  launching ranks 1-7..."
for n in 1 2 3 4 5 6 7; do
  ssh -o ConnectTimeout=10 -o BatchMode=yes spark-0$n "$(DC $n)" \
      > "$LOGDIR/rank$n.log" 2>&1 &
done
sleep 5
echo "  launching rank 0..."
ssh -o ConnectTimeout=10 -o BatchMode=yes HEAD_HOST_PLACEHOLDER "$(DC 0)" > "$LOGDIR/rank0.log" 2>&1
echo "  rank0 exit=$?"
wait
echo "  all ranks returned"
