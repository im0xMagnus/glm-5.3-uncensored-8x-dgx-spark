#!/bin/bash
# Shared helpers for the Spark cluster tooling. No secrets: root ops go through the scoped
# NOPASSWD rule in prism-spark-ops.sudoers (install once per node with install-sudoers.sh).
# Every helper FAILS LOUDLY -- a silent no-op here cost a boot on 2026-09-08.
SPARK_IPS="10 11 12 13 14 15 16 17"
SPARK_HEAD="10.100.128.10"
SPARK_API="http://$SPARK_HEAD:8888"
# drop_caches <ip-octet>  -- evict page cache on a node; prints before->after MB or FAILS
drop_caches() {
  ssh -o BatchMode=yes -o ConnectTimeout=15 "$USER@10.100.128.$1" \
    'b=$(free -m|awk "/^Mem:/{print \$6}"); sync; echo 3 | sudo -n tee /proc/sys/vm/drop_caches >/dev/null || { echo "DROP FAILED (no NOPASSWD rule? run install-sudoers.sh)"; exit 1; }; echo "cache ${b}->$(free -m|awk "/^Mem:/{print \$6}")MB"'
}
# gpu_clock <ip-octet> <maxMHz|reset>
gpu_clock() {
  local c; [ "$2" = reset ] && c="nvidia-smi -rgc" || c="nvidia-smi -lgc 0,$2"
  ssh -o BatchMode=yes -o ConnectTimeout=15 "$USER@10.100.128.$1" "sudo -n $c 2>&1 | tail -1; nvidia-smi --query-gpu=clocks.gr --format=csv,noheader"
}
