#!/bin/bash
# boot-flush: page cache refilling during a 433 GiB weight load competes with
# GPU/unified-memory allocation on GB10. Drop every 60s for the boot window, then exit.
# Uses the tee idiom matching /etc/sudoers.d/spark-ops (no password, no sh -c).
for i in $(seq 1 35); do
  sync; echo 3 | sudo -n tee /proc/sys/vm/drop_caches >/dev/null 2>&1
  sleep 60
done
