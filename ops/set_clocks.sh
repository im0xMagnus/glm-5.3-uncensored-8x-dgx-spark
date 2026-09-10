#!/bin/bash
# set_clocks.sh <maxMHz|reset>  - apply nvidia-smi -lgc 0,<max> (or -rgc) to all 8
# ranks and report what the driver actually accepted. GB10 exposes no power limit
# (-pl is N/A), so this is the only power/thermal lever available.
set -u
. "$(dirname "$0")/lib.sh"
V="${1:?usage: set_clocks.sh <maxMHz|reset>}"
if [ "$V" = reset ]; then CMD="nvidia-smi -rgc"; else CMD="nvidia-smi -lgc 0,$V"; fi
for ip in 10 11 12 13 14 15 16 17; do
  ( out=$(ssh -o StrictHostKeyChecking=no -o BatchMode=yes -o ConnectTimeout=10 $USER@NODE_PREFIX_PLACEHOLDER.$ip \
      "sudo -n $CMD 2>&1 | tail -1; \
       nvidia-smi --query-gpu=clocks.gr,clocks.max.gr --format=csv,noheader" 2>/dev/null)
    echo "  .$ip | $(echo "$out" | tr '\n' '|')" ) &
done; wait
