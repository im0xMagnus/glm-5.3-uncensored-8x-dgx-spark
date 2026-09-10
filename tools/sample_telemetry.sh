#!/bin/bash
# Sample clock/power/temp on all 8 ranks every 3s until killed. One CSV line per
# sample: ts,node,clock_mhz,power_w,temp_c,hw_thermal_slowdown,sw_power_cap
OUT="${1:-/tmp/telemetry.csv}"
: > "$OUT"
while :; do
  for ip in 10 11 12 13 14 15 16 17; do
    ( v=$(ssh -o StrictHostKeyChecking=no -o BatchMode=yes -o ConnectTimeout=6 $USER@<NODE_PREFIX>.$ip \
        "nvidia-smi --query-gpu=clocks.gr,power.draw,temperature.gpu,clocks_event_reasons.hw_thermal_slowdown,clocks_event_reasons.sw_power_cap --format=csv,noheader,nounits" 2>/dev/null)
      [ -n "$v" ] && echo "$(date +%s),$ip,$v" >> "$OUT" ) &
  done
  wait
  sleep 3
done
