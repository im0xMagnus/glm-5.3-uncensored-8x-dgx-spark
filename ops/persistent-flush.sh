#!/bin/bash
# Persistent low-rate page-cache flusher for the GB10 nodes: cron runs it every 5 minutes.
# Same tee idiom as boot-flush.sh -- matches /etc/sudoers.d/prism-spark-ops (no password, no sh -c).
# It never touches GPU / KV memory; it only stops host headroom eroding into page cache and keeps
# `free` honest. Log: /var/tmp/persistent-flush.log (self-trimmed). Exit 1 = the drop did not happen.
LOG=/var/tmp/persistent-flush.log
b=$(awk '/^Cached:/{print int($2/1024)}' /proc/meminfo)
if sync && echo 3 | sudo -n /usr/bin/tee /proc/sys/vm/drop_caches >/dev/null 2>>"$LOG"; then
  a=$(awk '/^Cached:/{print int($2/1024)}' /proc/meminfo); v=$(awk '/^MemAvailable:/{print int($2/1024)}' /proc/meminfo)
  echo "$(date +%FT%T) ok cached ${b}->${a}MB avail ${v}MB" >>"$LOG"
else
  echo "$(date +%FT%T) FAILED (sudo -n denied or tee error)" >>"$LOG"; exit 1
fi
if [ "$(wc -l <"$LOG")" -gt 600 ]; then tail -n 300 "$LOG" >"$LOG.tmp" && mv "$LOG.tmp" "$LOG"; fi
exit 0
