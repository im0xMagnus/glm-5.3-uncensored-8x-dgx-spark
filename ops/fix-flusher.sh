#!/bin/bash
# Ensure exactly ONE tee-idiom boot flusher runs. Kill by PID (never pkill -f a pattern
# that could match our own argv), start detached, then PROVE it: count + a real drop.
for p in $(pgrep -f "[n]vfp4-boot-flush"); do kill "$p" 2>/dev/null; done
sleep 1
b=$(free -m | awk '/^Mem:/{print $6}'); sync; echo 3 | sudo -n tee /proc/sys/vm/drop_caches >/dev/null 2>&1; a=$(free -m | awk '/^Mem:/{print $6}')
chmod +x "$HOME/boot-flush.sh"
setsid nohup "$HOME/boot-flush.sh" </dev/null >/dev/null 2>&1 &
sleep 1
printf 'flushers=%s drop=%s->%sMB avail=%sMB\n' "$(pgrep -fc "[n]vfp4-boot-flush")" "$b" "$a" "$(free -m | awk '/^Mem:/{print $7}')"
