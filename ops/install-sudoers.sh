#!/bin/bash
# One-time per node: grant passwordless page-cache drop + GPU clock control (nothing else).
# Validates with visudo BEFORE install; visudo and install run under ONE authenticated sudo.
# Verifies with `sudo -k` first -- a cached ticket makes `sudo -n` lie (learned 2026-09-08).
# Usage: SUDO_PW=... ./install-sudoers.sh 10 11 12 ...   (prompts if SUDO_PW unset)
set -u; cd "$(dirname "$0")"
[ -n "${SUDO_PW:-}" ] || { read -rsp "node sudo password: " SUDO_PW; echo; }
for ip in "$@"; do
  scp -q -o BatchMode=yes spark-ops.sudoers "$USER@NODE_PREFIX_PLACEHOLDER.$ip:/tmp/spark-ops.sudoers"
  r=$(printf '%s\n' "$SUDO_PW" | ssh -o BatchMode=yes -o ConnectTimeout=15 "$USER@NODE_PREFIX_PLACEHOLDER.$ip" '
    out=$(sudo -S -p "" sh -c "visudo -cf /tmp/spark-ops.sudoers >/dev/null 2>&1 && install -o root -g root -m 0440 /tmp/spark-ops.sudoers /etc/sudoers.d/spark-ops 2>&1 && echo INSTALLED" 2>&1 | tail -1)
    sudo -k; echo 3 | sudo -n tee /proc/sys/vm/drop_caches >/dev/null 2>&1 && t=OK || t=DENIED
    echo "install=${out:-FAILED} verify(sudo -k; sudo -n tee)=$t"')
  echo "  .$ip $r"
done
