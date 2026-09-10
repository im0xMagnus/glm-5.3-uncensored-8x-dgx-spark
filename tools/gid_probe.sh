#!/bin/bash
# Find the RoCE v2 GID index for each cabled fabric NIC.
# The usable index is the one whose GID encodes THIS node's own fabric IP
# AND whose type is "RoCE v2" (not IB/RoCE v1). Never assume - firmware renumbers it.

ip_to_gid_suffix() {  # NODE_PREFIX_PLACEHOLDER.10 -> 0a64:800a
  IFS=. read -r a b c d <<< "$1"
  printf "%02x%02x:%02x%02x" "$a" "$b" "$c" "$d"
}

echo "$(hostname)"
for pair in "enp1s0f0np0:rocep1s0f0" "enP2p1s0f0np0:roceP2p1s0f0"; do
  NETIF="${pair%%:*}"; RDEV="${pair##*:}"
  IP=$(ip -4 -br addr show "$NETIF" 2>/dev/null | awk '{print $3}' | cut -d/ -f1)
  [ -z "$IP" ] && { echo "  $NETIF: no IPv4"; continue; }
  SUF=$(ip_to_gid_suffix "$IP")
  GDIR="/sys/class/infiniband/$RDEV/ports/1"
  [ -d "$GDIR" ] || { echo "  $RDEV: no such rdma device"; continue; }

  FOUND=""
  for g in "$GDIR"/gids/*; do
    idx=$(basename "$g")
    gid=$(cat "$g" 2>/dev/null)
    case "$gid" in
      *ffff:$SUF) t=$(cat "$GDIR/gid_attrs/types/$idx" 2>/dev/null)
                  [ "$t" = "RoCE v2" ] && FOUND="$idx" && break ;;
    esac
  done
  STATE=$(cat "$GDIR/state" 2>/dev/null | awk '{print $NF}')
  RATE=$(cat "$GDIR/rate" 2>/dev/null)
  if [ -n "$FOUND" ]; then
    printf "  %-16s %-14s ip=%-15s GID_IDX=%-3s state=%-8s %s\n" \
      "$NETIF" "$RDEV" "$IP" "$FOUND" "$STATE" "$RATE"
  else
    printf "  %-16s %-14s ip=%-15s GID_IDX=NOT-FOUND state=%s\n" \
      "$NETIF" "$RDEV" "$IP" "$STATE"
  fi
done
