#!/bin/bash
# run_condition.sh <label> <maxMHz|default>
# One clock condition end-to-end: set clock -> sample telemetry on all 8 ranks
# via ONE persistent ssh each (never reconnect-per-sample; that is what caused
# the MaxStartups exhaustion earlier) -> run the bench -> aggregate.
set -u
cd "$(dirname "$0")"
LABEL="${1:?label}"; CLK="${2:?maxMHz|default}"
D=/tmp/clockab; mkdir -p $D

if [ "$CLK" = default ]; then ./set_clocks.sh reset >/dev/null 2>&1
else ./set_clocks.sh "$CLK" >/dev/null 2>&1; fi
sleep 5
echo "### condition=$LABEL requested=$CLK"
printf '  actual clocks: '
for ip in 10 11 12 13 14 15 16 17; do
  printf '%s ' "$(ssh -o BatchMode=yes -o ConnectTimeout=8 $USER@NODE_PREFIX_PLACEHOLDER.$ip \
    'nvidia-smi --query-gpu=clocks.gr --format=csv,noheader,nounits' 2>/dev/null)"
done; echo

# one persistent ssh per node, sampling for up to 8 min
for ip in 10 11 12 13 14 15 16 17; do
  ssh -o BatchMode=yes -o ConnectTimeout=8 $USER@NODE_PREFIX_PLACEHOLDER.$ip \
    "for i in \$(seq 1 160); do nvidia-smi --query-gpu=clocks.gr,power.draw,temperature.gpu --format=csv,noheader,nounits; sleep 3; done" \
    > $D/telem-$LABEL-$ip.csv 2>/dev/null &
done
SAMPLERS=$(jobs -p | tr '\n' ' ')

python3 bench_clocks.py "$LABEL" 3
BRC=$?

for p in $SAMPLERS; do kill $p 2>/dev/null; done
wait 2>/dev/null

echo "  --- telemetry under load (per-node mean of clock/power/temp) ---"
python3 - "$D" "$LABEL" <<'PY'
import sys, glob, os, statistics
d, label = sys.argv[1], sys.argv[2]
tot_p, rows = [], []
for f in sorted(glob.glob(os.path.join(d, f"telem-{label}-*.csv"))):
    ip = f.split("-")[-1].split(".")[0]
    c, p, t = [], [], []
    for line in open(f):
        parts = [x.strip() for x in line.split(",")]
        if len(parts) != 3: continue
        try: c.append(float(parts[0])); p.append(float(parts[1])); t.append(float(parts[2]))
        except ValueError: continue
    if not c: continue
    rows.append((ip, statistics.mean(c), statistics.mean(p), max(p), statistics.mean(t), max(t), len(c)))
    tot_p.append(statistics.mean(p))
for ip, mc, mp, xp, mt, xt, n in rows:
    print(f"    .{ip}  clk={mc:7.1f}MHz  pwr={mp:5.1f}W (max {xp:5.1f})  temp={mt:4.1f}C (max {xt:4.1f})  n={n}")
if tot_p:
    print(f"    CLUSTER mean power = {sum(tot_p):6.1f} W across 8 nodes  ({sum(tot_p)/8:.1f} W/node)")
PY
exit $BRC
