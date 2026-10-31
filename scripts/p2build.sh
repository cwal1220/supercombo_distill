#!/bin/bash
# Pulsar2 6.0 build of configs/<name>.json; prints max_cycle. usage: scripts/p2build.sh <name> [6.0|8.0]
ROOT=$(cd "$(dirname "$0")/.." && pwd)
n=$1; ver=${2:-6.0}
/home/chan/pulsar2/scripts/pulsar2-$ver.sh pulsar2 build --config /data/supercombo_distill/configs/$n.json > "$ROOT/logs/build_$n.log" 2>&1
rc=$?
echo "$n: exit $rc $(grep -h 'max_cycle' "$ROOT/logs/build_$n.log" | tail -1 | sed 's/.*max_cycle/max_cycle/')"
