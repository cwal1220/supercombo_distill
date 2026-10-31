#!/bin/sh
# Label the routes listed in $1 (raw drives under $2) in ONE process (the CoreML teacher loads once, ~4.5 min), after the
# PIDs in $3 (comma list) exit; ep_label.py --no-small with ORT_COREML=CPUAndNeuralEngine (~45 ms a frame), then compress
# the units this run wrote (zframes.py). Extra ep_label options (e.g. --tag ... --rate 2) go after $3.
# Logs: logs/label/batch_<list name>.log, then logs/label/DONE_<list name>.
list=$1; raw=$2; wait_pids=$3; shift 3
cd "$(dirname "$0")/.."
for p in $(echo "$wait_pids" | tr ',' ' '); do while kill -0 "$p" 2>/dev/null; do sleep 30; done; done
name=$(basename "$list" .txt); log=logs/label/batch_$name.log; mark=logs/label/.start_$name; touch "$mark"
set -- $(sed "s#^#$raw/#" "$list") --also-end "$@"
first=$1; shift
routes=""; while [ "$1" != "--also-end" ]; do routes="$routes $1"; shift; done; shift
ORT_COREML=CPUAndNeuralEngine ORT_THREADS=4 .venv/bin/python scripts/ep_label.py "$first" data/ep_label --no-small --also $routes "$@" > "$log" 2>&1
for u in data/ep_label/*; do
  if [ -f "$u/frames.npy" ] && [ "$u/frames.npy" -nt "$mark" ]; then .venv/bin/python scripts/zframes.py "$u" >> "$log" 2>&1; fi
done
rm -f "$mark"; echo done > logs/label/DONE_$name
