#!/bin/sh
# Label the routes listed in $1 (one per line, raw drives under $2) into data/ep_label on the Mac, $3 at a time, after
# the PIDs given in $4 (comma list) exit: ep_label.py --no-small with the teacher on the Neural Engine (ORT_COREML;
# compile it into outputs/coreml_cache once before), then the new units' frames compressed (zframes.py). A route is not
# started with less than 8 GB free. Logs: logs/label/<route>.log, then logs/label/QUEUE_DONE.
list=$1; raw=$2; par=${3:-2}; wait_pids=$4
cd "$(dirname "$0")/.."
for p in $(echo "$wait_pids" | tr ',' ' '); do while kill -0 "$p" 2>/dev/null; do sleep 30; done; done
xargs -P "$par" -n 1 sh scripts/label_one.sh "$raw" < "$list"
echo done > logs/label/QUEUE_DONE
