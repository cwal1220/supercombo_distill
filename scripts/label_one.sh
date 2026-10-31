#!/bin/sh
# One route for label_queue.sh: label_one.sh <raw dir> <route>. Only the units this run writes get compressed (older
# units of the route may be in use by a training run that still reads their frames.npy).
raw=$1; r=$2
free=$(df -g . | awk 'NR==2{print $4}')
if [ "$free" -lt 8 ]; then echo "$r: skipped, only ${free} GB free" >> logs/label/QUEUE_SKIPPED; exit 0; fi
mark=logs/label/.start_$r; touch "$mark"
ORT_COREML=CPUAndNeuralEngine ORT_THREADS=4 .venv/bin/python scripts/ep_label.py "$raw/$r" data/ep_label --no-small > "logs/label/$r.log" 2>&1
for u in data/ep_label/"$r"__*; do
  if [ -f "$u/frames.npy" ] && [ "$u/frames.npy" -nt "$mark" ]; then .venv/bin/python scripts/zframes.py "$u" >> "logs/label/$r.log" 2>&1; fi
done
rm -f "$mark"
