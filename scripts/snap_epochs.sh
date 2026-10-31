#!/bin/bash
# Keep a copy of every epoch's checkpoint: train.py overwrites <out>/last.pt after each epoch, so this watches its
# mtime and copies it to <out>/ep<N>.pt (N = 1, 2, ... counted within this run) once the write has settled.
# Stops when the training process for <out> is gone (after one last check) or after --max copies.
# With a third argument "eval", each copy is also evaluated on the held-out routes (eval_student.py on the CPU, in the
# background, so the GPU stays with training) into logs/eval_<run>_snaps.txt and the run's TensorBoard; "eval-s8" evaluates
# them with S8 weights (eval_student.py --s8, for QAT runs).
# usage (Mac): nohup scripts/snap_epochs.sh ckpt/mac_base_5ep 5 [eval|eval-s8] > logs/snap_mac_base_5ep.log 2>&1 < /dev/null &
cd "$(dirname "$0")/.."
out=${1:?usage: snap_epochs.sh <out dir> [max copies] [eval]}
max=${2:-100}
ev=${3:-}
ck=$out/last.pt
mtime() { stat -f %m "$ck" 2>/dev/null || echo 0; }
prev=$(mtime)
n=$(ls "$out"/ep*.pt 2>/dev/null | wc -l | tr -d ' ')
echo "$(date '+%F %T') watching $ck (mtime $prev, $n copies already)"
while [ "$n" -lt "$max" ]; do
  alive=$(ps -ax -o command | grep -c "[s]cripts/train.py --out $out")
  m=$(mtime)
  if [ "$m" != "$prev" ]; then
    sleep 15   # torch.save writes in place; let it finish
    m=$(mtime)
    n=$((n + 1))
    cp -p "$ck" "$out/ep$n.pt"
    echo "$(date '+%F %T') ep$n.pt <- last.pt (mtime $m, $(du -h "$out/ep$n.pt" | cut -f1))"
    if [ "$ev" = eval ] || [ "$ev" = eval-s8 ]; then
      (OMP_NUM_THREADS=4 .venv/bin/python scripts/eval_student.py --device cpu --ckpt "$out/ep$n.pt" $([ "$ev" = eval-s8 ] && echo --s8) 2>&1 \
        | grep -E "^model|^student|Error|Traceback" >> "logs/eval_$(basename "$out")_snaps.txt"
       echo "$(date '+%F %T') ep$n.pt evaluated") &
    fi
    prev=$m
  elif [ "$alive" = 0 ]; then
    echo "$(date '+%F %T') training process gone, stopping"
    break
  fi
  sleep 20
done
wait
echo "$(date '+%F %T') done, $n copies"
