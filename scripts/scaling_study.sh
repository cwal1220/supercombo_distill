#!/bin/bash
# Train the two selected student sizes (base ~5 ms, w48d 25.6 ms on MaixCAM2) with the same data and schedule,
# then evaluate both against the shipped model, the small model and the teacher on the held-out routes.
# usage: scripts/scaling_study.sh [epochs]   (default 1: the short comparison run)
set -u
cd "$(dirname "$0")/.."
EP=${1:-1}
declare -A ARCH=(
  [s05_base]='{}'
  [s26_w48d]='{"widths":[48,96,192,384,768],"depths":[1,3,6,9,3]}'
)
declare -A BATCH=([s05_base]=64 [s26_w48d]=32)
for name in s05_base s26_w48d; do
  grep -q "epoch $((EP-1)) done" ckpt/scale/$name/train.log 2>/dev/null && continue
  echo "== $name $(date)"
  scripts/py scripts/train.py --out ckpt/scale/$name --epochs $EP --batch ${BATCH[$name]} --workers 14 --arch "${ARCH[$name]}" \
    > logs/scale_$name.log 2>&1
  tail -2 logs/scale_$name.log
  scripts/py scripts/eval_student.py --ckpt ckpt/scale/$name/last.pt > logs/scale_eval_$name.txt 2>&1
  cat logs/scale_eval_$name.txt
done
echo SCALEDONE
