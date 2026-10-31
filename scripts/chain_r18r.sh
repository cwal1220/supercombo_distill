#!/bin/sh
# R18r: continue the final model (R18q EMA) with QAT at a lower lr on all replays (cal2 now has every night route),
# 4 epochs, then S8-evaluate against R18q EMA (logs/eval_mac_r18r.txt) and build the better of last/EMA if it improves
# plan_y2s without losing any of the six metrics to the shipped model (outputs/quant_r18r).
cd "$(dirname "$0")/.."
ARCH='{"backbone": "r18", "temporal_heads": ["lead", "lead_prob"], "lane_policy": true, "policy_tokens": 25}'
R=mac_r18r
(sleep 60; scripts/snap_epochs.sh ckpt/$R 4 eval-s8 > logs/snap_$R.log 2>&1) &
caffeinate -i .venv/bin/python scripts/train.py --out ckpt/$R --epochs 4 --epoch-samples 400000 --batch 64 --workers 4 --compile-refresh \
  --aug --wd 0.01 --flip 0.3 --curve-sampling --night-weight 2.5 --qat --gap-aug 0.2 --hist-noise 0.2 --ema 0.999 --lr 1e-4 \
  --refresh-every 2 --init ckpt/mac_r18q/ema.pt --tags r2,cal1,cal2 --plan-lat-weight 3 --arch "$ARCH" >> logs/$R.log 2>&1 && \
.venv/bin/python scripts/eval_student.py --s8 --ckpt ckpt/mac_r18q/ema.pt ckpt/$R/last.pt ckpt/$R/ema.pt > logs/eval_$R.txt 2>&1
echo DONE > logs/DONE_$R
