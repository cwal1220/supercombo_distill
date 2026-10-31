#!/bin/sh
# Low-rank FC students (scripts/lowrank.py: temporal.0 and vis_fc as rank-128/256 products, 2 ms less NPU latency):
#  1. LR:  R18q EMA factorized (ckpt/mac_lr/init_h128_v256.pt), QAT 4 epochs at lr 1e-4 on all replays.
#  2. L4:  ImageNet layer4 in place of the RepConv down (scripts/mimic.py hint-trained on R18q's neck, then factorized:
#          ckpt/mac_l4/init_lr.pt), QAT 4 epochs at lr 2e-4.
# Each run is S8-evaluated against R18q EMA (logs/eval_mac_lr.txt, logs/eval_mac_l4.txt), then both are built with
# Pulsar2 (outputs/quant_lr, outputs/quant_l4) for their cycle counts.
cd "$(dirname "$0")/.."
BASE='"backbone": "r18", "temporal_heads": ["lead", "lead_prob"], "lane_policy": true, "policy_tokens": 25, "hist_rank": 128, "vis_rank": 256'
run() {   # $1 run, $2 init, $3 lr, $4 arch
  (sleep 60; scripts/snap_epochs.sh ckpt/$1 4 eval-s8 > logs/snap_$1.log 2>&1) &
  caffeinate -i .venv/bin/python scripts/train.py --out ckpt/$1 --epochs 4 --epoch-samples 400000 --batch 64 --workers 4 --compile-refresh \
    --aug --wd 0.01 --flip 0.3 --curve-sampling --night-weight 2.5 --qat --gap-aug 0.2 --hist-noise 0.2 --ema 0.999 --lr $3 \
    --refresh-every 2 --init $2 --tags r2,cal1,cal2 --plan-lat-weight 3 --arch "$4" >> logs/$1.log 2>&1 && \
  .venv/bin/python scripts/eval_student.py --s8 --ckpt ckpt/mac_r18q/ema.pt ckpt/$1/last.pt ckpt/$1/ema.pt > logs/eval_$1.txt 2>&1
  echo DONE > logs/DONE_$1
}
run mac_lr ckpt/mac_lr/init_h128_v256.pt 1e-4 "{$BASE}"
while [ ! -f ckpt/mac_l4/init_lr.pt ]; do sleep 60; done
run mac_l4 ckpt/mac_l4/init_lr.pt 2e-4 "{$BASE, \"r18_layers\": 4}"
for r in lr l4; do
  .venv/bin/python scripts/quant_build.py --ckpt ckpt/mac_$r/ema.pt --name $r > logs/build_$r.out 2>&1
done
echo DONE > logs/DONE_chain_lr
