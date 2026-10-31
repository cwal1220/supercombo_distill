#!/bin/sh
# After run A (logs/DONE_mac_base_runA): the ImageNet ResNet-18 backbone (backbone "r18", 11.8 ms on the board) under
# run A's better (plan_y2s) heads and policy (--init-partial: backbone and neck fresh), with the replays labelled by then (r2, part of cal1) and
# run A's recipe, fp32 (QAT afterwards), 3 epochs of 400k samples at lr 1e-3, bank after epoch 2 (~6 h).
cd "$(dirname "$0")/.."
while [ ! -f logs/DONE_mac_base_runA ]; do sleep 60; done
init=$(.venv/bin/python - <<'PY'
import re
rows = {}
for l in open("logs/eval_mac_base_runA.txt"):
    m = re.match(r"student mac_base_runA/?(\S*) \(S8\)\s+(\S+)\s+(\S+)\s+(\S+)", l)
    if m: rows[m.group(1) or "last"] = float(m.group(4))
print("ckpt/mac_base_runA/ema.pt" if rows.get("ema", 99) < rows.get("last", 99) else "ckpt/mac_base_runA/last.pt")
PY
)
R=mac_r18
echo "r18 init (heads, policy): $init" > logs/$R.log
(sleep 60; scripts/snap_epochs.sh ckpt/$R 3 eval > logs/snap_$R.log 2>&1) &
caffeinate -i .venv/bin/python scripts/train.py --out ckpt/$R --epochs 3 --epoch-samples 400000 --batch 64 --workers 4 --compile-refresh --aug --wd 0.01 \
  --flip 0.3 --curve-sampling --gap-aug 0.2 --hist-noise 0.2 --ema 0.999 --lr 1e-3 --refresh-every 2 --init "$init" --init-partial \
  --tags r2,cal1 --arch '{"backbone": "r18", "temporal_heads": ["lead", "lead_prob"], "lane_policy": true, "policy_tokens": 25}' >> logs/$R.log 2>&1 && \
.venv/bin/python scripts/eval_student.py --ckpt ckpt/$R/last.pt ckpt/$R/ema.pt > logs/eval_$R.txt 2>&1
echo DONE > logs/DONE_$R
