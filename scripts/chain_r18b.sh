#!/bin/sh
# After the R18 run (logs/DONE_mac_r18): continue the R18 student (it beat the shipped model on lead_x at epoch 2 and
# was still improving fast) from its better checkpoint by plan_y2s: 4 more fp32 epochs of 400k samples at lr 5e-4 with
# the plan's lateral columns weighted 3x (plan_y2s and lat_acc are what it still loses), then QAT 2 epochs at lr 2e-4,
# then the S8 evaluation against run A.
cd "$(dirname "$0")/.."
while [ ! -f logs/DONE_mac_r18 ]; do sleep 60; done
best() {   # $1 run, $2 eval file, $3 " (S8)" or "": the better checkpoint by 5-route plan_y2s
.venv/bin/python - "$1" "$2" "$3" <<'PY'
import re, sys
r, f, tag = sys.argv[1], sys.argv[2], re.escape(sys.argv[3]); rows = {}
for l in open(f):
    m = re.match(rf"student {r}/?(\S*){tag}\s+(\S+)\s+(\S+)\s+(\S+)", l)
    if m: rows[m.group(1) or "last"] = float(m.group(4))
print(f"ckpt/{r}/ema.pt" if rows.get("ema", 99) < rows.get("last", 99) else f"ckpt/{r}/last.pt")
PY
}
ARCH='{"backbone": "r18", "temporal_heads": ["lead", "lead_prob"], "lane_policy": true, "policy_tokens": 25}'
init=$(best mac_r18 logs/eval_mac_r18.txt "")
R=mac_r18b
echo "r18b init: $init" > logs/$R.log
(sleep 60; scripts/snap_epochs.sh ckpt/$R 4 eval > logs/snap_$R.log 2>&1) &
caffeinate -i .venv/bin/python scripts/train.py --out ckpt/$R --epochs 4 --epoch-samples 400000 --batch 64 --workers 4 --compile-refresh \
  --aug --wd 0.01 --flip 0.3 --curve-sampling --gap-aug 0.2 --hist-noise 0.2 --ema 0.999 --lr 5e-4 --refresh-every 2 --init "$init" \
  --tags r2,cal1 --plan-lat-weight 3 --arch "$ARCH" >> logs/$R.log 2>&1 && \
.venv/bin/python scripts/eval_student.py --ckpt ckpt/$R/last.pt ckpt/$R/ema.pt > logs/eval_$R.txt 2>&1
echo DONE > logs/DONE_$R
init=$(best mac_r18b logs/eval_mac_r18b.txt "")
R=mac_r18q
echo "r18q init: $init" > logs/$R.log
(sleep 60; scripts/snap_epochs.sh ckpt/$R 2 eval-s8 > logs/snap_$R.log 2>&1) &
caffeinate -i .venv/bin/python scripts/train.py --out ckpt/$R --epochs 2 --epoch-samples 400000 --batch 64 --workers 4 --compile-refresh \
  --aug --wd 0.01 --flip 0.3 --curve-sampling --qat --gap-aug 0.2 --hist-noise 0.2 --ema 0.999 --lr 2e-4 --refresh-every 2 --init "$init" \
  --tags r2,cal1 --plan-lat-weight 3 --arch "$ARCH" >> logs/$R.log 2>&1 && \
.venv/bin/python scripts/eval_student.py --s8 --ckpt ckpt/mac_base_runA/ema.pt ckpt/$R/last.pt ckpt/$R/ema.pt > logs/eval_$R.txt 2>&1
echo DONE > logs/DONE_$R
