#!/bin/sh
# After the R18 run (logs/DONE_mac_r18): run B from run A's better (plan_y2s) checkpoint with the teacher-labelled
# replays (--tags r2,cal1: 2x playback and shaken calibration/camera offset, whatever is labelled by then), a 4.8 s
# policy history (policy_tokens 25, new history columns start at zero), the plan's lateral columns weighted 3x and the
# lead loss 1.0 (was 0.5); otherwise run A's recipe, 3 epochs QAT of 450k samples each.
cd "$(dirname "$0")/.."
while [ ! -f logs/DONE_mac_r18 ]; do sleep 60; done
best() {   # $1 run name: its better checkpoint by 5-route plan_y2s (S8)
.venv/bin/python - "$1" <<'PY'
import re, sys
r = sys.argv[1]; rows = {}
for l in open(f"logs/eval_{r}.txt"):
    m = re.match(rf"student {r}/?(\S*) \(S8\)\s+(\S+)\s+(\S+)\s+(\S+)", l)
    if m: rows[m.group(1) or "last"] = float(m.group(4))
print(f"ckpt/{r}/ema.pt" if rows.get("ema", 99) < rows.get("last", 99) else f"ckpt/{r}/last.pt")
PY
}
init=$(best mac_base_runA)
R=mac_base_runB
echo "run B init: $init" > logs/$R.log
(sleep 60; scripts/snap_epochs.sh ckpt/$R 3 eval-s8 > logs/snap_$R.log 2>&1) &
caffeinate -i .venv/bin/python scripts/train.py --out ckpt/$R --epochs 3 --epoch-samples 450000 --batch 64 --workers 4 --compile-refresh \
  --aug --wd 0.01 --flip 0.3 --curve-sampling --qat --gap-aug 0.2 --hist-noise 0.2 --ema 0.999 --lr 2e-4 --init "$init" \
  --tags r2,cal1 --plan-lat-weight 3 --lead-weight 1.0 \
  --arch '{"temporal_heads": ["lead", "lead_prob"], "lane_policy": true, "policy_tokens": 25}' >> logs/$R.log 2>&1 && \
.venv/bin/python scripts/eval_student.py --s8 --ckpt ckpt/mac_base_runA/ema.pt ckpt/$R/last.pt ckpt/$R/ema.pt > logs/eval_$R.txt 2>&1
echo DONE > logs/DONE_$R
