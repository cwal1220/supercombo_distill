#!/bin/sh
# After gap3b: wait for its train.py to exit (it holds the frames.npy maps), compress the training units' frames
# (zframes.py, 4 at a time; eval units stay frames.npy), wait for gap3b's final eval, then run A from the better of
# gap3b last.pt / ema.pt (plan_y2s): + the newly labelled routes, temporal lead heads, the lanes' and road edges'
# features into the policy (lane_policy), gap-aug v2, history noise 0.2, EMA; 4 epochs QAT. Replay units not yet.
cd "$(dirname "$0")/.."
while pgrep -f "train.py --out ckpt/mac_base_gap3b" > /dev/null; do sleep 30; done
ls -d data/ep_label/*/ | while read u; do
  case "$u" in *2026-10-25--22-42-26-175*|*2026-10-05--00-07-29-810*|*2026-10-22--23-22-40-918*|*2026-10-05--16-28-44-410*|*2026-09-27--02-44-40-679*) continue;; esac
  [ -f "$u/frames.npy" ] && [ -f "$u/labels.npz" ] && echo "$u"
done | xargs -P 4 -n 4 .venv/bin/python scripts/zframes.py > logs/zframes_convert.log 2>&1
df -h . | tail -1 >> logs/zframes_convert.log
while [ ! -f logs/DONE_mac_base_gap3b ]; do sleep 30; done
init=$(.venv/bin/python - <<'PY'
import re
rows = {}
for l in open("logs/eval_mac_base_gap3b.txt"):
    m = re.match(r"student mac_base_gap3b/?(\S*) \(S8\)\s+(\S+)\s+(\S+)\s+(\S+)", l)
    if m: rows[m.group(1) or "last"] = float(m.group(4))
print("ckpt/mac_base_gap3b/ema.pt" if rows.get("ema", 99) < rows.get("last", 99) else "ckpt/mac_base_gap3b/last.pt")
PY
)
echo "run A init: $init" > logs/mac_base_runA.log
R=mac_base_runA
(sleep 60; scripts/snap_epochs.sh ckpt/$R 4 eval-s8 > logs/snap_$R.log 2>&1) &
caffeinate -i .venv/bin/python scripts/train.py --out ckpt/$R --epochs 4 --batch 64 --workers 4 --compile-refresh --aug --wd 0.01 \
  --flip 0.3 --curve-sampling --qat --gap-aug 0.2 --hist-noise 0.2 --ema 0.999 --lr 2e-4 --init "$init" \
  --arch '{"temporal_heads": ["lead", "lead_prob"], "lane_policy": true}' >> logs/$R.log 2>&1 && \
.venv/bin/python scripts/eval_student.py --s8 --ckpt ckpt/mac_base_qat3/last.pt "$init" ckpt/$R/last.pt ckpt/$R/ema.pt > logs/eval_$R.txt 2>&1
echo DONE > logs/DONE_$R
