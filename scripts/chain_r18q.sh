#!/bin/sh
# Replaces the tail of chain_r18b.sh: wait for the R18b training (PID $1) to exit, evaluate it (fp32), then QAT the
# best of R18 EMA and R18b last/EMA (fp32 plan_y2s) for 2 epochs with night units drawn 2.5x (the shipped model still wins on the night eval
# route) and the night-route replays (cal2) as well; S8 evaluation against run A.
cd "$(dirname "$0")/.."
while kill -0 "$1" 2>/dev/null; do sleep 30; done
.venv/bin/python scripts/eval_student.py --ckpt ckpt/mac_r18/ema.pt ckpt/mac_r18b/last.pt ckpt/mac_r18b/ema.pt > logs/eval_mac_r18b.txt 2>&1
echo DONE > logs/DONE_mac_r18b
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
init=$(.venv/bin/python - <<'PY'       # the best by fp32 5-route plan_y2s of R18 (ep3 EMA) and R18b (last, EMA)
import re
rows = {}
for l in open("logs/eval_mac_r18b.txt"):
    m = re.match(r"student (mac_r18b?)(/ema)?\s+(\S+)\s+(\S+)\s+(\S+)", l)
    if m: rows[f"ckpt/{m.group(1)}/{'ema' if m.group(2) else 'last'}.pt"] = float(m.group(5))
print(min(rows, key=rows.get))
PY
)
R=mac_r18q
echo "r18q init: $init" > logs/$R.log
(sleep 60; scripts/snap_epochs.sh ckpt/$R 2 eval-s8 > logs/snap_$R.log 2>&1) &
caffeinate -i .venv/bin/python scripts/train.py --out ckpt/$R --epochs 2 --epoch-samples 400000 --batch 64 --workers 4 --compile-refresh \
  --aug --wd 0.01 --flip 0.3 --curve-sampling --night-weight 2.5 --qat --gap-aug 0.2 --hist-noise 0.2 --ema 0.999 --lr 2e-4 \
  --refresh-every 2 --init "$init" --tags r2,cal1,cal2 --plan-lat-weight 3 --arch "$ARCH" >> logs/$R.log 2>&1 && \
.venv/bin/python scripts/eval_student.py --s8 --ckpt ckpt/mac_base_runA/ema.pt ckpt/mac_r18/ema.pt ckpt/$R/last.pt ckpt/$R/ema.pt > logs/eval_$R.txt 2>&1
echo DONE > logs/DONE_$R
