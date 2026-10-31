#!/bin/sh
# After R18q (logs/DONE_mac_r18q) and its build (logs/DONE_final). Candidates by 5-route S8 metrics: R18q last/EMA and
# R18b EMA (PTQ, logs/eval_r18b_ema_s8.txt). If one beats the shipped model on all six metrics (lat, plan, lanes, width,
# lead, gas2), build it (unless chain_final already did) and measure it on the board. Otherwise continue R18b EMA in
# fp32 (R18's PTQ loss is small: plan +0.04): 4 epochs at lr 3e-4, night units 3x, all replays incl. the night cal2,
# then S8-evaluate, build the better checkpoint and measure on the board. SSHPASS comes from the environment.
cd "$(dirname "$0")/.."
while [ ! -f logs/DONE_mac_r18q ] || [ ! -f logs/DONE_final ]; do sleep 60; done
pick() {   # files...: "WIN|LOSE <ckpt>" for the candidate with the lowest plan_y2s among the S8 rows
.venv/bin/python - "$@" <<'PY'
import re, sys
rows, ship = {}, None
for f in sys.argv[1:]:
    for l in open(f):
        v = l.split()
        if l.startswith("board NPU (shipped)"):
            ship = [float(x) for x in v[3:10]]
        m = re.match(r"student (mac_r18[a-z]*)(/\S+)? \(S8\)\s+(.*)", l)
        if m:
            sub = (m.group(2) or "/last").lstrip("/")
            if not sub.startswith("ep"):
                rows[f"ckpt/{m.group(1)}/{sub}.pt"] = [float(x) for x in m.group(3).split()[:7]]
best = min(rows, key=lambda k: rows[k][2])
win = all(rows[best][i] < ship[i] for i in (0, 2, 3, 4, 5, 6))
print(("WIN " if win else "LOSE ") + best)
PY
}
build() {   # $1 checkpoint, $2 name
  .venv/bin/python scripts/quant_build.py --ckpt "$1" --name "$2" >> logs/final_$2.log 2>&1
  .venv/bin/python scripts/board_eval.py --ckpt "$1" --build "outputs/quant_$2" --name "$2" \
    --units 2026-10-25--22-42-26-175__015 > "logs/board_eval_$2.txt" 2>&1
}
verdict=$(pick logs/eval_mac_r18q.txt logs/eval_r18b_ema_s8.txt)
echo "$(date) candidates: $verdict" > logs/mac_r18c.log
case "$verdict" in
  WIN*) ck=${verdict#WIN }
        grep -q "final checkpoint: $ck" logs/final.log || build "$ck" r18win
        echo DONE > logs/DONE_mac_r18c; exit 0;;
esac
ARCH='{"backbone": "r18", "temporal_heads": ["lead", "lead_prob"], "lane_policy": true, "policy_tokens": 25}'
R=mac_r18c
(sleep 60; scripts/snap_epochs.sh ckpt/$R 4 eval-s8 > logs/snap_$R.log 2>&1) &
caffeinate -i .venv/bin/python scripts/train.py --out ckpt/$R --epochs 4 --epoch-samples 400000 --batch 64 --workers 4 --compile-refresh \
  --aug --wd 0.01 --flip 0.3 --curve-sampling --night-weight 3 --gap-aug 0.2 --hist-noise 0.2 --ema 0.999 --lr 3e-4 \
  --refresh-every 2 --init ckpt/mac_r18b/ema.pt --tags r2,cal1,cal2 --plan-lat-weight 3 --arch "$ARCH" >> logs/$R.log 2>&1 && \
.venv/bin/python scripts/eval_student.py --s8 --ckpt ckpt/$R/last.pt ckpt/$R/ema.pt > logs/eval_$R.txt 2>&1
verdict=$(pick logs/eval_mac_r18c.txt logs/eval_mac_r18q.txt logs/eval_r18b_ema_s8.txt)
echo "$(date) after R18c: $verdict" >> logs/$R.log
build "${verdict#* }" r18c
echo DONE > logs/DONE_$R
