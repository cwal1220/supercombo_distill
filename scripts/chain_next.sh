#!/bin/sh
# Takes over from chain_lr.sh after its LR training (wait for PID $1):
#  1. S8-evaluate LR (logs/eval_mac_lr.txt).
#  2. LRA: the better LR checkpoint, QAT 2 more epochs with --plan-abs-weight 0.5 (plan y up to 4 s in cm, so sharp turns
#     count as much as the metric counts them; the std-scaled plan loss underweights them), lr 1e-4 (logs/eval_mac_lra.txt).
#  3. L4: layer4 + low-rank (ckpt/mac_l4/init_lr.pt), QAT 4 epochs at lr 2e-4, with the plan-abs loss if LRA beat LR.
#  4. Pulsar2 builds of the best LR-family checkpoint (outputs/quant_lrbest) and of L4 EMA (outputs/quant_l4).
cd "$(dirname "$0")/.."
LR='{"backbone": "r18", "temporal_heads": ["lead", "lead_prob"], "lane_policy": true, "policy_tokens": 25, "hist_rank": 128, "vis_rank": 256}'
L4='{"backbone": "r18", "temporal_heads": ["lead", "lead_prob"], "lane_policy": true, "policy_tokens": 25, "hist_rank": 128, "vis_rank": 256, "r18_layers": 4}'
run() {   # $1 run, $2 init, $3 lr, $4 epochs, $5 arch, $6... extra train.py args
  r=$1 init=$2 lr=$3 ep=$4 arch=$5
  shift 5
  (sleep 60; scripts/snap_epochs.sh ckpt/$r "$ep" eval-s8 > logs/snap_$r.log 2>&1) &
  caffeinate -i .venv/bin/python scripts/train.py --out ckpt/$r --epochs "$ep" --epoch-samples 400000 --batch 64 --workers 4 --compile-refresh \
    --aug --wd 0.01 --flip 0.3 --curve-sampling --night-weight 2.5 --qat --gap-aug 0.2 --hist-noise 0.2 --ema 0.999 --lr "$lr" \
    --refresh-every 2 --init "$init" --tags r2,cal1,cal2 --plan-lat-weight 3 --arch "$arch" "$@" >> logs/$r.log 2>&1 && \
  .venv/bin/python scripts/eval_student.py --s8 --ckpt ckpt/mac_r18q/ema.pt ckpt/$r/last.pt ckpt/$r/ema.pt > logs/eval_$r.txt 2>&1
  echo DONE > logs/DONE_$r
}
best() {   # eval files...: "<ckpt> <plan_y2s>" of the student row with the lowest plan_y2s, R18q excluded
.venv/bin/python - "$@" <<'PY'
import re, sys
rows = {}
for f in sys.argv[1:]:
    for l in open(f):
        m = re.match(r"student (mac_\w+)(/ema)? \(S8\)\s+(.*)", l)
        if m and m.group(1) != "mac_r18q":
            rows[f"ckpt/{m.group(1)}/{'ema' if m.group(2) else 'last'}.pt"] = float(m.group(3).split()[2])
k = min(rows, key=rows.get)
print(k, rows[k])
PY
}
while kill -0 "$1" 2>/dev/null; do sleep 60; done
.venv/bin/python scripts/eval_student.py --s8 --ckpt ckpt/mac_r18q/ema.pt ckpt/mac_lr/last.pt ckpt/mac_lr/ema.pt > logs/eval_mac_lr.txt 2>&1
echo DONE > logs/DONE_mac_lr
set -- $(best logs/eval_mac_lr.txt)
lr_ck=$1 lr_plan=$2
echo "$(date) LR best $lr_ck plan $lr_plan" >> logs/chain_next.log
run mac_lra "$lr_ck" 1e-4 2 "$LR" --plan-abs-weight 0.5
set -- $(best logs/eval_mac_lra.txt)
abs=""
if .venv/bin/python -c "import sys; sys.exit(0 if $2 < $lr_plan else 1)"; then abs="--plan-abs-weight 0.5"; fi
echo "$(date) LRA best $1 plan $2; L4 extra args: '$abs'" >> logs/chain_next.log
run mac_l4 ckpt/mac_l4/init_lr.pt 2e-4 4 "$L4" $abs
set -- $(best logs/eval_mac_lr.txt logs/eval_mac_lra.txt)
.venv/bin/python scripts/quant_build.py --ckpt "$1" --name lrbest > logs/build_lrbest.out 2>&1
echo "$(date) built lrbest from $1" >> logs/chain_next.log
.venv/bin/python scripts/quant_build.py --ckpt ckpt/mac_l4/ema.pt --name l4 > logs/build_l4.out 2>&1
echo DONE > logs/DONE_chain_next
