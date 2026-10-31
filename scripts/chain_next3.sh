#!/bin/sh
# After chain_next2.sh (logs/DONE_chain_next2): LRB, against the far false leads. The student sees cars the teacher puts
# at 100-125 m at 70-90 m and then calls them leads; past 60 m the teacher's x std is 17-25 m, so the std-scaled lead
# loss barely sees those errors. LRB adds --lead-abs-weight 0.5 (current lead x, y in 3 m / 0.3 m units, every frame)
# to LRA's options and fine-tunes the best of LRA and L4 (lowest S8 plan_y2s) with QAT for 2 epochs at lr 1e-4.
# Then: S8 eval (logs/eval_mac_lrb.txt), frame dumps + lead quality (logs/lead_lrb.txt), Pulsar2 build (outputs/quant_lrb).
cd "$(dirname "$0")/.."
LR='{"backbone": "r18", "temporal_heads": ["lead", "lead_prob"], "lane_policy": true, "policy_tokens": 25, "hist_rank": 128, "vis_rank": 256}'
L4='{"backbone": "r18", "temporal_heads": ["lead", "lead_prob"], "lane_policy": true, "policy_tokens": 25, "hist_rank": 128, "vis_rank": 256, "r18_layers": 4}'
EVAL_UNITS="data/ep_label/2026-09-27--02-44-40-679__000 data/ep_label/2026-09-27--02-44-40-679__015
  data/ep_label/2026-10-05--00-07-29-810__000 data/ep_label/2026-10-05--16-28-44-410__000
  data/ep_label/2026-10-05--16-28-44-410__015 data/ep_label/2026-10-22--23-22-40-918__000
  data/ep_label/2026-10-25--22-42-26-175__000 data/ep_label/2026-10-25--22-42-26-175__015"
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
while [ ! -f logs/DONE_chain_next2 ]; do sleep 60; done
set -- $(best logs/eval_mac_lra.txt logs/eval_mac_l4.txt)
init=$1 arch=$LR
case "$init" in ckpt/mac_l4/*) arch=$L4 ;; esac
echo "$(date) LRB from $init (plan $2)" >> logs/chain_next3.log
(sleep 60; scripts/snap_epochs.sh ckpt/mac_lrb 2 eval-s8 > logs/snap_mac_lrb.log 2>&1) &
caffeinate -i .venv/bin/python scripts/train.py --out ckpt/mac_lrb --epochs 2 --epoch-samples 400000 --batch 64 --workers 4 --compile-refresh \
  --aug --wd 0.01 --flip 0.3 --curve-sampling --night-weight 2.5 --qat --gap-aug 0.2 --hist-noise 0.2 --ema 0.999 --lr 1e-4 \
  --refresh-every 2 --init "$init" --tags r2,cal1,cal2 --plan-lat-weight 3 --arch "$arch" \
  --plan-abs-weight 0.5 --lead-prob-weight 1.0 --lead-abs-weight 0.5 >> logs/mac_lrb.log 2>&1 && \
.venv/bin/python scripts/eval_student.py --s8 --ckpt ckpt/mac_r18q/ema.pt ckpt/mac_lrb/last.pt ckpt/mac_lrb/ema.pt > logs/eval_mac_lrb.txt 2>&1
echo DONE > logs/DONE_mac_lrb
set -- $(best logs/eval_mac_lrb.txt)
echo "$(date) LRB best $1 plan $2" >> logs/chain_next3.log
OMP_NUM_THREADS=4 .venv/bin/python scripts/diag_y2s_dump.py "$1" outputs/diag_y2s/dump/lrb_s8 $EVAL_UNITS --s8 > logs/dump_lrb.log 2>&1
.venv/bin/python scripts/lead_quality.py lrb_s8 lra_ema_s8 > logs/lead_lrb.txt 2>&1
.venv/bin/python scripts/quant_build.py --ckpt "$1" --name lrb > logs/build_lrb.out 2>&1
echo "$(date) built lrb from $1" >> logs/chain_next3.log
echo DONE > logs/DONE_chain_next3
