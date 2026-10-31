#!/bin/sh
# LRF (2026-10-31 14:55): lanes on hills. LD's lane targets were projected onto a flat road at the camera height, so a
# rising road's lanes widen with distance in them (x 1.35 at 30 m with 0.6 m of rise) and a falling road's narrow; the
# students copied that (LRC's lanes vs the teacher: 9 cm on the flat, 47 cm with > 0.6 m of rise; the shipped model
# 8-12 cm). The official lanes metric (vs flat LD) rewards the copy; against LD on the teacher's road surface
# (lane_geom.slope_fix) LRC loses to the shipped model. LRF: the better of LRC / LRD (S8 plan) QAT-fine-tuned 2 epochs
# with train.py --ld-slope-fix and LRD's options, S8 eval with the slope-fixed lane metrics (logs/eval_mac_lrf.txt),
# lanes by elevation (logs/diag_lane_grade_lrf.txt), then --extremes build, p2sim and the board, each when free.
# SSHPASS (the board) comes from the environment of whoever starts this script, never a file.
# usage: SSHPASS=<board password> nohup scripts/chain_lrf.sh > /dev/null 2>&1 &
cd "$(dirname "$0")/.."
U="2026-09-27--02-44-40-679__000 2026-09-27--02-44-40-679__015 2026-10-05--00-07-29-810__000 2026-10-05--16-28-44-410__000
  2026-10-05--16-28-44-410__015 2026-10-22--23-22-40-918__000 2026-10-25--22-42-26-175__000 2026-10-25--22-42-26-175__015"
LR='{"backbone": "r18", "temporal_heads": ["lead", "lead_prob"], "lane_policy": true, "policy_tokens": 25, "hist_rank": 128, "vis_rank": 256}'
while [ ! -f logs/DONE_mac_lrd ]; do sleep 30; done
init=$(.venv/bin/python - <<'PY'
import re
rows = {}
for f, run in (("logs/eval_mac_lrc.txt", "mac_lrc"), ("logs/eval_mac_lrd.txt", "mac_lrd")):
    for l in open(f):
        m = re.match(r"student (mac_lr[cd])(/ema)? \(S8\)\s+(.*)", l)
        if m and m.group(1) == run:
            rows[f"ckpt/{m.group(1)}/{'ema' if m.group(2) else 'last'}.pt"] = float(m.group(3).split()[2])
print(min(rows, key=rows.get))
PY
)
echo "$(date) LRF from $init" >> logs/chain_lrf.log
(sleep 60; scripts/snap_epochs.sh ckpt/mac_lrf 2 eval-s8 > logs/snap_mac_lrf.log 2>&1) &
caffeinate -i .venv/bin/python scripts/train.py --out ckpt/mac_lrf --epochs 2 --epoch-samples 400000 --batch 64 --workers 4 --compile-refresh \
  --aug --wd 0.01 --flip 0.3 --curve-sampling --night-weight 2.5 --qat --gap-aug 0.2 --hist-noise 0.2 --ema 0.999 --lr 1e-4 \
  --refresh-every 2 --init "$init" --tags r2,cal1,cal2 --plan-lat-weight 3 --arch "$LR" \
  --plan-abs-weight 0.5 --lead-prob-weight 1.0 --lead-abs-weight 0.5 --ld-slope-fix >> logs/mac_lrf.log 2>&1 && \
.venv/bin/python scripts/eval_student.py --s8 --ckpt "$init" ckpt/mac_lrf/last.pt ckpt/mac_lrf/ema.pt > logs/eval_mac_lrf.txt 2>&1
echo DONE > logs/DONE_mac_lrf
lrf=$(.venv/bin/python - <<'PY'
import re
rows = {}
for l in open("logs/eval_mac_lrf.txt"):
    m = re.match(r"student mac_lrf(/ema)? \(S8\)\s+(.*)", l)
    if m:
        v = m.group(2).split()
        rows[f"ckpt/mac_lrf/{'ema' if m.group(1) else 'last'}.pt"] = float(v[2]) + float(v[10])   # plan + slope-fixed lanes
print(min(rows, key=rows.get))
PY
)
echo "$(date) LRF best $lrf" >> logs/chain_lrf.log
.venv/bin/python scripts/diag_lane_grade.py --ckpt "$lrf" --name lrf $U > logs/diag_lane_grade_lrf.txt 2>&1
while pgrep -f "quant_build.py" > /dev/null; do sleep 30; done
.venv/bin/python scripts/quant_build.py --ckpt "$lrf" --name lrf_ext --per-unit 10 --extremes > logs/build_lrf_ext.out 2>&1
echo "$(date) built lrf_ext" >> logs/chain_lrf.log
OMP_NUM_THREADS=4 .venv/bin/python scripts/p2sim.py --no-s8 --ckpt "$lrf" --build outputs/quant_lrf_ext --units $U \
  --dump lrf_npu > logs/p2sim_lrf_ext_all.txt 2>&1
.venv/bin/python scripts/lead_quality.py lrf_npu > logs/lead_lrf_npu.txt 2>&1
echo "$(date) p2sim lrf_ext done" >> logs/chain_lrf.log
while [ ! -f logs/DONE_chain_data ] || pgrep -f "board_eval.py" > /dev/null; do sleep 60; done
scripts/board_queue.sh 0 "lrf_ext=$lrf"
echo DONE > logs/DONE_chain_lrf
