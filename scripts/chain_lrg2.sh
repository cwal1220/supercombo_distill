#!/bin/sh
# The rest of LRG (2026-10-31 19:30): chain_lrg.sh was stopped during training (the training process kept running)
# so that LRG gets the logistic far-lead calibration (lead_cal.py --logistic, see chain_lrf_calL.sh) instead of the
# grid fit. Waits for the training process, then: S8 eval (logs/eval_mac_lrg.txt), --extremes build, p2sim with the
# dump (lrg_npu, logs/lead_lrg_npu.txt), the logistic calibration with its leave-one-route-out estimate
# (logs/lead_cal_lrgL.txt), its build and p2sim, and the board after chain_lrf_calL.sh's board run.
# SSHPASS (the board) comes from the environment of whoever starts this script, never a file.
# usage: SSHPASS=<board password> nohup scripts/chain_lrg2.sh <training pid> > /dev/null 2>&1 &
cd "$(dirname "$0")/.."
U="2026-09-27--02-44-40-679__000 2026-09-27--02-44-40-679__015 2026-10-05--00-07-29-810__000 2026-10-05--16-28-44-410__000
  2026-10-05--16-28-44-410__015 2026-10-22--23-22-40-918__000 2026-10-25--22-42-26-175__000 2026-10-25--22-42-26-175__015"
while kill -0 "$1" 2>/dev/null; do sleep 30; done
grep -q "epoch 1 done" logs/mac_lrg.log || { echo "$(date) LRG training did not finish" >> logs/chain_lrg.log; exit 1; }
.venv/bin/python scripts/eval_student.py --s8 --ckpt ckpt/mac_lrf/ema.pt ckpt/mac_lrg/last.pt ckpt/mac_lrg/ema.pt > logs/eval_mac_lrg.txt 2>&1
echo DONE > logs/DONE_mac_lrg
lrg=$(.venv/bin/python - <<'PY'
import re
rows = {}
for l in open("logs/eval_mac_lrg.txt"):
    m = re.match(r"student mac_lrg(/ema)? \(S8\)\s+(.*)", l)
    if m:
        v = m.group(2).split()
        rows[f"ckpt/mac_lrg/{'ema' if m.group(1) else 'last'}.pt"] = float(v[2]) + float(v[10])   # plan + slope-fixed lanes
print(min(rows, key=rows.get))
PY
)
echo "$(date) LRG best $lrg" >> logs/chain_lrg.log
while pgrep -f "quant_build.py" > /dev/null; do sleep 30; done
.venv/bin/python scripts/quant_build.py --ckpt "$lrg" --name lrg_ext --per-unit 10 --extremes > logs/build_lrg_ext.out 2>&1
echo "$(date) built lrg_ext" >> logs/chain_lrg.log
OMP_NUM_THREADS=4 .venv/bin/python scripts/p2sim.py --no-s8 --ckpt "$lrg" --build outputs/quant_lrg_ext --units $U \
  --dump lrg_npu > logs/p2sim_lrg_ext_all.txt 2>&1
.venv/bin/python scripts/lead_quality.py lrg_npu > logs/lead_lrg_npu.txt 2>&1
echo "$(date) p2sim lrg_ext done" >> logs/chain_lrg.log
out=$(echo "$lrg" | sed 's/\.pt$/_calL.pt/')
.venv/bin/python scripts/lead_cal.py lrg_npu lrg_npu --loro --logistic --apply "$lrg" outputs/quant_lrg_ext "$out" > logs/lead_cal_lrgL.txt 2>&1
echo "$(date) LRG calibrated: $out" >> logs/chain_lrg.log
while pgrep -f "quant_build.py" > /dev/null; do sleep 30; done
.venv/bin/python scripts/quant_build.py --ckpt "$out" --name lrg_calL_ext --per-unit 10 --extremes > logs/build_lrg_calL_ext.out 2>&1
echo "$(date) built lrg_calL_ext" >> logs/chain_lrg.log
OMP_NUM_THREADS=4 .venv/bin/python scripts/p2sim.py --no-s8 --ckpt "$out" --build outputs/quant_lrg_calL_ext --units $U \
  --dump lrg_calL_npu > logs/p2sim_lrg_calL_ext_all.txt 2>&1
.venv/bin/python scripts/lead_quality.py lrg_calL_npu > logs/lead_lrg_calL_npu.txt 2>&1
echo "$(date) p2sim lrg_calL_ext done" >> logs/chain_lrg.log
while [ ! -f logs/DONE_chain_lrf_calL ] || pgrep -f "board_eval.py" > /dev/null; do sleep 60; done
scripts/board_queue.sh 0 "lrg_calL_ext=$out"
echo DONE > logs/DONE_chain_lrg
