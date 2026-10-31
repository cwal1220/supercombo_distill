#!/bin/sh
# LRF + the logistic far-lead calibration (2026-10-31 19:30). The grid-fitted calibration (chain_lrf_cal.sh, lrf_calE)
# reaches 4.98 / 9.75 % false / missed leads on the eval routes but only 4.52 / 11.06 % cross-validated; fitted by
# logistic regression with a temperature (lead_cal.py --logistic: logit' = 0.791 s - 0.567 relu(x - 80) / 20 - 0.105)
# it is 4.97 / 9.94 % and 4.81 / 10.13 % cross-validated, both better than the shipped model's 5.06 / 10.59
# (logs/lead_cal_lrfL.txt), with stable per-fold values. ckpt/mac_lrf/ema_calL.pt (already written, = the formula to
# 2e-6) → --extremes build, p2sim on all eval units, then the board after chain_lrf_cal.sh's board run.
# SSHPASS (the board) comes from the environment of whoever starts this script, never a file.
# usage: SSHPASS=<board password> nohup scripts/chain_lrf_calL.sh > /dev/null 2>&1 &
cd "$(dirname "$0")/.."
U="2026-09-27--02-44-40-679__000 2026-09-27--02-44-40-679__015 2026-10-05--00-07-29-810__000 2026-10-05--16-28-44-410__000
  2026-10-05--16-28-44-410__015 2026-10-22--23-22-40-918__000 2026-10-25--22-42-26-175__000 2026-10-25--22-42-26-175__015"
out=ckpt/mac_lrf/ema_calL.pt
while pgrep -f "quant_build.py" > /dev/null; do sleep 30; done
.venv/bin/python scripts/quant_build.py --ckpt "$out" --name lrf_calL_ext --per-unit 10 --extremes > logs/build_lrf_calL_ext.out 2>&1
echo "$(date) built lrf_calL_ext" >> logs/chain_lrf_calL.log
OMP_NUM_THREADS=4 .venv/bin/python scripts/p2sim.py --no-s8 --ckpt "$out" --build outputs/quant_lrf_calL_ext --units $U \
  --dump lrf_calL_npu > logs/p2sim_lrf_calL_ext_all.txt 2>&1
.venv/bin/python scripts/lead_quality.py lrf_calL_npu > logs/lead_lrf_calL_npu.txt 2>&1
echo "$(date) p2sim lrf_calL_ext done" >> logs/chain_lrf_calL.log
while [ ! -f logs/DONE_chain_lrf_cal ] || pgrep -f "board_eval.py" > /dev/null; do sleep 60; done
scripts/board_queue.sh 0 "lrf_calL_ext=$out"
echo DONE > logs/DONE_chain_lrf_calL
