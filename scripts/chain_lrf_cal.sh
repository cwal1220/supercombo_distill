#!/bin/sh
# LRF + far-lead calibration (2026-10-31 17:35): LRF fixes the lanes on hills (slope-fixed lanes 10.6 / width 14.3 cm
# vs shipped 14.9 / 16.5, S8) but keeps LRC's extra far false leads (8.08 %, shipped 5.06 %). LRC's calibration
# fitted on the eval routes (chain_final4.sh) took them to 5.03 / 9.77 % false / missed on the NPU (cross-validated
# estimate 4.81 / 10.15). The same procedure on LRF: fitted on chain_lrf.sh's NPU dump (lrf_npu) with the
# leave-one-route-out estimate (logs/lead_cal_lrfE.txt), ckpt/mac_lrf/ema_calE.pt, --extremes build, p2sim on all
# eval units, then the board once chain_lrf.sh's board run is done.
# SSHPASS (the board) comes from the environment of whoever starts this script, never a file.
# usage: SSHPASS=<board password> nohup scripts/chain_lrf_cal.sh > /dev/null 2>&1 &
cd "$(dirname "$0")/.."
U="2026-09-27--02-44-40-679__000 2026-09-27--02-44-40-679__015 2026-10-05--00-07-29-810__000 2026-10-05--16-28-44-410__000
  2026-10-05--16-28-44-410__015 2026-10-22--23-22-40-918__000 2026-10-25--22-42-26-175__000 2026-10-25--22-42-26-175__015"
until grep -q "p2sim lrf_ext done" logs/chain_lrf.log 2>/dev/null; do sleep 60; done
lrf=$(sed -n 's/.*LRF best //p' logs/chain_lrf.log | tail -1)
out=$(echo "$lrf" | sed 's/\.pt$/_calE.pt/')
.venv/bin/python scripts/lead_cal.py lrf_npu lrf_npu --loro --apply "$lrf" outputs/quant_lrf_ext "$out" > logs/lead_cal_lrfE.txt 2>&1
echo "$(date) LRF calibrated: $out" >> logs/chain_lrf_cal.log
while pgrep -f "quant_build.py" > /dev/null; do sleep 30; done
.venv/bin/python scripts/quant_build.py --ckpt "$out" --name lrf_calE_ext --per-unit 10 --extremes > logs/build_lrf_calE_ext.out 2>&1
echo "$(date) built lrf_calE_ext" >> logs/chain_lrf_cal.log
OMP_NUM_THREADS=4 .venv/bin/python scripts/p2sim.py --no-s8 --ckpt "$out" --build outputs/quant_lrf_calE_ext --units $U \
  --dump lrf_calE_npu > logs/p2sim_lrf_calE_ext_all.txt 2>&1
.venv/bin/python scripts/lead_quality.py lrf_calE_npu > logs/lead_lrf_calE_npu.txt 2>&1
echo "$(date) p2sim lrf_calE_ext done" >> logs/chain_lrf_cal.log
while [ ! -f logs/DONE_chain_lrf ] || pgrep -f "board_eval.py" > /dev/null; do sleep 60; done
scripts/board_queue.sh 0 "lrf_calE_ext=$out"
echo DONE > logs/DONE_chain_lrf_cal
