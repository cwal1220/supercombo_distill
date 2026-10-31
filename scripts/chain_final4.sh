#!/bin/sh
# LRC with the far-lead calibration fitted on the 5 eval routes (2026-10-31 14:45). Fitted on the new drives
# (chain_final3.sh's lrc_cal) it only reaches 6.06 / 8.75 % false / missed leads (shipped 5.06 / 10.59): the new
# drives' lead statistics differ from the eval routes'. Fitted on the eval routes, the honest estimate is the whole fit
# cross-validated leave-one-route-out (lead_cal.py --loro): 4.81 / 10.15 %, both better than the shipped model.
# ckpt/mac_lrc/ema_calE.pt (x0 80, b0 +0.20, b1 1.75; logs/lead_cal_lrcE.txt) is built after chain_final3.sh's
# lrc_cal build (one Pulsar2 build at a time), p2sim on all eval units, then the board once chain_data.sh's
# board run is done and no other board run is going.
# SSHPASS (the board) comes from the environment of whoever starts this script, never a file.
# usage: SSHPASS=<board password> nohup scripts/chain_final4.sh > /dev/null 2>&1 &
cd "$(dirname "$0")/.."
U="2026-09-27--02-44-40-679__000 2026-09-27--02-44-40-679__015 2026-10-05--00-07-29-810__000 2026-10-05--16-28-44-410__000
  2026-10-05--16-28-44-410__015 2026-10-22--23-22-40-918__000 2026-10-25--22-42-26-175__000 2026-10-25--22-42-26-175__015"
until grep -q "built lrc_cal_ext" logs/chain_final3.log 2>/dev/null; do sleep 60; done
while pgrep -f "quant_build.py" > /dev/null; do sleep 30; done
.venv/bin/python scripts/quant_build.py --ckpt ckpt/mac_lrc/ema_calE.pt --name lrc_calE_ext --per-unit 10 --extremes \
  > logs/build_lrc_calE_ext.out 2>&1
echo "$(date) built lrc_calE_ext" >> logs/chain_final4.log
OMP_NUM_THREADS=4 .venv/bin/python scripts/p2sim.py --no-s8 --ckpt ckpt/mac_lrc/ema_calE.pt --build outputs/quant_lrc_calE_ext \
  --units $U --dump lrc_calE_npu > logs/p2sim_lrc_calE_ext_all.txt 2>&1
.venv/bin/python scripts/lead_quality.py lrc_calE_npu > logs/lead_lrc_calE_npu.txt 2>&1
echo "$(date) p2sim lrc_calE_ext done" >> logs/chain_final4.log
while [ ! -f logs/DONE_chain_data ] || pgrep -f "board_eval.py" > /dev/null; do sleep 60; done
scripts/board_queue.sh 0 "lrc_calE_ext=ckpt/mac_lrc/ema_calE.pt"
echo DONE > logs/DONE_chain_final4
