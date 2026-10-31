#!/bin/sh
# Far-lead calibration builds (2026-10-31 14:20). The remaining gap to the shipped model is false leads beyond 60 m;
# scripts/lead_cal.py moves the lead_prob logit by b0 - b1 * relu(x - x0) / 20 inside the model (a dead lead_prob
# feature, same outputs), with (x0, b0, b1) fitted on the new drives that neither training nor the eval used.
#  1. LRC: fit on its S8 outputs on the new drives (dump lrc_s8_new, PID 17883), apply -> ckpt/mac_lrc/ema_cal.pt.
#  2. After chain_data.sh's LRD build (one Pulsar2 build at a time): --extremes build outputs/quant_lrc_cal_ext,
#     p2sim on all eval units (logs/p2sim_lrc_cal_ext_all.txt, dump lrc_cal_npu, logs/lead_lrc_cal_npu.txt).
#  3. LRD trained on the new drives, so it takes LRC's values (its starting point): <lrd>_cal.pt, the same build and
#     p2sim as lrd_cal once chain_data.sh's LRD p2sim is done.
#  4. Board runs of both once chain_data.sh is done.
# SSHPASS (the board) comes from the environment of whoever starts this script, never a file.
# usage: SSHPASS=<board password> nohup scripts/chain_final3.sh > /dev/null 2>&1 &
cd "$(dirname "$0")/.."
U="2026-09-27--02-44-40-679__000 2026-09-27--02-44-40-679__015 2026-10-05--00-07-29-810__000 2026-10-05--16-28-44-410__000
  2026-10-05--16-28-44-410__015 2026-10-22--23-22-40-918__000 2026-10-25--22-42-26-175__000 2026-10-25--22-42-26-175__015"
cal() {   # name, ckpt: --extremes build, p2sim on all eval units with frame dumps, lead quality
  .venv/bin/python scripts/quant_build.py --ckpt "$2" --name "$1_ext" --per-unit 10 --extremes > "logs/build_$1_ext.out" 2>&1
  echo "$(date) built $1_ext from $2" >> logs/chain_final3.log
  OMP_NUM_THREADS=4 .venv/bin/python scripts/p2sim.py --no-s8 --ckpt "$2" --build "outputs/quant_$1_ext" --units $U \
    --dump "$1_npu" > "logs/p2sim_$1_ext_all.txt" 2>&1
  .venv/bin/python scripts/lead_quality.py "$1_npu" > "logs/lead_$1_npu.txt" 2>&1
  echo "$(date) p2sim $1_ext done" >> logs/chain_final3.log
}
while kill -0 17883 2>/dev/null; do sleep 30; done
.venv/bin/python scripts/lead_cal.py lrc_s8_new lrc_npu --apply ckpt/mac_lrc/ema.pt outputs/quant_lrc_ext ckpt/mac_lrc/ema_cal.pt \
  > logs/lead_cal_lrc.txt 2>&1
set -- $(sed -n 's/^fitted on .*: far-lead x0 \([0-9.]*\) b0 \([-+0-9.]*\) b1 \([0-9.]*\);.*/\1 \2 \3/p' logs/lead_cal_lrc.txt)
x0=$1 b0=$2 b1=$3
echo "$(date) LRC far-lead calibration x0 $x0 b0 $b0 b1 $b1" >> logs/chain_final3.log
until grep -q "LRD best" logs/chain_data.log; do sleep 60; done
sleep 30
while pgrep -f "quant_build.py --ckpt .* --name lrd_ext" > /dev/null; do sleep 30; done
cal lrc_cal ckpt/mac_lrc/ema_cal.pt
until grep -q "p2sim lrd_ext done" logs/chain_data.log; do sleep 60; done
lrd=$(sed -n 's/.*LRD best \(.*\)/\1/p' logs/chain_data.log)
.venv/bin/python scripts/lead_cal.py lrc_s8_new lrd_npu --params "$x0" "$b0" "$b1" --apply "$lrd" outputs/quant_lrd_ext "${lrd%.pt}_cal.pt" \
  > logs/lead_cal_lrd.txt 2>&1
echo "$(date) LRD calibrated: ${lrd%.pt}_cal.pt" >> logs/chain_final3.log
cal lrd_cal "${lrd%.pt}_cal.pt"
while [ ! -f logs/DONE_chain_data ]; do sleep 60; done
scripts/board_queue.sh 0 "lrc_cal_ext=ckpt/mac_lrc/ema_cal.pt" "lrd_cal_ext=${lrd%.pt}_cal.pt"
echo DONE > logs/DONE_chain_final3
