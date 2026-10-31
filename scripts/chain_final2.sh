#!/bin/sh
# Replaces chain_final.sh (2026-10-31 08:58). The NPU's extra false leads (LRA: 8.8 % in the old S8 eval, 12.0 % on
# the NPU) came from the weight grid, not the U16 activations: LRA was QAT'd on the old grid (/127, [-127, 127]), and on
# Pulsar2's grid its outputs move (unit 09-27__015: 3.34 % -> 5.41 % false leads, = the NPU). The activations add a
# lead_prob logit error of 0.002 (std). So: QAT on Pulsar2's grid (fixed student.fake_quant_s8) and --extremes builds.
#  1. chain_next3.sh's random-calibration build of LRB (it clips on the NPU) is stopped before docker starts.
#  2. LRC (GPU, from logs/DONE_mac_lrb on): LRA (LR arch, 11.2 ms on the board) QAT'd for 2 epochs on Pulsar2's grid
#     with LRB's options (plan-abs, lead-prob 1.0, lead-abs 0.5); S8 eval logs/eval_mac_lrc.txt.
#  3. LRB's better checkpoint: --extremes build outputs/quant_lrb_ext, p2sim on all eval units (= the board;
#     logs/p2sim_lrb_ext_all.txt, frame dumps outputs/diag_y2s/dump/lrb_npu, logs/lead_lrb_npu.txt), board run.
#  4. The same for LRC (outputs/quant_lrc_ext, logs/p2sim_lrc_ext_all.txt, lrc_npu, logs/lead_lrc_npu.txt).
# SSHPASS (the board) comes from the environment of whoever starts this script, never a file.
# usage: SSHPASS=<board password> nohup scripts/chain_final2.sh > /dev/null 2>&1 &
cd "$(dirname "$0")/.."
U="2026-09-27--02-44-40-679__000 2026-09-27--02-44-40-679__015 2026-10-05--00-07-29-810__000 2026-10-05--16-28-44-410__000
  2026-10-05--16-28-44-410__015 2026-10-22--23-22-40-918__000 2026-10-25--22-42-26-175__000 2026-10-25--22-42-26-175__015"
LR='{"backbone": "r18", "temporal_heads": ["lead", "lead_prob"], "lane_policy": true, "policy_tokens": 25, "hist_rank": 128, "vis_rank": 256}'
best() {   # eval file, run: the run's checkpoint (last or ema) with the lower S8 plan_y2s
.venv/bin/python - "$1" "$2" <<'PY'
import re, sys
rows = {}
for l in open(sys.argv[1]):
    m = re.match(r"student (mac_\w+)(/ema)? \(S8\)\s+(.*)", l)
    if m and m.group(1) == sys.argv[2]:
        rows[f"ckpt/{m.group(1)}/{'ema' if m.group(2) else 'last'}.pt"] = float(m.group(3).split()[2])
print(min(rows, key=rows.get))
PY
}
npu() {   # name, ckpt: --extremes build, p2sim on all eval units with frame dumps, lead quality, board run
  .venv/bin/python scripts/quant_build.py --ckpt "$2" --name "$1_ext" --per-unit 10 --extremes > "logs/build_$1_ext.out" 2>&1
  echo "$(date) built $1_ext from $2" >> logs/chain_final2.log
  OMP_NUM_THREADS=4 .venv/bin/python scripts/p2sim.py --no-s8 --ckpt "$2" --build "outputs/quant_$1_ext" --units $U \
    --dump "$1_npu" > "logs/p2sim_$1_ext_all.txt" 2>&1
  .venv/bin/python scripts/lead_quality.py "$1_npu" > "logs/lead_$1_npu.txt" 2>&1
  echo "$(date) p2sim $1_ext done" >> logs/chain_final2.log
  scripts/board_queue.sh 0 "$1_ext=$2"
}
while [ ! -f logs/DONE_mac_lrb ]; do sleep 60; done
echo "$(date) LRB trained; LRC starts" >> logs/chain_final2.log
(
  (sleep 60; scripts/snap_epochs.sh ckpt/mac_lrc 2 eval-s8 > logs/snap_mac_lrc.log 2>&1) &
  caffeinate -i .venv/bin/python scripts/train.py --out ckpt/mac_lrc --epochs 2 --epoch-samples 400000 --batch 64 --workers 4 --compile-refresh \
    --aug --wd 0.01 --flip 0.3 --curve-sampling --night-weight 2.5 --qat --gap-aug 0.2 --hist-noise 0.2 --ema 0.999 --lr 1e-4 \
    --refresh-every 2 --init ckpt/mac_lra/last.pt --tags r2,cal1,cal2 --plan-lat-weight 3 --arch "$LR" \
    --plan-abs-weight 0.5 --lead-prob-weight 1.0 --lead-abs-weight 0.5 >> logs/mac_lrc.log 2>&1 && \
  .venv/bin/python scripts/eval_student.py --s8 --ckpt ckpt/mac_lra/last.pt ckpt/mac_lrc/last.pt ckpt/mac_lrc/ema.pt > logs/eval_mac_lrc.txt 2>&1
  echo DONE > logs/DONE_mac_lrc
) &
while [ ! -f logs/DONE_chain_next3 ]; do   # chain_next3's last step is the random-calibration build: stop it early
  p=$(pgrep -f "quant_build.py --ckpt .* --name lrb$")
  [ -n "$p" ] && kill $p && echo "$(date) stopped chain_next3's random-calibration build (pid $p)" >> logs/chain_final2.log
  sleep 10
done
lrb=$(best logs/eval_mac_lrb.txt mac_lrb)
echo "$(date) LRB best $lrb" >> logs/chain_final2.log
npu lrb "$lrb"
while [ ! -f logs/DONE_mac_lrc ]; do sleep 60; done
lrc=$(best logs/eval_mac_lrc.txt mac_lrc)
echo "$(date) LRC best $lrc" >> logs/chain_final2.log
npu lrc "$lrc"
echo DONE > logs/DONE_chain_final2
