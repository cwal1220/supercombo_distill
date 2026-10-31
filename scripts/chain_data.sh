#!/bin/sh
# More data (2026-10-31 11:00). The board had two drives no run has seen: 2026-10-30--19-05-02-089 (night, ~60 min) and
# 2026-10-31--05-17-39-046 (dawn, ~40 min), copied to the USB. The student's remaining weakness is false leads on
# unseen night roads (it beats the shipped model on the training drives), so new night driving is the most direct fix.
#  1. Label both with the teacher on the Neural Engine (ep_label.py --no-small --zstd; ~45 ms a frame).
#  2. LRD: the better of LRB / LRC (lower S8 plan_y2s; its arch) QAT-fine-tuned 2 epochs with the new units in the pool,
#     LRB's options (plan-abs 0.5, lead-prob 1.0, lead-abs 0.5, Pulsar2's weight grid); S8 eval logs/eval_mac_lrd.txt.
#  3. --extremes build outputs/quant_lrd_ext, p2sim on all eval units (logs/p2sim_lrd_ext_all.txt, dump lrd_npu,
#     logs/lead_lrd_npu.txt), board run once chain_final2.sh is done (one board job at a time).
# SSHPASS (the board) comes from the environment of whoever starts this script, never a file.
# usage: SSHPASS=<board password> nohup scripts/chain_data.sh > /dev/null 2>&1 &
cd "$(dirname "$0")/.."
RAW=/Volumes/Untitled/edgepilot_drives/experiments
U="2026-09-27--02-44-40-679__000 2026-09-27--02-44-40-679__015 2026-10-05--00-07-29-810__000 2026-10-05--16-28-44-410__000
  2026-10-05--16-28-44-410__015 2026-10-22--23-22-40-918__000 2026-10-25--22-42-26-175__000 2026-10-25--22-42-26-175__015"
LR='{"backbone": "r18", "temporal_heads": ["lead", "lead_prob"], "lane_policy": true, "policy_tokens": 25, "hist_rank": 128, "vis_rank": 256}'
L4='{"backbone": "r18", "temporal_heads": ["lead", "lead_prob"], "lane_policy": true, "policy_tokens": 25, "hist_rank": 128, "vis_rank": 256, "r18_layers": 4}'
best() {   # eval files...: "<ckpt> <plan_y2s>" of the student row (mac_lrb / mac_lrc / mac_lrd) with the lowest plan_y2s
.venv/bin/python - "$@" <<'PY'
import re, sys
rows = {}
for f in sys.argv[1:]:
    for l in open(f):
        m = re.match(r"student (mac_lr[bcd])(/ema)? \(S8\)\s+(.*)", l)
        if m:
            rows[f"ckpt/{m.group(1)}/{'ema' if m.group(2) else 'last'}.pt"] = float(m.group(3).split()[2])
k = min(rows, key=rows.get)
print(k, rows[k])
PY
}
mkdir -p logs/label
ORT_COREML=CPUAndNeuralEngine ORT_THREADS=4 .venv/bin/python scripts/ep_label.py "$RAW/2026-10-30--19-05-02-089" data/ep_label \
  --also "$RAW/2026-10-31--05-17-39-046" --no-small --zstd > logs/label/batch_mac_new1031.log 2>&1
echo "$(date) labelled the new drives (exit $?): $(ls -d data/ep_label/2026-10-30--19-05* data/ep_label/2026-10-31--05-17* | tr '\n' ' ')" >> logs/chain_data.log
while [ ! -f logs/DONE_mac_lrc ]; do sleep 60; done
set -- $(best logs/eval_mac_lrb.txt logs/eval_mac_lrc.txt)
init=$1 arch=$LR
case "$init" in ckpt/mac_lrb/*) arch=$L4 ;; esac
echo "$(date) LRD from $init (plan $2)" >> logs/chain_data.log
(sleep 60; scripts/snap_epochs.sh ckpt/mac_lrd 2 eval-s8 > logs/snap_mac_lrd.log 2>&1) &
caffeinate -i .venv/bin/python scripts/train.py --out ckpt/mac_lrd --epochs 2 --epoch-samples 400000 --batch 64 --workers 4 --compile-refresh \
  --aug --wd 0.01 --flip 0.3 --curve-sampling --night-weight 2.5 --qat --gap-aug 0.2 --hist-noise 0.2 --ema 0.999 --lr 1e-4 \
  --refresh-every 2 --init "$init" --tags r2,cal1,cal2 --plan-lat-weight 3 --arch "$arch" \
  --plan-abs-weight 0.5 --lead-prob-weight 1.0 --lead-abs-weight 0.5 >> logs/mac_lrd.log 2>&1 && \
.venv/bin/python scripts/eval_student.py --s8 --ckpt "$init" ckpt/mac_lrd/last.pt ckpt/mac_lrd/ema.pt > logs/eval_mac_lrd.txt 2>&1
echo DONE > logs/DONE_mac_lrd
grep -v "^student mac_lr[bc]" logs/eval_mac_lrd.txt > logs/eval_mac_lrd_only.txt   # the eval file also has the init's row
set -- $(best logs/eval_mac_lrd_only.txt)
lrd=$1
echo "$(date) LRD best $lrd" >> logs/chain_data.log
.venv/bin/python scripts/quant_build.py --ckpt "$lrd" --name lrd_ext --per-unit 10 --extremes > logs/build_lrd_ext.out 2>&1
OMP_NUM_THREADS=4 .venv/bin/python scripts/p2sim.py --no-s8 --ckpt "$lrd" --build outputs/quant_lrd_ext --units $U \
  --dump lrd_npu > logs/p2sim_lrd_ext_all.txt 2>&1
.venv/bin/python scripts/lead_quality.py lrd_npu > logs/lead_lrd_npu.txt 2>&1
echo "$(date) p2sim lrd_ext done" >> logs/chain_data.log
while [ ! -f logs/DONE_chain_final2 ]; do sleep 60; done
scripts/board_queue.sh 0 "lrd_ext=$lrd"
echo DONE > logs/DONE_chain_data
