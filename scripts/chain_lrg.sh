#!/bin/sh
# LRG (2026-10-31 19:05): LRF's lanes, better lead detection. LRF on the NPU still has more far false leads than the
# shipped model (8.09 vs 5.06 %, AUC 0.9740 vs 0.9772); the far-lead calibration fitted on the eval routes reaches
# 4.98 / 9.75 % but only 4.52 / 11.06 % cross-validated (leave-one-route-out), a tie. A calibration only moves along
# the ROC; LRG tries to move the ROC: LRF EMA QAT-fine-tuned 2 epochs with LRF's options, lead_prob BCE weighted by
# |p - teacher p|^2 (train.py --lead-prob-focal 2, Quality Focal Loss, Li et al. 2020; the frames the student gets
# wrong count most) and lead-prob weight 2. Then S8 eval (logs/eval_mac_lrg.txt), --extremes build, p2sim with the
# dump (lrg_npu, logs/lead_lrg_npu.txt), the far-lead calibration fitted and cross-validated as for LRF
# (logs/lead_cal_lrgE.txt), its build and p2sim, and the board, each when free.
# SSHPASS (the board) comes from the environment of whoever starts this script, never a file.
# usage: SSHPASS=<board password> nohup scripts/chain_lrg.sh > /dev/null 2>&1 &
cd "$(dirname "$0")/.."
U="2026-09-27--02-44-40-679__000 2026-09-27--02-44-40-679__015 2026-10-05--00-07-29-810__000 2026-10-05--16-28-44-410__000
  2026-10-05--16-28-44-410__015 2026-10-22--23-22-40-918__000 2026-10-25--22-42-26-175__000 2026-10-25--22-42-26-175__015"
LR='{"backbone": "r18", "temporal_heads": ["lead", "lead_prob"], "lane_policy": true, "policy_tokens": 25, "hist_rank": 128, "vis_rank": 256}'
init=ckpt/mac_lrf/ema.pt
echo "$(date) LRG from $init" >> logs/chain_lrg.log
(sleep 60; scripts/snap_epochs.sh ckpt/mac_lrg 2 eval-s8 > logs/snap_mac_lrg.log 2>&1) &
caffeinate -i .venv/bin/python scripts/train.py --out ckpt/mac_lrg --epochs 2 --epoch-samples 400000 --batch 64 --workers 4 --compile-refresh \
  --aug --wd 0.01 --flip 0.3 --curve-sampling --night-weight 2.5 --qat --gap-aug 0.2 --hist-noise 0.2 --ema 0.999 --lr 1e-4 \
  --refresh-every 2 --init "$init" --tags r2,cal1,cal2 --plan-lat-weight 3 --arch "$LR" \
  --plan-abs-weight 0.5 --lead-prob-weight 2.0 --lead-prob-focal 2 --lead-abs-weight 0.5 --ld-slope-fix >> logs/mac_lrg.log 2>&1 && \
.venv/bin/python scripts/eval_student.py --s8 --ckpt "$init" ckpt/mac_lrg/last.pt ckpt/mac_lrg/ema.pt > logs/eval_mac_lrg.txt 2>&1
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
out=$(echo "$lrg" | sed 's/\.pt$/_calE.pt/')
.venv/bin/python scripts/lead_cal.py lrg_npu lrg_npu --loro --apply "$lrg" outputs/quant_lrg_ext "$out" > logs/lead_cal_lrgE.txt 2>&1
echo "$(date) LRG calibrated: $out" >> logs/chain_lrg.log
while pgrep -f "quant_build.py" > /dev/null; do sleep 30; done
.venv/bin/python scripts/quant_build.py --ckpt "$out" --name lrg_calE_ext --per-unit 10 --extremes > logs/build_lrg_calE_ext.out 2>&1
echo "$(date) built lrg_calE_ext" >> logs/chain_lrg.log
OMP_NUM_THREADS=4 .venv/bin/python scripts/p2sim.py --no-s8 --ckpt "$out" --build outputs/quant_lrg_calE_ext --units $U \
  --dump lrg_calE_npu > logs/p2sim_lrg_calE_ext_all.txt 2>&1
.venv/bin/python scripts/lead_quality.py lrg_calE_npu > logs/lead_lrg_calE_npu.txt 2>&1
echo "$(date) p2sim lrg_calE_ext done" >> logs/chain_lrg.log
while [ ! -f logs/DONE_chain_lrf_cal ] || pgrep -f "board_eval.py" > /dev/null; do sleep 60; done
scripts/board_queue.sh 0 "lrg_calE_ext=$out"
echo DONE > logs/DONE_chain_lrg
