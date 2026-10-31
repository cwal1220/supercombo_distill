#!/bin/sh
# After chain_next3.sh (logs/DONE_chain_next3): the deployable builds and their board-equivalent evaluation.
#  1. Pulsar2 builds of L4 and LRB (their better checkpoint by S8 plan) with the range-covering calibration
#     (quant_build.py --extremes; the random calibration clipped desire_enc and the lane features on the NPU):
#     outputs/quant_l4_ext, outputs/quant_lrb_ext.
#  2. The random-calibration builds' calib/ and quant/dataset/ (1.6 GB each) move to the backup disk.
#  3. p2sim.py (= the board, to 1e-4) on all eval units: logs/p2sim_<name>_all.txt.
#  4. Board latency and outputs of both (scripts/board_queue.sh; SSHPASS from the environment, never a file).
cd "$(dirname "$0")/.."
U="2026-09-27--02-44-40-679__000 2026-09-27--02-44-40-679__015 2026-10-05--00-07-29-810__000 2026-10-05--16-28-44-410__000
  2026-10-05--16-28-44-410__015 2026-10-22--23-22-40-918__000 2026-10-25--22-42-26-175__000 2026-10-25--22-42-26-175__015"
OFF=/Volumes/Backup/supercombo_offload_20261031
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
while [ ! -f logs/DONE_chain_next3 ]; do sleep 60; done
l4=$(best logs/eval_mac_l4.txt mac_l4)
lrb=$(best logs/eval_mac_lrb.txt mac_lrb)
echo "$(date) l4 $l4, lrb $lrb" >> logs/chain_final.log
for b in lrbest l4 lrb; do
  if [ -d "outputs/quant_$b/calib" ]; then
    mkdir -p "$OFF/quant_$b/quant" && mv "outputs/quant_$b/calib" "$OFF/quant_$b/calib" && mv "outputs/quant_$b/quant/dataset" "$OFF/quant_$b/quant/dataset"
  fi
done
.venv/bin/python scripts/quant_build.py --ckpt "$lrb" --name lrb_ext --per-unit 10 --extremes > logs/build_lrb_ext.out 2>&1
.venv/bin/python scripts/quant_build.py --ckpt "$l4" --name l4_ext --per-unit 10 --extremes > logs/build_l4_ext.out 2>&1
echo "$(date) built lrb_ext, l4_ext" >> logs/chain_final.log
OMP_NUM_THREADS=4 .venv/bin/python scripts/p2sim.py --no-s8 --ckpt "$lrb" --build outputs/quant_lrb_ext --units $U > logs/p2sim_lrb_ext_all.txt 2>&1
OMP_NUM_THREADS=4 .venv/bin/python scripts/p2sim.py --no-s8 --ckpt "$l4" --build outputs/quant_l4_ext --units $U > logs/p2sim_l4_ext_all.txt 2>&1
echo "$(date) p2sim done" >> logs/chain_final.log
scripts/board_queue.sh 0 "lrb_ext=$lrb" "l4_ext=$l4"
echo DONE > logs/DONE_chain_final
