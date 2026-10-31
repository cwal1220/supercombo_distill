#!/bin/bash
# Mac training-throughput sweep: latest PyTorch (py3.14 .venv) with batch / dtype / channels_last / torch.compile,
# and MLX, for base and w48d. One line per run in logs/bench_sweep.txt. (The 2026-10-27 run also had PyTorch 2.8
# from a py3.9 venv as baseline; that venv is gone.)
# usage (Mac, ~/Documents/supercombo_distill): scripts/bench_mac_sweep.sh
cd "$(dirname "$0")/.."
W='{"widths":[48,96,192,384,768],"depths":[1,3,6,9,3]}'
out=logs/bench_sweep.txt
run() { echo "== $1 | ${*:2}" >> $out; timeout 900 "${@:2}" 2>&1 | grep -E "samples/s|Error|error" | tail -2 >> $out; }
echo "# sweep $(date)" >> $out
for arch in base w48d; do
  if [ $arch = base ]; then A='{}'; b1=64; b2=128; else A=$W; b1=32; b2=64; fi
  run "$arch torch-new" .venv/bin/python scripts/bench_train.py --device mps --arch "$A" --batch $b1 --steps 15
  run "$arch torch-new" .venv/bin/python scripts/bench_train.py --device mps --arch "$A" --batch $b2 --steps 15
  run "$arch torch-new" .venv/bin/python scripts/bench_train.py --device mps --arch "$A" --batch $b2 --dtype fp16 --steps 15
  run "$arch torch-new" .venv/bin/python scripts/bench_train.py --device mps --arch "$A" --batch $b2 --channels-last --steps 15
  run "$arch torch-new" .venv/bin/python scripts/bench_train.py --device mps --arch "$A" --batch $b2 --compile --steps 15
  run "$arch mlx" .venv/bin/python scripts/bench_mlx.py --arch "$A" --batch $b1 --steps 15
  run "$arch mlx" .venv/bin/python scripts/bench_mlx.py --arch "$A" --batch $b2 --steps 15
  run "$arch mlx" .venv/bin/python scripts/bench_mlx.py --arch "$A" --batch $b2 --dtype fp16 --steps 15
  run "$arch mlx" .venv/bin/python scripts/bench_mlx.py --arch "$A" --batch $b1 --dtype fp32 --steps 15
  run "$arch mlx" .venv/bin/python scripts/bench_mlx.py --arch "$A" --batch $b2 --no-compile --steps 15
done
echo "# done $(date)" >> $out
