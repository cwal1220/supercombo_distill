#!/bin/sh
# Measure the final model on the MaixCAM2 once it is reachable again (checks every minute for up to a day).
# SSHPASS comes from the environment of whoever starts this script (never a file).
cd "$(dirname "$0")/.."
i=0
until ping -c 1 -t 3 192.168.219.107 > /dev/null 2>&1; do sleep 60; i=$((i + 1)); [ $i -gt 1440 ] && exit 1; done
sleep 120                                   # let it finish booting
.venv/bin/python scripts/board_eval.py --ckpt ckpt/mac_r18q/ema.pt --build outputs/quant_r18final --name r18final \
  --units 2026-10-25--22-42-26-175__015 > logs/board_eval_r18final.txt 2>&1
echo DONE > logs/DONE_board_r18final
