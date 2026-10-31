#!/bin/sh
# Measure Pulsar2 builds on the MaixCAM2 once it is reachable, after another board job (PID $1, e.g. board_when_up.sh)
# has finished, so that only one run holds the NPU. Each further argument is <name>=<ckpt>: outputs/quant_<name> is
# measured with scripts/board_eval.py (which stops edgepilot.service and starts it again) into logs/board_eval_<name>.txt;
# builds that do not exist yet are skipped. Checks every minute for up to a day.
# SSHPASS comes from the environment of whoever starts this script (never a file).
# usage: SSHPASS=<board password> nohup scripts/board_queue.sh <pid|0> lra=ckpt/mac_lra/last.pt ... &
cd "$(dirname "$0")/.."
wait_pid=$1
shift
while [ "$wait_pid" != 0 ] && kill -0 "$wait_pid" 2>/dev/null; do sleep 60; done
i=0
until ping -c 1 -t 3 192.168.219.107 > /dev/null 2>&1; do sleep 60; i=$((i + 1)); [ $i -gt 1440 ] && exit 1; done
sleep 120                                   # let it finish booting
for job in "$@"; do
  name=${job%%=*} ck=${job#*=}
  [ -f "outputs/quant_$name/m.axmodel" ] || { echo "$(date) skip $name (no build)" >> logs/board_queue.log; continue; }
  .venv/bin/python scripts/board_eval.py --ckpt "$ck" --build "outputs/quant_$name" --name "$name" \
    --units 2026-10-25--22-42-26-175__015 > "logs/board_eval_$name.txt" 2>&1
  echo "$(date) measured $name (exit $?)" >> logs/board_queue.log
done
echo DONE > logs/DONE_board_queue
