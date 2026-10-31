#!/bin/bash
# One line a minute: GPU temperature / power / utilisation and the hottest thermal zone (millidegrees).
# usage: setsid nohup scripts/thermal_log.sh >> logs/thermal.log 2>&1 < /dev/null &
while true; do
  g=$(nvidia-smi --query-gpu=temperature.gpu,power.draw,utilization.gpu --format=csv,noheader 2>/dev/null)
  c=$(cat /sys/class/thermal/thermal_zone*/temp 2>/dev/null | sort -n | tail -1)
  echo "$(date '+%m-%d %H:%M:%S') gpu $g cpu_max $c"
  sleep 60
done
