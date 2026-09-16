#!/usr/bin/env bash
# Wait for chunk 1 (collected on C:) to be pushed, then move the rest of the run to D:.
set -u
cd "/c/Users/vieth/Documents/lablab hackathon"
LOG=out/volume_v2.log
until grep -q "convert/push chunk 1 rc=" "$LOG"; do sleep 15; done
rc=$(grep "convert/push chunk 1 rc=" "$LOG" | tail -1 | sed 's/.*rc=//')
echo "$(date +%T) swap: chunk 1 push rc=$rc" | tee -a "$LOG"
if [ "$rc" != "0" ]; then echo "$(date +%T) swap: chunk 1 push failed, not swapping" | tee -a "$LOG"; exit 1; fi
# stop the C: driver and any chunk-2 collection it may have started
powershell -NoProfile -Command "Get-CimInstance Win32_Process | Where-Object { (\$_.Name -eq 'bash.exe' -and \$_.CommandLine -like '*run_volume.sh*') -or (\$_.Name -eq 'python.exe' -and \$_.CommandLine -like '*collect.py*') } | ForEach-Object { Stop-Process -Id \$_.ProcessId -Force -ErrorAction SilentlyContinue; 'stopped ' + \$_.ProcessId + ' ' + \$_.Name }" | tee -a "$LOG"
sleep 3
rm -rf out/episodes_v2_chunk2
mkdir -p /d/lerobot /d/episodes
if [ -d out/lerobot/VietHwang__dinner-table-v2 ]; then
  mv out/lerobot/VietHwang__dinner-table-v2 /d/lerobot/ && echo "$(date +%T) swap: dataset root moved to D:" | tee -a "$LOG"
fi
echo "$(date +%T) swap: starting D: driver at chunk 2" | tee -a "$LOG"
exec bash sim/run_volume_D.sh 2
