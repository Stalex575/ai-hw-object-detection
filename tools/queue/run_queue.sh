#!/usr/bin/env bash
# Sequential GPU queue. Runs $Q/jobs/*.sh in name order, one at a time; a failed job does not stop the queue.
# Per job: $Q/state/<job>.{running,done,failed,log}; START/END lines in $Q/queue.log.
# While $Q/HOLD exists, polls every 60 s for new jobs instead of exiting.
# Launch (the venv must be on PATH: the Jupyter kernelspec runs `python`):
#   PATH=$PWD/.venv/bin:$PATH setsid nohup tools/queue/run_queue.sh >> outputs/queue/nohup.out 2>&1 < /dev/null & disown
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
Q="${QUEUE_DIR:-$REPO/outputs/queue}"   # git-ignored
cd "$REPO"
mkdir -p "$Q/jobs" "$Q/state"
echo "$(date -Is) queue start pid $$" >> "$Q/queue.log"
while true; do
  next=""
  for j in "$Q"/jobs/*.sh; do
    [ -e "$j" ] || continue
    n=$(basename "$j" .sh)
    [ -e "$Q/state/$n.done" ] || [ -e "$Q/state/$n.failed" ] || [ -e "$Q/state/$n.running" ] || { next=$j; break; }
  done
  if [ -z "$next" ]; then
    [ -e "$Q/HOLD" ] && { sleep 60; continue; }
    echo "$(date -Is) queue empty, exiting" >> "$Q/queue.log"; exit 0
  fi
  n=$(basename "$next" .sh)
  touch "$Q/state/$n.running"
  echo "$(date -Is) START $n" >> "$Q/queue.log"
  bash "$next" > "$Q/state/$n.log" 2>&1; rc=$?
  rm -f "$Q/state/$n.running"
  if [ $rc -eq 0 ]; then touch "$Q/state/$n.done"; else echo $rc > "$Q/state/$n.failed"; fi
  echo "$(date -Is) END $n rc=$rc" >> "$Q/queue.log"
done
