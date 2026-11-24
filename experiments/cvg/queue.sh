#!/bin/bash
# CVG: run a queue of diagnostic processes one after another through launch.sh (one tag-cvg unit at a time).
#   bash experiments/cvg/queue.sh <queue file>
# Each non-empty, non-# line of the queue file: <id> <script.py> [args...].  Before each item: wait until no tag-cvg
# unit is active and MemAvailable >= 6 GiB.  An item whose launch directory already exists is skipped (never rerun);
# a failed item is logged and the queue goes on with the next (independent) item.  Log: $RES/queue.log.
set -uo pipefail
REPO=/home/rmcfz/code/python_projects/seq_vp_tag
RES=/home/rmcfz/seq_vp_tag_results/cvg_v1
Q=$1
log() { echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) $*" >> "$RES/queue.log"; }
log "queue start $Q"
while read -r line; do
  [ -z "$line" ] && continue
  case "$line" in \#*) continue;; esac
  id=${line%% *}
  if [ -e "$RES/launch/$id" ]; then log "skip $id (exists)"; continue; fi
  while systemctl --user list-units --all --no-legend 'tag-cvg-*' | grep -q 'active\|activating' \
        || [ "$(awk '/MemAvailable/ {print int($2/1024)}' /proc/meminfo)" -lt 6144 ]; do sleep 15; done
  log "start $id"
  # shellcheck disable=SC2086
  bash "$REPO/experiments/cvg/launch.sh" $line < /dev/null
  log "end $id exit $?"
done < "$Q"
log "queue end $Q"
