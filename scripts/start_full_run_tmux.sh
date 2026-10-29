#!/usr/bin/env bash
# Author: Coda
# Start the complete five-dataset run in a detached tmux session.
set -euo pipefail

SESSION_NAME="${MMCYBER_TMUX_SESSION:-mmcyber-full}"
REPOSITORY_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIRECTORY="$REPOSITORY_ROOT/runs/logs"
LOG_FILE="$LOG_DIRECTORY/full-run-$(date +%Y%m%d-%H%M%S).log"

if ! command -v tmux >/dev/null 2>&1; then
  echo "tmux is required. Install it (for example: sudo apt-get install tmux)." >&2
  exit 1
fi
if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
  echo "tmux session '$SESSION_NAME' already exists; attach with: tmux attach -t $SESSION_NAME" >&2
  exit 1
fi

mkdir -p "$LOG_DIRECTORY"
tmux new-session -d -s "$SESSION_NAME" -c "$REPOSITORY_ROOT" \
  "bash scripts/run_all_full_datasets.sh > '$LOG_FILE' 2>&1"

echo "Started full run in tmux session '$SESSION_NAME'."
echo "Attach: tmux attach -t $SESSION_NAME"
echo "Live log: tail -F '$LOG_FILE'"
