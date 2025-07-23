#!/bin/bash

WATCH_DIR="${1:-.}"
SLEEP_TIME=2  # seconds between checks

declare -A mtimes

# Initialize mtimes for all .tex files
for f in $(find "$WATCH_DIR" -type f -name "*.tex"); do
  mtimes["$f"]=$(stat -c %Y "$f")
done

echo "Watching $WATCH_DIR for changes in .tex files (polling every $SLEEP_TIME seconds)..."

while true; do
  changed=0
  for f in $(find "$WATCH_DIR" -type f -name "*.tex"); do
    new_mtime=$(stat -c %Y "$f")
    old_mtime=${mtimes["$f"]}
    if [[ "$new_mtime" != "$old_mtime" ]]; then
      echo "Change detected in $f"
      mtimes["$f"]=$new_mtime
      changed=1
    fi
  done

  if [[ $changed -eq 1 ]]; then
    echo "Running make"
	make > /dev/null
    echo "Done."
  fi

  sleep $SLEEP_TIME
done
