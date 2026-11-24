#!/usr/bin/env bash
# Fail if any tracked or staged file is larger than the limit (default 2 MB).
# Gzipped SUMO networks (*.net.xml.gz) are exempt.
# Usage: scripts/check_large_files.sh [limit_bytes]
set -euo pipefail
LIMIT="${1:-2097152}"
cd "$(git rev-parse --show-toplevel)"
fail=0
total=0
count=0
while IFS= read -r -d '' f; do
  [ -f "$f" ] || continue
  size=$(wc -c < "$f" | tr -d ' ')
  total=$((total + size))
  count=$((count + 1))
  if [ "$size" -gt "$LIMIT" ] && [[ "$f" != *.net.xml.gz ]]; then
    echo "TOO LARGE: $f ($size bytes > $LIMIT)"
    fail=1
  fi
done < <(git ls-files -z --cached)
echo "checked $count files, total $total bytes"
exit "$fail"
