#!/usr/bin/env bash
# Docker foundation check (L0): build the image, run input/ twice offline with the scorer's limits,
# report per-task timing and whether both runs produced byte-identical JSON and PNG.
#   tools/docker_check.sh [image-tag] [input-dir]
set -euo pipefail
tag=${1:-chameleon-docker-check}
inp=$(realpath "${2:-input}")
out=$(mktemp -d)
docker build -q -t "$tag" . >/dev/null
for k in 1 2; do
  mkdir -p "$out/run$k"
  s=$(date +%s.%N)
  docker run --rm --network none --cpus 2 --memory 3g -v "$inp":/app/input:ro -v "$out/run$k":/app/output "$tag" \
    2>&1 | grep -v Fontconfig | sed "s/^/run$k: /"
  echo "run$k wall $(echo "$(date +%s.%N) - $s" | bc | cut -c1-6)s"
done
(cd "$out/run1" && sha256sum * | sort) > "$out/h1"
(cd "$out/run2" && sha256sum * | sort) > "$out/h2"
n=$(wc -l < "$out/h1")
if diff -q "$out/h1" "$out/h2" >/dev/null; then echo "deterministic: $n files identical across runs"; else
  echo "NOT deterministic:"; diff "$out/h1" "$out/h2" || true; fi
echo "outputs in $out"
