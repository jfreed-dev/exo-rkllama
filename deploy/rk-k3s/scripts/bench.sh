#!/usr/bin/env bash
# Bench for the exo RK DaemonSet (issue #6): tok/s plus proof the NPU (not the CPU)
# did the work, via /sys/kernel/debug/rknpu/load sampled mid-generation.
#
#   scripts/bench.sh [model-id]
#
# Env: NS (default exo-rk), API_PORT (52415), PROMPT, MAX_SAMPLES (12).
# Run scripts/smoke.sh first; this assumes the instance is already placed.
set -euo pipefail

NS="${NS:-exo-rk}"
MODEL="${1:-qwen2.5-7b-rkllm}"
API_PORT="${API_PORT:-52415}"
PROMPT="${PROMPT:-Write a 200-word story about a cluster of small computers.}"
MAX_SAMPLES="${MAX_SAMPLES:-12}"

POD=$(kubectl -n "${NS}" get pods -l app.kubernetes.io/name=exo \
  -o jsonpath='{.items[0].metadata.name}')
NODE_IP=$(kubectl -n "${NS}" get pods -l app.kubernetes.io/name=exo \
  -o jsonpath='{.items[0].status.hostIP}')
API="http://${NODE_IP}:${API_PORT}"

printf '>> rknpu driver: '
kubectl -n "${NS}" exec "${POD}" -- cat /sys/kernel/debug/rknpu/version || echo "unavailable"

sample_npu_load() {
  for _ in $(seq "${MAX_SAMPLES}"); do
    kubectl -n "${NS}" exec "${POD}" -- cat /sys/kernel/debug/rknpu/load 2>/dev/null || true
    sleep 1
  done
}

printf '>> generating with %s (NPU load sampled concurrently)\n' "${MODEL}"
sample_npu_load >/tmp/npu-load.$$ &
SAMPLER=$!

START=$(date +%s.%N)
CHUNKS=$(curl -fsS -N --max-time 300 "${API}/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d "{\"model\": \"${MODEL}\", \"stream\": true, \"messages\": [{\"role\": \"user\", \"content\": \"${PROMPT}\"}]}" |
  grep -c '^data: {' || true)
END=$(date +%s.%N)
wait "${SAMPLER}" || true

ELAPSED=$(echo "${END} ${START}" | awk '{printf "%.1f", $1 - $2}')
# One SSE chunk per token piece; close enough for a first tok/s figure until the
# engine reports real usage counts (issue #8).
TOKS=$(echo "${CHUNKS} ${ELAPSED}" | awk '{printf "%.1f", $1 / $2}')
printf '>> %s chunks in %ss => ~%s tok/s\n' "${CHUNKS}" "${ELAPSED}" "${TOKS}"

printf '>> NPU load samples during generation:\n'
sort -u /tmp/npu-load.$$ | head -10
if ! grep -qv 'Core0:  0%' /tmp/npu-load.$$; then
  printf '>> WARNING: NPU load stayed at 0%% — generation likely ran on the CPU\n'
fi
rm -f /tmp/npu-load.$$
