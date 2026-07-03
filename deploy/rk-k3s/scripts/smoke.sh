#!/usr/bin/env bash
# Smoke test for the exo RK DaemonSet (issue #6).
# Asserts: DaemonSet ready -> RKLLM instance places -> chat completion streams tokens.
#
#   scripts/smoke.sh [model-id]
#
# Env: NS (default exo-rk), API_PORT (52415), TIMEOUT_S (300).
set -euo pipefail

NS="${NS:-exo-rk}"
MODEL="${1:-qwen2.5-7b-rkllm}"
API_PORT="${API_PORT:-52415}"
TIMEOUT_S="${TIMEOUT_S:-300}"

say() { printf '>> %s\n' "$*"; }

say "waiting for DaemonSet rollout in ${NS}"
kubectl -n "${NS}" rollout status ds/exo --timeout="${TIMEOUT_S}s"

NODE_IP=$(kubectl -n "${NS}" get pods -l app.kubernetes.io/name=exo \
  -o jsonpath='{.items[0].status.hostIP}')
API="http://${NODE_IP}:${API_PORT}"
say "API endpoint: ${API}"

say "checking NPU device + driver on the node"
POD=$(kubectl -n "${NS}" get pods -l app.kubernetes.io/name=exo \
  -o jsonpath='{.items[0].metadata.name}')
kubectl -n "${NS}" exec "${POD}" -- sh -c \
  'test -e /dev/rknpu && cat /sys/kernel/debug/rknpu/version' ||
  say "WARNING: /dev/rknpu or rknpu debugfs missing (CPU fallback likely)"

say "placing instance for ${MODEL}"
curl -fsS -X POST "${API}/place_instance" \
  -H 'Content-Type: application/json' \
  -d "{\"model_id\": \"${MODEL}\"}" >/dev/null

say "waiting for instance to become ready"
curl -fsS "${API}/instance/await?model_id=${MODEL}&timeout_seconds=${TIMEOUT_S}" >/dev/null

say "requesting a streamed completion"
RESPONSE=$(curl -fsS -N --max-time 120 "${API}/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d "{\"model\": \"${MODEL}\", \"stream\": true, \"messages\": [{\"role\": \"user\", \"content\": \"Say hello in five words.\"}]}")

CHUNKS=$(printf '%s\n' "${RESPONSE}" | grep -c '^data: {' || true)
if [ "${CHUNKS}" -lt 2 ]; then
  say "FAIL: expected streamed chunks, got ${CHUNKS}"
  printf '%s\n' "${RESPONSE}" | head -20
  exit 1
fi
say "PASS: ${CHUNKS} streamed chunks from ${MODEL}"
