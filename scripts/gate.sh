#!/usr/bin/env bash
# Tier-1 gate (x86, no NPU): type-check, lint, format-check, unit tests.
# Mirrors the checks in CLAUDE.md. Run before committing and after an upstream sync.
#
#   scripts/gate.sh            # run everything, report a summary
#   SKIP_PYRIGHT=1 scripts/gate.sh
#
# Notes:
#   * basedpyright needs the MLX stack resolvable (macOS, or Linux with the
#     `mlx-cpu` extra synced: `uv sync --extra mlx-cpu`). Our RK code is clean
#     regardless; unresolved `mlx.*` on a bare box is an environment gap.
#   * EXO_DASHBOARD_DIR lets unit tests import without a built dashboard.
set -uo pipefail
cd "$(dirname "$0")/.."

fail=0
run() {
	echo
	echo "▶ $*"
	if ! "$@"; then
		echo "✗ FAILED: $*"
		fail=1
	fi
}

[ "${SKIP_RUFF:-0}" = 1 ] || run uv run ruff check
[ "${SKIP_PYRIGHT:-0}" = 1 ] || run uv run basedpyright
[ "${SKIP_TESTS:-0}" = 1 ] || run env EXO_DASHBOARD_DIR=placeholder uv run pytest -q
if command -v nix >/dev/null 2>&1 && [ "${SKIP_FMT:-0}" != 1 ]; then
	run nix fmt
fi

echo
if [ "$fail" = 0 ]; then echo "✓ gate passed"; else echo "✗ gate failed"; fi
exit "$fail"
