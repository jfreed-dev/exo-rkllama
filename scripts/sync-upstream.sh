#!/usr/bin/env bash
# Sync this RK feature branch with upstream exo, then run the Tier-1 gate.
#
# Branch model:
#   main            clean mirror of upstream/main — no RK changes land here
#   rk-integration  long-lived RK feature line (merge upstream into it regularly)
#
# Usage:
#   scripts/sync-upstream.sh                # merge $UPSTREAM_REMOTE/main into current branch
#   scripts/sync-upstream.sh upstream/v1.2  # merge a specific ref
#   UPSTREAM_REMOTE=upstream scripts/sync-upstream.sh
set -euo pipefail
cd "$(dirname "$0")/.."

UPSTREAM_REMOTE="${UPSTREAM_REMOTE:-upstream}"
UPSTREAM_REF="${1:-${UPSTREAM_REMOTE}/main}"
branch="$(git rev-parse --abbrev-ref HEAD)"

if [ -n "$(git status --porcelain)" ]; then
	echo "✗ Working tree not clean — commit or stash before syncing." >&2
	exit 1
fi

if ! git remote get-url "${UPSTREAM_REMOTE}" >/dev/null 2>&1; then
	echo "✗ No '${UPSTREAM_REMOTE}' remote. Add it with:" >&2
	echo "    git remote add ${UPSTREAM_REMOTE} https://github.com/exo-explore/exo.git" >&2
	exit 1
fi

echo "▶ Fetching ${UPSTREAM_REMOTE}…"
git fetch "${UPSTREAM_REMOTE}" --tags

base="$(git merge-base HEAD "${UPSTREAM_REF}")"
count="$(git rev-list --count "${base}..${UPSTREAM_REF}")"
if [ "${count}" = 0 ]; then
	echo "✓ '${branch}' already up to date with ${UPSTREAM_REF}."
	exit 0
fi

echo "▶ ${count} upstream commit(s) to merge into '${branch}':"
git --no-pager log --oneline "${base}..${UPSTREAM_REF}" | sed 's/^/    /' | head -40
echo

if ! git merge --no-edit "${UPSTREAM_REF}"; then
	cat >&2 <<'EOF'
✗ Merge conflicts. The RK port is almost entirely additive; conflicts can only
  occur in the handful of shared files it hooks into:
    src/exo/shared/types/backends.py                  (RkllmNpu enum)
    src/exo/shared/types/worker/instances.py          (Instance union member)
    src/exo/master/placement.py                       (RkllmSingleNode placement)
    src/exo/worker/runner/bootstrap.py                (engine dispatch branch)
    src/exo/utils/info_gatherer/info_gatherer.py      (RK3588 detection)
    src/exo/api/main.py                               (placement previews)
    src/exo/worker/engines/mlx/utils_mlx.py           (match exhaustiveness arm)
  Resolve those, `git add`, `git commit`, then run: scripts/gate.sh
EOF
	exit 1
fi

echo "▶ Merge clean. Running gate…"
exec scripts/gate.sh
