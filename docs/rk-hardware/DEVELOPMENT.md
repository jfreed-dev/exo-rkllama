# RK hardware development workflow

This fork adds **Rockchip RK3588/RK3576 NPU** inference (RKLLM) to exo while tracking
upstream `exo-explore/exo`. RKLLM runs a *whole model on one NPU* (no cross-node
sharding), so a Turing Pi cluster of RK1s is used as a **data-parallel pool** — one
whole model per node, the master routes requests.

## Branch model

| Branch | Purpose |
|---|---|
| `main` | Clean mirror of `upstream/main`. **No RK changes land here.** |
| `rk-integration` | Long-lived RK feature line. Upstream is merged into it regularly. |
| `rk/<feature>` | Short-lived branches off `rk-integration`, merged back via PR. |

> `rk-integration` is the repository's **default branch** (so scheduled/dispatch
> workflows run); `main` is kept as a clean mirror of `upstream/main`.

Remotes:

```bash
git remote add upstream https://github.com/exo-explore/exo.git   # one time
git remote -v   # origin = your fork, upstream = exo-explore/exo
```

## Keeping `main` synced to upstream

```bash
git checkout main
git fetch upstream
git merge --ff-only upstream/main     # main must stay a clean mirror
git push origin main
```

## Pulling upstream into the RK line

```bash
git checkout rk-integration
scripts/sync-upstream.sh              # fetch + merge upstream/main, then run the gate
```

The port is **~95% additive** (the `src/exo/worker/engines/rkllm/` package and the
`resources/inference_model_cards/*-rkllm.toml` cards never conflict). The only conflict
surface is the seven shared files the engine hooks into — `sync-upstream.sh` lists them
if a merge conflicts. To shrink that surface toward zero, see *Engine registry* below.

## The gate (Tier-1 testing — x86, no NPU)

```bash
scripts/gate.sh        # ruff + basedpyright + pytest (+ nix fmt if available)
```

What it proves on any dev box / CI runner without an NPU:

- **RK code is type-clean** under basedpyright strict and lint-clean under ruff.
- **Unit tests** pass: NPU detection (mocked device-tree), the engine's
  submit/step/cancel loop (fake backend), backend selection, and that placement routes
  an `RkllmNpu` model to a single-node `RkllmSingleNodeInstance`.

Caveats:

- basedpyright needs the MLX stack resolvable. On a bare Linux box `mlx.*` may not
  resolve (CI's macOS target does); sync the extra with `uv sync --extra mlx-cpu`. Our
  RK files are clean regardless.
- Unit tests import without a built dashboard via `EXO_DASHBOARD_DIR` (the gate sets it).
  Full app runs still need `cd dashboard && npm install && npm run build`.

## Tier-2 testing — on the Turing Pi RK1 cluster

The real NPU path can only be validated on hardware. Target flow (automation TBD):

1. **provision** — RKNPU driver ≥ 0.9.8 (`cat /sys/kernel/debug/rknpu/version`),
   `librkllmrt.so` / `rkllama`, deps.
2. **deploy** — push the built exo + sync `.rkllm` models (HTTP backend: rkllama models
   dir; ctypes backend: `~/RKLLAMA/models/<model_id>/` or `RKLLM_MODEL_PATH`).
3. **smoke** — start exo, hit the OpenAI endpoint, assert a streamed token response.
4. **bench** — tok/s per node and aggregate data-parallel throughput; record driver +
   `librkllmrt` versions per build.

Planned automation: **Ansible** (inventory of the 4 RK1 nodes) driving the Turing Pi 2
BMC via the [`tpi`](https://github.com/turing-machines/tpi) CLI for power/flash/console
between runs; Molecule to test the playbooks; a self-hosted GitHub Actions runner on one
node so Tier-2 can run from CI.

## Backend selection

```bash
EXO_RKLLM_BACKEND=http     # default: talk to a rkllama server (RKLLM_SERVER_HOST/PORT)
EXO_RKLLM_BACKEND=ctypes   # in-process librkllmrt.so (RKLLM_LIB_PATH / RKLLM_MODEL_PATH)
```

## Versioning builds

Tag RK builds with the upstream base they were cut from, e.g.
`rk-v0.2.0+exo-g<short-sha>` where `<short-sha>` is `git rev-parse --short upstream/main`
at sync time. Keep a short RK changelog of deltas against upstream.

## Next steps (not yet built)

- **CI**: GitHub Actions running `scripts/gate.sh` on push/PR, plus a scheduled job that
  runs `scripts/sync-upstream.sh` to catch upstream drift early.
- **Engine registry**: refactor dispatch (`bootstrap.py`) and detection
  (`info_gatherer.py`) to a plugin registry / entry points (the pre-zenoh tree had
  `plugin_discovery.py`) so RK support registers itself instead of editing shared files —
  dropping the merge-conflict surface to ~zero.
- **Ansible + tpi** scaffold for the Tier-2 flow above.
