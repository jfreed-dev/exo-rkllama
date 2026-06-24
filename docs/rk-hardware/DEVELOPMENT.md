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

## Tier-2 testing — on the Turing Pi RK1 cluster (Talos / Kubernetes)

The cluster runs **Talos Linux** (immutable, API-driven, Kubernetes-only — no SSH, no
shell, no package manager), so the NPU path is validated via Kubernetes, **not** by
installing/running exo on the node directly (an Ansible-over-SSH approach does not apply).
Feasibility is confirmed (sources below).

**Gating item — the Talos image must include the NPU driver.** Stock Talos has no RK3588
NPU support. Build a [Talos Image Factory](https://factory.talos.dev) image for **Turing
RK1** (the `siderolabs/sbc-rockchip` overlay supports `turingrk1`; Talos ≥ v1.10 added
RK3588 kernel support) **with the `siderolabs/rockchip-rknn` system extension**
(`ghcr.io/siderolabs/rockchip-rknn`), which ships the `rknpu` kernel modules. **Verify the
extension's driver is ≥ 0.9.6 (ideally 0.9.8)** for RKLLM — the one unverified dependency.

**Deploy model.** exo runs as a **privileged DaemonSet** (one worker per RK1 node). NPU
access in a pod needs `securityContext.privileged: true` and a host `/dev` (or
`/dev/rknpu`) mount — a non-privileged device-plugin path currently falls back to CPU.
`librkllmrt.so` + the `.rkllm` models ship in the image (or a PVC): use
`EXO_RKLLM_BACKEND=ctypes` with `RKLLM_MODEL_PATH`, or run `rkllama` as a sidecar for the
HTTP backend. Talos Pod Security must allow privileged in the namespace.

**Validate.** Confirm the NPU is actually used (not CPU fallback) — tok/s + `librkllmrt`/
driver versions — then confirm data-parallel routing across nodes (one whole model each).

Precedent: RKLLM has been run in Kubernetes on RK3588 NPUs (Orange Pi 5 + MicroK8s +
RKLLama, privileged pods). Ansible can still *orchestrate* `talosctl`/`kubectl` (+ the
Turing Pi 2 BMC `tpi` for power/flash), but "provision" = Talos machine config + Image
Factory image, not apt/SSH.

Sources: [siderolabs/sbc-rockchip](https://github.com/siderolabs/sbc-rockchip) ·
[Turing RK1 (Sidero docs)](https://docs.siderolabs.com/talos/v1.9/platform-specific-installations/single-board-computers/turing_rk1) ·
[siderolabs/extensions `rockchip-rknn`](https://github.com/siderolabs/extensions) ·
[RKLLama on MicroK8s / RK3588](https://www.sngular.com/insights/471/the-definitive-guide-to-deploying-qwen3-on-the-npu-of-the-orange-pi-5-pro-max-plus-ultra-using-rkllama-and-microk8s) ·
[privileged-pod NPU requirement (immich #25057)](https://github.com/immich-app/immich/issues/25057)

## Backend selection

```bash
EXO_RKLLM_BACKEND=http     # default: talk to a rkllama server (RKLLM_SERVER_HOST/PORT)
EXO_RKLLM_BACKEND=ctypes   # in-process librkllmrt.so (RKLLM_LIB_PATH / RKLLM_MODEL_PATH)
```

## Versioning builds

Tag RK builds with the upstream base they were cut from, e.g.
`rk-v0.2.0+exo-g<short-sha>` where `<short-sha>` is `git rev-parse --short upstream/main`
at sync time. Keep a short RK changelog of deltas against upstream.

## Status & next steps

**Done:** CI is live and green on the default branch — `rk-ci` (fast Linux lint + RK unit
tests), the upstream `ci-pipeline` (`nix flake check` ×3 platforms + macOS pytest), and
`rk-upstream-sync` (scheduled/dispatch drift detector → opens a PR or fails on conflict).

**Not yet built** (tracked as GitHub issues):

- **Engine registry**: refactor dispatch (`bootstrap.py`) and detection
  (`info_gatherer.py`) to a plugin registry / entry points (the pre-zenoh tree had
  `plugin_discovery.py`) so RK support registers itself instead of editing shared files —
  dropping the merge-conflict surface to ~zero.
- **Talos NPU image**: build/verify a Turing RK1 Image Factory image with the
  `siderolabs/rockchip-rknn` extension and confirm its rknpu driver version (Tier-2 gate).
- **Tier-2 k8s automation**: privileged exo DaemonSet + `.rkllm` models, NPU smoke/bench,
  driven via `talosctl`/`kubectl`.
- **Model download**, **`token_id` fidelity (HTTP)**, **ctypes chat templating**, and a
  **RKLLM 1.2.3 → 1.3.0 / driver ≥ 0.9.8** bump (see the engine package + issues).
