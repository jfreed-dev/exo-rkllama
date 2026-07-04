# RK fork changelog

Deltas of the `rk-integration` line against upstream `exo-explore/exo`, newest first.
PR numbers reference [freed-dev-llc/exo-rkllama](https://github.com/freed-dev-llc/exo-rkllama).

## 2026-07-04: rk-v0.2.0 — NPU-gated model selection

Second release: `ghcr.io/freed-dev-llc/exo-rkllama-rk:rk-v0.2.0` (arm64), same
upstream base as rk-v0.1.0 (`exo-explore/exo` `54596e6d`).

- **#24**: NPU-gated model catalog and search. The catalog (`/models`,
  `/v1/models`) and HuggingFace search (`/models/search`) defaulted to
  mlx-community, which cannot run on the RK3588/RK3576 NPU. On a Rockchip NPU
  host (`detect_rockchip_npu`), `get_models` now limits the catalog to
  RKLLM-engine cards (`is_rkllm_model`) and `search_models` queries HF's
  `rkllm` library (`filter=rkllm`) instead of `author=mlx-community`; both fall
  back to all of HuggingFace when empty. Non-NPU hosts keep the mlx-community
  default. Dashboard search placeholder and trending header made
  library-agnostic.
- **#25**: DaemonSet image pin bumped `rk-v0.1.0` → `rk-v0.2.0`.

## 2026-07-03: Tier-2 hardware validation (issues #5, #6 closed)

First tokens from the RK3588 NPU: 4x Turing RK1 flashed to Armbian Trixie
(vendor kernel 6.1.115, rknpu driver 0.9.8), K3s v1.36.2, exo as a privileged
DaemonSet. Smoke PASS on `llama3.2-3b-rkllm` (w8a8_g128, toolkit 1.2.1); NPU at
~90% on all three cores during generation; 3.46 tok/s generate / 27 tok/s
prefill per the runtime's counters; two replicas on distinct nodes served six
concurrent requests split 3/3. Procedure: [`RUNBOOK.md`](RUNBOOK.md).

- **#21**: end-to-end runbook (bare metal to tokens)
- **#20**: hardware validation results recorded in the README
- **#19**: placement anti-affinity so same-model RKLLM replicas spread across
  nodes (they stacked onto the first node that resolved the artifact);
  `bench.sh` samples every pod's NPU instead of the first
- **#18**: ctypes bindings rewritten for the RKLLM **1.2.3 ABI** (the 1.1.x-era
  structs segfaulted in `rkllm_init`); banked real `token_id`s through
  TokenPiece, runtime perf logging at finish, and `rkllm_abort` on cancel
- **#17**: lazy backend selection in `RkllmBuilder.load` (single-node instances
  never receive ConnectToGroup, so `connect()` never ran); smoke-script fixes
  (await timeout cap, NPU detection via DRM by-path node)
- **#16**: image build fix (git for uv git dependencies, cmake/perl for -sys
  crates); first successful arm64 image
- **#15**: `rk-image` workflow (native arm64 GHCR builds) and the
  `llama3.2-3b-rkllm` model card (runtime-1.2.x-compatible smoke model; most
  community conversions are toolkit 1.1.4 and will not load)
- **#14**: `deploy/rk-k3s/` (privileged DaemonSet, Dockerfile with librkllmrt
  baked in, smoke/bench scripts)

## 2026-07-03: Tier-2 platform decision (issue #5 re-scoped)

- **#13**: verified stock Talos cannot run RKLLM: the `rockchip-rknn` extension
  ships only the mainline `rocket.ko` (Teflon/TFLite), while librkllmrt needs
  the downstream rknpu driver that exists only for vendor 5.10/6.1 kernels.
  Decision: **Armbian + K3s** on the NPU nodes. Issues #5/#6 re-scoped from
  Talos to K3s.

## 2026-07-03: engine hardening after adversarial review

- **#12**: two blockers plus five review findings. Custom HF cards no longer
  force-pin to the RKLLM engine (`is_rkllm_model` = sole-backend test; download
  and engine dispatch both key off the placed instance type). `.rkllm`
  artifacts resolve locally instead of an impossible HF download
  (`engines/rkllm/models.py`). Coordinator sweep skips RKLLM cards (it
  overwrote worker download state every 60s); DeleteDownload refuses
  hand-placed RKLLM artifacts; invalid `RKLLM_MODEL_PATH` /
  `EXO_RKLLM_BACKEND` fail with the real configuration error.

## 2026-06-24: port re-established on the zenoh architecture

- RK engine package (`src/exo/worker/engines/rkllm/`): `RkllmBackend`
  abstraction with HTTP (rkllama) and in-process ctypes transports,
  `Backend.RkllmNpu`, `RkllmSingleNodeInstance`, RK3588 detection, ~7 hooks
  into shared files. RK CI (`rk-ci`, `rk-upstream-sync`); `rk-integration` as
  the default branch with `main` mirroring upstream.
