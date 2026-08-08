# RK fork changelog

Deltas of the `rk-integration` line against upstream `exo-explore/exo`, newest first.
PR numbers reference [freed-dev-llc/exo-rkllama](https://github.com/freed-dev-llc/exo-rkllama).

## Unreleased

- **#40** (fixes #37): stalled event-log sync now triggers a master
  re-election instead of retrying forever. Pods restarted at different times
  could stay synced to a dead session (node ids are ephemeral): the router
  nacked the same missing index forever while its session filter dropped every
  replayed answer, so instance launches were acknowledged but never
  materialized until every pod was deleted at once. `EventRouter` now counts
  unanswered nack attempts (default 8, roughly 45 s under the existing
  0.5 s-base/10 s-cap backoff) and consecutive foreign-session events with
  zero in-session progress (default 100); past either threshold it sends
  itself a local `ConnectionMessage` to re-run the master election. The
  connection-messages topic never publishes to the network, so only the local
  campaign ignites and the election protocol fans out from there. Incumbents
  win re-election with `is_new_master=False` (no disruption on healthy
  nodes); diverged nodes rebuild router and worker onto the winning session.
  The master also logs a warning when a requested event-log range is beyond
  its tail, which was previously indistinguishable from silence.
- **#39**: on-hardware verification of the `deepseek-r1-distill-qwen-14b-rkllm`
  flow (14B places, loads, and completes a streamed chat on the RK1 cluster;
  ~0.8 tok/s decode). MODELS.md's reasoning-model guidance corrected to the
  measured wire shape: the reasoning block is terminated by `</think>` with no
  opening `<think>` on the wire (the chat template pre-fills it), so clients
  split on `</think>`. `smoke.sh`'s chat `curl --max-time` is now
  `CHAT_MAX_TIME_S` (default 300) — a hardcoded 120 s could never outlast an
  R1-class preamble.
- **#36**: adversarial review of the `deepseek-r1-distill-qwen-14b-rkllm`
  onboarding and the RKLLM generation path it exercises. The engine now
  enforces `max_tokens` (previously ignored everywhere outside MLX): it cancels
  the backend at the cap and finishes the stream with reason `length`, and it
  passes backend finish reasons through instead of hardcoding `stop`. The
  ctypes backend initializes the runtime from the model card —
  `context_length` sizes `max_context_len` and `sampling_defaults` set the
  init-time sampling (`temperature`/`top_p`/`top_k`/penalties) — so declared
  card values now take effect (behavior change: library defaults were used
  before). Per-request sampling overrides remain dropped by both backends, and
  runtime-cap truncations still report `stop` (RKLLM's FINISH carries no
  reason); both are documented in MODELS.md. The new card declares
  `reasoning_dialect = "post_last_user"`. MODELS.md also fixes the
  quantization/artifact-filename mismatch and the smoke-script path, and
  scopes the `<think>`-preamble guidance to what each backend actually does.

## 2026-07-06: rk-v0.2.3 (hub-add error guidance)

Point release: `ghcr.io/freed-dev-llc/exo-rkllama-rk:rk-v0.2.3` (arm64), same
upstream base as rk-v0.2.2 (`exo-explore/exo` `54596e6d`).

- **#33**: adding a model from the hub on an NPU host failed with an opaque
  `Failed to add model (400: Bad Request)`. The NPU-host search is gated to
  HF's `rkllm` library (#24), but `POST /models/add` builds cards from
  `config.json` + safetensors, which `.rkllm` repos don't have, and the
  dashboard read `err.detail` while the API wraps errors as
  `{error: {message}}`, so no detail ever reached the user. `EnginePlugin`
  grows `hub_add_guidance`, `plugin_for_hf_model()` matches a hub repo's
  library/tags against each plugin's `hf_search_filter`, `add_custom_model`
  returns the plugin's installation guidance for plugin-library repos, and the
  dashboard reads the error envelope. Downloading `.rkllm` models from the hub
  end to end is #34.
- **#35**: DaemonSet image pin bumped `rk-v0.2.2` → `rk-v0.2.3`.

## 2026-07-06: rk-v0.2.2 (engine plugin registry)

Point release: `ghcr.io/freed-dev-llc/exo-rkllama-rk:rk-v0.2.2` (arm64), same
upstream base as rk-v0.2.1 (`exo-explore/exo` `54596e6d`). No behavior change:
this is an internal refactor that shrinks the upstream-merge surface.

- **#31**: engine dispatch and detection now route through a plugin registry
  (`src/exo/shared/plugins.py`, `EnginePlugin` protocol) that the RKLLM port
  registers with (`src/exo/worker/engines/rkllm/plugin.py`), instead of editing
  shared upstream files with RKLLM-specific imports and predicates. Bootstrap
  builder dispatch, `info_gatherer` backend detection, worker download
  resolution, placement (pinning, single-node constraints, instance
  construction, `INSTANCE_META_BACKENDS`), and the API catalog/search/preview
  paths call generic registry hooks. `ModelCard.is_rkllm_model`,
  `BoundInstance.is_rkllm_model`, and `FETCHED_CARD_BACKENDS` are replaced by
  `plugin_for_card` / `plugin_for_instance` / `fetched_card_backends()`. The
  only RKLLM-specific lines left in shared files are the wire-format types
  (`Backend.RkllmNpu`, `InstanceMeta.RkllmSingleNode`, the `Instance` union
  member) that every node must parse. Verified on the RK1 cluster on a branch
  image before merge: catalog filtered to RKLLM-only, placement preview yields a
  single `Pipeline`/`RkllmSingleNode` combination, `llama3.2-3b-rkllm` placed as
  an `RkllmSingleNodeInstance` and streamed 118 chunks with the serving node's
  NPU at 87% while the others stayed idle.
- **#32**: DaemonSet image pin bumped `rk-v0.2.1` → `rk-v0.2.2`.

## 2026-07-04: rk-v0.2.1 (NPU onboarding and docs)

Point release: `ghcr.io/freed-dev-llc/exo-rkllama-rk:rk-v0.2.1` (arm64), same
upstream base as rk-v0.2.0 (`exo-explore/exo` `54596e6d`).

- **#29**: the onboarding wizard pins the bundled `llama3.2-3b-rkllm` card as its
  fast-loading small option on a Rockchip NPU host (the catalog is RKLLM-only
  there, so the previous mlx-community pin was a dead id), falling back to the mlx
  build off-NPU.
- **#27, #28**: model-onboarding guide (`docs/rk-hardware/MODELS.md`), un-drafted
  the deploy README, shipped `exo-service.yaml` + a self-healing preload CronJob in
  `deploy/rk-k3s/`, and a "different hardware" note in the RUNBOOK.

## 2026-07-04: rk-v0.2.0 (NPU-gated model selection)

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
