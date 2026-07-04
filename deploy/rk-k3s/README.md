# exo on RK1 NPU nodes (Armbian + K3s)

Runs exo as a privileged DaemonSet, one worker per RK1 node, with the RKLLM ctypes
backend talking to the NPU through `/dev/rknpu`.

**Status: validated on hardware.** Shipped as
`ghcr.io/freed-dev-llc/exo-rkllama-rk:rk-v0.2.0` (arm64) and running on a 4-node
Turing Pi 2 RK1 cluster (Armbian Trixie, vendor kernel 6.1.115, rknpu driver 0.9.8,
K3s v1.36.2). For the end-to-end bring-up from bare metal see
[`../../docs/rk-hardware/RUNBOOK.md`](../../docs/rk-hardware/RUNBOOK.md); for adding
and running your own models see
[`../../docs/rk-hardware/MODELS.md`](../../docs/rk-hardware/MODELS.md).

## Prerequisites

1. RK1 node(s) provisioned with Armbian (vendor kernel) and joined to K3s
   (issue #5; see freed-dev-llc/turing-rk1-cluster `docs/INSTALLATION-K3S.md`).
   Confirm on the node: `/dev/rknpu` exists and
   `cat /sys/kernel/debug/rknpu/version` reports >= 0.9.6 (0.9.8 for RKLLM 1.3).
2. Label each NPU node so the DaemonSet schedules there:

   ```bash
   kubectl label node <node> exo.freed.dev/rk-npu=true
   ```

3. Pre-place the converted models on each node (exo does not download `.rkllm`
   files; see `docs/rk-hardware/DEVELOPMENT.md`):

   ```bash
   # on the node (or scp from elsewhere)
   mkdir -p /var/lib/exo/rkllm-models/qwen2.5-7b-rkllm
   cp qwen2.5-7b-w8a8.rkllm /var/lib/exo/rkllm-models/qwen2.5-7b-rkllm/
   ```

   The DaemonSet mounts `/var/lib/exo/rkllm-models` at `/root/RKLLAMA/models`,
   which is where the engine's resolver looks.

## Build and push the image

```bash
docker buildx build --platform linux/arm64 \
  -f deploy/rk-k3s/Dockerfile \
  -t ghcr.io/freed-dev-llc/exo-rkllama-rk:latest \
  --push .
```

The build context is the repository root. `RKLLM_VERSION` pins the
airockchip/rknn-llm tag that supplies `librkllmrt.so`; it should match the RKLLM
version the engine targets (1.2.3 today, 1.3.0 tracked in issue #9).

## Deploy

```bash
kubectl apply -f deploy/rk-k3s/exo-daemonset.yaml
kubectl -n exo-rk rollout status ds/exo
```

Pods use `hostNetwork`, so zenoh peer discovery between exo nodes works as on bare
metal and the API/dashboard is reachable at `http://<node-ip>:52415`.

## Test

```bash
deploy/rk-k3s/scripts/smoke.sh            # place instance, assert streamed tokens
deploy/rk-k3s/scripts/bench.sh            # tok/s + NPU-load proof (not CPU fallback)
```

Both default to the `qwen2.5-7b-rkllm` card and take a model id as `$1`. The bench
approximates tok/s by counting SSE chunks until the engine reports real usage
numbers (issue #8).

## Design notes

- **Privileged + hostPath `/dev`**: the non-privileged device-plugin path currently
  falls back to CPU; revisit once a rknpu device plugin exists.
- **ctypes backend** (`EXO_RKLLM_BACKEND=ctypes`): no sidecar to manage. The rkllama
  HTTP sidecar remains an option for server-side chat templating (issue #8 tracks
  ctypes templating); add a second container and set `EXO_RKLLM_BACKEND=http` +
  `RKLLM_SERVER_HOST=127.0.0.1` to switch.
- **`/sys/kernel/debug` mount** exists solely so the bench can read
  `/sys/kernel/debug/rknpu/load` and prove the NPU did the work.
- **State under `/var/lib/exo/state`** keeps model cards and exo state across pod
  restarts.
