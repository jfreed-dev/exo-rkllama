# RK1 cluster runbook: bare metal to tokens

End-to-end procedure for standing up the exo NPU pool on a Turing Pi 2 with RK1
nodes, exactly as validated on 2026-07-03 (issues #5, #6). Every command here was
run against the real cluster. Assumes a workstation on the same LAN with `ssh`,
`kubectl`, `tpi`, and this repository checked out.

Naming used throughout: BMC at `turing-bmc` (10.10.88.70), nodes `rk1-1` through
`rk1-4` (10.10.88.73 through .76 via DHCP leases keyed to the boards' MACs).

## 1. Prepare the Armbian image (once per image refresh)

Use the **vendor kernel** build. The mainline (`current`/`edge`) kernels lack the
rknpu driver, and so does every stock Talos image; only the Rockchip 6.1 BSP
kernel carries it (see DEVELOPMENT.md for the verification trail).

```bash
mkdir -p ~/Downloads/rk1 && cd ~/Downloads/rk1
curl -fL -o armbian-trixie-vendor-minimal.img.xz \
  "https://dl.armbian.com/turing-rk1/Trixie_vendor_minimal"
unxz -k armbian-trixie-vendor-minimal.img.xz
```

Preseed it so nodes boot straight to SSH (no first-boot wizard):

```bash
sudo losetup -P -f --show armbian-trixie-vendor-minimal.img   # prints /dev/loopN
sudo mkdir -p /mnt/rk1img && sudo mount /dev/loop0p1 /mnt/rk1img

sudo tee /mnt/rk1img/root/.not_logged_in_yet >/dev/null <<'EOF'
PRESET_LOCALE="en_US.UTF-8"
PRESET_TIMEZONE="America/New_York"
PRESET_ROOT_PASSWORD="<bootstrap-password>"
PRESET_USER_NAME="jon"
PRESET_USER_PASSWORD="<bootstrap-password>"
PRESET_DEFAULT_REALNAME="Jonathan Freed"
PRESET_USER_SHELL="bash"
EOF

sudo mkdir -p /mnt/rk1img/root/.ssh
cat ~/.ssh/id_ed25519.pub | sudo tee /mnt/rk1img/root/.ssh/authorized_keys >/dev/null
sudo chmod 700 /mnt/rk1img/root/.ssh
sudo chmod 600 /mnt/rk1img/root/.ssh/authorized_keys

sudo umount /mnt/rk1img && sudo losetup -d /dev/loop0
```

Note: the first-run script may not preserve `authorized_keys` on every node;
section 3 re-installs keys over password auth, so this is belt-and-braces.
Rotate the bootstrap password after provisioning.

## 2. Flash all nodes from the BMC

Flashing erases the node eMMC (the 2026-07-03 run replaced a Talos cluster;
anything on the old cluster, e.g. Longhorn volumes, is gone).

```bash
scp ~/Downloads/rk1/armbian-trixie-vendor-minimal.img turing-bmc:/mnt/sdcard/images/
ssh turing-bmc 'for n in 1 2 3 4; do
  tpi flash --local --image-path /mnt/sdcard/images/armbian-trixie-vendor-minimal.img --node $n
done; for n in 1 2 3 4; do tpi power on --node $n; done'
```

Each flash writes ~1.7GB plus a CRC pass; `tpi flash` boots the node into the
new OS when it finishes. First boot takes a couple of minutes (rootfs resize +
preseed consumption). Nodes reappear on their usual DHCP addresses.

## 3. First-boot provisioning

Install the SSH key (password bootstrap), set hostnames, and verify the NPU on
every node. If your `~/.ssh/config` sets `IdentitiesOnly` for other hosts, pass
`-i` explicitly as shown; without it key auth can silently not be offered.

```bash
PUB=$(cat ~/.ssh/id_ed25519.pub)
for ip in 10.10.88.73 10.10.88.74 10.10.88.75 10.10.88.76; do
  sshpass -p '<bootstrap-password>' ssh -o StrictHostKeyChecking=accept-new \
    -o PreferredAuthentications=password root@$ip \
    "mkdir -p /root/.ssh && chmod 700 /root/.ssh && echo '$PUB' > /root/.ssh/authorized_keys && chmod 600 /root/.ssh/authorized_keys"
done

i=1
for ip in 10.10.88.73 10.10.88.74 10.10.88.75 10.10.88.76; do
  ssh -i ~/.ssh/id_ed25519 -o IdentitiesOnly=yes root@$ip \
    "hostnamectl set-hostname rk1-$i && sed -i 's/turing-rk1/rk1-$i/g' /etc/hosts \
     && apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get full-upgrade -y -qq \
     && uname -r && cat /sys/kernel/debug/rknpu/version"
  i=$((i + 1))
done
```

Expected on every node: kernel `6.1.115-vendor-rk35xx` (or newer vendor build)
and `RKNPU driver: v0.9.8`. The NPU appears as DRM render nodes
(`/dev/dri/by-path/platform-fdab0000.npu-*`); there is no `/dev/rknpu` on this
BSP. Reboot any node the upgrade asks to.

## 4. K3s

Server on rk1-1, agents on the rest:

```bash
ssh -i ~/.ssh/id_ed25519 -o IdentitiesOnly=yes root@10.10.88.73 \
  'curl -sfL https://get.k3s.io | INSTALL_K3S_EXEC="server --write-kubeconfig-mode 644" sh -'
TOKEN=$(ssh -i ~/.ssh/id_ed25519 -o IdentitiesOnly=yes root@10.10.88.73 \
  'cat /var/lib/rancher/k3s/server/node-token')
for ip in 10.10.88.74 10.10.88.75 10.10.88.76; do
  ssh -i ~/.ssh/id_ed25519 -o IdentitiesOnly=yes root@$ip \
    "curl -sfL https://get.k3s.io | K3S_URL=https://10.10.88.73:6443 K3S_TOKEN=$TOKEN sh -"
done

ssh -i ~/.ssh/id_ed25519 -o IdentitiesOnly=yes root@10.10.88.73 \
  'cat /etc/rancher/k3s/k3s.yaml' | sed 's/127.0.0.1/10.10.88.73/' > ~/.kube/rk1-k3s.yaml
chmod 600 ~/.kube/rk1-k3s.yaml
export KUBECONFIG=~/.kube/rk1-k3s.yaml
kubectl get nodes   # all four Ready
```

If an agent join dies with "connection closed", the apt upgrade restarted sshd
mid-install; rerun the join for that node.

## 5. Models

exo does not download `.rkllm` artifacts; place them on every NPU node.
Conversions must match the runtime line the image ships (**1.2.x**; most
community conversions are toolkit 1.1.4 and will not load). Known-good source:
the [jamescallander RK3588 collection](https://huggingface.co/collections/jamescallander/rk3588-rkllm-models)
(toolkit v1.2.1). The directory name must equal the exo model card id.

```bash
# Download once on rk1-1
ssh -i ~/.ssh/id_ed25519 -o IdentitiesOnly=yes root@10.10.88.73 \
  'mkdir -p /var/lib/exo/rkllm-models/llama3.2-3b-rkllm && curl -fL \
   -o /var/lib/exo/rkllm-models/llama3.2-3b-rkllm/Llama-3.2-3B-Instruct_w8a8_g128_rk3588.rkllm \
   "https://huggingface.co/jamescallander/Llama-3.2-3B-Instruct_w8a8_g128_rk3588.rkllm/resolve/main/Llama-3.2-3B-Instruct_w8a8_g128_rk3588.rkllm"'

# Fan out over the LAN (rk1-1 needs a keypair authorized on the others)
ssh -i ~/.ssh/id_ed25519 -o IdentitiesOnly=yes root@10.10.88.73 \
  'test -f /root/.ssh/id_ed25519 || ssh-keygen -t ed25519 -N "" -f /root/.ssh/id_ed25519 -q; cat /root/.ssh/id_ed25519.pub'
# ...append that key to /root/.ssh/authorized_keys on rk1-2..4, then:
ssh -i ~/.ssh/id_ed25519 -o IdentitiesOnly=yes root@10.10.88.73 \
  'for ip in 10.10.88.74 10.10.88.75 10.10.88.76; do
     ssh -o StrictHostKeyChecking=accept-new root@$ip "mkdir -p /var/lib/exo/rkllm-models/llama3.2-3b-rkllm"
     scp /var/lib/exo/rkllm-models/llama3.2-3b-rkllm/*.rkllm root@$ip:/var/lib/exo/rkllm-models/llama3.2-3b-rkllm/
   done'
```

New models additionally need a card in `resources/inference_model_cards/`
(`backends = ["RkllmNpu"]`; see `llama3.2-3b-rkllm.toml` as the template). For the
card fields, where the card must live (custom cards get pruned), and the
launch/preload flow, see [`MODELS.md`](MODELS.md).

## 6. Deploy exo and validate

```bash
export KUBECONFIG=~/.kube/rk1-k3s.yaml
kubectl label node rk1-1 rk1-2 rk1-3 rk1-4 exo.freed.dev/rk-npu=true
kubectl apply -f deploy/rk-k3s/exo-daemonset.yaml
kubectl -n exo-rk rollout status ds/exo

deploy/rk-k3s/scripts/smoke.sh llama3.2-3b-rkllm   # place, await, streamed tokens
deploy/rk-k3s/scripts/bench.sh llama3.2-3b-rkllm   # tok/s + NPU-load proof
```

Data-parallel check (replicas must land on distinct nodes and share traffic):

```bash
API=http://10.10.88.73:52415
curl -fsS -X POST "$API/place_instance" -H 'Content-Type: application/json' \
  -d '{"model_id": "llama3.2-3b-rkllm"}'    # once per desired replica
# fire N concurrent chat completions, then count generations per pod:
for pod in $(kubectl -n exo-rk get pods -o name); do
  echo "$pod: $(kubectl -n exo-rk logs $pod --since=5m | grep -c 'RKLLM perf')"
done
```

Reference numbers from the validated run (Llama-3.2-3B w8a8_g128): 3.46 tok/s
generate, 27 tok/s prefill per node, NPU ~90% on all three cores, 6 concurrent
requests split 3/3 across two replicas.

## 7. Updating exo on the cluster

The `rk-image` workflow builds `ghcr.io/freed-dev-llc/exo-rkllama-rk` on a
native arm64 runner. It runs automatically only when the Dockerfile changes;
**after source changes, dispatch it manually**:

```bash
gh workflow run rk-image.yml --repo freed-dev-llc/exo-rkllama --ref rk-integration
```

Roll the cluster by immutable SHA tag, not `latest` (the DaemonSet uses
`imagePullPolicy: IfNotPresent`, so `latest` will not repull):

```bash
kubectl -n exo-rk set image ds/exo exo=ghcr.io/freed-dev-llc/exo-rkllama-rk:<full-commit-sha>
kubectl -n exo-rk rollout status ds/exo
```

## Troubleshooting notes from the first bring-up

- **Runner crash-loop, AssertionError in builder.load**: fixed in #17 (single-node
  instances get no ConnectToGroup). Present in images before `fefdc714`.
- **SIGSEGV in rkllm_init**: ctypes structs must match the librkllmrt ABI line;
  fixed for 1.2.3 in #18. Any future RKLLM bump must re-verify `runtime.py`
  against the matching `rkllm.h`.
- **Model loads then no tokens / instance vanishes**: check
  `kubectl -n exo-rk logs <pod>` for the runner traceback; five failures delete
  the instance.
- **Replicas stack on one node**: fixed in #19 (placement anti-affinity).
- **`/instance/await` returns 422**: `timeout_seconds` is capped at 300 by the API.
- **bench reports NPU 0%**: pre-#19 script sampled only the first pod; the
  instance may live elsewhere.
