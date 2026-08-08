# Adding and running your own models

exo on the RK3588/RK3576 NPU runs pre-converted single-file `.rkllm` artifacts
through the RKLLM ctypes backend. It never downloads them: you place the file on
each NPU node, give it a model card, and launch an instance. This guide covers the
parts that are easy to get wrong.

## 1. Get a compatible `.rkllm`

The artifact must be converted with an RKLLM toolkit version **at or below the
runtime the image ships** (`1.2.x` today; a model from a newer toolkit fails
`rkllm_init`). Most community conversions on HuggingFace are toolkit `1.1.x`: those
load on the `1.2.3` runtime but log `model version too old` and are slower. Filter
HuggingFace by the `rkllm` library (`https://huggingface.co/models?library=rkllm`,
API `?filter=rkllm`) rather than by author. Known-good, version-matched sources:

| Source | Toolkit | Notes |
|--------|---------|-------|
| [jamescallander](https://huggingface.co/jamescallander) | 1.2.1 | naming `*_w8a8_g128_rk3588.rkllm` |
| [thanhtantran](https://huggingface.co/thanhtantran) | 1.2.0 | includes larger Qwen builds |
| [c01zaut](https://huggingface.co/c01zaut) | 1.1.x | broad catalog up to ~27B; older format |

Size vs RAM: a `w8a8` model is roughly 1 byte/param, so a 14B is ~15 GB and a 20B is
~20 GB. Each instance loads the whole model on one node, so on 32 GB nodes the
practical ceiling is ~20B (a 27B `.rkllm` at ~30 GB will not fit alongside the KV
cache and OS).

## 2. Place the file on every node

Placement picks a node by free memory, not by which node holds the file, so every
NPU node needs the complete artifact or the instance can land on a node that fails
to load it. The directory name **must equal the model card id** (slashes become
`--`):

```bash
# on each node
mkdir -p /var/lib/exo/rkllm-models/qwen2.5-14b-rkllm
curl -fL -o /var/lib/exo/rkllm-models/qwen2.5-14b-rkllm/Qwen2.5-14B-Instruct-1M-rk3588-w8a8_g128.rkllm \
  "https://huggingface.co/.../resolve/main/....rkllm"
```

The DaemonSet mounts `/var/lib/exo/rkllm-models` at `/root/RKLLAMA/models`, where the
resolver looks.

## 3. Write the model card

A card is a hand-written TOML with `backends = ["RkllmNpu"]` (exactly that one
backend is the `is_rkllm_model` test). The architecture fields come from the base
model's HuggingFace `config.json`:

```toml
model_id = "qwen2.5-14b-rkllm"
n_layers = 48                 # num_hidden_layers
hidden_size = 5120            # hidden_size
num_key_value_heads = 8       # num_key_value_heads
supports_tensor = false
tasks = ["TextGeneration"]
backends = ["RkllmNpu"]
family = "qwen"
quantization = "w8a8_g128"       # matches the artifact filename
base_model = "Qwen2.5-14B-Instruct-1M"
capabilities = ["text"]
context_length = 4096

[storage_size]
in_bytes = 15606658452        # exact size of the .rkllm file (stat -c %s)

[sampling_defaults]
temperature = 0.7
top_p = 0.8
top_k = 20
```

Use `resources/inference_model_cards/llama3.2-3b-rkllm.toml` as the template.

Which card fields take effect on the NPU path: `context_length` sizes the
runtime's context at load and `[sampling_defaults]` initializes its sampling
(`temperature`, `top_p`, `top_k`, and the three penalties; `min_p` has no RKLLM
equivalent). Per-request sampling overrides from the API are still dropped by
both backends, and on the rkllama HTTP path sampling is the server's, not the
card's. `quantization`, `family`, and `capabilities` are informational.

### Where the card must live (the gotcha)

Cards load from two places: the built-in `resources/inference_model_cards/` and the
per-node custom-cards dir (`$XDG_DATA_HOME/exo/custom_model_cards`). **Do not use the
custom-cards dir for hand-placed rkllm cards.** The master reconciles custom cards
against cluster state and deletes any it did not create through the API, so a TOML
you drop there is removed on the next restart. Use a built-in card instead, by one of:

- **Bake it into the image** (durable, needs a rebuild): add the TOML to
  `resources/inference_model_cards/` and rebuild the arm64 image. Best for models you
  want in every deployment.
- **Mount a host dir over the built-in dir** (no rebuild): seed a host directory with
  the image's existing cards plus yours, and mount it at
  `/app/resources/inference_model_cards`. Adding a later model is then just a file
  drop plus a pod restart. Cards only load at startup, so restart the pod after
  changing them (`kubectl -n exo-rk rollout restart ds/exo`).

Confirm the card loaded: `curl -s http://<node-ip>:52415/v1/models | jq '.data[].id'`
should list your `model_id`. Note that `/v1/models` shows every card the node knows
about; `/models?status=downloaded` and `/ollama/api/tags` list a model only once exo
has resolved its `.rkllm` on a node (which happens when an instance launches), and
keep listing it afterwards.

## 4. Launch an instance (exo does not lazy-load)

A chat request against a model with no running instance returns
`404 No instance found`; you must create an instance first. An instance is
single-node and whole-model: it loads the full model on one node's NPU. Launch flow:

```bash
API=http://<node-ip>:52415
MODEL=qwen2.5-14b-rkllm            # or deepseek-r1-distill-qwen-14b-rkllm

# 1. compute a placement, 2. create the instance, 3. wait for it to load
placement=$(curl -s "$API/instance/placement?model_id=$MODEL&sharding=Pipeline&instance_meta=RkllmSingleNode&min_nodes=1")
curl -s -X POST "$API/instance" -H 'Content-Type: application/json' \
  -d "$(printf '%s' "$placement" | jq -c '{instance: .}')"
curl -sN "$API/instance/await?model_id=$MODEL"   # emits {"type":"ready"} when loaded

# then chat as usual
curl -s "$API/v1/chat/completions" -H 'Content-Type: application/json' \
  -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}"
```

Reasoning models such as `deepseek-r1-distill-qwen-14b-rkllm` emit a reasoning
preamble before the final answer, on both backends (on the rkllama HTTP path the
server's chat template forces it; on the ctypes path the model reasons anyway).
On the wire the block is **terminated by `</think>` with no opening `<think>`** —
the opening tag is pre-filled by the chat template, never generated — so clients
should split on `</think>`, not match a `<think>...</think>` span. Budget
`max_tokens` for both the reasoning block and the visible answer: the engine
stops the generation at the cap and finishes with `finish_reason="length"`, so
a budget smaller than the preamble leaves no answer. Two bounds apply
regardless: the runtime caps a generation at 2048 tokens (init-time
`max_new_tokens`, reported as a normal stop), and a long preamble takes minutes
at NPU decode rates — measured on the RK1 cluster: the 14B spent ~250 tokens /
~5 minutes (~0.8 tok/s) reasoning about "Say hello in five words" before a
one-word answer. Stream responses and set generous client timeouts.

`deploy/rk-k3s/scripts/smoke.sh` wraps this for the bundled cards (pass a model
id as `$1`). The smoke bounds its chat with `CHAT_MAX_TOKENS` (default 128), so
reasoning models finish inside `CHAT_MAX_TIME_S` (300 s) with a `length`
finish on images ≥ rk-v0.3.0; on older images `max_tokens` is ignored and an
R1-class chat can run for tens of minutes regardless of either timeout knob.

## 5. Parallelism and keeping instances loaded

Each instance processes one generation at a time (the NPU is serial), and the master
routes each request to the least-loaded instance. Running **one instance per node**
gives N-way parallelism: with four nodes, up to four requests run concurrently and
the rest queue on the least-loaded node.

Instance state is ephemeral. A reboot, pod restart, or image roll drops every loaded
instance, and exo does not restore them (the `.rkllm` files and cards survive, the
loaded instances do not). To keep a model warm across restarts, run a reconciler that
re-launches missing instances. [`exo-preload-cronjob.yaml`](../../deploy/rk-k3s/exo-preload-cronjob.yaml)
is a self-healing CronJob that ensures one instance per node; it is idempotent (a
no-op when fully loaded) and waits for each instance to finish loading before the
next so placement spreads them across nodes. It reaches the API through the
[`exo-service.yaml`](../../deploy/rk-k3s/exo-service.yaml) ClusterIP Service:

```bash
kubectl apply -f deploy/rk-k3s/exo-service.yaml
kubectl apply -f deploy/rk-k3s/exo-preload-cronjob.yaml   # set MODEL in the manifest
```
