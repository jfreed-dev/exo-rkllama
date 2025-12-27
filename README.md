# exo-rkllama

A fork of [exo-explore/exo](https://github.com/exo-explore/exo) that adds **RKLLM inference engine support** for Rockchip RK3588/RK3576 NPU devices.

## What is exo?

[exo](https://github.com/exo-explore/exo) is a distributed inference framework that lets you run large language models across multiple heterogeneous devices (iPhones, iPads, Macs, NVIDIA GPUs, Raspberry Pis). Key features:

- **P2P Architecture**: No master-worker hierarchy - all devices are equal peers
- **ChatGPT-Compatible API**: Drop-in replacement at `localhost:52415/v1/chat/completions`
- **Automatic Device Discovery**: Zero-configuration networking via UDP, Tailscale, or manual config
- **Dynamic Model Partitioning**: Automatically splits models across devices based on available memory

## What This Fork Adds

This fork adds the **RKLLM inference engine** that enables LLM inference on Rockchip NPUs (Neural Processing Units), specifically targeting:

- **RK3588** (6 TOPS INT8)
- **RK3576** (6 TOPS INT8)

This allows devices like Orange Pi 5, Rock 5B, Turing Pi 2 RK1 modules, and other RK3588-based SBCs to participate in exo clusters or run inference standalone.

### Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         Request Flow                             │
│                                                                  │
│  User ──▶ ChatGPT API ──▶ Node ──▶ RKLLMEngine ──▶ HTTP Client  │
│             :52415                                      │        │
│                                                         ▼        │
│                                               ┌─────────────────┐│
│                                               │ RKLLAMA Server  ││
│                                               │    :8080        ││
│                                               └────────┬────────┘│
│                                                        ▼         │
│                                               ┌─────────────────┐│
│                                               │  RK3588 NPU     ││
│                                               │  6 TOPS INT8    ││
│                                               └─────────────────┘│
└─────────────────────────────────────────────────────────────────┘
```

The RKLLM engine communicates with a separate [rkllama](https://github.com/jfreed-dev/rkllama) server that handles the actual NPU inference via Rockchip's RKLLM runtime.

## Performance

Tested on RK3588 with RKLLM SDK 1.1.4:

| Model | Tokens/sec | Notes |
|-------|------------|-------|
| Qwen2.5-1.5B-Instruct (w8a8) | ~7.8 | Concise responses, ideal for APIs |
| DeepSeek-R1-1.5B | ~7.9 | Chain-of-thought reasoning |

## Installation

**Requires Python >= 3.12.0**

```bash
git clone https://github.com/jfreed-dev/exo-rkllama.git
cd exo-rkllama
pip install -e .
```

## Quick Start

### 1. Start RKLLAMA Server (on RK3588 device)

```bash
cd /opt/rkllama
source venv/bin/activate
python server.py --target_platform rk3588 --port 8080
```

### 2. Start exo with RKLLM Engine

```bash
# Optional: configure server location (defaults shown)
export RKLLM_SERVER_HOST=localhost
export RKLLM_SERVER_PORT=8080

# Start exo
exo --inference-engine rkllm --disable-tui
```

### 3. Send Requests

```bash
curl http://localhost:52415/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
     "model": "qwen2.5-1.5b-instruct-rkllm",
     "messages": [{"role": "user", "content": "Hello!"}],
     "temperature": 0.7
   }'
```

## Supported RKLLM Models

| Model ID | Description |
|----------|-------------|
| `qwen2.5-1.5b-instruct-rkllm` | Qwen 2.5 1.5B Instruct |
| `qwen2.5-3b-rkllm` | Qwen 2.5 3B |
| `qwen2.5-7b-rkllm` | Qwen 2.5 7B |
| `deepseek-r1-1.5b-rkllm` | DeepSeek R1 1.5B (chain-of-thought) |
| `phi-3-mini-rkllm` | Phi-3 Mini |

Models require pre-converted `.rkllm` files in `~/RKLLAMA/models/`. See [rkllm-toolkit](https://github.com/jfreed-dev/rkllm-toolkit) for conversion.

## Status Matrix

### Working

| Feature | Status | Notes |
|---------|--------|-------|
| Basic inference | ✅ | Via HTTP client to rkllama server |
| ChatGPT-compatible API | ✅ | `/v1/chat/completions` endpoint |
| Token caching | ✅ | Handles RKLLM batch-style generation |
| Model loading/unloading | ✅ | Hot-swap models via HTTP API |
| HuggingFace tokenizers | ✅ | Auto-downloads tokenizers |
| Qwen2.5-1.5B-Instruct | ✅ | Tested, ~7.8 tok/s, recommended |
| Systemd services | ✅ | Auto-start on boot |
| Web UI (tinychat) | ✅ | Works at localhost:52415 |

### Not Working / Known Issues

| Feature | Status | Notes |
|---------|--------|-------|
| DeepSeek-R1-1.5B | ⚠️ | Long chain-of-thought causes timeouts, needs streaming |
| Streaming responses | ⚠️ | Code exists, not fully integrated |
| Layer sharding | ❌ | By design - RKLLM loads full models only |
| Multi-node distribution | ❌ | By design - use load balancer instead |
| Direct ctypes mode | ⚠️ | Fallback only, HTTP mode recommended |
| Qwen2.5-3B+ models | ❓ | Not yet converted/tested |

### Planned / TODO

| Feature | Priority | Description |
|---------|----------|-------------|
| Streaming support | Medium | SSE/chunked responses for real-time output |
| Convert 3B/7B models | Medium | Larger models for better quality |
| Multi-node load balancing | Medium | HAProxy/nginx config for request parallelism |
| Health monitoring | Low | Prometheus metrics, Grafana dashboard |
| Deployment guide | Low | Full setup documentation |

### Recently Fixed

| Feature | Date | Notes |
|---------|------|-------|
| RKLLM runtime 1.2.3 | 2025-12-27 | Updated rkllama fork with correct ABI structures |

See [TODO.md](TODO.md) for detailed task list.

## Limitations

**Single Node Only**: RKLLM loads complete models - no layer sharding across nodes. For multiple RK3588 devices, use request-level parallelism (load balancer) instead of layer-level distribution.

## Other Inference Engines

This fork retains all original exo inference engines:

| Engine | Platform | Status |
|--------|----------|--------|
| **MLX** | Apple Silicon | Supported |
| **tinygrad** | Cross-platform | Supported |
| **RKLLM** | Rockchip NPU | **Added in this fork** |

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `RKLLM_SERVER_HOST` | `localhost` | RKLLAMA server hostname |
| `RKLLM_SERVER_PORT` | `8080` | RKLLAMA server port |
| `EXO_HOME` | `~/.cache/exo` | Model cache directory |
| `DEBUG` | `0` | Debug verbosity (0-9) |

## Documentation

- [RKLLM Engine Details](exo/inference/rkllm/README.md) - Full architecture, benchmarks, troubleshooting
- [Systemd Services](systemd/README.md) - Auto-start configuration

## Related Repositories

- [exo-explore/exo](https://github.com/exo-explore/exo) - Original exo project
- [jfreed-dev/rkllama](https://github.com/jfreed-dev/rkllama) - RKLLAMA server for NPU inference
- [jfreed-dev/rkllm-toolkit](https://github.com/jfreed-dev/rkllm-toolkit) - Model conversion toolkit

## License

GPL-3.0 (same as upstream exo)
