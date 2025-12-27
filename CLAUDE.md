# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Exo is a distributed inference framework for running large language models across multiple heterogeneous devices (iPhones, iPads, Macs, NVIDIA GPUs, Raspberry Pis). Key features:
- P2P architecture (no master-worker hierarchy)
- ChatGPT-compatible API at `localhost:52415/v1/chat/completions`
- Multi-engine support: MLX (Apple Silicon), tinygrad (cross-platform), RKLLM (Rockchip RK3588 NPU)
- Automatic device discovery via UDP, Tailscale, or manual config

## Build & Installation

```bash
# From source (recommended)
pip install -e .

# Or with venv
source install.sh
```

**Requires Python >= 3.12.0** (asyncio improvements)

## Running

```bash
exo                                    # Start node with auto-discovery
exo --inference-engine mlx             # Use MLX engine (Apple Silicon)
exo --inference-engine tinygrad        # Use tinygrad engine
exo --inference-engine rkllm           # Use RKLLM engine (Rockchip RK3588 NPU)
exo --run-model llama-3.2-3b           # Single-device inference mode
```

Key CLI options: `--chatgpt-api-port`, `--discovery-module {udp|tailscale|manual}`, `--disable-tui`, `--models-seed-dir`

## Testing

```bash
# Inference engine tests
python3 -m exo.inference.test_inference_engine

# Tokenizer tests
python3 ./test/test_tokenizers.py

# Model helper tests
python3 ./test/test_model_helpers.py
```

Test environment variables:
- `DEBUG=9` - Full debug output
- `TINYGRAD_DEBUG=2` - tinygrad-specific debug
- `TEMPERATURE=0` - Deterministic sampling
- `VERBOSE=1` - Verbose tokenizer output

## Code Formatting

```bash
pip install -e '.[formatting]'
python3 format.py ./exo       # Format entire module
python3 format.py ./file.py   # Format single file
```

Uses YAPF with 2-space indentation and 200-char line limit (see `.style.yapf`).

## Architecture

### Core Components

```
exo/
├── main.py              # CLI entry point, argument parser
├── models.py            # Model definitions (layers, HuggingFace repos)
├── orchestration/
│   └── node.py          # Central Node class - manages distributed inference
├── inference/
│   ├── mlx/             # MLX engine (Apple Silicon GPU)
│   ├── tinygrad/        # tinygrad engine (cross-platform)
│   ├── rkllm/           # RKLLM engine (Rockchip RK3588 NPU)
│   └── shard.py         # Model shard representation
├── networking/
│   ├── grpc/            # gRPC peer communication
│   ├── udp/             # UDP device discovery
│   ├── tailscale/       # Tailscale discovery
│   └── manual/          # Manual topology config
├── topology/
│   └── ring_memory_weighted_partitioning_strategy.py  # Default partitioning
├── download/
│   └── new_shard_download.py  # HuggingFace model downloads
├── api/
│   └── chatgpt_api.py   # aiohttp ChatGPT-compatible server
└── tinychat/            # Web UI (Alpine.js)
```

### Key Abstractions

- **Node** (`orchestration/node.py`): Central orchestrator managing inference across the cluster, handles peer communication and topology
- **InferenceEngine**: Abstract interface for model inference (MLX, tinygrad, RKLLM, dummy implementations)
- **Shard**: Represents a partition of a model (start_layer, end_layer)
- **PartitioningStrategy**: Distributes model layers across devices based on memory/capabilities

### Adding a New Model

Edit `exo/models.py`:
```python
"model-name": {
  "layers": 32,
  "repo": {
    "MLXDynamicShardInferenceEngine": "hub-id",
    "TinygradDynamicShardInferenceEngine": "hub-id",
    "RKLLMInferenceEngine": "hub-id-for-tokenizer",  # For Rockchip NPU
  },
}
```

Note: RKLLM models require pre-converted `.rkllm` files in `~/RKLLAMA/models/`. Use rkllm-toolkit to convert models.

## Environment Variables

- `EXO_HOME` - Override default model cache directory (`~/.cache/exo/downloads/`)
- `HF_ENDPOINT` - Custom HuggingFace endpoint
- `DEBUG`, `DEBUG_DISCOVERY` - Debug verbosity levels
- `RKLLM_LIB_PATH` - Path to librkllmrt.so for RKLLM engine (default: `~/RKLLAMA/lib/librkllmrt.so`)

## Related Repositories

Forked repos for Rockchip NPU support:
- https://github.com/jfreed-dev/rkllama
- https://github.com/jfreed-dev/rkllm-toolkit

Turing Pi 2 cluster (test environment):
- https://github.com/jfreed-dev/turning-ansible-cluster

## Commit Messages

- Do NOT include any references to Claude, AI, or automated generation
- Do NOT include "Co-Authored-By" lines referencing Claude or Anthropic
- Keep commit messages concise and descriptive of the actual changes
