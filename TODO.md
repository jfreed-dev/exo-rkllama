# RKLLM Integration TODO List

**Last Updated:** 2025-12-28

## High Priority

### Bug Fixes
- [x] ~~**Fix rkllama upstream compatibility (RKLLM 1.2.3)**~~ **FIXED 2025-12-27**
  - Updated jfreed-dev/rkllama fork to support RKLLM 1.2.3
  - All ctypes structures updated to match new ABI
  - See commit `e5ecd35` in rkllama fork

  <details>
  <summary>Root Cause Analysis (resolved)</summary>

  The issue was **ABI incompatibility** - ctypes structure definitions didn't match the new `librkllmrt.so` binary. Key breaking changes:

  | Structure | 1.1.4 | 1.2.3 | Impact |
  |-----------|-------|-------|--------|
  | `RKLLMResult` | `text, size, last_hidden_layer` | `text, token_id, last_hidden_layer, logits, perf` | Memory layout mismatch |
  | `RKLLMExtendParam` | `base_domain_id, reserved[112]` | `+embed_flash, cpus_num, cpus_mask, n_batch, use_cross_attn` | Struct size change |
  | `RKLLMInput` | `input_mode, input_data` | `role, enable_thinking, input_type, input_data` | New fields prepended |
  | `RKLLMParam` | (old fields) | Added `n_keep`, `use_gpu` | Additional fields |

  </details>

## Known Issues

### DeepSeek-R1-1.5B Model Issues ⚠️
- **Status:** Understood - architectural limitation
- **Tested:** 2025-12-28

| Issue | Cause | Status |
|-------|-------|--------|
| `[PAD151935]` tokens | RKLLM-internal thinking tokens (ID 151935 > vocab max 151664) | ✅ Now filtered in rkllama callback |
| Long generation | DeepSeek-R1 generates 2000+ thinking tokens before answer | ⚠️ By design |
| Request timeouts | Default max_new_tokens=2048 exhausted by thinking | ✅ Added MAX_NEW_TOKENS to Modelfile |

**Root Cause Analysis (2025-12-28):**
- Token 151935 is an **RKLLM-internal token** added during model conversion
- HuggingFace tokenizer vocabulary max is 151664; token 151935 = 151664 + 271
- RKLLM runtime outputs `[PAD151935]` as fallback for undecoded tokens
- These represent DeepSeek's internal chain-of-thought reasoning markers
- Even "What is 2+2?" generates 2000+ thinking tokens before "4"

**Fixes Applied:**
- [x] ~~Increase `max_new_tokens` for DeepSeek~~ Added `MAX_NEW_TOKENS=8192` to Modelfile
- [x] ~~Implement thinking token filter~~ Updated rkllama callback to wrap `[PAD...]` in `<think>...</think>`
- [x] ~~Add MAX_NEW_TOKENS support~~ server.py now reads from Modelfile

**Remaining Limitations:**
- DeepSeek responses take 3-5+ minutes due to extensive thinking
- Not practical for interactive use; better for batch/reasoning tasks
- Recommend Qwen2.5-1.5B-Instruct for general use (~8 tok/s, instant responses)

## Modularization (Upstream Compatibility)

**Goal:** Make RKLLM a cleanly separable module that can be maintained independently of the main exo-explore/exo project, enabling easy upstream syncing.

### Phase 1: Consolidate RKLLM Configuration ✅ DONE (2025-12-28)
- [x] **Create `exo/inference/rkllm/models.py`**
  - Moved all RKLLM model definitions from `exo/models.py`
  - Includes `RKLLM_MODELS` dict and `RKLLM_PRETTY_NAMES` dict
  - Self-contained, no dependencies on core exo

- [x] **Create `exo/inference/rkllm/detection.py`**
  - Moved `detect_rockchip_npu()` from `exo/helpers.py`
  - Includes device-tree and library detection logic
  - Export function for use by helpers.py

- [x] **Create `exo/inference/rkllm/metrics.py`**
  - Moved RKLLM-specific Prometheus metrics from `exo/api/prometheus_metrics.py`
  - 5 metrics: `RKLLM_SERVER_UP`, `RKLLM_INFERENCE_SECONDS`, etc.
  - Imported conditionally in prometheus_metrics.py

- [x] **Update core files with conditional imports**
  - `exo/models.py`: Conditionally imports from rkllm/models.py
  - `exo/helpers.py`: Conditionally imports detection function
  - `exo/api/prometheus_metrics.py`: Conditional RKLLM metrics import
  - `exo/inference/rkllm/__init__.py`: Updated exports

### Phase 2: Upstream Plugin System PR
- [ ] **Draft PR for exo-explore/exo**
  - Add entry point discovery for inference engines
  - Add entry point discovery for model providers
  - Add entry point discovery for device detectors
  - See `docs/PR_PLUGIN_SYSTEM.md` for full proposal

- [ ] **Create reference implementation**
  - Demonstrate plugin pattern with dummy engine
  - Add documentation for third-party engine authors

### Phase 3: Extract to Separate Package
- [ ] **Create `exo-rkllm` PyPI package**
  - Standalone package with entry point registration
  - Zero changes required to upstream exo
  - `pip install exo exo-rkllm` enables RKLLM support

### Current Integration Points (to be modularized)
| File | Changes | Status |
|------|---------|--------|
| `exo/main.py` | CLI arg + auto-detect (2 lines) | Keep minimal |
| `exo/models.py` | 6 model defs (~28 lines) | → `rkllm/models.py` |
| `exo/inference/inference_engine.py` | Factory registration (3 lines) | Keep (or plugin) |
| `exo/helpers.py` | NPU detection (~25 lines) | → `rkllm/detection.py` |
| `exo/topology/device_capabilities.py` | Device specs (~30 lines) | → conditional |
| `exo/api/prometheus_metrics.py` | 5 metrics (~30 lines) | → `rkllm/metrics.py` |

---

## Model Expansion

### New Model Conversions
- [ ] **Set up x86_64 environment for model conversion**
  - RKLLM-Toolkit only works on x86_64
  - Need Docker or remote x86_64 machine
  - Toolkit: https://github.com/airockchip/rknn-llm (release-v1.1.4)

- [ ] **Convert Qwen2.5-3B model**
  - Larger model for better quality
  - Check if RK3588 has sufficient memory (~4GB+ needed)
  - Use w8a8 quantization

- [ ] **Convert Phi-3-mini model**
  - Microsoft's efficient small model
  - Good for instruction following
  - ~3.8B parameters

- [ ] **Benchmark larger models (3B, 7B)**
  - Test memory limits on RK3588 (8GB RAM)
  - Compare quality vs speed tradeoffs

## Feature Enhancements

### Streaming Support
- [x] ~~**Add streaming support to RKLLM HTTP client**~~ **DONE 2025-12-27**
  - Added `generate_stream()` async generator for SSE/chunked responses
  - Added `generate_from_prompt_stream()` for pre-templated prompts
  - Auto-enabled for DeepSeek models via `STREAMING_MODELS` set
  - Qwen models continue to use token caching (faster for short responses)

### Multi-Node Support
- [x] **Implement multi-node load balancing** (2025-12-27)
  - RKLLM can't do layer sharding
  - Use request-level parallelism instead
  - See `nginx/README.md` for configuration

- [ ] **Add automatic model switching**
  - Switch models based on request type
  - Use Qwen for quick responses
  - Use DeepSeek for reasoning tasks

## DevOps

### Service Management
- [x] **Create systemd service files**
  - `rkllama.service` for RKLLAMA server
  - `exo-rkllm.service` for Exo
  - Auto-restart on failure
  - Proper logging configuration
  - See `systemd/README.md` for usage

- [x] **Add health monitoring** (2025-12-27)
  - Prometheus metrics endpoint at `/metrics`
  - Request metrics: count, latency, tokens, errors
  - RKLLM-specific: server health, inference timing, model load time
  - See `exo/api/prometheus_metrics.py` for all metrics

## Documentation

- [x] ~~Add benchmark results to README~~
- [x] ~~Document model comparison~~
- [x] ~~Add troubleshooting guide~~
- [x] **Create deployment guide** (2025-12-27)
  - Step-by-step setup instructions
  - Network configuration
  - Security considerations
  - See `docs/DEPLOYMENT.md`
- [x] **Add architecture diagrams** (2025-12-27)
  - Network topology diagram (Mermaid)
  - Data flow for storage and monitoring
  - Component interaction diagram
  - See `docs/ARCHITECTURE.md`

## Completed

- [x] Integrate RKLLM inference engine with exo
- [x] Implement token caching mechanism
- [x] Add Qwen2.5-1.5B-Instruct model support
- [x] Benchmark Qwen vs DeepSeek performance
- [x] Document benchmark results and findings
- [x] Fix RKLLM 1.2.3 ABI compatibility in rkllama fork
- [x] Add streaming support to HTTP client and engine (2025-12-27)
- [x] Add Prometheus health monitoring (2025-12-27)
- [x] Create deployment guide (2025-12-27)
- [x] Add nginx load balancer config (2025-12-27)
- [x] Add architecture diagrams (2025-12-27)

---

## Quick Reference

### Current Working Configuration
- **Runtime:** RKLLM SDK 1.2.3
- **rkllama:** 31cc4d4 (jfreed-dev/rkllama v0.0.5)
- **Speed:** ~7.8-8.0 tok/s on RK3588 NPU

### Model Status
| Model | Status | Notes |
|-------|--------|-------|
| Qwen2.5-1.5B-Instruct | ✅ Working | Recommended for general use |
| DeepSeek-R1-1.5B | ⚠️ Issues | Long chain-of-thought, timeouts (see Known Issues) |

### Model Sources
- Qwen2.5-1.5B-Instruct: https://huggingface.co/c01zaut/Qwen2.5-1.5B-Instruct-RK3588-1.1.4
- RKLLM Toolkit: https://github.com/airockchip/rknn-llm/releases/tag/release-v1.2.3
