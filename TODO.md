# RKLLM Integration TODO List

**Last Updated:** 2025-12-27

## High Priority

### Bug Fixes
- [ ] **Fix rkllama upstream compatibility**
  - New rkllama (runtime 1.2.3) causes DeepSeek output corruption (&&&&& characters)
  - Currently using old version (d0392d7) with runtime 1.1.4
  - Need to investigate root cause or find compatible models

- [ ] **Report bug to rkllama upstream**
  - Document reproduction steps
  - Include model versions and runtime info
  - Submit issue at https://github.com/jfreed-dev/rkllama

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
- [ ] **Add streaming support to RKLLM HTTP client**
  - Currently only non-streaming mode works
  - Implement SSE/chunked response handling
  - Update `generate()` method in `rkllm_http_client.py`

### Multi-Node Support
- [ ] **Implement multi-node load balancing**
  - RKLLM can't do layer sharding
  - Use request-level parallelism instead
  - Add nginx/HAProxy configuration example

- [ ] **Add automatic model switching**
  - Switch models based on request type
  - Use Qwen for quick responses
  - Use DeepSeek for reasoning tasks

## DevOps

### Service Management
- [ ] **Create systemd service files**
  - `rkllama.service` for RKLLAMA server
  - `exo-rkllm.service` for Exo
  - Auto-restart on failure
  - Proper logging configuration

- [ ] **Add health monitoring**
  - Prometheus metrics endpoint
  - Grafana dashboard for NPU utilization
  - Alert on service failures

## Documentation

- [x] ~~Add benchmark results to README~~
- [x] ~~Document model comparison~~
- [x] ~~Add troubleshooting guide~~
- [ ] **Create deployment guide**
  - Step-by-step setup instructions
  - Network configuration
  - Security considerations

## Completed

- [x] Integrate RKLLM inference engine with exo
- [x] Implement token caching mechanism
- [x] Add Qwen2.5-1.5B-Instruct model support
- [x] Benchmark Qwen vs DeepSeek performance
- [x] Document benchmark results and findings

---

## Quick Reference

### Current Working Configuration
- **Runtime:** RKLLM SDK 1.1.4
- **rkllama:** d0392d7 (old version)
- **Models:** DeepSeek-R1-1.5B, Qwen2.5-1.5B-Instruct
- **Speed:** ~7.8-8.0 tok/s on RK3588 NPU

### Model Sources
- Qwen2.5-1.5B-Instruct: https://huggingface.co/c01zaut/Qwen2.5-1.5B-Instruct-RK3588-1.1.4
- RKLLM Toolkit: https://github.com/airockchip/rknn-llm/releases/tag/release-v1.1.4
