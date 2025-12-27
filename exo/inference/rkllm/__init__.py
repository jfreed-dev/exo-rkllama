"""
RKLLM Inference Engine for Rockchip RK3588/RK3576 NPU devices.

This module provides inference capabilities using the RKLLM runtime
for Rockchip NPU devices. The recommended mode is HTTP, which connects
to a running rkllama server.

Usage:
  # Start rkllama server on RK3588 device:
  python server.py --target_platform rk3588 --port 8080

  # Then use exo with rkllm engine:
  exo --inference-engine rkllm

Environment variables:
  RKLLM_SERVER_HOST: Host of rkllama server (default: localhost)
  RKLLM_SERVER_PORT: Port of rkllama server (default: 8080)
"""

from exo.inference.rkllm.rkllm_engine import RKLLMInferenceEngine
from exo.inference.rkllm.rkllm_http_client import RKLLMHTTPClient, RKLLMServerConfig

__all__ = ["RKLLMInferenceEngine", "RKLLMHTTPClient", "RKLLMServerConfig"]
