"""RKLLM inference engine for Rockchip RK3588/RK3576 NPUs.

RKLLM loads a complete model and exchanges tokens (not hidden states), so it runs
whole-model on a single NPU node — there is no cross-node layer sharding. The engine
plugs into the worker runner like the MLX and image engines, selecting between an HTTP
backend (a separate ``rkllama`` server) and an in-process ``ctypes`` backend
(``librkllmrt.so``) via the ``EXO_RKLLM_BACKEND`` environment variable.
"""

from exo.worker.engines.rkllm.detection import detect_rockchip_npu

__all__ = ["detect_rockchip_npu"]
