"""Rockchip NPU detection for the RKLLM inference engine.

Detects RK3588/RK3576 SoCs via the device tree or the presence of the RKLLM
runtime library. Safe to call on any platform: returns ``False`` off-NPU.
"""

import os

# Known RKLLM runtime library locations.
RKLLM_LIB_PATHS: list[str] = [
    os.path.expanduser("~/RKLLAMA/lib/librkllmrt.so"),
    "/usr/lib/librkllmrt.so",
    "/usr/local/lib/librkllmrt.so",
    "/usr/lib/aarch64-linux-gnu/librkllmrt.so",
]

# Rockchip SoC identifiers with NPU support that RKLLM targets.
ROCKCHIP_NPU_SOCS: list[str] = ["rk3588", "rk3576"]

_COMPATIBLE_PATH = "/proc/device-tree/compatible"


def _read_device_tree_compatible() -> str | None:
    if not os.path.exists(_COMPATIBLE_PATH):
        return None
    try:
        with open(_COMPATIBLE_PATH, "rb") as f:
            return f.read().decode("utf-8", errors="ignore").lower()
    except OSError:
        return None


def detect_rockchip_npu() -> bool:
    """Return True when running on a Rockchip RK3588/RK3576 with NPU support.

    Detection order: device-tree ``compatible`` string, then a fallback check for
    the RKLLM runtime library at known paths.
    """
    compatible = _read_device_tree_compatible()
    if compatible is not None and any(soc in compatible for soc in ROCKCHIP_NPU_SOCS):
        return True

    return any(os.path.exists(path) for path in RKLLM_LIB_PATHS)


def get_rockchip_soc_name() -> str:
    """Return the detected Rockchip SoC name (e.g. ``"RK3588"``) or ``"Unknown"``."""
    compatible = _read_device_tree_compatible()
    if compatible is not None:
        for soc in ROCKCHIP_NPU_SOCS:
            if soc in compatible:
                return soc.upper()

    if any(os.path.exists(path) for path in RKLLM_LIB_PATHS):
        return "Rockchip (Unknown SoC)"

    return "Unknown"


def get_rkllm_library_path() -> str | None:
    """Return the path to ``librkllmrt.so`` if present, else ``None``."""
    env_path = os.environ.get("RKLLM_LIB_PATH")
    if env_path and os.path.exists(env_path):
        return env_path
    for path in RKLLM_LIB_PATHS:
        if os.path.exists(path):
            return path
    return None
