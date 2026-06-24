from pathlib import Path

import pytest

from exo.worker.engines.rkllm import detection
from exo.worker.engines.rkllm.detection import (
    detect_rockchip_npu,
    get_rkllm_library_path,
    get_rockchip_soc_name,
)


def test_detect_false_off_npu(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(detection, "_COMPATIBLE_PATH", str(tmp_path / "absent"))
    monkeypatch.setattr(detection, "RKLLM_LIB_PATHS", [str(tmp_path / "absent.so")])
    monkeypatch.delenv("RKLLM_LIB_PATH", raising=False)
    assert detect_rockchip_npu() is False
    assert get_rockchip_soc_name() == "Unknown"
    assert get_rkllm_library_path() is None


def test_detect_true_on_rk3588(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    compatible = tmp_path / "compatible"
    compatible.write_bytes(b"rockchip,rk3588\x00rockchip,rk3588-evb\x00")
    monkeypatch.setattr(detection, "_COMPATIBLE_PATH", str(compatible))
    assert detect_rockchip_npu() is True
    assert get_rockchip_soc_name() == "RK3588"


def test_library_path_from_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    lib = tmp_path / "librkllmrt.so"
    lib.write_bytes(b"")
    monkeypatch.setenv("RKLLM_LIB_PATH", str(lib))
    assert get_rkllm_library_path() == str(lib)
