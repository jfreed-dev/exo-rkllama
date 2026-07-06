"""Tests for local .rkllm artifact discovery and download resolution."""

from pathlib import Path

import pytest

from exo.shared.models.model_cards import (
    ModelCard,
    ModelTask,
)
from exo.shared.plugins import (
    fetched_card_backends,
    plugin_for_card,
    plugin_for_instance,
)
from exo.shared.types.backends import Backend
from exo.shared.types.common import ModelId, NodeId
from exo.shared.types.memory import Memory
from exo.shared.types.worker.downloads import DownloadCompleted, DownloadFailed
from exo.shared.types.worker.instances import (
    BoundInstance,
    InstanceId,
    MlxRingInstance,
    RkllmSingleNodeInstance,
)
from exo.shared.types.worker.runners import RunnerId, ShardAssignments
from exo.shared.types.worker.shards import PipelineShardMetadata
from exo.worker.engines.rkllm import models

MODEL_ID = ModelId("qwen2.5-7b-rkllm")


def _card(backends: list[Backend]) -> ModelCard:
    return ModelCard(
        model_id=MODEL_ID,
        storage_size=Memory.from_kb(1000),
        n_layers=4,
        hidden_size=64,
        supports_tensor=False,
        tasks=[ModelTask.TextGeneration],
        backends=backends,
    )


def _shard(backends: list[Backend]) -> PipelineShardMetadata:
    card = _card(backends)
    return PipelineShardMetadata(
        model_card=card,
        device_rank=0,
        world_size=1,
        start_layer=0,
        end_layer=card.n_layers,
        n_layers=card.n_layers,
    )


@pytest.fixture
def isolated_dirs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Path]:
    exo_dir = tmp_path / "exo-models"
    rkllama_dir = tmp_path / "rkllama-models"
    exo_dir.mkdir()
    rkllama_dir.mkdir()
    monkeypatch.setattr(models, "EXO_MODELS_DIRS", (exo_dir,))
    monkeypatch.setattr(models, "EXO_MODELS_READ_ONLY_DIRS", ())
    monkeypatch.setattr(models, "RKLLAMA_MODELS_DIR", rkllama_dir)
    monkeypatch.delenv("RKLLM_MODEL_PATH", raising=False)
    return {"exo": exo_dir, "rkllama": rkllama_dir}


@pytest.mark.usefixtures("isolated_dirs")
def test_find_returns_none_when_nothing_present() -> None:
    assert models.find_rkllm_model_file(MODEL_ID) is None


def test_find_in_exo_models_dir(isolated_dirs: dict[str, Path]) -> None:
    model_dir = isolated_dirs["exo"] / MODEL_ID.normalize()
    model_dir.mkdir()
    artifact = model_dir / "model.rkllm"
    artifact.write_bytes(b"stub")

    assert models.find_rkllm_model_file(MODEL_ID) == artifact


def test_find_in_rkllama_models_dir(isolated_dirs: dict[str, Path]) -> None:
    model_dir = isolated_dirs["rkllama"] / MODEL_ID.normalize()
    model_dir.mkdir()
    artifact = model_dir / "model.rkllm"
    artifact.write_bytes(b"stub")

    assert models.find_rkllm_model_file(MODEL_ID) == artifact


@pytest.mark.usefixtures("isolated_dirs")
def test_find_env_override_wins(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact = tmp_path / "elsewhere.rkllm"
    artifact.write_bytes(b"stub")
    monkeypatch.setenv("RKLLM_MODEL_PATH", str(artifact))

    assert models.find_rkllm_model_file(MODEL_ID) == artifact


def test_find_ignores_dir_without_rkllm_file(
    isolated_dirs: dict[str, Path],
) -> None:
    model_dir = isolated_dirs["exo"] / MODEL_ID.normalize()
    model_dir.mkdir()
    (model_dir / "README.md").write_text("no artifact here")

    assert models.find_rkllm_model_file(MODEL_ID) is None


def test_resolve_download_completes_with_local_artifact(
    isolated_dirs: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("EXO_RKLLM_BACKEND", "ctypes")
    model_dir = isolated_dirs["exo"] / MODEL_ID.normalize()
    model_dir.mkdir()
    (model_dir / "model.rkllm").write_bytes(b"stub")

    progress = models.resolve_rkllm_download(NodeId(), _shard([Backend.RkllmNpu]))

    assert isinstance(progress, DownloadCompleted)
    assert progress.model_directory == str(model_dir)
    assert progress.read_only is True


@pytest.mark.usefixtures("isolated_dirs")
def test_resolve_download_trusts_server_for_http_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("EXO_RKLLM_BACKEND", "http")

    progress = models.resolve_rkllm_download(NodeId(), _shard([Backend.RkllmNpu]))

    assert isinstance(progress, DownloadCompleted)


@pytest.mark.usefixtures("isolated_dirs")
def test_resolve_download_fails_for_ctypes_without_artifact(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("EXO_RKLLM_BACKEND", "ctypes")

    progress = models.resolve_rkllm_download(NodeId(), _shard([Backend.RkllmNpu]))

    assert isinstance(progress, DownloadFailed)
    assert "RKLLM_MODEL_PATH" in progress.error_message


def test_card_owned_by_plugin_only_for_rkllm_only_backends() -> None:
    rkllm_plugin = plugin_for_card(_card([Backend.RkllmNpu]))
    assert rkllm_plugin is not None
    assert rkllm_plugin.backend == Backend.RkllmNpu
    assert plugin_for_card(_card([Backend.MlxMetal])) is None
    # A permissive card listing every backend (e.g. an old fetched card) is not
    # treated as an RKLLM model.
    assert plugin_for_card(_card(list(Backend))) is None


def test_fetched_card_backends_exclude_rkllm() -> None:
    # The default used by ModelCard.fetch_from_hf: an arbitrary HF safetensors
    # repo can never run on the RKLLM engine.
    assert Backend.RkllmNpu not in fetched_card_backends()
    assert plugin_for_card(_card(fetched_card_backends())) is None


@pytest.mark.usefixtures("isolated_dirs")
def test_env_pointing_at_missing_file_fails_loudly(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("EXO_RKLLM_BACKEND", "ctypes")
    monkeypatch.setenv("RKLLM_MODEL_PATH", str(tmp_path / "missing.rkllm"))

    with pytest.raises(ValueError, match="RKLLM_MODEL_PATH"):
        models.find_rkllm_model_file(MODEL_ID)

    progress = models.resolve_rkllm_download(NodeId(), _shard([Backend.RkllmNpu]))
    assert isinstance(progress, DownloadFailed)
    assert "RKLLM_MODEL_PATH" in progress.error_message
    assert "does not point at" in progress.error_message


@pytest.mark.usefixtures("isolated_dirs")
def test_invalid_backend_env_reports_real_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("EXO_RKLLM_BACKEND", "htpp")

    progress = models.resolve_rkllm_download(NodeId(), _shard([Backend.RkllmNpu]))

    assert isinstance(progress, DownloadFailed)
    assert "EXO_RKLLM_BACKEND" in progress.error_message


def test_bound_instance_dispatch_follows_instance_type() -> None:
    # Engine dispatch keys off the placed instance type, not the card's backends:
    # a permissive card on an MLX placement must not select the RKLLM engine.
    runner_id = RunnerId()
    node_id = NodeId()
    assignments = ShardAssignments(
        model_id=MODEL_ID,
        runner_to_shard={runner_id: _shard(list(Backend))},
        node_to_runner={node_id: runner_id},
    )
    mlx_bound = BoundInstance(
        instance=MlxRingInstance(
            instance_id=InstanceId(),
            shard_assignments=assignments,
            hosts_by_node={},
            ephemeral_port=50000,
        ),
        bound_runner_id=runner_id,
        bound_node_id=node_id,
    )
    assert plugin_for_instance(mlx_bound.instance) is None

    rkllm_bound = BoundInstance(
        instance=RkllmSingleNodeInstance(
            instance_id=InstanceId(),
            shard_assignments=assignments,
        ),
        bound_runner_id=runner_id,
        bound_node_id=node_id,
    )
    rkllm_plugin = plugin_for_instance(rkllm_bound.instance)
    assert rkllm_plugin is not None
    assert rkllm_plugin.backend == Backend.RkllmNpu
