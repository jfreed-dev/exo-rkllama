"""Tests for the RKLLM engine plugin and the shared plugin registry hooks."""

import pytest

from exo.master.placement import INSTANCE_META_BACKENDS
from exo.shared import plugins
from exo.shared.models.model_cards import ModelCard, ModelTask
from exo.shared.types.backends import Backend
from exo.shared.types.common import ModelId, NodeId
from exo.shared.types.memory import Memory
from exo.shared.types.worker.instances import (
    InstanceId,
    InstanceMeta,
    MlxRingInstance,
    RkllmSingleNodeInstance,
)
from exo.shared.types.worker.runners import RunnerId, ShardAssignments
from exo.shared.types.worker.shards import PipelineShardMetadata
from exo.worker.engines.rkllm import plugin as rkllm_plugin_module
from exo.worker.engines.rkllm.plugin import RKLLM_PLUGIN

MODEL_ID = ModelId("qwen2.5-7b-rkllm")


def _assignments() -> ShardAssignments:
    card = ModelCard(
        model_id=MODEL_ID,
        storage_size=Memory.from_kb(1000),
        n_layers=4,
        hidden_size=64,
        supports_tensor=False,
        tasks=[ModelTask.TextGeneration],
        backends=[Backend.RkllmNpu],
    )
    runner_id = RunnerId()
    shard = PipelineShardMetadata(
        model_card=card,
        device_rank=0,
        world_size=1,
        start_layer=0,
        end_layer=card.n_layers,
        n_layers=card.n_layers,
    )
    return ShardAssignments(
        model_id=MODEL_ID,
        runner_to_shard={runner_id: shard},
        node_to_runner={NodeId(): runner_id},
    )


def test_registry_contains_rkllm_plugin_once() -> None:
    registered = [
        plugin
        for plugin in plugins.load_plugins()
        if plugin.backend == Backend.RkllmNpu
    ]
    assert registered == [RKLLM_PLUGIN]


def test_plugin_for_instance_meta() -> None:
    assert (
        plugins.plugin_for_instance_meta(InstanceMeta.RkllmSingleNode) is RKLLM_PLUGIN
    )
    assert plugins.plugin_for_instance_meta(InstanceMeta.MlxRing) is None
    assert plugins.plugin_for_instance_meta(InstanceMeta.MlxJaccl) is None


def test_placement_backends_include_plugin_meta() -> None:
    assert INSTANCE_META_BACKENDS[InstanceMeta.RkllmSingleNode] == [Backend.RkllmNpu]


def test_make_instance_builds_single_node_instance() -> None:
    assignments = _assignments()
    instance = RKLLM_PLUGIN.make_instance(InstanceId(), assignments)
    assert isinstance(instance, RkllmSingleNodeInstance)
    assert instance.shard_assignments == assignments


def test_owns_instance_matches_only_rkllm_instances() -> None:
    assignments = _assignments()
    rkllm_instance = RkllmSingleNodeInstance(
        instance_id=InstanceId(), shard_assignments=assignments
    )
    mlx_instance = MlxRingInstance(
        instance_id=InstanceId(),
        shard_assignments=assignments,
        hosts_by_node={},
        ephemeral_port=50000,
    )
    assert RKLLM_PLUGIN.owns_instance(rkllm_instance)
    assert not RKLLM_PLUGIN.owns_instance(mlx_instance)
    assert plugins.plugin_for_instance(rkllm_instance) is RKLLM_PLUGIN
    assert plugins.plugin_for_instance(mlx_instance) is None
    assert plugins.plugin_for_instance(None) is None


def test_detect_plugin_backends_follows_hardware_detection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(rkllm_plugin_module, "detect_rockchip_npu", lambda: True)
    assert plugins.detect_plugin_backends() == [Backend.RkllmNpu]

    monkeypatch.setattr(rkllm_plugin_module, "detect_rockchip_npu", lambda: False)
    assert plugins.detect_plugin_backends() == []


def test_detected_host_plugin_caches_detection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plugins.detected_host_plugin.cache_clear()
    try:
        monkeypatch.setattr(rkllm_plugin_module, "detect_rockchip_npu", lambda: True)
        assert plugins.detected_host_plugin() is RKLLM_PLUGIN
        # Cached: a later (impossible at runtime) hardware change is not observed.
        monkeypatch.setattr(rkllm_plugin_module, "detect_rockchip_npu", lambda: False)
        assert plugins.detected_host_plugin() is RKLLM_PLUGIN
    finally:
        plugins.detected_host_plugin.cache_clear()
