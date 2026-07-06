"""Engine plugin registry: the seam between shared exo code and engine ports.

An engine port (e.g. RKLLM on the Rockchip NPU) hooks into exo at a handful of
dispatch points: hardware detection, model-card ownership, placement, download
resolution, and runner builder selection. Shared code calls the generic hooks in
this module instead of naming concrete engines, so a port registers itself in
``load_plugins`` and leaves upstream files untouched apart from its wire-format
types (a ``Backend`` member, an ``InstanceMeta`` member, and an ``Instance``
union member, which must exist on every node regardless of hardware).

Current plugin semantics: a plugin engine runs a whole model on a single node
(no cross-node sharding). Placement pins plugin-owned cards to a single-node
pipeline instance of the plugin's ``InstanceMeta``.

This module stays import-light on purpose: engine modules are imported only
when ``load_plugins`` first runs, never when this module is imported.
"""

from __future__ import annotations

from functools import cache
from typing import TYPE_CHECKING, Protocol

from exo.shared.types.backends import Backend

if TYPE_CHECKING:
    from exo.shared.models.model_cards import ModelCard
    from exo.shared.types.common import NodeId
    from exo.shared.types.events import Event
    from exo.shared.types.tasks import TaskId
    from exo.shared.types.worker.downloads import DownloadCompleted, DownloadFailed
    from exo.shared.types.worker.instances import Instance, InstanceId, InstanceMeta
    from exo.shared.types.worker.runners import ShardAssignments, ShardMetadata
    from exo.utils.channels import MpReceiver, MpSender
    from exo.worker.engines.base import Builder


class EnginePlugin(Protocol):
    """Hooks an engine port implements to run under exo without editing shared code."""

    @property
    def backend(self) -> Backend:
        """The backend this engine provides; hosts that pass ``detect`` report it."""
        ...

    @property
    def instance_meta(self) -> InstanceMeta:
        """The instance meta placement pins plugin-owned cards to."""
        ...

    @property
    def hf_search_filter(self) -> str:
        """HuggingFace ``list_models`` library filter for models this engine can run."""
        ...

    def detect(self) -> bool:
        """Whether this engine's hardware/runtime is present on the current host."""
        ...

    def owns_instance(self, instance: Instance) -> bool:
        """Whether a placed instance belongs to this engine."""
        ...

    def make_instance(
        self, instance_id: InstanceId, shard_assignments: ShardAssignments
    ) -> Instance:
        """Build this engine's instance for a single-node pipeline placement."""
        ...

    def make_builder(
        self, event_sender: MpSender[Event], cancel_receiver: MpReceiver[TaskId]
    ) -> Builder:
        """Build the runner's engine builder.

        Runs in the runner subprocess; implementations import their engine code
        lazily here so unselected engines are never loaded.
        """
        ...

    def resolve_download(
        self, node_id: NodeId, shard_metadata: ShardMetadata
    ) -> DownloadCompleted | DownloadFailed:
        """Resolve a ``DownloadModel`` task for an engine-managed artifact.

        Plugin-owned cards are never fetched from HuggingFace; the engine
        locates (or refuses) its own artifacts.
        """
        ...


@cache
def load_plugins() -> tuple[EnginePlugin, ...]:
    """All registered engine plugins.

    In-tree registration: add an import here. Imports are function-level so
    that importing this module never pulls engine code eagerly.
    """
    from exo.worker.engines.rkllm.plugin import RKLLM_PLUGIN

    return (RKLLM_PLUGIN,)


def detect_plugin_backends() -> list[Backend]:
    """Backends whose engine hardware/runtime is present on this host."""
    return [plugin.backend for plugin in load_plugins() if plugin.detect()]


@cache
def detected_host_plugin() -> EnginePlugin | None:
    """The plugin whose hardware this host runs on, if any.

    Cached because the host hardware does not change at runtime. Hosts detected
    as plugin hardware serve only that engine's model catalog.
    """
    for plugin in load_plugins():
        if plugin.detect():
            return plugin
    return None


def plugin_for_card(card: ModelCard) -> EnginePlugin | None:
    """The plugin that exclusively owns a model card, if any.

    A card belongs to a plugin when the plugin's backend is its only backend:
    such cards are hand-written for that engine's artifact format. A card that
    merely lists the backend among others keeps generic handling.
    """
    card_backends = set(card.backends)
    for plugin in load_plugins():
        if card_backends == {plugin.backend}:
            return plugin
    return None


def plugin_for_instance(instance: Instance | None) -> EnginePlugin | None:
    """The plugin that owns a placed instance, if any."""
    if instance is None:
        return None
    for plugin in load_plugins():
        if plugin.owns_instance(instance):
            return plugin
    return None


def plugin_for_instance_meta(instance_meta: InstanceMeta) -> EnginePlugin | None:
    """The plugin providing an instance meta, if any."""
    for plugin in load_plugins():
        if plugin.instance_meta == instance_meta:
            return plugin
    return None


def fetched_card_backends() -> list[Backend]:
    """Backends assigned to cards fetched from HuggingFace.

    Everything except plugin backends: those need pre-converted artifacts and
    hand-written cards, never a raw safetensors repo.
    """
    plugin_backends = {plugin.backend for plugin in load_plugins()}
    return [backend for backend in Backend if backend not in plugin_backends]
