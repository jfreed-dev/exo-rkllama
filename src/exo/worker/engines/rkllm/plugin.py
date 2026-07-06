"""RKLLM engine plugin: registers the Rockchip NPU port with exo's registry.

Registered in ``exo.shared.plugins.load_plugins``. Shared code dispatches to
this plugin through the generic registry hooks, keeping RKLLM specifics out of
upstream files.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, final

from exo.shared.types.backends import Backend
from exo.shared.types.worker.instances import (
    Instance,
    InstanceId,
    InstanceMeta,
    RkllmSingleNodeInstance,
)
from exo.worker.engines.rkllm.detection import detect_rockchip_npu

if TYPE_CHECKING:
    from exo.shared.types.common import NodeId
    from exo.shared.types.events import Event
    from exo.shared.types.tasks import TaskId
    from exo.shared.types.worker.downloads import DownloadCompleted, DownloadFailed
    from exo.shared.types.worker.runners import ShardAssignments, ShardMetadata
    from exo.utils.channels import MpReceiver, MpSender
    from exo.worker.engines.base import Builder


@final
class RkllmEnginePlugin:
    """Whole-model inference on a single Rockchip RK3588/RK3576 NPU node."""

    @property
    def backend(self) -> Backend:
        return Backend.RkllmNpu

    @property
    def instance_meta(self) -> InstanceMeta:
        return InstanceMeta.RkllmSingleNode

    @property
    def hf_search_filter(self) -> str:
        return "rkllm"

    def detect(self) -> bool:
        return detect_rockchip_npu()

    def owns_instance(self, instance: Instance) -> bool:
        return isinstance(instance, RkllmSingleNodeInstance)

    def make_instance(
        self, instance_id: InstanceId, shard_assignments: ShardAssignments
    ) -> Instance:
        # Single-node pipeline assignment: one shard, world_size 1, rank 0.
        return RkllmSingleNodeInstance(
            instance_id=instance_id, shard_assignments=shard_assignments
        )

    def make_builder(
        self, event_sender: MpSender[Event], cancel_receiver: MpReceiver[TaskId]
    ) -> Builder:
        # Lazy import: this runs in the runner subprocess, which loads engine
        # code only once the engine is selected.
        from exo.worker.engines.rkllm.builder import RkllmBuilder

        return RkllmBuilder(event_sender, cancel_receiver)

    def resolve_download(
        self, node_id: NodeId, shard_metadata: ShardMetadata
    ) -> DownloadCompleted | DownloadFailed:
        from exo.worker.engines.rkllm.models import resolve_rkllm_download

        return resolve_rkllm_download(node_id, shard_metadata)


RKLLM_PLUGIN = RkllmEnginePlugin()
