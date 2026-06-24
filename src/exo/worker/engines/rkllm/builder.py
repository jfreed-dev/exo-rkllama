"""Builder for the RKLLM engine.

Single-node, whole-model: ``connect`` selects the backend (no distributed group),
``load`` readies the model on the chosen backend and yields coarse progress, and
``build`` returns the :class:`RkllmEngine`.
"""

from collections.abc import Generator
from dataclasses import dataclass

from exo.shared.types.events import Event
from exo.shared.types.tasks import TaskId
from exo.shared.types.worker.instances import BoundInstance
from exo.shared.types.worker.runner_response import ModelLoadingResponse
from exo.shared.types.worker.shards import ShardMetadata
from exo.utils.channels import MpReceiver, MpSender
from exo.worker.engines.base import Builder, Engine
from exo.worker.engines.rkllm.backend import RkllmBackend, select_backend
from exo.worker.engines.rkllm.engine import RkllmEngine


@dataclass
class RkllmBuilder(Builder):
    event_sender: MpSender[Event]
    cancel_receiver: MpReceiver[TaskId]
    shard_metadata: ShardMetadata | None = None
    backend: RkllmBackend | None = None

    def connect(self, bound_instance: BoundInstance) -> None:
        # Whole-model on one NPU: no MLX distributed group to initialize.
        self.backend = select_backend()

    def load(self, bound_instance: BoundInstance) -> Generator[ModelLoadingResponse]:
        assert self.backend is not None
        self.shard_metadata = bound_instance.bound_shard
        yield from self.backend.load(bound_instance.bound_shard.model_card)

    def build(self) -> Engine:
        assert self.backend is not None
        assert self.shard_metadata is not None
        return RkllmEngine(
            self.backend,
            self.shard_metadata,
            self.cancel_receiver,
        )

    def close(self) -> None:
        if self.backend is not None:
            self.backend.close()
