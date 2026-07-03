"""Regression test: single-node instances never receive ConnectToGroup, so the
builder must be able to load without connect() having run first."""

from collections.abc import Iterable, Iterator

import pytest

from exo.shared.models.model_cards import ModelCard, ModelId, ModelTask
from exo.shared.types.backends import Backend
from exo.shared.types.common import NodeId
from exo.shared.types.events import Event
from exo.shared.types.memory import Memory
from exo.shared.types.tasks import TaskId
from exo.shared.types.text_generation import TextGenerationTaskParams
from exo.shared.types.worker.instances import (
    BoundInstance,
    InstanceId,
    RkllmSingleNodeInstance,
)
from exo.shared.types.worker.runner_response import ModelLoadingResponse
from exo.shared.types.worker.runners import RunnerId, ShardAssignments
from exo.shared.types.worker.shards import PipelineShardMetadata
from exo.utils.channels import mp_channel
from exo.worker.engines.rkllm import builder as builder_module
from exo.worker.engines.rkllm.backend import RkllmBackend, TokenPiece
from exo.worker.engines.rkllm.builder import RkllmBuilder


class FakeBackend(RkllmBackend):
    def __init__(self) -> None:
        self.loaded: bool = False

    def load(self, model_card: ModelCard) -> Iterable[ModelLoadingResponse]:
        self.loaded = True
        yield ModelLoadingResponse(
            layers_loaded=model_card.n_layers, total=model_card.n_layers
        )

    def generate(self, params: TextGenerationTaskParams) -> Iterator[TokenPiece]:
        yield TokenPiece(text="", token_id=0, finished=True, finish_reason="stop")

    def cancel(self) -> None:
        pass

    def close(self) -> None:
        pass


def _bound_instance() -> BoundInstance:
    card = ModelCard(
        model_id=ModelId("llama3.2-3b-rkllm"),
        storage_size=Memory.from_kb(1000),
        n_layers=4,
        hidden_size=64,
        supports_tensor=False,
        tasks=[ModelTask.TextGeneration],
        backends=[Backend.RkllmNpu],
    )
    shard = PipelineShardMetadata(
        model_card=card,
        device_rank=0,
        world_size=1,
        start_layer=0,
        end_layer=card.n_layers,
        n_layers=card.n_layers,
    )
    runner_id = RunnerId()
    node_id = NodeId()
    return BoundInstance(
        instance=RkllmSingleNodeInstance(
            instance_id=InstanceId(),
            shard_assignments=ShardAssignments(
                model_id=card.model_id,
                runner_to_shard={runner_id: shard},
                node_to_runner={node_id: runner_id},
            ),
        ),
        bound_runner_id=runner_id,
        bound_node_id=node_id,
    )


def test_load_without_connect_selects_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = FakeBackend()
    monkeypatch.setattr(builder_module, "select_backend", lambda: fake)
    event_sender, _ = mp_channel[Event]()
    _, cancel_receiver = mp_channel[TaskId]()
    rkllm_builder = RkllmBuilder(event_sender, cancel_receiver)

    # No connect() call: the worker plan skips ConnectToGroup for world size 1.
    responses = list(rkllm_builder.load(_bound_instance()))

    assert fake.loaded
    assert responses[-1].layers_loaded == responses[-1].total
    engine = rkllm_builder.build()
    assert engine is not None
