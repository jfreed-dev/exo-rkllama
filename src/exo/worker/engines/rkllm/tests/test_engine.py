from collections.abc import Iterable, Iterator

import pytest

from exo.shared.models.model_cards import ModelCard, ModelId, ModelTask
from exo.shared.types.backends import Backend
from exo.shared.types.chunks import Chunk, TokenChunk
from exo.shared.types.common import CommandId
from exo.shared.types.memory import Memory
from exo.shared.types.tasks import TaskId, TextGeneration
from exo.shared.types.text_generation import (
    InputMessage,
    InputMessageContent,
    TextGenerationTaskParams,
)
from exo.shared.types.worker.instances import InstanceId
from exo.shared.types.worker.runner_response import (
    CancelledResponse,
    FinishedResponse,
    ModelLoadingResponse,
)
from exo.shared.types.worker.shards import PipelineShardMetadata
from exo.utils.channels import mp_channel
from exo.worker.engines.rkllm.backend import RkllmBackend, TokenPiece
from exo.worker.engines.rkllm.ctypes_backend import RkllmCtypesBackend
from exo.worker.engines.rkllm.engine import RkllmEngine

Response = Chunk | CancelledResponse | FinishedResponse


class FakeBackend(RkllmBackend):
    def __init__(self, pieces: list[TokenPiece]) -> None:
        self.pieces: list[TokenPiece] = pieces
        self.cancelled: bool = False
        self.closed: bool = False

    def load(self, model_card: ModelCard) -> Iterable[ModelLoadingResponse]:
        yield ModelLoadingResponse(
            layers_loaded=model_card.n_layers, total=model_card.n_layers
        )

    def generate(self, params: TextGenerationTaskParams) -> Iterator[TokenPiece]:
        yield from self.pieces

    def cancel(self) -> None:
        self.cancelled = True

    def close(self) -> None:
        self.closed = True


def _model_card() -> ModelCard:
    return ModelCard(
        model_id=ModelId("qwen2.5-7b-rkllm"),
        storage_size=Memory.from_kb(1000),
        n_layers=4,
        hidden_size=64,
        supports_tensor=False,
        tasks=[ModelTask.TextGeneration],
        backends=[Backend.RkllmNpu],
    )


def _task(max_output_tokens: int | None = None) -> TextGeneration:
    return TextGeneration(
        instance_id=InstanceId(),
        command_id=CommandId(),
        task_params=TextGenerationTaskParams(
            model=ModelId("qwen2.5-7b-rkllm"),
            input=[InputMessage(role="user", content=InputMessageContent("hi"))],
            max_output_tokens=max_output_tokens,
        ),
    )


def _make_engine(backend: RkllmBackend) -> RkllmEngine:
    card = _model_card()
    shard = PipelineShardMetadata(
        model_card=card,
        device_rank=0,
        world_size=1,
        start_layer=0,
        end_layer=card.n_layers,
        n_layers=card.n_layers,
    )
    _, cancel_recv = mp_channel[TaskId]()
    return RkllmEngine(backend, shard, cancel_recv)


def _drain(engine: RkllmEngine, task: TextGeneration) -> list[Response]:
    engine.submit(task)
    out: list[Response] = []
    for _ in range(100):
        responses = list(engine.step())
        out.extend(resp for _task_id, resp in responses)
        if any(isinstance(resp, (FinishedResponse, CancelledResponse)) for resp in out):
            break
    return out


def test_engine_streams_then_finishes() -> None:
    backend = FakeBackend(
        [
            TokenPiece(text="Hello", token_id=0, finished=False),
            TokenPiece(text=" world", token_id=1, finished=True, finish_reason="stop"),
        ]
    )
    engine = _make_engine(backend)
    out = _drain(engine, _task())

    tokens = [r for r in out if isinstance(r, TokenChunk)]
    assert [t.text for t in tokens] == ["Hello", " world"]
    assert tokens[-1].finish_reason == "stop"
    assert isinstance(out[-1], FinishedResponse)


def test_engine_cancellation() -> None:
    backend = FakeBackend(
        [TokenPiece(text=f"t{i}", token_id=i, finished=False) for i in range(50)]
    )
    engine = _make_engine(backend)
    task = _task()
    # Pre-seed the cancellation set so the first step observes it.
    engine._cancelled_tasks.add(task.task_id)  # pyright: ignore[reportPrivateUsage]

    out = _drain(engine, task)

    assert any(isinstance(r, CancelledResponse) for r in out)
    assert not any(isinstance(r, FinishedResponse) for r in out)
    assert backend.cancelled is True


def test_engine_respects_max_output_tokens() -> None:
    backend = FakeBackend(
        [TokenPiece(text=f"t{i}", token_id=i, finished=False) for i in range(5)]
    )
    engine = _make_engine(backend)

    out = _drain(engine, _task(max_output_tokens=2))

    tokens = [r for r in out if isinstance(r, TokenChunk)]
    assert [t.text for t in tokens] == ["t0", "t1", ""]
    assert tokens[-1].finish_reason == "length"
    assert backend.cancelled is True
    assert isinstance(out[-1], FinishedResponse)


def test_engine_passes_through_backend_finish_reason() -> None:
    backend = FakeBackend(
        [TokenPiece(text="t", token_id=0, finished=True, finish_reason="length")]
    )
    engine = _make_engine(backend)

    out = _drain(engine, _task())

    tokens = [r for r in out if isinstance(r, TokenChunk)]
    assert tokens[-1].finish_reason == "length"
    assert isinstance(out[-1], FinishedResponse)


def test_select_backend_ctypes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EXO_RKLLM_BACKEND", "ctypes")
    from exo.worker.engines.rkllm.backend import select_backend

    assert isinstance(select_backend(), RkllmCtypesBackend)


def test_select_backend_invalid(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EXO_RKLLM_BACKEND", "bogus")
    from exo.worker.engines.rkllm.backend import select_backend

    with pytest.raises(ValueError):
        _ = select_backend()
