"""RKLLM engine: drives a single whole-model NPU generation at a time.

Mirrors the single-task loop of the image engine: ``submit`` enqueues a text task,
``step`` advances the active generation and yields one response per call. There is no
batching (the NPU runs one model, one stream) and no disaggregated prefill.
"""

from collections import deque
from collections.abc import Generator, Iterable
from dataclasses import dataclass, field
from typing import BinaryIO

from loguru import logger

from exo.shared.types.chunks import Chunk, ErrorChunk, TokenChunk
from exo.shared.types.tasks import GenerationTask, TaskId, TextGeneration
from exo.shared.types.text_generation import TextGenerationTaskParams
from exo.shared.types.worker.runner_response import (
    CancelledResponse,
    FinishedResponse,
)
from exo.shared.types.worker.shards import ShardMetadata
from exo.utils.channels import MpReceiver
from exo.worker.disaggregated.server import PrefillRequest
from exo.worker.engines.base import Engine
from exo.worker.engines.rkllm.backend import RkllmBackend


@dataclass
class RkllmEngine(Engine):
    backend: RkllmBackend
    shard_metadata: ShardMetadata
    cancel_receiver: MpReceiver[TaskId]
    current_gen: (
        Generator[tuple[TaskId, Chunk | FinishedResponse | CancelledResponse]] | None
    ) = field(init=False, default=None)
    queue: deque[TextGeneration] = field(init=False, default_factory=deque)
    _cancelled_tasks: set[TaskId] = field(init=False, default_factory=set)

    def warmup(self) -> None:
        # The model is already initialized during load(); nothing to calibrate.
        logger.info("RKLLM engine ready")

    def submit(self, task: GenerationTask) -> None:
        assert isinstance(task, TextGeneration), (
            f"RKLLM engine only handles text generation, got {type(task).__name__}"
        )
        self.queue.append(task)

    def step(
        self,
    ) -> Iterable[tuple[TaskId, Chunk | CancelledResponse | FinishedResponse]]:
        resp = None
        if self.current_gen is not None:
            resp = next(self.current_gen, None)
        if resp is None and len(self.queue) > 0:
            task = self.queue.popleft()
            self.current_gen = self._run_text_task(task.task_id, task.task_params)
            resp = next(self.current_gen, None)
        return (resp,) if resp is not None else ()

    def close(self) -> None:
        self.backend.close()

    def serve_prefill(self, request: PrefillRequest, wfile: BinaryIO) -> None:
        raise NotImplementedError() from None

    def _run_text_task(
        self,
        task_id: TaskId,
        task_params: TextGenerationTaskParams,
    ) -> Generator[tuple[TaskId, Chunk | FinishedResponse | CancelledResponse]]:
        model_id = self.shard_metadata.model_card.model_id

        def drain_cancels() -> None:
            for cancel_id in self.cancel_receiver.collect():
                self._cancelled_tasks.add(cancel_id)

        cancelled = False
        try:
            for piece in self.backend.generate(task_params):
                drain_cancels()
                if self.should_cancel(task_id):
                    self.backend.cancel()
                    cancelled = True
                    yield (task_id, CancelledResponse())
                    return
                if piece.text or piece.finished:
                    yield (
                        task_id,
                        TokenChunk(
                            model=model_id,
                            text=piece.text,
                            token_id=piece.token_id,
                            usage=None,
                            finish_reason="stop" if piece.finished else None,
                        ),
                    )
        except Exception as e:
            logger.exception("RKLLM generation failed")
            yield (
                task_id,
                ErrorChunk(model=model_id, error_message=str(e)),
            )
            raise
        finally:
            if not cancelled:
                yield (task_id, FinishedResponse())
