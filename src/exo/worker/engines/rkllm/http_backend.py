"""HTTP backend: drives a rkllama server, exposing a synchronous streaming API.

The runner calls the engine synchronously, so this backend owns a private asyncio
event loop on a daemon thread and bridges each async call back to sync code.
"""

import asyncio
import threading
from collections.abc import Coroutine, Iterable, Iterator

from loguru import logger

from exo.shared.models.model_cards import ModelCard
from exo.shared.types.text_generation import TextGenerationTaskParams
from exo.shared.types.worker.runner_response import ModelLoadingResponse
from exo.worker.engines.rkllm.backend import RkllmBackend, TokenPiece
from exo.worker.engines.rkllm.http_client import RKLLMHTTPClient, RKLLMServerConfig


class RkllmHttpBackend(RkllmBackend):
    def __init__(self, config: RKLLMServerConfig | None = None) -> None:
        self._client: RKLLMHTTPClient = RKLLMHTTPClient(
            config or RKLLMServerConfig.from_env()
        )
        self._loop: asyncio.AbstractEventLoop = asyncio.new_event_loop()
        self._thread: threading.Thread = threading.Thread(
            target=self._loop.run_forever, name="rkllm-http", daemon=True
        )
        self._thread.start()
        self._cancelled: threading.Event = threading.Event()

    def _block[T](self, coro: Coroutine[object, object, T]) -> T:
        return asyncio.run_coroutine_threadsafe(coro, self._loop).result()

    def load(self, model_card: ModelCard) -> Iterable[ModelLoadingResponse]:
        if not self._block(self._client.health_check()):
            raise RuntimeError(
                f"rkllama server not reachable at {self._client.config.base_url}; "
                "start it with `rkllama serve` on the NPU node"
            )
        if not self._block(self._client.load_model(model_card.model_id)):
            available = self._block(self._client.list_models())
            raise RuntimeError(
                f"rkllama failed to load {model_card.model_id}; available: {available}"
            )
        total = model_card.n_layers
        yield ModelLoadingResponse(layers_loaded=total, total=total)

    def generate(self, params: TextGenerationTaskParams) -> Iterator[TokenPiece]:
        self._cancelled.clear()
        messages = _messages_from_params(params)
        agen = self._client.generate_stream(messages)
        index = 0
        try:
            while not self._cancelled.is_set():
                try:
                    fragment, finished = self._block(agen.__anext__())
                except StopAsyncIteration:
                    break
                yield TokenPiece(
                    text=fragment,
                    token_id=index,
                    finished=finished,
                    finish_reason="stop" if finished else None,
                )
                index += 1
                if finished:
                    break
        finally:
            self._block(agen.aclose())

    def cancel(self) -> None:
        self._cancelled.set()

    def close(self) -> None:
        try:
            self._block(self._client.close())
        except Exception as e:  # noqa: BLE001 - best-effort teardown
            logger.debug(f"RKLLM http close error: {e}")
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=5)
        if not self._loop.is_running():
            self._loop.close()


def _messages_from_params(
    params: TextGenerationTaskParams,
) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    if params.instructions is not None:
        messages.append({"role": "system", "content": str(params.instructions)})
    for message in params.input:
        messages.append({"role": message.role, "content": str(message.content)})
    return messages
