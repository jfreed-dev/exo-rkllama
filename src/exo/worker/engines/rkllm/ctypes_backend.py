"""In-process ctypes backend: loads ``librkllmrt.so`` and runs the model directly.

No external daemon. Note: unlike the rkllama server, this path does not apply a
model-specific chat template — it builds a minimal generic prompt. Proper templating
(via the model tokenizer) is a follow-up; the HTTP backend templates server-side.
"""

import os
import threading
from collections.abc import Iterable, Iterator
from pathlib import Path

from exo.shared.models.model_cards import ModelCard
from exo.shared.types.text_generation import TextGenerationTaskParams
from exo.shared.types.worker.runner_response import ModelLoadingResponse
from exo.worker.engines.rkllm.backend import RkllmBackend, TokenPiece
from exo.worker.engines.rkllm.runtime import RKLLMRuntime


class RkllmCtypesBackend(RkllmBackend):
    def __init__(self) -> None:
        self._runtime: RKLLMRuntime | None = None
        self._cancelled: threading.Event = threading.Event()

    def load(self, model_card: ModelCard) -> Iterable[ModelLoadingResponse]:
        model_path = _resolve_model_path(model_card.model_id)
        self._runtime = RKLLMRuntime(model_path)
        total = model_card.n_layers
        yield ModelLoadingResponse(layers_loaded=total, total=total)

    def generate(self, params: TextGenerationTaskParams) -> Iterator[TokenPiece]:
        if self._runtime is None:
            raise RuntimeError("RKLLM ctypes backend used before load()")
        self._cancelled.clear()
        prompt = _prompt_from_params(params)
        index = 0
        for fragment in self._runtime.generate_stream(prompt):
            if self._cancelled.is_set():
                break
            yield TokenPiece(text=fragment, token_id=index, finished=False)
            index += 1
        if not self._cancelled.is_set():
            yield TokenPiece(
                text="", token_id=index, finished=True, finish_reason="stop"
            )

    def cancel(self) -> None:
        self._cancelled.set()

    def close(self) -> None:
        if self._runtime is not None:
            self._runtime.release()
            self._runtime = None


def _resolve_model_path(model_id: str) -> str:
    env_path = os.environ.get("RKLLM_MODEL_PATH")
    if env_path:
        return env_path
    base = Path(os.path.expanduser("~/RKLLAMA/models")) / model_id
    if base.is_dir():
        candidates = sorted(base.glob("*.rkllm"))
        if candidates:
            return str(candidates[0])
    raise RuntimeError(
        f"No .rkllm file for {model_id}; set RKLLM_MODEL_PATH or place the model "
        f"under ~/RKLLAMA/models/{model_id}/"
    )


def _prompt_from_params(params: TextGenerationTaskParams) -> str:
    parts: list[str] = []
    if params.instructions is not None:
        parts.append(f"System: {params.instructions}")
    for message in params.input:
        parts.append(f"{message.role.capitalize()}: {message.content}")
    parts.append("Assistant:")
    return "\n".join(parts)
