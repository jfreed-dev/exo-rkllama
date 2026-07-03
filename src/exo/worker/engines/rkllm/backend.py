"""Backend abstraction for the RKLLM engine.

Two transports implement :class:`RkllmBackend`:

* ``http``  — talk to a separate ``rkllama`` server over HTTP (default).
* ``ctypes`` — load ``librkllmrt.so`` in-process.

The engine is synchronous (the runner drives ``step()`` in a loop), so backends
expose a synchronous streaming ``generate()`` and hide any async/threading inside.
Select the transport with ``EXO_RKLLM_BACKEND=http|ctypes``.
"""

import os
from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator
from dataclasses import dataclass

from exo.shared.models.model_cards import ModelCard
from exo.shared.types.text_generation import TextGenerationTaskParams
from exo.shared.types.worker.runner_response import ModelLoadingResponse

# RKLLM does not expose token ids in its result struct (only text fragments), so
# backends emit a running counter here and the engine treats it as opaque.
type FinishReasonRkllm = str | None


@dataclass(frozen=True)
class TokenPiece:
    """One streamed unit of output from an RKLLM backend."""

    text: str
    token_id: int
    finished: bool
    finish_reason: FinishReasonRkllm = None


class RkllmBackend(ABC):
    """A transport to an RKLLM runtime that runs a whole model on one NPU."""

    @abstractmethod
    def load(self, model_card: ModelCard) -> Iterable[ModelLoadingResponse]:
        """Ready the model. Yields coarse progress; the final yield means loaded."""
        ...

    @abstractmethod
    def generate(self, params: TextGenerationTaskParams) -> Iterator[TokenPiece]:
        """Stream output for one request as :class:`TokenPiece` values."""
        ...

    @abstractmethod
    def cancel(self) -> None:
        """Request the in-flight generation to stop as soon as possible."""
        ...

    @abstractmethod
    def close(self) -> None:
        """Release all backend resources."""
        ...


def backend_choice() -> str:
    """Return the transport named by ``EXO_RKLLM_BACKEND``: ``http`` or ``ctypes``."""
    choice = os.environ.get("EXO_RKLLM_BACKEND", "http").strip().lower()
    if choice in ("http", ""):
        return "http"
    if choice != "ctypes":
        raise ValueError(
            f"Unknown EXO_RKLLM_BACKEND={choice!r}; expected 'http' or 'ctypes'"
        )
    return choice


def select_backend() -> RkllmBackend:
    """Construct the backend named by ``EXO_RKLLM_BACKEND`` (default ``http``)."""
    if backend_choice() == "ctypes":
        from exo.worker.engines.rkllm.ctypes_backend import RkllmCtypesBackend

        return RkllmCtypesBackend()
    from exo.worker.engines.rkllm.http_backend import RkllmHttpBackend

    return RkllmHttpBackend()
