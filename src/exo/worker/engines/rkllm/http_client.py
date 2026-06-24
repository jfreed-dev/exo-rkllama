"""Async HTTP client for a ``rkllama`` server (Flask wrapper around librkllmrt).

Ported from the pre-zenoh plugin; the dead ``exo.helpers.DEBUG`` import was replaced
with loguru and the types were tightened for strict checking. The client only speaks
to the server — message construction and token assembly live in the backend.
"""

import os
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from typing import cast

import aiohttp
from loguru import logger


@dataclass(frozen=True)
class RKLLMServerConfig:
    """Connection settings for a rkllama server."""

    host: str = "localhost"
    port: int = 8080
    timeout: float = 300.0  # generation can be slow on the NPU

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    @classmethod
    def from_env(cls) -> "RKLLMServerConfig":
        return cls(
            host=os.environ.get("RKLLM_SERVER_HOST", "localhost"),
            port=int(os.environ.get("RKLLM_SERVER_PORT", "8080")),
        )


class RKLLMHTTPClient:
    """Minimal async client for the rkllama HTTP API."""

    def __init__(self, config: RKLLMServerConfig | None = None) -> None:
        self.config: RKLLMServerConfig = config or RKLLMServerConfig()
        self._session: aiohttp.ClientSession | None = None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=self.config.timeout)
            self._session = aiohttp.ClientSession(timeout=timeout)
        return self._session

    async def close(self) -> None:
        if self._session is not None and not self._session.closed:
            await self._session.close()
        self._session = None

    async def health_check(self) -> bool:
        try:
            session = await self._get_session()
            async with session.get(f"{self.config.base_url}/") as resp:
                return resp.status == 200
        except aiohttp.ClientError as e:
            logger.debug(f"RKLLM health check failed: {e}")
            return False

    async def get_current_model(self) -> str | None:
        try:
            session = await self._get_session()
            async with session.get(f"{self.config.base_url}/current_model") as resp:
                if resp.status != 200:
                    return None
                data = cast("dict[str, object]", await resp.json())
                name = data.get("model_name")
                return name if isinstance(name, str) else None
        except aiohttp.ClientError as e:
            logger.debug(f"RKLLM get_current_model failed: {e}")
            return None

    async def load_model(
        self,
        model_name: str,
        huggingface_path: str | None = None,
        from_file: str | None = None,
    ) -> bool:
        """Load ``model_name`` (a directory under ``~/RKLLAMA/models/``)."""
        if await self.get_current_model() == model_name:
            return True
        await self.unload_model()

        payload: dict[str, str] = {"model_name": model_name}
        if huggingface_path:
            payload["huggingface_path"] = huggingface_path
        if from_file:
            payload["from"] = from_file

        try:
            session = await self._get_session()
            async with session.post(
                f"{self.config.base_url}/load_model", json=payload
            ) as resp:
                if resp.status == 200:
                    logger.info(f"RKLLM model {model_name} loaded")
                    return True
                logger.error(
                    f"RKLLM load_model {model_name} failed: {await resp.text()}"
                )
                return False
        except aiohttp.ClientError as e:
            logger.error(f"RKLLM load_model {model_name} failed: {e}")
            return False

    async def unload_model(self) -> bool:
        try:
            session = await self._get_session()
            async with session.post(f"{self.config.base_url}/unload_model") as resp:
                return resp.status == 200
        except aiohttp.ClientError as e:
            logger.debug(f"RKLLM unload_model failed: {e}")
            return False

    async def list_models(self) -> list[str]:
        try:
            session = await self._get_session()
            async with session.get(f"{self.config.base_url}/models") as resp:
                if resp.status != 200:
                    return []
                data = cast("dict[str, object]", await resp.json())
                models = data.get("models")
                if isinstance(models, list):
                    return [
                        m for m in cast("list[object]", models) if isinstance(m, str)
                    ]
                return []
        except aiohttp.ClientError as e:
            logger.debug(f"RKLLM list_models failed: {e}")
            return []

    async def generate_stream(
        self, messages: list[dict[str, str]]
    ) -> AsyncGenerator[tuple[str, bool]]:
        """Stream ``(text_fragment, is_finished)`` from the rkllama ``/generate`` API.

        The server templates the messages and streams JSON lines shaped like
        ``{"choices": [{"content": "...", "finish_reason": "stop"|null}]}``.
        """
        session = await self._get_session()
        payload: dict[str, object] = {"messages": messages, "stream": True}
        async with session.post(
            f"{self.config.base_url}/generate", json=payload
        ) as resp:
            if resp.status != 200:
                logger.error(f"RKLLM generate failed: {await resp.text()}")
                return
            async for raw in resp.content:
                line = raw.decode("utf-8", errors="replace").strip()
                if not line:
                    continue
                for fragment, finished in _parse_stream_line(line):
                    yield fragment, finished
                    if finished:
                        return


def _parse_stream_line(line: str) -> list[tuple[str, bool]]:
    """Parse a (possibly multi-object) rkllama stream line into fragments."""
    import json

    out: list[tuple[str, bool]] = []
    for chunk in line.split("\n\n"):
        chunk = chunk.strip()
        if not chunk:
            continue
        try:
            data = cast("dict[str, object]", json.loads(chunk))
        except json.JSONDecodeError:
            continue
        choices = data.get("choices")
        if not isinstance(choices, list) or not choices:
            continue
        choice = cast("list[object]", choices)[0]
        if not isinstance(choice, dict):
            continue
        choice_d = cast("dict[str, object]", choice)
        content = choice_d.get("content")
        finished = choice_d.get("finish_reason") == "stop"
        if isinstance(content, str) and content:
            out.append((content, finished))
        elif finished:
            out.append(("", True))
    return out
