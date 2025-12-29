import numpy as np
import os
from exo.helpers import DEBUG  # Make sure to import DEBUG

from typing import Tuple, Optional, List
from abc import ABC, abstractmethod
from .shard import Shard
from exo.download.shard_download import ShardDownloader


class InferenceEngine(ABC):
  session = {}

  @abstractmethod
  async def encode(self, shard: Shard, prompt: str) -> np.ndarray:
    pass

  @abstractmethod
  async def sample(self, x: np.ndarray) -> np.ndarray:
    pass

  @abstractmethod
  async def decode(self, shard: Shard, tokens: np.ndarray) -> str:
    pass

  @abstractmethod
  async def infer_tensor(self, request_id: str, shard: Shard, input_data: np.ndarray, inference_state: Optional[dict] = None) -> tuple[np.ndarray, Optional[dict]]:
    pass

  @abstractmethod
  async def load_checkpoint(self, shard: Shard, path: str):
    pass

  async def save_checkpoint(self, shard: Shard, path: str):
    pass

  async def save_session(self, key, value):
    self.session[key] = value

  async def clear_session(self):
    self.session.empty()

  async def infer_prompt(self, request_id: str, shard: Shard, prompt: str, inference_state: Optional[dict] = None) -> tuple[np.ndarray, Optional[dict]]:
    tokens = await self.encode(shard, prompt)
    if shard.model_id != 'stable-diffusion-2-1-base':
      x = tokens.reshape(1, -1)
    else:
      x = tokens
    output_data, inference_state = await self.infer_tensor(request_id, shard, x, inference_state)

    return output_data, inference_state


# Import plugin discovery system
from exo.inference.plugin_discovery import (
  discover_inference_engines,
  load_inference_engine,
  list_available_engines,
)

# Backward-compatible class name mapping (now derived from plugin discovery)
def _get_inference_engine_classes() -> dict:
  """Get engine name to class name mapping for backward compatibility."""
  engines = discover_inference_engines()
  return {name: info[1] for name, info in engines.items()}

inference_engine_classes = _get_inference_engine_classes()


def get_inference_engine(inference_engine_name: str, shard_downloader: ShardDownloader):
  """
  Get an inference engine instance by name.

  Supports both built-in engines (mlx, tinygrad, rkllm, dummy) and
  plugin engines registered via entry points.

  Args:
    inference_engine_name: Name of the engine
    shard_downloader: ShardDownloader for model loading

  Returns:
    Instantiated inference engine
  """
  if DEBUG >= 2:
    print(f"get_inference_engine called with: {inference_engine_name}")

  return load_inference_engine(inference_engine_name, shard_downloader)
