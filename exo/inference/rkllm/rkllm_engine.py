"""
RKLLM Inference Engine for Rockchip RK3588 NPU.

This engine provides inference capabilities using the RKLLM runtime library
for Rockchip NPU devices. It supports hidden state extraction for distributed
inference across multiple devices.
"""

import numpy as np
import asyncio
from typing import Optional, Tuple
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from exo.inference.inference_engine import InferenceEngine
from exo.inference.shard import Shard
from exo.inference.tokenizers import resolve_tokenizer
from exo.download.shard_download import ShardDownloader
from exo.helpers import DEBUG

from .rkllm_ctypes_wrapper import RKLLMWrapper, find_rkllm_library


class RKLLMInferenceEngine(InferenceEngine):
  """
  RKLLM-based inference engine for Rockchip RK3588 NPU.

  Key characteristics:
  - Loads complete .rkllm models (no partial layer loading)
  - Supports hidden state extraction for pipeline parallelism
  - Thread-safe with dedicated executor for NPU operations
  """

  def __init__(self, shard_downloader: ShardDownloader):
    self.shard: Optional[Shard] = None
    self.shard_downloader = shard_downloader
    self._wrapper: Optional[RKLLMWrapper] = None
    self._tokenizer = None
    self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="rkllm")
    self._shard_lock = asyncio.Lock()
    self.session = {}

    # Check if RKLLM library is available
    lib_path = find_rkllm_library()
    if lib_path is None and DEBUG >= 1:
      print("Warning: RKLLM library not found. Engine will fail on first use.")

  async def encode(self, shard: Shard, prompt: str) -> np.ndarray:
    """Encode prompt to tokens using model's tokenizer."""
    await self.ensure_shard(shard)
    tokens = await asyncio.get_running_loop().run_in_executor(
      self._executor,
      self._tokenizer.encode,
      prompt
    )
    return np.array(tokens)

  async def sample(self, x: np.ndarray, temp: float = 0.0, top_p: float = 1.0) -> np.ndarray:
    """
    Sample next token from logits.

    Note: RKLLM handles sampling internally during generation.
    This method is provided for interface compliance and for cases
    where we need to sample from returned hidden states.
    """
    # For RKLLM, sampling is typically done internally by the runtime
    # This is a fallback implementation for when we have raw logits
    logits = x[:, -1, :] if x.ndim == 3 else x

    if temp == 0:
      return np.argmax(logits, axis=-1, keepdims=True).astype(np.int32)

    # Apply temperature
    logits = logits / max(temp, 1e-8)

    # Softmax
    exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
    probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

    # Top-p (nucleus) sampling
    if top_p < 1.0:
      sorted_indices = np.argsort(probs, axis=-1)[:, ::-1]
      sorted_probs = np.take_along_axis(probs, sorted_indices, axis=-1)
      cumulative_probs = np.cumsum(sorted_probs, axis=-1)
      mask = cumulative_probs > top_p
      mask[:, 1:] = mask[:, :-1].copy()
      mask[:, 0] = False
      sorted_probs[mask] = 0.0
      probs = np.zeros_like(probs)
      np.put_along_axis(probs, sorted_indices, sorted_probs, axis=-1)
      probs = probs / np.sum(probs, axis=-1, keepdims=True)

    # Sample
    sampled = np.array([
      np.random.choice(len(p), p=p) for p in probs
    ]).reshape(-1, 1)

    return sampled.astype(np.int32)

  async def decode(self, shard: Shard, tokens: np.ndarray) -> str:
    """Decode tokens to string."""
    await self.ensure_shard(shard)
    token_list = tokens.flatten().tolist()
    return await asyncio.get_running_loop().run_in_executor(
      self._executor,
      self._tokenizer.decode,
      token_list
    )

  async def infer_tensor(
    self,
    request_id: str,
    shard: Shard,
    input_data: np.ndarray,
    inference_state: Optional[dict] = None
  ) -> Tuple[np.ndarray, Optional[dict]]:
    """
    Run inference on input tensor.

    For RKLLM, input_data is typically token IDs for the first node,
    or hidden states for subsequent nodes in a pipeline.

    Args:
      request_id: Unique identifier for this request
      shard: Model shard specification
      input_data: Token IDs (shape: batch, seq_len) or hidden states
      inference_state: Optional state dict for continuation

    Returns:
      Tuple of (output tensor, updated inference_state)
    """
    await self.ensure_shard(shard)

    is_first_layer = shard.is_first_layer()
    is_last_layer = shard.is_last_layer()

    if DEBUG >= 2:
      print(f"RKLLM infer_tensor: request_id={request_id}, "
            f"input_shape={input_data.shape}, "
            f"first_layer={is_first_layer}, last_layer={is_last_layer}")

    def run_inference():
      if is_first_layer and is_last_layer:
        # Single node - full generation from tokens
        # Decode tokens to text, run generation, return as "logits" placeholder
        if input_data.dtype in [np.int32, np.int64]:
          # Input is token IDs
          text = self._tokenizer.decode(input_data.flatten().tolist())
          result_text = self._wrapper.run_generate(text)
          # Encode result back to tokens for compatibility
          result_tokens = self._tokenizer.encode(result_text)
          return np.array(result_tokens).reshape(1, -1)
        else:
          # Input is embeddings - run from embeddings
          result_text = self._wrapper.run_from_embeddings(input_data)
          result_tokens = self._tokenizer.encode(result_text)
          return np.array(result_tokens).reshape(1, -1)

      elif is_first_layer:
        # First in pipeline - extract hidden states
        if input_data.dtype in [np.int32, np.int64]:
          text = self._tokenizer.decode(input_data.flatten().tolist())
          hidden_states = self._wrapper.run_with_hidden_state(text)
          return hidden_states
        else:
          raise ValueError("First layer expects token IDs, not embeddings")

      elif is_last_layer:
        # Last in pipeline - continue from hidden states to generate
        if input_data.dtype in [np.float32, np.float16, np.float64]:
          result_text = self._wrapper.run_from_embeddings(input_data.astype(np.float32))
          result_tokens = self._tokenizer.encode(result_text)
          return np.array(result_tokens).reshape(1, -1)
        else:
          raise ValueError("Last layer expects hidden states, not token IDs")

      else:
        # Middle of pipeline - hidden state in, hidden state out
        # Note: RKLLM may not fully support this mode
        # For now, pass through hidden states
        if DEBUG >= 1:
          print("Warning: Middle pipeline position may not be fully supported")
        return input_data

    output = await asyncio.get_running_loop().run_in_executor(
      self._executor,
      run_inference
    )

    return output, inference_state

  async def load_checkpoint(self, shard: Shard, path: str):
    """Load model from checkpoint path."""
    async with self._shard_lock:
      if self._wrapper:
        await asyncio.get_running_loop().run_in_executor(
          self._executor,
          self._wrapper.release
        )

      self._wrapper = await asyncio.get_running_loop().run_in_executor(
        self._executor,
        RKLLMWrapper,
        path
      )
      self.shard = shard

  async def ensure_shard(self, shard: Shard):
    """
    Ensure the model for the given shard is loaded.

    Note: RKLLM loads complete models, so we normalize any partial
    shard to cover all layers.
    """
    async with self._shard_lock:
      if self.shard == shard:
        return

      # RKLLM requires full model loading
      if not (shard.start_layer == 0 and shard.end_layer == shard.n_layers - 1):
        if DEBUG >= 1:
          print(f"RKLLM loads complete models. "
                f"Requested shard {shard.start_layer}-{shard.end_layer}/{shard.n_layers} "
                f"will load full model.")

      # Download/locate the model
      model_path = await self.shard_downloader.ensure_shard(
        shard, self.__class__.__name__
      )

      if DEBUG >= 2:
        print(f"Loading RKLLM model from: {model_path}")

      # Find .rkllm file in the model path
      rkllm_path = await self._find_rkllm_file(model_path, shard)

      # Release previous model if any
      if self._wrapper:
        await asyncio.get_running_loop().run_in_executor(
          self._executor,
          self._wrapper.release
        )

      # Load new model
      self._wrapper = await asyncio.get_running_loop().run_in_executor(
        self._executor,
        RKLLMWrapper,
        str(rkllm_path)
      )

      # Load tokenizer from HuggingFace repo or local path
      self._tokenizer = await resolve_tokenizer(model_path)

      # Store normalized shard (full model)
      self.shard = Shard(
        shard.model_id,
        0,
        shard.n_layers - 1,
        shard.n_layers
      )

      self.session = {}

      if DEBUG >= 1:
        print(f"RKLLM model loaded: {shard.model_id}")

  async def _find_rkllm_file(self, model_path, shard: Optional[Shard] = None) -> Path:
    """Find .rkllm model file in the given path or RKLLAMA models directory."""
    model_path = Path(model_path)
    rkllama_models = Path.home() / 'RKLLAMA' / 'models'

    # If path is directly a .rkllm file
    if model_path.suffix == '.rkllm':
      return model_path

    # Search for .rkllm file in directory
    if model_path.is_dir():
      rkllm_files = list(model_path.glob('*.rkllm'))
      if rkllm_files:
        return rkllm_files[0]

    # Check RKLLAMA models directory by path name
    rkllama_path = rkllama_models / model_path.name
    if rkllama_path.is_dir():
      rkllm_files = list(rkllama_path.glob('*.rkllm'))
      if rkllm_files:
        return rkllm_files[0]

    # Search RKLLAMA models by shard model_id
    if shard and rkllama_models.is_dir():
      model_id = shard.model_id.lower().replace('-rkllm', '')
      for model_dir in rkllama_models.iterdir():
        if model_dir.is_dir():
          dir_name_lower = model_dir.name.lower()
          # Match by partial name (e.g., "deepseek-r1-1.5b" matches "DeepSeek-R1-1.5B")
          if model_id in dir_name_lower or dir_name_lower in model_id:
            rkllm_files = list(model_dir.glob('*.rkllm'))
            if rkllm_files:
              if DEBUG >= 2:
                print(f"Found RKLLM model: {rkllm_files[0]}")
              return rkllm_files[0]

    # Search all RKLLAMA models for any .rkllm file as last resort
    if rkllama_models.is_dir():
      for model_dir in rkllama_models.iterdir():
        if model_dir.is_dir():
          rkllm_files = list(model_dir.glob('*.rkllm'))
          if rkllm_files:
            if DEBUG >= 1:
              print(f"Using first available RKLLM model: {rkllm_files[0]}")
            return rkllm_files[0]

    raise FileNotFoundError(
      f"No .rkllm file found in {model_path} or ~/RKLLAMA/models/. "
      f"Please ensure the model is converted to RKLLM format."
    )

  async def cleanup(self):
    """Release all resources."""
    if self._wrapper:
      await asyncio.get_running_loop().run_in_executor(
        self._executor,
        self._wrapper.release
      )
      self._wrapper = None

    self._executor.shutdown(wait=True)

  def __del__(self):
    if self._wrapper:
      self._wrapper.release()
