"""
HTTP Client for RKLLM/RKLLama server.

Provides an async HTTP client to interact with the rkllama server
which exposes the RKLLM runtime via a Flask API.
"""

import aiohttp
import asyncio
from typing import Optional, List, Dict, Any
from dataclasses import dataclass

from exo.helpers import DEBUG


@dataclass
class RKLLMServerConfig:
  """Configuration for connecting to rkllama server."""
  host: str = "localhost"
  port: int = 8080
  timeout: float = 300.0  # 5 minute timeout for generation

  @property
  def base_url(self) -> str:
    return f"http://{self.host}:{self.port}"


class RKLLMHTTPClient:
  """
  Async HTTP client for rkllama server.

  The rkllama server handles:
  - Model loading/unloading
  - Tokenization (via HuggingFace tokenizers)
  - Inference on the RKLLM runtime
  """

  def __init__(self, config: Optional[RKLLMServerConfig] = None):
    self.config = config or RKLLMServerConfig()
    self._session: Optional[aiohttp.ClientSession] = None
    self._current_model: Optional[str] = None

  async def _get_session(self) -> aiohttp.ClientSession:
    """Get or create aiohttp session."""
    if self._session is None or self._session.closed:
      timeout = aiohttp.ClientTimeout(total=self.config.timeout)
      self._session = aiohttp.ClientSession(timeout=timeout)
    return self._session

  async def close(self):
    """Close the HTTP session."""
    if self._session and not self._session.closed:
      await self._session.close()
      self._session = None

  async def health_check(self) -> bool:
    """Check if the rkllama server is running."""
    try:
      session = await self._get_session()
      async with session.get(f"{self.config.base_url}/") as resp:
        return resp.status == 200
    except aiohttp.ClientError:
      return False
    except Exception as e:
      if DEBUG >= 2:
        print(f"RKLLM health check failed: {e}")
      return False

  async def list_models(self) -> List[str]:
    """Get list of available models on the server."""
    try:
      session = await self._get_session()
      async with session.get(f"{self.config.base_url}/models") as resp:
        if resp.status == 200:
          data = await resp.json()
          return data.get("models", [])
        return []
    except Exception as e:
      if DEBUG >= 1:
        print(f"Failed to list RKLLM models: {e}")
      return []

  async def get_current_model(self) -> Optional[str]:
    """Get the currently loaded model name."""
    try:
      session = await self._get_session()
      async with session.get(f"{self.config.base_url}/current_model") as resp:
        if resp.status == 200:
          data = await resp.json()
          return data.get("model_name")
        return None
    except Exception as e:
      if DEBUG >= 2:
        print(f"Failed to get current model: {e}")
      return None

  async def load_model(
    self,
    model_name: str,
    huggingface_path: Optional[str] = None,
    from_file: Optional[str] = None
  ) -> bool:
    """
    Load a model on the rkllama server.

    Args:
      model_name: Name of the model directory in ~/RKLLAMA/models/
      huggingface_path: Optional HuggingFace repo for tokenizer
      from_file: Optional .rkllm filename

    Returns:
      True if model loaded successfully
    """
    # Check if model is already loaded
    current = await self.get_current_model()
    if current == model_name:
      if DEBUG >= 2:
        print(f"Model {model_name} already loaded")
      return True

    # Unload current model if one is loaded
    if current:
      await self.unload_model()

    try:
      session = await self._get_session()
      payload: Dict[str, Any] = {"model_name": model_name}

      if huggingface_path:
        payload["huggingface_path"] = huggingface_path
      if from_file:
        payload["from"] = from_file

      async with session.post(
        f"{self.config.base_url}/load_model",
        json=payload
      ) as resp:
        if resp.status == 200:
          self._current_model = model_name
          if DEBUG >= 1:
            print(f"RKLLM model {model_name} loaded successfully")
          return True
        else:
          error = await resp.json()
          if DEBUG >= 1:
            print(f"Failed to load model: {error}")
          return False
    except Exception as e:
      if DEBUG >= 1:
        print(f"Failed to load RKLLM model: {e}")
      return False

  async def unload_model(self) -> bool:
    """Unload the current model."""
    try:
      session = await self._get_session()
      async with session.post(f"{self.config.base_url}/unload_model") as resp:
        if resp.status == 200:
          self._current_model = None
          return True
        return False
    except Exception as e:
      if DEBUG >= 2:
        print(f"Failed to unload model: {e}")
      return False

  async def generate(
    self,
    messages: List[Dict[str, str]],
    stream: bool = False
  ) -> str:
    """
    Generate text from messages.

    Args:
      messages: List of message dicts with 'role' and 'content' keys
                e.g., [{"role": "user", "content": "Hello"}]
      stream: Whether to stream the response

    Returns:
      Generated text response
    """
    try:
      session = await self._get_session()
      payload = {
        "messages": messages,
        "stream": stream
      }

      async with session.post(
        f"{self.config.base_url}/generate",
        json=payload
      ) as resp:
        if resp.status == 200:
          if stream:
            # Handle streaming response
            full_text = ""
            async for line in resp.content:
              if line:
                try:
                  import json
                  data = json.loads(line.decode('utf-8').strip())
                  if data.get("choices"):
                    content = data["choices"][0].get("content", "")
                    full_text += content
                except (json.JSONDecodeError, KeyError):
                  continue
            return full_text
          else:
            data = await resp.json()
            if data.get("choices"):
              return data["choices"][0].get("content", "")
            return ""
        else:
          error = await resp.text()
          if DEBUG >= 1:
            print(f"Generate failed: {error}")
          return ""
    except asyncio.TimeoutError:
      if DEBUG >= 1:
        print("RKLLM generate timed out")
      return ""
    except Exception as e:
      if DEBUG >= 1:
        print(f"RKLLM generate failed: {e}")
      return ""

  async def generate_from_prompt(self, prompt: str) -> str:
    """
    Generate from a prompt string, handling pre-templated prompts.

    If the prompt contains chat template markers (e.g., from exo),
    extract the user content to avoid double-templating.

    Args:
      prompt: The user prompt text (may be pre-templated)

    Returns:
      Generated text response
    """
    import re

    # Check if prompt is already templated (exo uses special tokens)
    # Common patterns: <｜User｜>, <|user|>, [INST], etc.
    user_markers = [
      (r'<｜User｜>(.*?)<｜Assistant｜>', 1),
      (r'<\|user\|>(.*?)<\|assistant\|>', 1),
      (r'\[INST\](.*?)\[/INST\]', 1),
      (r'<\|im_start\|>user\n(.*?)<\|im_end\|>', 1),
    ]

    extracted_content = None
    for pattern, group in user_markers:
      match = re.search(pattern, prompt, re.DOTALL | re.IGNORECASE)
      if match:
        extracted_content = match.group(group).strip()
        break

    if extracted_content:
      # Use extracted content to avoid double-templating
      if DEBUG >= 2:
        print(f"Extracted user content from template: {extracted_content[:100]}...")
      messages = [{"role": "user", "content": extracted_content}]
    else:
      # Use prompt as-is
      messages = [{"role": "user", "content": prompt}]

    return await self.generate(messages, stream=False)
