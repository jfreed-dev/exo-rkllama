"""Local ``.rkllm`` artifact discovery.

RKLLM models are pre-converted single-file artifacts, not HF safetensors repos, so
exo cannot download them. The worker resolves a local copy at "download" time instead
(:func:`resolve_rkllm_download`) and the ctypes backend reuses the same search to find
the file it loads.
"""

import os
from pathlib import Path

from exo.shared.constants import EXO_MODELS_DIRS, EXO_MODELS_READ_ONLY_DIRS
from exo.shared.types.common import ModelId, NodeId
from exo.shared.types.worker.downloads import DownloadCompleted, DownloadFailed
from exo.shared.types.worker.shards import ShardMetadata
from exo.worker.engines.rkllm.backend import backend_choice

# Default models directory of the rkllama server; the ctypes backend uses the same
# convention (one directory per model, containing the .rkllm file).
RKLLAMA_MODELS_DIR = Path("~/RKLLAMA/models").expanduser()


def find_rkllm_model_file(model_id: ModelId) -> Path | None:
    """Return the local ``.rkllm`` file for ``model_id``, or ``None``.

    Search order: ``RKLLM_MODEL_PATH`` (points at the file itself), the exo model
    directories, then the rkllama models directory. Model directories are named by
    the normalized model id (slashes become ``--``).
    """
    env_path = os.environ.get("RKLLM_MODEL_PATH")
    if env_path and Path(env_path).is_file():
        return Path(env_path)
    for base in (*EXO_MODELS_READ_ONLY_DIRS, *EXO_MODELS_DIRS, RKLLAMA_MODELS_DIR):
        candidate = base / model_id.normalize()
        if candidate.is_dir():
            files = sorted(candidate.glob("*.rkllm"))
            if files:
                return files[0]
    return None


def resolve_rkllm_download(
    node_id: NodeId, shard_metadata: ShardMetadata
) -> DownloadCompleted | DownloadFailed:
    """Resolve a ``DownloadModel`` task for an RKLLM model without downloading.

    A local ``.rkllm`` file marks the model complete. Without one, the HTTP backend
    still completes (the rkllama server owns its model files and ``load`` verifies
    the model exists server-side), while the ctypes backend fails with instructions.
    """
    model_card = shard_metadata.model_card
    found = find_rkllm_model_file(model_card.model_id)
    if found is not None:
        return DownloadCompleted(
            node_id=node_id,
            shard_metadata=shard_metadata,
            model_directory=str(found.parent),
            total=model_card.storage_size,
            # exo did not download this artifact and must never delete it.
            read_only=True,
        )
    try:
        transport = backend_choice()
    except ValueError:
        transport = "ctypes"  # invalid env: fall through to the actionable error
    if transport == "http":
        return DownloadCompleted(
            node_id=node_id,
            shard_metadata=shard_metadata,
            total=model_card.storage_size,
            read_only=True,
        )
    return DownloadFailed(
        node_id=node_id,
        shard_metadata=shard_metadata,
        error_message=(
            f"No .rkllm file found for {model_card.model_id}. Place the pre-converted "
            f"model under {RKLLAMA_MODELS_DIR}/{model_card.model_id.normalize()}/ or an "
            "exo models directory, or set RKLLM_MODEL_PATH to the .rkllm file."
        ),
    )
