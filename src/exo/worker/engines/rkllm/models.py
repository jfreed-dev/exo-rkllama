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
# convention (one directory per model, containing the .rkllm file). os.path.expanduser
# never raises: with no resolvable home it leaves the path unchanged (unlike
# Path.expanduser), and this module is imported by every worker.
RKLLAMA_MODELS_DIR = Path(os.path.expanduser("~/RKLLAMA/models"))


def find_rkllm_model_file(model_id: ModelId) -> Path | None:
    """Return the local ``.rkllm`` file for ``model_id``, or ``None``.

    Search order: ``RKLLM_MODEL_PATH`` (points at the file itself), the exo model
    directories, then the rkllama models directory. Model directories are named by
    the normalized model id (slashes become ``--``).

    Raises ``ValueError`` when ``RKLLM_MODEL_PATH`` is set but does not point at a
    file, so a typo'd or unmounted override fails loudly instead of silently falling
    back to a different artifact.
    """
    env_path = os.environ.get("RKLLM_MODEL_PATH")
    if env_path:
        path = Path(env_path)
        if not path.is_file():
            raise ValueError(
                f"RKLLM_MODEL_PATH={env_path!r} does not point at a .rkllm file"
            )
        return path
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
    A bad ``RKLLM_MODEL_PATH`` or ``EXO_RKLLM_BACKEND`` value fails with the actual
    configuration error rather than a missing-file message.
    """
    model_card = shard_metadata.model_card
    try:
        found = find_rkllm_model_file(model_card.model_id)
        transport = backend_choice()
    except ValueError as error:
        return DownloadFailed(
            node_id=node_id,
            shard_metadata=shard_metadata,
            error_message=str(error),
        )
    if found is not None:
        return DownloadCompleted(
            node_id=node_id,
            shard_metadata=shard_metadata,
            model_directory=str(found.parent),
            total=model_card.storage_size,
            # exo did not download this artifact and must never delete it.
            read_only=True,
        )
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
            "exo models directory (or set RKLLM_MODEL_PATH to the file), then relaunch "
            "the model instance or restart exo on this node to retry."
        ),
    )
