# pyright: reportAny=false, reportExplicitAny=false, reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportUnknownParameterType=false, reportMissingParameterType=false, reportAttributeAccessIssue=false
"""Low-level ctypes bindings for ``librkllmrt.so`` (Rockchip RKLLM runtime).

Struct layouts match the RKLLM **1.2.3** ABI exactly (airockchip/rknn-llm,
``rkllm-runtime/Linux/librkllm_api/include/rkllm.h`` at ``release-v1.2.3``); the
1.1.x-era layouts this file originally shipped segfault inside ``rkllm_init`` when
loaded against the 1.2.x library. Defaults come from ``rkllm_createDefaultParam()``
rather than hand-rolled values so library-internal fields keep their blessed
settings. The library is loaded lazily so this module imports cleanly on non-NPU
hosts. ctypes defeats strict static typing, so the gnarly rules are relaxed for
this file only (see the pragma above).
"""

import ctypes
import queue
import threading
from collections.abc import Iterator

from loguru import logger

from exo.worker.engines.rkllm.detection import get_rkllm_library_path


class LLMCallState:
    RKLLM_RUN_NORMAL = 0
    RKLLM_RUN_WAITING = 1
    RKLLM_RUN_FINISH = 2
    RKLLM_RUN_ERROR = 3


class RKLLMInputType:
    RKLLM_INPUT_PROMPT = 0
    RKLLM_INPUT_TOKEN = 1
    RKLLM_INPUT_EMBED = 2
    RKLLM_INPUT_MULTIMODAL = 3


class RKLLMInferMode:
    RKLLM_INFER_GENERATE = 0


class RKLLMExtendParam(ctypes.Structure):
    _fields_ = [
        ("base_domain_id", ctypes.c_int32),
        ("embed_flash", ctypes.c_int8),
        ("enabled_cpus_num", ctypes.c_int8),
        ("enabled_cpus_mask", ctypes.c_uint32),
        ("n_batch", ctypes.c_uint8),
        ("use_cross_attn", ctypes.c_int8),
        ("reserved", ctypes.c_uint8 * 104),
    ]


class RKLLMParam(ctypes.Structure):
    _fields_ = [
        ("model_path", ctypes.c_char_p),
        ("max_context_len", ctypes.c_int32),
        ("max_new_tokens", ctypes.c_int32),
        ("top_k", ctypes.c_int32),
        ("n_keep", ctypes.c_int32),
        ("top_p", ctypes.c_float),
        ("temperature", ctypes.c_float),
        ("repeat_penalty", ctypes.c_float),
        ("frequency_penalty", ctypes.c_float),
        ("presence_penalty", ctypes.c_float),
        ("mirostat", ctypes.c_int32),
        ("mirostat_tau", ctypes.c_float),
        ("mirostat_eta", ctypes.c_float),
        ("skip_special_token", ctypes.c_bool),
        ("is_async", ctypes.c_bool),
        ("img_start", ctypes.c_char_p),
        ("img_end", ctypes.c_char_p),
        ("img_content", ctypes.c_char_p),
        ("extend_param", RKLLMExtendParam),
    ]


class RKLLMEmbedInput(ctypes.Structure):
    _fields_ = [
        ("embed", ctypes.POINTER(ctypes.c_float)),
        ("n_tokens", ctypes.c_size_t),
    ]


class RKLLMTokenInput(ctypes.Structure):
    _fields_ = [
        ("input_ids", ctypes.POINTER(ctypes.c_int32)),
        ("n_tokens", ctypes.c_size_t),
    ]


class RKLLMMultiModalInput(ctypes.Structure):
    _fields_ = [
        ("prompt", ctypes.c_char_p),
        ("image_embed", ctypes.POINTER(ctypes.c_float)),
        ("n_image_tokens", ctypes.c_size_t),
        ("n_image", ctypes.c_size_t),
        ("image_width", ctypes.c_size_t),
        ("image_height", ctypes.c_size_t),
    ]


class _RKLLMInputUnion(ctypes.Union):
    # Every member must be declared or the union (and the enclosing struct) is
    # sized too small and the library scribbles past it.
    _fields_ = [
        ("prompt_input", ctypes.c_char_p),
        ("embed_input", RKLLMEmbedInput),
        ("token_input", RKLLMTokenInput),
        ("multimodal_input", RKLLMMultiModalInput),
    ]


class RKLLMInput(ctypes.Structure):
    _anonymous_ = ("_union",)
    _fields_ = [
        ("role", ctypes.c_char_p),
        ("enable_thinking", ctypes.c_bool),
        ("input_type", ctypes.c_int),
        ("_union", _RKLLMInputUnion),
    ]


class RKLLMLoraParam(ctypes.Structure):
    _fields_ = [("lora_adapter_name", ctypes.c_char_p)]


class RKLLMPromptCacheParam(ctypes.Structure):
    _fields_ = [
        ("save_prompt_cache", ctypes.c_int),
        ("prompt_cache_path", ctypes.c_char_p),
    ]


class RKLLMInferParam(ctypes.Structure):
    _fields_ = [
        ("mode", ctypes.c_int),
        ("lora_params", ctypes.POINTER(RKLLMLoraParam)),
        ("prompt_cache_params", ctypes.POINTER(RKLLMPromptCacheParam)),
        ("keep_history", ctypes.c_int),
    ]


class RKLLMResultLastHiddenLayer(ctypes.Structure):
    _fields_ = [
        ("hidden_states", ctypes.POINTER(ctypes.c_float)),
        ("embd_size", ctypes.c_int),
        ("num_tokens", ctypes.c_int),
    ]


class RKLLMResultLogits(ctypes.Structure):
    _fields_ = [
        ("logits", ctypes.POINTER(ctypes.c_float)),
        ("vocab_size", ctypes.c_int),
        ("num_tokens", ctypes.c_int),
    ]


class RKLLMPerfStat(ctypes.Structure):
    _fields_ = [
        ("prefill_time_ms", ctypes.c_float),
        ("prefill_tokens", ctypes.c_int),
        ("generate_time_ms", ctypes.c_float),
        ("generate_tokens", ctypes.c_int),
        ("memory_usage_mb", ctypes.c_float),
    ]


class RKLLMResult(ctypes.Structure):
    _fields_ = [
        ("text", ctypes.c_char_p),
        ("token_id", ctypes.c_int32),
        ("last_hidden_layer", RKLLMResultLastHiddenLayer),
        ("logits", RKLLMResultLogits),
        ("perf", RKLLMPerfStat),
    ]


RKLLM_Handle_t = ctypes.c_void_p
# 1.2.x callbacks return int: 0 continues, 1 suspends the inference.
RKLLMCallback = ctypes.CFUNCTYPE(
    ctypes.c_int, ctypes.POINTER(RKLLMResult), ctypes.c_void_p, ctypes.c_int
)

# Sentinel pushed onto the stream queue when generation finishes.
_FINISH: object = object()


def load_rkllm_library(lib_path: str | None = None) -> ctypes.CDLL:
    path = lib_path or get_rkllm_library_path()
    if path is None:
        raise RuntimeError(
            "Could not find librkllmrt.so. Set RKLLM_LIB_PATH or install the RKLLM "
            "runtime (~/RKLLAMA/lib/librkllmrt.so)."
        )
    logger.info(f"Loading RKLLM runtime from {path}")
    return ctypes.CDLL(path)


class RKLLMRuntime:
    """Thread-safe holder for a single loaded RKLLM model."""

    def __init__(
        self,
        model_path: str,
        max_context_len: int = 4096,
        max_new_tokens: int = 2048,
    ) -> None:
        self._lib = load_rkllm_library()
        self._configure_signatures()
        self._handle = RKLLM_Handle_t()
        self._lock = threading.Lock()
        self._queue: queue.Queue[object] = queue.Queue()
        self._error: str | None = None
        # Keep a reference so the callback is not garbage collected.
        self._callback = RKLLMCallback(self._on_result)
        self._init_model(model_path, max_context_len, max_new_tokens)

    def _configure_signatures(self) -> None:
        self._lib.rkllm_createDefaultParam.argtypes = []
        self._lib.rkllm_createDefaultParam.restype = RKLLMParam
        self._lib.rkllm_init.argtypes = [
            ctypes.POINTER(RKLLM_Handle_t),
            ctypes.POINTER(RKLLMParam),
            RKLLMCallback,
        ]
        self._lib.rkllm_init.restype = ctypes.c_int
        self._lib.rkllm_run.argtypes = [
            RKLLM_Handle_t,
            ctypes.POINTER(RKLLMInput),
            ctypes.POINTER(RKLLMInferParam),
            ctypes.c_void_p,
        ]
        self._lib.rkllm_run.restype = ctypes.c_int
        self._lib.rkllm_abort.argtypes = [RKLLM_Handle_t]
        self._lib.rkllm_abort.restype = ctypes.c_int
        self._lib.rkllm_destroy.argtypes = [RKLLM_Handle_t]
        self._lib.rkllm_destroy.restype = ctypes.c_int

    def _on_result(self, result_ptr, _userdata, state: int) -> int:
        if state == LLMCallState.RKLLM_RUN_NORMAL and result_ptr:
            result = result_ptr.contents
            if result.text:
                self._queue.put(
                    (result.text.decode("utf-8", errors="replace"), result.token_id)
                )
        elif state == LLMCallState.RKLLM_RUN_FINISH:
            if result_ptr:
                perf = result_ptr.contents.perf
                if perf.generate_time_ms > 0:
                    logger.info(
                        f"RKLLM perf: prefill {perf.prefill_tokens} tok / "
                        f"{perf.prefill_time_ms:.0f} ms, generate "
                        f"{perf.generate_tokens} tok / {perf.generate_time_ms:.0f} ms "
                        f"({perf.generate_tokens / perf.generate_time_ms * 1000:.2f} tok/s)"
                    )
            self._queue.put(_FINISH)
        elif state == LLMCallState.RKLLM_RUN_ERROR:
            self._error = "RKLLM inference error"
            self._queue.put(_FINISH)
        return 0

    def _init_model(
        self, model_path: str, max_context_len: int, max_new_tokens: int
    ) -> None:
        param = self._lib.rkllm_createDefaultParam()
        param.model_path = model_path.encode("utf-8")
        param.max_context_len = max_context_len
        param.max_new_tokens = max_new_tokens
        param.skip_special_token = True
        param.is_async = False

        ret = self._lib.rkllm_init(
            ctypes.byref(self._handle), ctypes.byref(param), self._callback
        )
        if ret != 0:
            raise RuntimeError(f"rkllm_init failed: error code {ret}")
        logger.info(f"RKLLM model initialized: {model_path}")

    def generate_stream(self, prompt: str) -> Iterator[tuple[str, int]]:
        """Stream ``(text_fragment, token_id)`` for ``prompt``.

        Blocks the ``rkllm_run`` call in a thread; the callback feeds the queue.
        """
        with self._lock:
            self._error = None
            while not self._queue.empty():
                _ = self._queue.get_nowait()

            rkllm_input = RKLLMInput()
            rkllm_input.role = b"user"
            rkllm_input.enable_thinking = False
            rkllm_input.input_type = RKLLMInputType.RKLLM_INPUT_PROMPT
            rkllm_input.prompt_input = prompt.encode("utf-8")

            infer_param = RKLLMInferParam()
            infer_param.mode = RKLLMInferMode.RKLLM_INFER_GENERATE
            infer_param.lora_params = None
            infer_param.prompt_cache_params = None
            # exo resends the full conversation with each request.
            infer_param.keep_history = 0

            def _run() -> None:
                ret = self._lib.rkllm_run(
                    self._handle,
                    ctypes.byref(rkllm_input),
                    ctypes.byref(infer_param),
                    None,
                )
                if ret != 0:
                    self._error = f"rkllm_run failed: error code {ret}"
                    self._queue.put(_FINISH)

            worker = threading.Thread(target=_run, name="rkllm-run", daemon=True)
            worker.start()
            while True:
                item = self._queue.get()
                if item is _FINISH:
                    break
                if isinstance(item, tuple):
                    yield item
            worker.join(timeout=5)
            if self._error is not None:
                raise RuntimeError(self._error)

    def abort(self) -> None:
        """Ask the runtime to stop the in-flight generation."""
        if self._handle:
            _ = self._lib.rkllm_abort(self._handle)

    def release(self) -> None:
        if self._handle:
            _ = self._lib.rkllm_destroy(self._handle)
            self._handle = RKLLM_Handle_t()
