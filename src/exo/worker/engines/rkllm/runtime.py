# pyright: reportAny=false, reportExplicitAny=false, reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportUnknownParameterType=false, reportMissingParameterType=false, reportAttributeAccessIssue=false
"""Low-level ctypes bindings for ``librkllmrt.so`` (Rockchip RKLLM runtime).

Ported from the pre-zenoh plugin and trimmed to what the in-process backend needs:
model init, streaming text generation, and release. The library is loaded lazily so
this module imports cleanly on non-NPU hosts. ctypes defeats strict static typing, so
the gnarly rules are relaxed for this file only (see the pragma above).
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


class RKLLMInputMode:
    RKLLM_INPUT_PROMPT = 0


class RKLLMInferMode:
    RKLLM_INFER_GENERATE = 0


class RKLLMExtendParam(ctypes.Structure):
    _fields_ = [("base_domain_id", ctypes.c_int32), ("reserved", ctypes.c_uint8 * 112)]


class RKLLMParam(ctypes.Structure):
    _fields_ = [
        ("model_path", ctypes.c_char_p),
        ("max_context_len", ctypes.c_int32),
        ("max_new_tokens", ctypes.c_int32),
        ("top_k", ctypes.c_int32),
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


class RKLLMInputUnion(ctypes.Union):
    _fields_ = [("prompt_input", ctypes.c_char_p)]


class RKLLMInput(ctypes.Structure):
    _fields_ = [("input_mode", ctypes.c_int), ("input_data", RKLLMInputUnion)]


class RKLLMInferParam(ctypes.Structure):
    _fields_ = [
        ("mode", ctypes.c_int),
        ("lora_params", ctypes.c_void_p),
        ("prompt_cache_params", ctypes.c_void_p),
    ]


class RKLLMResultLastHiddenLayer(ctypes.Structure):
    _fields_ = [
        ("hidden_states", ctypes.POINTER(ctypes.c_float)),
        ("embd_size", ctypes.c_int),
        ("num_tokens", ctypes.c_int),
    ]


class RKLLMResult(ctypes.Structure):
    _fields_ = [
        ("text", ctypes.c_char_p),
        ("size", ctypes.c_int),
        ("last_hidden_layer", RKLLMResultLastHiddenLayer),
    ]


RKLLM_Handle_t = ctypes.c_void_p
RKLLMCallback = ctypes.CFUNCTYPE(
    None, ctypes.POINTER(RKLLMResult), ctypes.c_void_p, ctypes.c_int
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
        self._handle = RKLLM_Handle_t()
        self._lock = threading.Lock()
        self._queue: queue.Queue[object] = queue.Queue()
        self._error: str | None = None
        # Keep a reference so the callback is not garbage collected.
        self._callback = RKLLMCallback(self._on_result)
        self._init_model(model_path, max_context_len, max_new_tokens)

    def _on_result(self, result_ptr, _userdata, state: int) -> None:
        if state == LLMCallState.RKLLM_RUN_NORMAL and result_ptr:
            result = result_ptr.contents
            if result.text:
                self._queue.put(result.text.decode("utf-8", errors="replace"))
        elif state == LLMCallState.RKLLM_RUN_FINISH:
            self._queue.put(_FINISH)
        elif state == LLMCallState.RKLLM_RUN_ERROR:
            self._error = "RKLLM inference error"
            self._queue.put(_FINISH)

    def _init_model(
        self, model_path: str, max_context_len: int, max_new_tokens: int
    ) -> None:
        param = RKLLMParam()
        param.model_path = model_path.encode("utf-8")
        param.max_context_len = max_context_len
        param.max_new_tokens = max_new_tokens
        param.top_k = 40
        param.top_p = 0.9
        param.temperature = 0.8
        param.repeat_penalty = 1.1
        param.skip_special_token = True
        param.is_async = False

        self._lib.rkllm_init.argtypes = [
            ctypes.POINTER(RKLLM_Handle_t),
            ctypes.POINTER(RKLLMParam),
            RKLLMCallback,
        ]
        self._lib.rkllm_init.restype = ctypes.c_int
        ret = self._lib.rkllm_init(
            ctypes.byref(self._handle), ctypes.byref(param), self._callback
        )
        if ret != 0:
            raise RuntimeError(f"rkllm_init failed: error code {ret}")
        logger.info(f"RKLLM model initialized: {model_path}")

    def generate_stream(self, prompt: str) -> Iterator[str]:
        """Stream text fragments for ``prompt``; blocks the rkllm_run in a thread."""
        with self._lock:
            self._error = None
            while not self._queue.empty():
                _ = self._queue.get_nowait()

            rkllm_input = RKLLMInput()
            rkllm_input.input_mode = RKLLMInputMode.RKLLM_INPUT_PROMPT
            rkllm_input.input_data.prompt_input = prompt.encode("utf-8")

            infer_param = RKLLMInferParam()
            infer_param.mode = RKLLMInferMode.RKLLM_INFER_GENERATE
            infer_param.lora_params = None
            infer_param.prompt_cache_params = None

            self._lib.rkllm_run.argtypes = [
                RKLLM_Handle_t,
                ctypes.POINTER(RKLLMInput),
                ctypes.POINTER(RKLLMInferParam),
                ctypes.c_void_p,
            ]
            self._lib.rkllm_run.restype = ctypes.c_int

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
                if isinstance(item, str):
                    yield item
            worker.join(timeout=5)
            if self._error is not None:
                raise RuntimeError(self._error)

    def release(self) -> None:
        if self._handle:
            self._lib.rkllm_destroy.argtypes = [RKLLM_Handle_t]
            self._lib.rkllm_destroy.restype = ctypes.c_int
            _ = self._lib.rkllm_destroy(self._handle)
            self._handle = RKLLM_Handle_t()
