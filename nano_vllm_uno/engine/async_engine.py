from __future__ import annotations

import concurrent.futures
import queue
import threading
import uuid
from dataclasses import dataclass
from typing import Any

from nano_vllm_uno.sampling_params import SamplingParams


@dataclass(frozen=True)
class GenerationResult:
    request_id: str
    text: str
    token_ids: list[int]
    stats: dict[str, int]


@dataclass
class PendingRequest:
    request_id: str
    prompt: str | list[int]
    future: concurrent.futures.Future[GenerationResult]
    max_tokens: int | None = None
    seq_id: int | None = None


_STOP = object()


class AsyncLLMEngine:
    """Run one synchronous LLM engine behind a persistent request queue."""

    def __init__(
        self,
        engine: Any,
        sampling_params: SamplingParams,
    ) -> None:
        self.engine = engine
        self.sampling_params = sampling_params
        self._incoming: queue.Queue[PendingRequest | object] = queue.Queue()
        self._seq_to_request: dict[int, PendingRequest] = {}
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._accepting = False
        self._fatal_error: BaseException | None = None

    def start(self) -> None:
        with self._lock:
            if self._thread is not None:
                if self._thread.is_alive():
                    return
                raise RuntimeError("AsyncLLMEngine cannot be restarted after shutdown")
            self._accepting = True
            self._thread = threading.Thread(
                target=self._run_forever,
                name="nano-vllm-uno-engine",
                daemon=True,
            )
            self._thread.start()

    def submit(
        self,
        prompt: str | list[int],
        *,
        request_id: str | None = None,
        max_tokens: int | None = None,
    ) -> concurrent.futures.Future[GenerationResult]:
        with self._lock:
            if not self._accepting:
                if self._fatal_error is not None:
                    raise RuntimeError("AsyncLLMEngine has failed") from self._fatal_error
                raise RuntimeError("AsyncLLMEngine is not running")
            future: concurrent.futures.Future[GenerationResult] = (
                concurrent.futures.Future()
            )
            request = PendingRequest(
                request_id=request_id or uuid.uuid4().hex,
                prompt=prompt,
                future=future,
                max_tokens=max_tokens,
            )
            self._incoming.put(request)
            return future

    def shutdown(self, timeout: float | None = 30.0) -> None:
        with self._lock:
            self._accepting = False
            thread = self._thread
            if thread is None:
                return
            self._incoming.put(_STOP)
        thread.join(timeout=timeout)
        if thread.is_alive():
            raise TimeoutError("Timed out waiting for AsyncLLMEngine to stop")

    @property
    def is_running(self) -> bool:
        with self._lock:
            thread = self._thread
            return bool(
                self._accepting
                and self._fatal_error is None
                and thread is not None
                and thread.is_alive()
            )

    @property
    def fatal_error(self) -> BaseException | None:
        with self._lock:
            return self._fatal_error

    def _admit(self, request: PendingRequest) -> None:
        if not request.future.set_running_or_notify_cancel():
            return
        try:
            seq_id = self.engine.add_request(
                request.prompt,
                self.sampling_params,
                max_tokens=request.max_tokens,
            )
        except BaseException as exc:
            request.future.set_exception(exc)
            return
        if seq_id in self._seq_to_request:
            error = RuntimeError(f"Duplicate scheduler sequence id: {seq_id}")
            request.future.set_exception(error)
            raise error
        request.seq_id = seq_id
        self._seq_to_request[seq_id] = request

    def _drain_incoming(self) -> bool:
        stop_requested = False
        while True:
            try:
                item = self._incoming.get_nowait()
            except queue.Empty:
                break
            if item is _STOP:
                stop_requested = True
            else:
                self._admit(item)
        return stop_requested

    def _resolve_finished(
        self,
        outputs: list[tuple[int, list[int], dict[str, int]]],
    ) -> None:
        for seq_id, token_ids, stats in outputs:
            request = self._seq_to_request.pop(seq_id, None)
            if request is None:
                raise RuntimeError(
                    f"Completed scheduler sequence {seq_id} has no pending request"
                )
            output = self.engine.finalize_output(
                token_ids,
                stats,
                self.sampling_params,
            )
            request.future.set_result(
                GenerationResult(
                    request_id=request.request_id,
                    text=str(output["text"]),
                    token_ids=list(output["token_ids"]),
                    stats=dict(output["stats"]),
                )
            )

    def _fail_pending(self, error: BaseException) -> None:
        for request in list(self._seq_to_request.values()):
            if not request.future.done():
                request.future.set_exception(error)
        self._seq_to_request.clear()
        while True:
            try:
                item = self._incoming.get_nowait()
            except queue.Empty:
                break
            if isinstance(item, PendingRequest) and not item.future.done():
                item.future.set_exception(error)

    def _run_forever(self) -> None:
        try:
            stop_requested = False
            while not stop_requested:
                if self.engine.is_finished():
                    item = self._incoming.get()
                    if item is _STOP:
                        break
                    self._admit(item)

                stop_requested = self._drain_incoming()
                if stop_requested:
                    break

                if not self.engine.is_finished():
                    outputs, _ = self.engine.step()
                    self._resolve_finished(outputs)
        except BaseException as exc:
            with self._lock:
                self._fatal_error = exc
            self._fail_pending(exc)
        finally:
            with self._lock:
                self._accepting = False
            self._fail_pending(
                RuntimeError("AsyncLLMEngine stopped before request completion")
            )
            self.engine.exit()
