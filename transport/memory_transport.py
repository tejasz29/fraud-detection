"""In-process transport for tests and single-process demos.

Uses a shared module-level queue so two ``MemoryTransport`` instances created in
the *same* process communicate. It needs no external broker, which makes the
producer/consumer logic testable without Kafka. It is not for cross-process use.
"""
from __future__ import annotations

import queue
from typing import Iterator

from transport import Transport

_SHARED: "queue.Queue[dict]" = queue.Queue()


class MemoryTransport(Transport):
    """Passes transactions through an in-memory queue (same process only)."""

    def __init__(self) -> None:
        self._queue = _SHARED

    def connect(self) -> None:
        return None

    def send(self, transaction: dict) -> None:
        self._queue.put(transaction)

    def receive(self, should_stop) -> Iterator[dict]:
        while not should_stop():
            try:
                yield self._queue.get(timeout=0.2)
            except queue.Empty:
                continue

    def close(self) -> None:
        return None
