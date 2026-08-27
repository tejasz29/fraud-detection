"""Pluggable message-transport abstraction.

The producer and consumer depend only on :class:`Transport`, never on Kafka
directly. Swap implementations by changing the ``TRANSPORT`` setting — no code
changes in the services.

Implementations:
    * :class:`KafkaTransport`  — real broker (default).
    * :class:`MemoryTransport` — in-process queue, for tests / single-process demos.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

from config import Settings


class Transport(ABC):
    """A point-to-point channel that carries transaction dicts producer -> consumer."""

    @abstractmethod
    def connect(self) -> None:
        """Open any underlying connections (producer, consumer, or both)."""

    @abstractmethod
    def send(self, transaction: dict) -> None:
        """Publish one transaction to the channel."""

    @abstractmethod
    def receive(self, should_stop) -> "Iterator[dict]":
        """Yield transactions until ``should_stop()`` is true.

        ``should_stop`` is a zero-arg callable checked between messages so the
        loop can exit promptly on shutdown signals.
        """

    def flush(self) -> None:
        """Best-effort flush of any buffered outgoing messages."""

    def close(self) -> None:
        """Release connections / resources."""


def get_transport(settings: Settings) -> Transport:
    """Construct the configured transport implementation."""
    if settings.transport == "kafka":
        from transport.kafka_transport import KafkaTransport

        return KafkaTransport(settings.kafka_bootstrap_servers, settings.kafka_topic)
    if settings.transport == "memory":
        from transport.memory_transport import MemoryTransport

        return MemoryTransport()
    raise ValueError(f"unknown TRANSPORT={settings.transport!r} (expected 'kafka' or 'memory')")
