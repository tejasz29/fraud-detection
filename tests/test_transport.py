"""Tests for the pluggable Transport abstraction."""
from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import load_settings  # noqa: E402
from transport import Transport, get_transport  # noqa: E402
from transport.kafka_transport import KafkaTransport  # noqa: E402
from transport.memory_transport import MemoryTransport  # noqa: E402


def _settings(transport: str):
    return replace(load_settings(), transport=transport)


def test_get_transport_selects_kafka() -> None:
    assert isinstance(get_transport(_settings("kafka")), KafkaTransport)


def test_get_transport_selects_memory() -> None:
    assert isinstance(get_transport(_settings("memory")), MemoryTransport)


def test_get_transport_rejects_unknown() -> None:
    with pytest.raises(ValueError):
        get_transport(_settings("rabbitmq"))


def test_memory_transport_is_a_transport() -> None:
    assert isinstance(MemoryTransport(), Transport)


def test_memory_roundtrip() -> None:
    t = MemoryTransport()
    t.connect()
    t.send({"transaction_id": "txn-000001", "Amount": 12.0})

    received: list[dict] = []
    for msg in t.receive(should_stop=lambda: len(received) >= 1):
        received.append(msg)
        break  # stop after the first message

    assert received == [{"transaction_id": "txn-000001", "Amount": 12.0}]
    t.close()
