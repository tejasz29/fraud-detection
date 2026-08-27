"""Tests for centralized configuration (config.py)."""
from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import config  # noqa: E402


def test_defaults() -> None:
    s = config.load_settings()
    assert s.kafka_bootstrap_servers == "localhost:9092"
    assert s.kafka_topic == "transactions"
    assert s.transport == "kafka"
    assert s.dashboard_port == 8501
    assert s.model_path.name == "model.pkl"
    assert s.kafka_enabled is True


def test_env_overrides(monkeypatch) -> None:
    monkeypatch.setenv("TRANSPORT", "memory")
    monkeypatch.setenv("KAFKA_BOOTSTRAP_SERVERS", "broker:9092")
    monkeypatch.setenv("DASHBOARD_PORT", "9000")
    s = config.load_settings()
    assert s.transport == "memory"
    assert s.kafka_enabled is False
    assert s.kafka_bootstrap_servers == "broker:9092"
    assert s.dashboard_port == 9000


def test_paths_resolve_absolute() -> None:
    s = config.load_settings()
    assert s.db_path.is_absolute()
    assert s.model_path.is_absolute()
