"""Tests for training helpers in consumer/train.py (pure-numpy, no dataset needed)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from consumer.train import tune_threshold  # noqa: E402


def test_tune_threshold_keeps_default_when_already_optimal() -> None:
    # Perfectly separated data: default 0.5 is already the best cut.
    y = np.array([0] * 100 + [1] * 100)
    proba = np.array([0.1] * 100 + [0.9] * 100)
    assert tune_threshold(y, proba, default=0.5, min_gain=0.01) == 0.5


def test_tune_threshold_returns_valid_float() -> None:
    rng = np.random.default_rng(2)
    y = rng.integers(0, 2, size=500)
    proba = rng.uniform(0.01, 0.99, size=500)
    t = tune_threshold(y, proba, default=0.5, min_gain=0.01)
    assert isinstance(t, float)
    assert 0.0 <= t <= 1.0
