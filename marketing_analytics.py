"""Lowercase runtime alias for the Marketing_analytics package."""

from __future__ import annotations

import importlib
import sys

_module = importlib.import_module("Marketing_analytics")
sys.modules[__name__] = _module
