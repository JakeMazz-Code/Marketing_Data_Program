"""Bootstrap imports so CLI scripts can reach the package without sys.path hacks."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType


_PACKAGE_NAME = "Marketing_analytics"


def ensure_package_imported() -> ModuleType:
    """Load the Marketing_analytics package when invoked as a script."""

    module = sys.modules.get(_PACKAGE_NAME)
    if module is not None:
        return module

    package_dir = Path(__file__).resolve().parents[1] / _PACKAGE_NAME
    init_path = package_dir / "__init__.py"
    if not init_path.exists():
        raise ModuleNotFoundError(f"Cannot locate {_PACKAGE_NAME} package at {package_dir}")

    spec = importlib.util.spec_from_file_location(
        _PACKAGE_NAME,
        init_path,
        submodule_search_locations=[str(package_dir)],
    )
    if spec is None or spec.loader is None:
        raise ModuleNotFoundError(f"Unable to load {_PACKAGE_NAME} from {init_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[_PACKAGE_NAME] = module
    spec.loader.exec_module(module)
    return module
