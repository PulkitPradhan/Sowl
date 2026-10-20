"""Phase 0 smoke test: the src/ag skeleton imports.

If this fails, the scaffold or the packaging config (pyproject/hatchling) is
broken -- it should never fail because of later code inside the packages.
"""

import importlib

import pytest

SUBPACKAGES = [
    "daemon",
    "client",
    "wire",
    "models",
    "skills",
    "router",
    "planner",
    "laya",
    "browser",
    "popup",
    "workspace",
    "eval",
]

MODULES = ["ag", *(f"ag.{name}" for name in SUBPACKAGES)]


@pytest.mark.parametrize("module", MODULES)
def test_module_imports(module: str) -> None:
    importlib.import_module(module)
