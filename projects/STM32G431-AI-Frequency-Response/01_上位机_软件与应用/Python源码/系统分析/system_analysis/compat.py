from __future__ import annotations

from importlib import import_module
from types import ModuleType
from typing import Any


def export_module(module_name: str, namespace: dict[str, Any]) -> ModuleType:
    module = import_module(module_name)
    names = [name for name in vars(module) if not (name.startswith("__") and name.endswith("__"))]
    namespace.update({name: getattr(module, name) for name in names})
    namespace["__all__"] = names
    namespace["__doc__"] = module.__doc__
    return module

