from __future__ import annotations

import importlib
from typing import Callable, Dict, Type

_REGISTRY: Dict[str, object] = {}

# Lazy loading mappings for optional blocks
_LAZY_IMPORTS = {"ingest.uproot_loader": "mlpipe.blocks.ingest.uproot_loader"}


def register(name: str) -> Callable[[Type], Type]:
    def deco(cls: Type) -> Type:
        if name in _REGISTRY:
            raise ValueError(f"Block name already registered: {name}")
        _REGISTRY[name] = cls
        return cls

    return deco


def get(name: str):
    if name not in _REGISTRY:
        # Try lazy loading if it's a known optional block
        if name in _LAZY_IMPORTS:
            try:
                module_path = _LAZY_IMPORTS[name]
                importlib.import_module(module_path)
                # After import, the block should be registered
                if name in _REGISTRY:
                    return _REGISTRY[name]
                else:
                    raise ImportError(
                        f"Block {name} could not be registered (likely missing dependencies)"
                    )
            except ImportError as e:
                raise ImportError(
                    f"Block {name} requires additional dependencies. "
                    f"Try: pip install hep-ml-templates[data-uproot]. "
                    f"Error: {e}"
                )

        raise KeyError(f"Unknown block: {name}. Known: {list(_REGISTRY)}")
    return _REGISTRY[name]


def list_blocks():
    return sorted(_REGISTRY.keys())
