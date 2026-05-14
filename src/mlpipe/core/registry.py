from __future__ import annotations

import importlib
from collections.abc import Callable

_REGISTRY: dict[str, object] = {}

# Lazy loading mappings for optional blocks
_LAZY_IMPORTS = {
    "ingest.uproot_loader": "mlpipe.blocks.ingest.uproot_loader",
    # Allow lazy loading of gnn module so top-level import doesn't hard fail
    "model.gnn_gcn": "mlpipe.blocks.model.gnn_pyg",
    "model.gnn_gat": "mlpipe.blocks.model.gnn_pyg",
}


def register(name: str) -> Callable[[type], type]:
    def deco(cls: type) -> type:
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
                ) from e

        raise KeyError(f"Unknown block: {name}. Known: {list(_REGISTRY)}")
    return _REGISTRY[name]


def list_blocks():
    return sorted(_REGISTRY.keys())
