from __future__ import annotations
from typing import Callable, Dict, Type

_REGISTRY: Dict[str, object] = {}

def register(name: str) -> Callable[[Type], Type]:
    def deco(cls: Type) -> Type:
        if name in _REGISTRY:
            raise ValueError(f"Block name already registered: {name}")
        _REGISTRY[name] = cls
        return cls
    return deco

def get(name: str):
    if name not in _REGISTRY:
        raise KeyError(f"Unknown block: {name}. Known: {list(_REGISTRY)}")
    return _REGISTRY[name]

def list_blocks():
    return sorted(_REGISTRY.keys())
