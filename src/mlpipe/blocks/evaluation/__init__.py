# Import all evaluation blocks to register them
from . import classification_metrics  # noqa: F401

try:
    from . import reconstruction_metrics  # noqa: F401
except ImportError:
    pass  # Optional dependency
