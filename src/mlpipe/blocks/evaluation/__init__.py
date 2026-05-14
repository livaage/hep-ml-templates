# Import all evaluation blocks to register them
try:
    from . import classification_metrics  # noqa: F401
except ImportError:
    pass  # Module may not be installed in local installation

try:
    from . import reconstruction_metrics  # noqa: F401
except ImportError:
    pass  # Optional dependency
