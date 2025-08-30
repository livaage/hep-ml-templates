# Import all ingest blocks to register them
try:
    from . import csv_loader  # noqa: F401
except ImportError:
    pass

try:
    from . import higgs_loader  # noqa: F401
except ImportError:
    pass  # Optional dependency

try:
    from . import uproot_loader  # noqa: F401
except ImportError:
    pass  # Optional dependency
