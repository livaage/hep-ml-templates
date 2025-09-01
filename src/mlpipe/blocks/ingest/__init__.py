# Import all ingest blocks to register them
try:
    from . import csv_loader  # noqa: F401
except ImportError:
    pass

try:
    from . import graph_csv_loader  # noqa: F401
except ImportError:
    pass

# NOTE: uproot_loader is NOT imported by default to avoid dependency issues
# It will only be imported when explicitly needed through registry.get() 
# when a configuration specifies "ingest.uproot_loader"
