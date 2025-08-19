# Import all ingest blocks to register them
try:
    from . import csv_loader
except ImportError:
    pass

try:
    from . import higgs_loader
except ImportError:
    pass  # Optional dependency