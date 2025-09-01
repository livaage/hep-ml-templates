# Import all preprocessing blocks to register them
from . import (
    data_split,  # noqa: F401
    standard_scaler,  # noqa: F401
)

try:
    from . import onehot_encoder  # noqa: F401
except ImportError:
    pass  # Optional dependency
