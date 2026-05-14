# Import preprocessing blocks for their @register side effects.
from . import data_split, standard_scaler  # noqa: F401

try:
    from . import onehot_encoder  # noqa: F401
except ImportError:
    pass  # Optional dependency
