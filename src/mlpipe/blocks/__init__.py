# Always available blocks
from .ingest import csv_loader                 # registers "ingest.csv"  # noqa: F401
from .preprocessing import standard_scaler     # registers "preprocessing.standard_scaler"  # noqa: F401  
from .feature_eng import column_selector       # registers "feature.column_selector"  # noqa: F401
from .training import sklearn_trainer          # registers "train.sklearn"  # noqa: F401
from .evaluation import classification_metrics  # registers "eval.classification"  # noqa: F401

# Import model blocks (they handle their own optional imports)
from . import model  # This will import available models based on dependencies


# Function to register commonly used blocks for testing/demo purposes
def register_all_available_blocks():
    """
    Register all available blocks. Use this for testing or when you want
    to use the registry-based block access pattern.

    This function only imports blocks that have their dependencies available.
    """
    # Blocks are already imported at module level, so this function
    # is mainly for backward compatibility. The imports at the top
    # of this file already register all available blocks.
    pass


# For backward compatibility, you can call register_all_available_blocks()
# But users should prefer importing blocks directly for better modularity
