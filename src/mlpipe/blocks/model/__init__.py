# Optional imports - only load if dependencies are available
try:
    from . import xgb_classifier          # registers "model.xgb_classifier"
except ImportError:
    pass  # XGBoost not available

try:
    from . import decision_tree           # registers "model.decision_tree" 
except ImportError:
    pass  # scikit-learn not available (though it should be in core dependencies)

# Stub imports for future neural network models
# These will only work if torch/lightning dependencies are available
try:
    from . import ae_lightning            # registers "model.ae_lightning"
except ImportError:
    pass  # PyTorch Lightning not available

try:
    from . import gnn_pyg                 # registers "model.gnn_pyg"
except ImportError:
    pass  # PyTorch Geometric not available
