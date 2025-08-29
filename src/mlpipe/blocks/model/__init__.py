# Optional imports - only load if dependencies are available
try:
    from . import xgb_classifier          # registers "model.xgb_classifier"
except ImportError:
    pass  # XGBoost not available

try:
    from . import decision_tree           # registers "model.decision_tree"
except ImportError:
    pass  # scikit-learn not available (though it should be in core dependencies)

try:
    from . import ensemble_models         # registers "model.random_forest", "model.adaboost", "model.ensemble_voting"
except ImportError:
    pass  # scikit-learn not available

try:
    from . import svm                     # registers "model.svm"
except ImportError:
    pass  # scikit-learn not available

try:
    from . import mlp                     # registers "model.mlp"
except ImportError:
    pass  # scikit-learn not available

# Neural network models - require torch/lightning dependencies
try:
    from . import ae_lightning            # registers "model.ae_vanilla", "model.ae_variational"
except ImportError:
    pass  # PyTorch Lightning not available

try:
    from . import gnn_pyg                 # registers "model.gnn_gcn", "model.gnn_gat"
except ImportError:
    pass  # PyTorch Geometric not available

try:
    from . import hep_neural              # registers "model.transformer_hep", "model.cnn_hep"
except ImportError:
    pass  # PyTorch/Lightning not available
