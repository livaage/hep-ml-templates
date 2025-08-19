import pandas as pd
from mlpipe.core.interfaces import FeatureBlock
from mlpipe.core.registry import register

@register("feature.column_selector")
class ColumnSelector(FeatureBlock):
    """Column selector that works with sensible defaults.
    
    Can be used standalone:
        selector = ColumnSelector()  # Selects all columns
        X_selected = selector.transform(X)
    
    Or with specific columns:
        selector = ColumnSelector(include=['energy', 'momentum'])
        X_selected = selector.transform(X)
    """
    def __init__(self, include=None, exclude=None):
        """Initialize with optional column filters."""
        self.include = include or []
        self.exclude = exclude or []

    def transform(self, X: pd.DataFrame, config: dict = None) -> pd.DataFrame:
        """Transform data by selecting columns. Config is optional."""
        include = self.include
        exclude = self.exclude
        
        # Allow config override for pipeline usage
        if config:
            include = config.get('include', include) or include
            exclude = config.get('exclude', exclude) or exclude
        
        cols = list(X.columns)
        
        if include:
            cols = [c for c in cols if c in include]
        if exclude:
            cols = [c for c in cols if c not in exclude]
        
        return X[cols]
