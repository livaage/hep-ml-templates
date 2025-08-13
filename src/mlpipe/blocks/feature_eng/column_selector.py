import pandas as pd
from mlpipe.core.interfaces import FeatureBlock
from mlpipe.core.registry import register

@register("feature.column_selector")
class ColumnSelector(FeatureBlock):
    def __init__(self, include=None, exclude=None):
        self.include = include or []
        self.exclude = exclude or []

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        cols = list(X.columns)
        if self.include:
            cols = [c for c in cols if c in self.include]
        if self.exclude:
            cols = [c for c in cols if c not in self.exclude]
        return X[cols]
