import pandas as pd
from sklearn.preprocessing import StandardScaler

from mlpipe.core.interfaces import Preprocessor
from mlpipe.core.registry import register


@register("preprocessing.standard_scaler")
class StandardScalerBlock(Preprocessor):
    def __init__(self):
        self.scaler = StandardScaler()
        self.columns = None

    def fit(self, X: pd.DataFrame, y=None):
        self.columns = X.columns
        self.scaler.fit(X.values)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        Xs = self.scaler.transform(X.values)
        return pd.DataFrame(Xs, columns=self.columns, index=X.index)
