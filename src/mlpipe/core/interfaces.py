from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple

import pandas as pd


class DataIngestor(ABC):
    @abstractmethod
    def load(self) -> Tuple[pd.DataFrame, pd.Series, Dict[str, Any]]:
        """Load data and return features, target, and metadata.

        Returns:
            Tuple of (features_dataframe, target_series, metadata_dict)
        """
        ...


class Preprocessor(ABC):
    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "Preprocessor": ...

    @abstractmethod
    def transform(self, X: pd.DataFrame) -> pd.DataFrame: ...


class FeatureBlock(ABC):
    @abstractmethod
    def transform(self, X: pd.DataFrame) -> pd.DataFrame: ...


class ModelBlock(ABC):
    @abstractmethod
    def build(self, config: Dict[str, Any]) -> None: ...

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series) -> None: ...

    @abstractmethod
    def predict(self, X: pd.DataFrame): ...


class Trainer(ABC):
    @abstractmethod
    def train(self, model: ModelBlock, X: pd.DataFrame, y: pd.Series, config: Dict[str, Any]): ...


class Evaluator(ABC):
    @abstractmethod
    def evaluate(self, y_true, y_pred, config: Dict[str, Any]) -> Dict[str, float]: ...
