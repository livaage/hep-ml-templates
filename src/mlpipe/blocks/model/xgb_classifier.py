from typing import Any, Dict
from xgboost import XGBClassifier
from mlpipe.core.interfaces import ModelBlock
from mlpipe.core.registry import register

@register("model.xgb_classifier")
class XGBClassifierBlock(ModelBlock):
    def __init__(self):
        self.model: XGBClassifier | None = None

    def build(self, config: Dict[str, Any]) -> None:
        self.model = XGBClassifier(**config)

    def fit(self, X, y) -> None:
        assert self.model is not None, "Call build(config) before fit."
        self.model.fit(X, y)

    def predict(self, X):
        assert self.model is not None
        # Prefer probabilities if available; fallback to decision_function/labels
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X)[:, 1]
        return self.model.predict(X)
