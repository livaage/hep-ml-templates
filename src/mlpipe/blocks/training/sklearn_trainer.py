from typing import Dict, Any
from mlpipe.core.interfaces import Trainer, ModelBlock
from mlpipe.core.registry import register

@register("train.sklearn")
class SklearnTrainer(Trainer):
    def train(self, model: ModelBlock, X, y, config: Dict[str, Any]):
        # config dict is accepted for symmetry, but XGB fit uses its own params already
        model.fit(X, y)
        return model
