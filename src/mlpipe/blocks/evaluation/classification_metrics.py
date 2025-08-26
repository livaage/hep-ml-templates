from typing import Dict, Any
from sklearn.metrics import roc_auc_score, accuracy_score
from mlpipe.core.interfaces import Evaluator
from mlpipe.core.registry import register
import numpy as np


@register("eval.classification")
class ClassificationEvaluator(Evaluator):
    def evaluate(self, y_true, y_pred, config: Dict[str, Any]) -> Dict[str, float]:
        threshold = float(config.get("threshold", 0.5))
        # If y_pred are probabilities, threshold; if labels, handle gracefully
        if y_pred.ndim == 1 and np.issubdtype(y_pred.dtype, np.floating):
            probs = y_pred
            preds = (probs >= threshold).astype(int)
            try:
                auc = roc_auc_score(y_true, probs)
            except ValueError:
                auc = float("nan")
        else:
            preds = y_pred
            auc = float("nan")
        acc = accuracy_score(y_true, preds)
        return {"auc": float(auc), "accuracy": float(acc)}
