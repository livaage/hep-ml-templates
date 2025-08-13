from mlpipe.core.registry import get
def test_known_blocks():
    assert get("preprocessing.standard_scaler")
    assert get("feature.column_selector")
    assert get("model.xgb_classifier")
    assert get("train.sklearn")
    assert get("eval.classification")
