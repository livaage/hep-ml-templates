from pathlib import Path

from mlpipe.pipelines.xgb_basic.run import run_pipeline


def test_xgb_pipeline(tmp_path: Path, capsys):
    # use default configs; demo data will be created
    run_pipeline(pipeline="xgb_basic", config_path="configs", config_name="pipeline")
    out = capsys.readouterr().out
    assert "Metrics" in out
    assert "accuracy" in out.lower()
