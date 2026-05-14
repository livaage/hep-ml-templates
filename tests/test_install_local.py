"""Verify the `install-local` scaffold produces a working project layout.

This test calls `install_local()` directly (no subprocess) and asserts that
the expected files are written into the target directory. It does NOT try
to `pip install` the resulting project — that's covered by manual smoke
testing because it's slow and OS-dependent.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from mlpipe.cli.local_install import install_local


def test_install_local_pipeline_xgb_scaffolds_a_project(tmp_path: Path):
    target = tmp_path / "my-project"
    ok = install_local(extras=["pipeline-xgb"], target_dir=str(target))
    assert ok is True

    # The scaffold should produce a buildable Python package + configs.
    assert (target / "setup.py").is_file()
    assert (target / "mlpipe").is_dir()
    assert (target / "configs" / "pipeline.yaml").is_file()
    # The XGB pipeline pulls in these blocks/configs:
    assert (target / "mlpipe" / "blocks" / "model" / "xgb_classifier.py").is_file()
    assert (target / "configs" / "model" / "xgb_classifier.yaml").is_file()


def test_install_local_rejects_unknown_extra(tmp_path: Path):
    target = tmp_path / "bogus-project"
    ok = install_local(extras=["definitely-not-an-extra"], target_dir=str(target))
    # install_local catches everything and returns False on failure; we just
    # need to confirm it doesn't silently succeed.
    assert ok is False or not (target / "setup.py").is_file()


@pytest.mark.parametrize(
    "extra,expected_block",
    [
        ("pipeline-decision-tree", "mlpipe/blocks/model/decision_tree.py"),
        ("pipeline-ensemble", "mlpipe/blocks/model/ensemble_models.py"),
    ],
)
def test_install_local_other_pipelines(tmp_path: Path, extra: str, expected_block: str):
    target = tmp_path / extra
    ok = install_local(extras=[extra], target_dir=str(target))
    assert ok is True
    assert (target / expected_block).is_file()
    assert (target / "configs" / "pipeline.yaml").is_file()
