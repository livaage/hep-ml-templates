"""Comprehensive pipeline tests for all HEP ML Templates pipeline types."""

import shutil
import pytest
import subprocess
import sys
import tempfile
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def run_pipeline_test(
    pipeline_name: str,
    model_block: str,
    training_block: str,
    eval_block: str,
    data_block: str = "ingest.csv",
    data_file: str = "demo_tabular.csv",
    model_overrides: str = None,
    return_output: bool = False,
):
    """Run a pipeline test using the modular installation method."""
    print(f"🔍 Testing {pipeline_name} Pipeline...")

    # Create temporary directory for this test
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        try:
            # Copy the main framework
            main_src = Path(__file__).parent.parent / "src"
            configs_src = Path(__file__).parent.parent / "configs"
            data_src = Path(__file__).parent.parent / "data"

            # Set up test environment
            test_src = tmp_path / "src"
            test_configs = tmp_path / "configs"
            test_data = tmp_path / "data"

            shutil.copytree(main_src, test_src)
            shutil.copytree(configs_src, test_configs)
            shutil.copytree(data_src, test_data)

            # Create setup.py for local installation
            setup_content = """
from setuptools import setup, find_packages

setup(
    name="hep_ml_templates_local",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "numpy", "pandas", "scikit-learn", "pyyaml", "omegaconf", "hydra-core",
        "torch", "pytorch-lightning", "xgboost", "matplotlib", "seaborn"
    ]
)
"""
            with open(tmp_path / "setup.py", "w") as f:
                f.write(setup_content)

            # Create CLI script
            cli_content = """#!/usr/bin/env python3
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from mlpipe.cli.main import main

if __name__ == "__main__":
    main()
"""
            cli_path = tmp_path / "mlpipe_cli.py"
            with open(cli_path, "w") as f:
                f.write(cli_content)
            cli_path.chmod(0o755)

            # Install locally
            result = subprocess.run(
                [sys.executable, "-m", "pip", "install", "-e", "."],
                cwd=tmp_path,
                capture_output=True,
                text=True,
            )

            if result.returncode != 0:
                print(f"  ❌ Local installation failed: {result.stderr}")
                return False

            # Create overrides for the specific pipeline
            overrides = [
                "data=csv_demo",
                f"model={model_block.split('.')[-1]}",  # Extract just the model name
                f"training={training_block.split('.')[-1]}",  # Extract just the training name
                f"evaluation={eval_block.split('.')[-1]}",  # Extract just the eval name
            ]

            # Handle special data cases
            if data_block == "ingest.graph_csv":
                overrides[0] = "data=graph_demo"

            # Add model-specific parameter overrides
            if model_overrides:
                # Split multiple overrides by spaces and add each one
                override_list = model_overrides.split()
                overrides.extend(override_list)

            # Run the pipeline
            cmd = [
                sys.executable,
                "mlpipe_cli.py",
                "run",
                "--config-path",
                str(test_configs),
                "--config-name",
                "pipeline",
                "--overrides",
            ] + overrides

            result = subprocess.run(cmd, cwd=tmp_path, capture_output=True, text=True, timeout=300)

            success = (
                result.returncode == 0
                and "Pipeline execution completed successfully" in result.stdout
            )

            if success:
                print(f"  ✅ {pipeline_name} pipeline completed successfully")
            else:
                print(f"  ❌ {pipeline_name} pipeline failed:")
                print(f"    Return code: {result.returncode}")
                if result.stderr:
                    print(f"    Error: {result.stderr[:500]}...")  # Show more error details
                if result.stdout:
                    print(f"    Output: {result.stdout[-300:]}")  # Show end of stdout

            if return_output:
                return success, result.stdout, result.stderr
            return success

        except subprocess.TimeoutExpired:
            print(f"  ❌ {pipeline_name} pipeline timed out after 5 minutes")
            return False
        except Exception as e:
            print(f"  ❌ {pipeline_name} pipeline failed with exception: {e}")
            return False


def test_xgb_pipeline():
    """Test XGBoost classification pipeline."""
    assert run_pipeline_test(
        "XGBoost", "model.xgb_classifier", "train.sklearn", "eval.classification"
    )


def test_decision_tree_pipeline():
    """Test Decision Tree classification pipeline."""
    assert run_pipeline_test(
        "Decision Tree", "model.decision_tree", "train.sklearn", "eval.classification"
    )


def test_random_forest_pipeline():
    """Test Random Forest classification pipeline."""
    assert run_pipeline_test(
        "Random Forest", "model.random_forest", "train.sklearn", "eval.classification"
    )


def test_ensemble_pipeline():
    """Test Ensemble Voting pipeline."""
    assert run_pipeline_test(
        "Ensemble", "model.ensemble_voting", "train.sklearn", "eval.classification"
    )


def test_neural_pipeline():
    """Test MLP Neural Network pipeline."""
    assert run_pipeline_test("Neural Network", "model.mlp", "train.sklearn", "eval.classification")


def test_gnn_pipeline():
    """Test Graph Neural Network pipeline.

    If torch_geometric is unavailable, assert we surface a clear, actionable error message
    instead of crashing or providing a cryptic failure.
    """
    tg = pytest.importorskip if False else None  # silence linter about unused importorskip
    try:
        import torch_geometric  # type: ignore
        has_tg = True
    except Exception:
        has_tg = False

    if has_tg:
        assert run_pipeline_test(
            "GNN",
            "model.gnn_gcn",
            "train.sklearn",
            "eval.classification",
            data_block="ingest.graph_csv",
            data_file="graph_nodes_demo.csv",
            model_overrides="model.params.task=node",
        )
    else:
        success, out, err = run_pipeline_test(
            "GNN",
            "model.gnn_gcn",
            "train.sklearn",
            "eval.classification",
            data_block="ingest.graph_csv",
            data_file="graph_nodes_demo.csv",
            model_overrides="model.params.task=node",
            return_output=True,
        )
        assert success is False
        combined = (err or "") + "\n" + (out or "")
        assert "Torch Geometric" in combined or "torch_geometric" in combined


def test_vanilla_autoencoder_pipeline():
    """Test Vanilla Autoencoder pipeline."""
    assert run_pipeline_test(
        "Vanilla Autoencoder",
        "model.ae_vanilla",
        "train.pytorch",
        "eval.reconstruction",
        model_overrides="training.params.accelerator=cpu training.params.max_epochs=5",
    )


def test_variational_autoencoder_pipeline():
    """Test Variational Autoencoder pipeline."""
    assert run_pipeline_test(
        "Variational Autoencoder",
        "model.ae_variational",
        "train.pytorch",
        "eval.reconstruction",
        model_overrides="training.params.accelerator=cpu training.params.max_epochs=5",
    )


def main():
    """Run comprehensive pipeline tests."""
    print("🧪 Running Comprehensive HEP ML Pipeline Tests")
    print("=" * 60)

    test_functions = [
        test_xgb_pipeline,
        test_decision_tree_pipeline,
        test_random_forest_pipeline,
        test_ensemble_pipeline,
        test_neural_pipeline,
        # test_gnn_pipeline,
        test_vanilla_autoencoder_pipeline,
        test_variational_autoencoder_pipeline,
    ]

    results = []

    for test_func in test_functions:
        try:
            success = test_func()
            results.append(success)
        except Exception as e:
            print(f"❌ {test_func.__name__} crashed: {e}")
            results.append(False)

        print()  # Add spacing between tests

    # Summary
    passed = sum(results)
    total = len(results)

    print("=" * 60)
    print(f"📊 Pipeline Test Results: {passed}/{total} pipelines passed")

    if passed == total:
        print("🎉 All pipeline tests passed! HEP ML Templates is working perfectly!")
        return 0
    else:
        print(f"⚠️  {total - passed} pipeline tests failed. Check logs above.")
        return 1


if __name__ == "__main__":
    exit(main())
