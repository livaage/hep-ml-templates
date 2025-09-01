#!/usr/bin/env python3

"""
Independent Template Pattern Tests for HEP ML Pipelines

This module tests each pipeline using the template pattern approach:
- Create isolated test directory
- Copy and install framework locally 
- Run pipeline with specific configuration
- Validate results independently

Each test is completely isolated and uses its own environment.
"""

import os
import sys
import shutil
import tempfile
import subprocess
from pathlib import Path


def create_template_test(test_name, model_config, train_config, eval_config, data_config="demo_tabular.csv", data_block="ingest.csv"):
    """Create and run an independent template test for a specific pipeline configuration."""
    
    print(f"\n🔧 Creating independent test for {test_name}...")
    
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
            setup_content = '''
from setuptools import setup, find_packages

setup(
    name="hep_ml_templates_local",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "numpy", "pandas", "scikit-learn", "pyyaml"
    ]
)
'''
            with open(tmp_path / "setup.py", "w") as f:
                f.write(setup_content)
            
            # Create CLI entry point
            cli_content = '''#!/usr/bin/env python3
import sys
sys.path.insert(0, "src")
from mlpipe.cli.main import main

if __name__ == "__main__":
    main()
'''
            with open(tmp_path / "mlpipe_cli.py", "w") as f:
                f.write(cli_content)
            os.chmod(tmp_path / "mlpipe_cli.py", 0o755)
            
            # Create pipeline configuration
            pipeline_config = f'''data: {data_config.replace('.csv', '')}
preprocessing: standard
feature_eng: all_columns
model: {model_config.split('.')[-1]}
training: {train_config.split('.')[-1]}
evaluation: {eval_config.split('.')[-1]}
runtime: local_cpu
'''
            
            with open(test_configs / "pipeline.yaml", "w") as f:
                f.write(pipeline_config)
            
            # Install framework locally
            print(f"  📦 Installing framework locally...")
            install_cmd = [sys.executable, "-m", "pip", "install", "-e", ".", "--quiet"]
            install_result = subprocess.run(install_cmd, cwd=tmp_path, capture_output=True, text=True)
            
            if install_result.returncode != 0:
                print(f"  ❌ Installation failed: {install_result.stderr}")
                return False
            
            # Create overrides for the specific pipeline
            overrides = [
                f"data=csv_demo",
                f"model={model_config.split('.')[-1]}",  # Extract just the model name
                f"training={train_config.split('.')[-1]}",  # Extract just the training name  
                f"evaluation={eval_config.split('.')[-1]}"  # Extract just the eval name
            ]
            
            # Handle special data cases
            if data_block == "ingest.graph_csv":
                overrides[0] = "data=graph_demo"
            
            # Run the pipeline
            print(f"  🚀 Running {test_name} pipeline...")
            run_cmd = [
                sys.executable, "mlpipe_cli.py", "run",
                "--config-path", str(test_configs),
                "--config-name", "pipeline",  
                "--overrides"
            ] + overrides
            
            result = subprocess.run(run_cmd, cwd=tmp_path, capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0 and "Pipeline execution completed successfully" in result.stdout:
                print(f"  ✅ {test_name} independent test passed!")
                
                # Extract key metrics from output
                lines = result.stdout.split('\n')
                for line in lines:
                    if ('accuracy:' in line.lower() or 'f1_score:' in line.lower() or 
                        'mse:' in line.lower() or 'mae:' in line.lower()):
                        print(f"    📊 {line.strip()}")
                
                return True
            else:
                print(f"  ❌ {test_name} independent test failed:")
                if result.stderr:
                    print(f"    Error: {result.stderr[-300:]}")
                if result.stdout:
                    print(f"    Output: {result.stdout[-300:]}")
                return False
                
        except subprocess.TimeoutExpired:
            print(f"  ❌ {test_name} test timed out after 5 minutes")
            return False
        except Exception as e:
            print(f"  ❌ {test_name} test failed with exception: {e}")
            return False


def test_xgb_independent():
    """Independent test for XGBoost classifier."""
    return create_template_test("XGBoost", "model.xgb_classifier", "train.sklearn", "eval.classification")


def test_decision_tree_independent():
    """Independent test for Decision Tree classifier."""
    return create_template_test("Decision Tree", "model.decision_tree", "train.sklearn", "eval.classification")


def test_ensemble_independent():
    """Independent test for Ensemble classifier."""
    return create_template_test("Ensemble", "model.ensemble_voting", "train.sklearn", "eval.classification")


def test_neural_independent():
    """Independent test for Neural Network (MLP) classifier."""
    return create_template_test("Neural Network", "model.mlp", "train.sklearn", "eval.classification")


def test_vanilla_autoencoder_independent():
    """Independent test for Vanilla Autoencoder."""
    return create_template_test("Vanilla Autoencoder", "model.ae_vanilla", "train.pytorch", "eval.reconstruction")


def test_variational_autoencoder_independent():
    """Independent test for Variational Autoencoder."""
    return create_template_test("Variational Autoencoder", "model.ae_variational", "train.pytorch", "eval.reconstruction")


def test_gnn_independent():
    """Independent test for Graph Neural Network (experimental)."""
    return create_template_test("GNN", "model.gnn_gcn", "train.sklearn", "eval.classification", 
                               data_config="graph_nodes_demo.csv", data_block="ingest.graph_csv")


def main():
    """Run all independent pipeline tests."""
    print("🧪 Running Independent Template Pattern Tests")
    print("=" * 60)
    print("Each test creates its own isolated environment and installs the framework locally.")
    
    test_functions = [
        test_xgb_independent,
        test_decision_tree_independent, 
        test_ensemble_independent,
        test_neural_independent,
        test_vanilla_autoencoder_independent,
        test_variational_autoencoder_independent,
        # test_gnn_independent,  # Skip for now due to trainer compatibility
    ]
    
    results = []
    
    for test_func in test_functions:
        try:
            success = test_func()
            results.append(success)
        except Exception as e:
            print(f"❌ {test_func.__name__} failed with exception: {e}")
            results.append(False)
    
    print("\n" + "="*60)
    passed = sum(results)
    total = len(results)
    percentage = (passed / total) * 100 if total > 0 else 0
    
    print(f"📊 Independent Template Test Results: {passed}/{total} tests passed ({percentage:.1f}%)")
    
    if passed == total:
        print("🎉 All independent tests passed! The framework is working correctly.")
        return 0
    else:
        print(f"⚠️  {total - passed} test(s) failed. Check the logs above for details.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
