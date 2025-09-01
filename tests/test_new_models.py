"""
Test script to validate new HEP model implementations.
Tests model registration, basic instantiation, and interface compliance.
"""

import sys
from typing import List, Tuple


def test_model_imports() -> List[Tuple[str, bool, str]]:
    """Test that all models can be imported without errors."""
    results = []

    # Traditional models (should always work with sklearn)
    traditional_models = [
        ("model.xgb_classifier", "XGBoost"),
        ("model.decision_tree", "Decision Tree"),
        ("model.random_forest", "Random Forest"),
        ("model.svm", "Support Vector Machine"),
        ("model.mlp", "Multi-Layer Perceptron"),
        ("model.adaboost", "AdaBoost"),
        ("model.ensemble_voting", "Voting Ensemble"),
    ]

    # Neural network models (require torch/lightning)
    neural_models = [
        ("model.ae_vanilla", "Vanilla Autoencoder"),
        ("model.ae_variational", "Variational Autoencoder"),
        ("model.gnn_gcn", "Graph Convolutional Network"),
        ("model.gnn_gat", "Graph Attention Network"),
        ("model.transformer_hep", "HEP Transformer"),
        ("model.cnn_hep", "HEP CNN"),
    ]

    # Test model registry imports
    try:
        from mlpipe.core.registry import get_registry

        registry = get_registry()

        # Test traditional models
        for model_key, model_name in traditional_models:
            try:
                if model_key in registry:
                    model_class = registry[model_key]
                    # Try to instantiate
                    model = model_class()
                    results.append((model_name, True, "✅ Imported and instantiated"))
                else:
                    results.append((model_name, False, "❌ Not found in registry"))
            except ImportError as e:
                results.append((model_name, False, f"❌ Import error: {str(e)}"))
            except Exception as e:
                results.append((model_name, False, f"❌ Instantiation error: {str(e)}"))

        # Test neural models (may fail due to missing dependencies)
        for model_key, model_name in neural_models:
            try:
                if model_key in registry:
                    model_class = registry[model_key]
                    model = model_class()
                    results.append((model_name, True, "✅ Imported and instantiated"))
                else:
                    results.append(
                        (model_name, False, "❌ Not found in registry (missing dependencies?)")
                    )
            except ImportError as e:
                results.append(
                    (model_name, False, f"🔶 Missing dependencies: {str(e).split('.')[-1]}")
                )
            except Exception as e:
                results.append((model_name, False, f"❌ Error: {str(e)}"))

    except Exception as e:
        results.append(("Registry", False, f"❌ Could not access registry: {str(e)}"))

    return results


def test_model_interfaces():
    """Test that models implement the required interface."""
    from mlpipe.core.registry import get_registry

    results = []
    registry = get_registry()

    for model_key, model_class in registry.items():
        if model_key.startswith("model."):
            try:
                model = model_class()

                # Check interface compliance
                has_build = hasattr(model, "build") and callable(model.build)
                has_fit = hasattr(model, "fit") and callable(model.fit)
                has_predict = hasattr(model, "predict") and callable(model.predict)

                if has_build and has_fit and has_predict:
                    results.append((model_key, True, "✅ Implements ModelBlock interface"))
                else:
                    missing = []
                    if not has_build:
                        missing.append("build")
                    if not has_fit:
                        missing.append("fit")
                    if not has_predict:
                        missing.append("predict")
                    results.append((model_key, False, f"❌ Missing methods: {missing}"))

            except Exception as e:
                results.append((model_key, False, f"❌ Interface test error: {str(e)}"))

    return results


def main():
    """Run all tests and display results."""
    print("🧪 Testing HEP ML Templates - Model Extensions")
    print("=" * 60)

    # Test imports
    print("\n📦 Testing Model Imports:")
    print("-" * 40)
    import_results = test_model_imports()

    success_count = 0
    for model_name, success, message in import_results:
        print(f"{model_name:25} | {message}")
        if success:
            success_count += 1

    print(f"\n✅ Successfully imported: {success_count}/{len(import_results)} models")

    # Test interfaces
    print("\n🔍 Testing Model Interfaces:")
    print("-" * 40)
    interface_results = test_model_interfaces()

    interface_success = 0
    for model_key, success, message in interface_results:
        print(f"{model_key:25} | {message}")
        if success:
            interface_success += 1

    print(f"\n✅ Interface compliant: {interface_success}/{len(interface_results)} models")

    # Summary
    total_success = success_count + interface_success
    total_tests = len(import_results) + len(interface_results)

    print("\n" + "=" * 60)
    print(f"📊 Overall Results: {total_success}/{total_tests} tests passed")

    if total_success == total_tests:
        print("🎉 All tests passed! Your HEP ML models are ready to use.")
        return 0
    else:
        print("⚠️  Some tests failed. Check dependencies and implementations.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
