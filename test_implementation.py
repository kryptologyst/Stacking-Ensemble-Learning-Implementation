#!/usr/bin/env python3
"""Simple test script to verify the implementation works."""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")
    
    try:
        from src.data import DataLoader, set_random_seeds
        from src.models import get_ensemble_models
        from src.train import run_quick_experiment
        from src.metrics import EnsembleEvaluator
        from src.viz import EnsembleVisualizer
        print("✅ All imports successful")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_data_loading():
    """Test data loading functionality."""
    print("Testing data loading...")
    
    try:
        from src.data import DataLoader
        
        loader = DataLoader(random_state=42)
        X_train, X_test, y_train, y_test = loader.load_dataset("breast_cancer")
        
        print(f"✅ Data loaded: {X_train.shape[0]} train, {X_test.shape[0]} test samples")
        return True
    except Exception as e:
        print(f"❌ Data loading error: {e}")
        return False

def test_models():
    """Test model creation."""
    print("Testing model creation...")
    
    try:
        from src.models import get_ensemble_models
        
        models = get_ensemble_models(random_state=42)
        print(f"✅ Created {len(models)} models: {list(models.keys())}")
        return True
    except Exception as e:
        print(f"❌ Model creation error: {e}")
        return False

def test_quick_experiment():
    """Test quick experiment."""
    print("Testing quick experiment...")
    
    try:
        from src.train import run_quick_experiment
        
        results = run_quick_experiment(
            dataset_name="breast_cancer",
            models=["simple_stacking", "sklearn_stacking"],
            random_state=42
        )
        
        print(f"✅ Quick experiment completed: {len(results['results'])} models tested")
        print(f"   Best accuracy: {results['summary']['best_accuracy']:.4f}")
        return True
    except Exception as e:
        print(f"❌ Quick experiment error: {e}")
        return False

def main():
    """Run all tests."""
    print("=" * 50)
    print("STACKING ENSEMBLE LEARNING - TEST SUITE")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_data_loading,
        test_models,
        test_quick_experiment
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 50)
    print(f"TEST RESULTS: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The implementation is working correctly.")
        return 0
    else:
        print("❌ Some tests failed. Please check the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
