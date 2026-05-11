#!/usr/bin/env python3
"""Simple demo script showcasing the stacking ensemble implementation."""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

def main():
    """Run a simple demonstration."""
    print("=" * 60)
    print("STACKING ENSEMBLE LEARNING - DEMONSTRATION")
    print("=" * 60)
    print()
    
    print("⚠️  RESEARCH DEMO DISCLAIMER:")
    print("   This is for educational purposes only.")
    print("   Not intended for production use or real-world decisions.")
    print()
    
    try:
        from src.train import run_quick_experiment
        from src.data import get_available_datasets
        
        print("📊 Available datasets:", ", ".join(get_available_datasets()))
        print()
        
        print("🚀 Running experiment on breast cancer dataset...")
        print("   Testing: Simple Stacking, Advanced Stacking, Blending")
        print()
        
        # Run experiment
        results = run_quick_experiment(
            dataset_name="breast_cancer",
            models=["simple_stacking", "advanced_stacking", "blending", "sklearn_stacking", "voting"],
            random_state=42
        )
        
        print("✅ Experiment completed!")
        print()
        
        # Display results
        print("📈 RESULTS SUMMARY:")
        print("-" * 40)
        print(f"Dataset: {results['dataset']}")
        print(f"Models tested: {len(results['results'])}")
        print(f"Best accuracy: {results['summary']['best_accuracy']:.4f}")
        print(f"Mean accuracy: {results['summary']['mean_accuracy']:.4f}")
        print(f"Training time: {results['summary']['total_training_time']:.2f}s")
        print()
        
        print("🏆 LEADERBOARD:")
        print("-" * 40)
        for i, (model, score, std) in enumerate(results['leaderboard'], 1):
            print(f"{i:2d}. {model:<20} {score:.4f} ± {std:.4f}")
        print()
        
        print("🔍 DETAILED METRICS:")
        print("-" * 40)
        for model_name, metrics in results['results'].items():
            print(f"\n{model_name.upper()}:")
            for metric, value in metrics.items():
                if isinstance(value, float):
                    print(f"  {metric}: {value:.4f}")
        
        print()
        print("=" * 60)
        print("🎉 DEMONSTRATION COMPLETED SUCCESSFULLY!")
        print()
        print("Next steps:")
        print("1. Run 'python run_experiment.py --help' for more options")
        print("2. Launch 'streamlit run demo/app.py' for interactive demo")
        print("3. Check the README.md for detailed documentation")
        print()
        print("Author: kryptologyst | GitHub: https://github.com/kryptologyst")
        print("=" * 60)
        
    except Exception as e:
        print(f"❌ Error running demonstration: {e}")
        print()
        print("Please ensure all dependencies are installed:")
        print("pip install -r requirements.txt")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
