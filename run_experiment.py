#!/usr/bin/env python3
"""Main experiment script for stacking ensemble learning."""

import argparse
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.data.loader import DataLoader, set_random_seeds
from src.models.ensembles import get_ensemble_models
from src.train.trainer import EnsembleTrainer, run_quick_experiment
from src.metrics.evaluator import EnsembleEvaluator
from src.viz.visualizer import create_summary_plots

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Main experiment function."""
    parser = argparse.ArgumentParser(description="Stacking Ensemble Learning Experiment")
    parser.add_argument(
        "--dataset", 
        type=str, 
        default="breast_cancer",
        choices=["breast_cancer", "wine", "iris", "synthetic"],
        help="Dataset to use for experiment"
    )
    parser.add_argument(
        "--models", 
        nargs="+", 
        default=None,
        help="Specific models to test (default: all)"
    )
    parser.add_argument(
        "--random-state", 
        type=int, 
        default=42,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--cv-folds", 
        type=int, 
        default=5,
        help="Number of cross-validation folds"
    )
    parser.add_argument(
        "--ablation", 
        action="store_true",
        help="Run ablation study"
    )
    parser.add_argument(
        "--hyperparameter-search", 
        action="store_true",
        help="Run hyperparameter search"
    )
    parser.add_argument(
        "--quick", 
        action="store_true",
        help="Run quick experiment with default settings"
    )
    parser.add_argument(
        "--save-dir", 
        type=str, 
        default="assets",
        help="Directory to save results"
    )
    
    args = parser.parse_args()
    
    # Set random seeds
    set_random_seeds(args.random_state)
    logger.info(f"Random seed set to: {args.random_state}")
    
    # Create save directory
    save_path = Path(args.save_dir)
    save_path.mkdir(exist_ok=True)
    
    if args.quick:
        # Run quick experiment
        logger.info("Running quick experiment")
        results = run_quick_experiment(
            dataset_name=args.dataset,
            models=args.models,
            random_state=args.random_state
        )
        
        # Create visualizations
        create_summary_plots(results, str(save_path))
        
        logger.info("Quick experiment completed successfully")
        return
    
    # Initialize trainer
    trainer = EnsembleTrainer(
        random_state=args.random_state,
        cv_folds=args.cv_folds,
        save_dir=str(save_path)
    )
    
    # Run main experiment
    logger.info(f"Starting experiment on {args.dataset} dataset")
    experiment_results = trainer.run_experiment(
        dataset_name=args.dataset,
        models_to_test=args.models,
        save_results=True
    )
    
    # Create visualizations
    create_summary_plots(experiment_results, str(save_path))
    
    # Run additional studies if requested
    if args.ablation:
        logger.info("Running ablation study")
        ablation_results = trainer.run_ablation_study(
            dataset_name=args.dataset,
            base_model="simple_stacking"
        )
        
        # Create ablation visualization
        from src.viz.visualizer import EnsembleVisualizer
        visualizer = EnsembleVisualizer()
        visualizer.plot_ablation_results(
            ablation_results,
            save_path=str(save_path / "ablation_results.png")
        )
    
    if args.hyperparameter_search:
        logger.info("Running hyperparameter search")
        search_results = trainer.run_hyperparameter_search(
            dataset_name=args.dataset,
            model_name="simple_stacking"
        )
    
    logger.info("Experiment completed successfully")
    
    # Print summary
    print("\n" + "="*60)
    print("EXPERIMENT SUMMARY")
    print("="*60)
    print(f"Dataset: {args.dataset}")
    print(f"Models tested: {len(experiment_results['results'])}")
    print(f"Best accuracy: {experiment_results['summary']['best_accuracy']:.4f}")
    print(f"Results saved to: {save_path}")
    print("="*60)


if __name__ == "__main__":
    main()
