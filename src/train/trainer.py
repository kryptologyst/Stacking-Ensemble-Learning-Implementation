"""Training utilities for ensemble learning experiments."""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
import logging
import time
from pathlib import Path
import joblib
import yaml

from ..data.loader import DataLoader
from ..models.ensembles import get_ensemble_models
from ..metrics.evaluator import EnsembleEvaluator, generate_evaluation_report

logger = logging.getLogger(__name__)


class EnsembleTrainer:
    """Trainer for ensemble learning experiments."""
    
    def __init__(
        self,
        random_state: int = 42,
        cv_folds: int = 5,
        save_dir: str = "assets"
    ):
        """Initialize trainer.
        
        Args:
            random_state: Random seed for reproducibility
            cv_folds: Number of CV folds
            save_dir: Directory to save results
        """
        self.random_state = random_state
        self.cv_folds = cv_folds
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        
        self.data_loader = DataLoader(random_state=random_state)
        self.evaluator = EnsembleEvaluator(cv_folds=cv_folds, random_state=random_state)
        
    def run_experiment(
        self,
        dataset_name: str = "breast_cancer",
        models_to_test: Optional[List[str]] = None,
        save_results: bool = True
    ) -> Dict[str, Any]:
        """Run complete ensemble learning experiment.
        
        Args:
            dataset_name: Name of dataset to use
            models_to_test: List of model names to test (None for all)
            save_results: Whether to save results to disk
            
        Returns:
            Dictionary containing all results
        """
        logger.info(f"Starting ensemble learning experiment on {dataset_name}")
        
        # Load data
        X_train, X_test, y_train, y_test, metadata = self.data_loader.load_dataset(
            dataset_name, return_metadata=True
        )
        
        # Get models
        all_models = get_ensemble_models(random_state=self.random_state)
        
        if models_to_test is None:
            models_to_test = list(all_models.keys())
        
        models = {name: all_models[name] for name in models_to_test if name in all_models}
        
        logger.info(f"Testing {len(models)} models: {list(models.keys())}")
        
        # Train and evaluate models
        results = {}
        training_times = {}
        
        for model_name, model in models.items():
            logger.info(f"Training {model_name}")
            
            start_time = time.time()
            
            try:
                # Train model
                model.fit(X_train, y_train)
                
                # Evaluate on test set
                y_pred = model.predict(X_test)
                
                # Get probabilities if available
                if hasattr(model, 'predict_proba'):
                    y_proba = model.predict_proba(X_test)
                else:
                    y_proba = None
                
                # Calculate metrics
                metrics = self.evaluator._calculate_metrics(y_test, y_pred, y_proba)
                
                # Add cross-validation results
                cv_scores = self.evaluator._cross_validate(model, X_train, y_train)
                metrics.update({
                    'cv_mean': cv_scores.mean(),
                    'cv_std': cv_scores.std(),
                    'cv_scores': cv_scores.tolist()
                })
                
                training_time = time.time() - start_time
                training_times[model_name] = training_time
                
                results[model_name] = metrics
                
                logger.info(f"{model_name} completed in {training_time:.2f}s - Accuracy: {metrics['accuracy']:.4f}")
                
            except Exception as e:
                logger.error(f"Error training {model_name}: {e}")
                results[model_name] = {'error': str(e)}
        
        # Create comprehensive results
        experiment_results = {
            'dataset': dataset_name,
            'metadata': metadata,
            'models_tested': models_to_test,
            'results': results,
            'training_times': training_times,
            'leaderboard': self.evaluator.create_leaderboard(results, 'accuracy'),
            'summary': self._create_summary(results, training_times)
        }
        
        # Save results if requested
        if save_results:
            self._save_results(experiment_results, dataset_name)
        
        logger.info("Experiment completed successfully")
        return experiment_results
        
    def run_ablation_study(
        self,
        dataset_name: str = "breast_cancer",
        base_model: str = "simple_stacking"
    ) -> Dict[str, Any]:
        """Run ablation study on ensemble components.
        
        Args:
            dataset_name: Name of dataset to use
            base_model: Base ensemble model to ablate
            
        Returns:
            Ablation study results
        """
        logger.info(f"Running ablation study on {base_model}")
        
        # Load data
        X_train, X_test, y_train, y_test, metadata = self.data_loader.load_dataset(
            dataset_name, return_metadata=True
        )
        
        # Get base model
        all_models = get_ensemble_models(random_state=self.random_state)
        if base_model not in all_models:
            raise ValueError(f"Model {base_model} not found")
        
        base_ensemble = all_models[base_model]
        
        # Test individual base models
        ablation_results = {}
        
        if hasattr(base_ensemble, 'base_models'):
            # Test individual base models
            for name, model in base_ensemble.base_models:
                logger.info(f"Testing individual model: {name}")
                
                try:
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    
                    if hasattr(model, 'predict_proba'):
                        y_proba = model.predict_proba(X_test)
                    else:
                        y_proba = None
                    
                    metrics = self.evaluator._calculate_metrics(y_test, y_pred, y_proba)
                    ablation_results[f"individual_{name}"] = metrics
                    
                except Exception as e:
                    logger.error(f"Error testing {name}: {e}")
                    ablation_results[f"individual_{name}"] = {'error': str(e)}
        
        # Test full ensemble
        logger.info("Testing full ensemble")
        try:
            base_ensemble.fit(X_train, y_train)
            y_pred = base_ensemble.predict(X_test)
            
            if hasattr(base_ensemble, 'predict_proba'):
                y_proba = base_ensemble.predict_proba(X_test)
            else:
                y_proba = None
            
            metrics = self.evaluator._calculate_metrics(y_test, y_pred, y_proba)
            ablation_results['full_ensemble'] = metrics
            
        except Exception as e:
            logger.error(f"Error testing full ensemble: {e}")
            ablation_results['full_ensemble'] = {'error': str(e)}
        
        # Create ablation summary
        ablation_summary = {
            'dataset': dataset_name,
            'base_model': base_model,
            'results': ablation_results,
            'improvement': self._calculate_improvement(ablation_results)
        }
        
        # Save ablation results
        self._save_ablation_results(ablation_summary, dataset_name, base_model)
        
        return ablation_summary
        
    def run_hyperparameter_search(
        self,
        dataset_name: str = "breast_cancer",
        model_name: str = "simple_stacking",
        param_grid: Optional[Dict[str, List]] = None
    ) -> Dict[str, Any]:
        """Run hyperparameter search for ensemble model.
        
        Args:
            dataset_name: Name of dataset to use
            model_name: Name of model to optimize
            param_grid: Parameter grid for search
            
        Returns:
            Hyperparameter search results
        """
        logger.info(f"Running hyperparameter search for {model_name}")
        
        # Load data
        X_train, X_test, y_train, y_test, metadata = self.data_loader.load_dataset(
            dataset_name, return_metadata=True
        )
        
        # Default parameter grid
        if param_grid is None:
            param_grid = {
                'cv_folds': [3, 5, 7],
                'meta_model': ['logistic', 'ridge', 'svm']
            }
        
        # Run grid search
        from sklearn.model_selection import ParameterGrid
        
        best_score = -np.inf
        best_params = None
        search_results = []
        
        for params in ParameterGrid(param_grid):
            logger.info(f"Testing parameters: {params}")
            
            try:
                # Create model with current parameters
                model = self._create_model_with_params(model_name, params)
                
                # Cross-validation
                cv_scores = self.evaluator._cross_validate(model, X_train, y_train)
                mean_score = cv_scores.mean()
                
                search_results.append({
                    'params': params,
                    'cv_mean': mean_score,
                    'cv_std': cv_scores.std(),
                    'cv_scores': cv_scores.tolist()
                })
                
                if mean_score > best_score:
                    best_score = mean_score
                    best_params = params
                    
            except Exception as e:
                logger.error(f"Error with parameters {params}: {e}")
                search_results.append({
                    'params': params,
                    'error': str(e)
                })
        
        # Test best model
        if best_params is not None:
            logger.info(f"Testing best parameters: {best_params}")
            best_model = self._create_model_with_params(model_name, best_params)
            best_model.fit(X_train, y_train)
            
            y_pred = best_model.predict(X_test)
            if hasattr(best_model, 'predict_proba'):
                y_proba = best_model.predict_proba(X_test)
            else:
                y_proba = None
            
            test_metrics = self.evaluator._calculate_metrics(y_test, y_pred, y_proba)
        else:
            test_metrics = {}
        
        # Create search summary
        search_summary = {
            'dataset': dataset_name,
            'model': model_name,
            'param_grid': param_grid,
            'best_params': best_params,
            'best_cv_score': best_score,
            'test_metrics': test_metrics,
            'all_results': search_results
        }
        
        # Save search results
        self._save_search_results(search_summary, dataset_name, model_name)
        
        return search_summary
        
    def _create_summary(
        self, 
        results: Dict[str, Dict[str, Any]], 
        training_times: Dict[str, float]
    ) -> Dict[str, Any]:
        """Create summary statistics.
        
        Args:
            results: Model results
            training_times: Training times
            
        Returns:
            Summary statistics
        """
        # Calculate summary metrics
        accuracies = [r.get('accuracy', 0) for r in results.values() if 'accuracy' in r]
        f1_scores = [r.get('f1', 0) for r in results.values() if 'f1' in r]
        
        summary = {
            'num_models': len(results),
            'best_accuracy': max(accuracies) if accuracies else 0,
            'worst_accuracy': min(accuracies) if accuracies else 0,
            'mean_accuracy': np.mean(accuracies) if accuracies else 0,
            'std_accuracy': np.std(accuracies) if accuracies else 0,
            'best_f1': max(f1_scores) if f1_scores else 0,
            'mean_f1': np.mean(f1_scores) if f1_scores else 0,
            'total_training_time': sum(training_times.values()),
            'mean_training_time': np.mean(list(training_times.values())) if training_times else 0
        }
        
        return summary
        
    def _calculate_improvement(self, ablation_results: Dict[str, Dict[str, Any]]) -> Dict[str, float]:
        """Calculate improvement from individual models to ensemble.
        
        Args:
            ablation_results: Results from ablation study
            
        Returns:
            Improvement metrics
        """
        if 'full_ensemble' not in ablation_results:
            return {}
        
        ensemble_acc = ablation_results['full_ensemble'].get('accuracy', 0)
        
        individual_accs = []
        for key, result in ablation_results.items():
            if key.startswith('individual_') and 'accuracy' in result:
                individual_accs.append(result['accuracy'])
        
        if not individual_accs:
            return {}
        
        best_individual = max(individual_accs)
        mean_individual = np.mean(individual_accs)
        
        return {
            'vs_best_individual': ensemble_acc - best_individual,
            'vs_mean_individual': ensemble_acc - mean_individual,
            'improvement_percent': ((ensemble_acc - mean_individual) / mean_individual) * 100
        }
        
    def _create_model_with_params(self, model_name: str, params: Dict[str, Any]) -> Any:
        """Create model with given parameters.
        
        Args:
            model_name: Name of model
            params: Parameters to set
            
        Returns:
            Model instance
        """
        from ..models.ensembles import SimpleStackingEnsemble, AdvancedStackingEnsemble, BlendingEnsemble
        from sklearn.linear_model import LogisticRegression, RidgeClassifier
        from sklearn.svm import SVC
        
        # Create base model
        if model_name == "simple_stacking":
            meta_model = None
            if 'meta_model' in params:
                meta_name = params['meta_model']
                if meta_name == 'logistic':
                    meta_model = LogisticRegression(random_state=self.random_state)
                elif meta_name == 'ridge':
                    meta_model = RidgeClassifier(random_state=self.random_state)
                elif meta_name == 'svm':
                    meta_model = SVC(probability=True, random_state=self.random_state)
            
            return SimpleStackingEnsemble(
                meta_model=meta_model,
                cv_folds=params.get('cv_folds', 5),
                random_state=self.random_state
            )
        else:
            # For other models, use default parameters
            all_models = get_ensemble_models(random_state=self.random_state)
            return all_models[model_name]
        
    def _save_results(self, results: Dict[str, Any], dataset_name: str) -> None:
        """Save experiment results.
        
        Args:
            results: Results to save
            dataset_name: Name of dataset
        """
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        
        # Save detailed results
        results_file = self.save_dir / f"experiment_{dataset_name}_{timestamp}.yaml"
        with open(results_file, 'w') as f:
            yaml.dump(results, f, default_flow_style=False)
        
        # Save models
        models_dir = self.save_dir / "models" / f"{dataset_name}_{timestamp}"
        models_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate and save report
        report = generate_evaluation_report(results['results'])
        report_file = self.save_dir / f"report_{dataset_name}_{timestamp}.txt"
        with open(report_file, 'w') as f:
            f.write(report)
        
        logger.info(f"Results saved to {results_file}")
        logger.info(f"Report saved to {report_file}")
        
    def _save_ablation_results(self, results: Dict[str, Any], dataset_name: str, model_name: str) -> None:
        """Save ablation study results.
        
        Args:
            results: Ablation results
            dataset_name: Name of dataset
            model_name: Name of model
        """
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        
        ablation_file = self.save_dir / f"ablation_{dataset_name}_{model_name}_{timestamp}.yaml"
        with open(ablation_file, 'w') as f:
            yaml.dump(results, f, default_flow_style=False)
        
        logger.info(f"Ablation results saved to {ablation_file}")
        
    def _save_search_results(self, results: Dict[str, Any], dataset_name: str, model_name: str) -> None:
        """Save hyperparameter search results.
        
        Args:
            results: Search results
            dataset_name: Name of dataset
            model_name: Name of model
        """
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        
        search_file = self.save_dir / f"search_{dataset_name}_{model_name}_{timestamp}.yaml"
        with open(search_file, 'w') as f:
            yaml.dump(results, f, default_flow_style=False)
        
        logger.info(f"Search results saved to {search_file}")


def run_quick_experiment(
    dataset_name: str = "breast_cancer",
    models: Optional[List[str]] = None,
    random_state: int = 42
) -> Dict[str, Any]:
    """Run a quick ensemble learning experiment.
    
    Args:
        dataset_name: Name of dataset to use
        models: List of models to test
        random_state: Random seed
        
    Returns:
        Experiment results
    """
    trainer = EnsembleTrainer(random_state=random_state)
    return trainer.run_experiment(dataset_name=dataset_name, models_to_test=models)
