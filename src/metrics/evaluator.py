"""Comprehensive evaluation metrics for ensemble learning."""

import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, log_loss,
    confusion_matrix, classification_report
)
from sklearn.model_selection import cross_val_score, StratifiedKFold
import logging

logger = logging.getLogger(__name__)


class EnsembleEvaluator:
    """Comprehensive evaluator for ensemble models."""
    
    def __init__(self, cv_folds: int = 5, random_state: int = 42):
        """Initialize evaluator.
        
        Args:
            cv_folds: Number of CV folds for cross-validation
            random_state: Random seed
        """
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
        
    def evaluate_model(
        self, 
        model: Any, 
        X: np.ndarray, 
        y: np.ndarray,
        model_name: str = "Model"
    ) -> Dict[str, Any]:
        """Evaluate a single model comprehensively.
        
        Args:
            model: Model to evaluate
            X: Features
            y: Targets
            model_name: Name of the model
            
        Returns:
            Dictionary of evaluation metrics
        """
        logger.info(f"Evaluating {model_name}")
        
        # Cross-validation scores
        cv_scores = self._cross_validate(model, X, y)
        
        # Single train-test evaluation
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.random_state, stratify=y
        )
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        # Get probabilities if available
        if hasattr(model, 'predict_proba'):
            y_proba = model.predict_proba(X_test)
        else:
            y_proba = None
            
        # Calculate metrics
        metrics = self._calculate_metrics(y_test, y_pred, y_proba)
        
        # Add cross-validation results
        metrics.update({
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'cv_scores': cv_scores.tolist()
        })
        
        return metrics
        
    def compare_models(
        self, 
        models: Dict[str, Any], 
        X: np.ndarray, 
        y: np.ndarray
    ) -> Dict[str, Dict[str, Any]]:
        """Compare multiple models.
        
        Args:
            models: Dictionary of model_name -> model
            X: Features
            y: Targets
            
        Returns:
            Dictionary of evaluation results for each model
        """
        logger.info(f"Comparing {len(models)} models")
        
        results = {}
        for name, model in models.items():
            results[name] = self.evaluate_model(model, X, y, name)
            
        return results
        
    def create_leaderboard(
        self, 
        results: Dict[str, Dict[str, Any]], 
        metric: str = 'accuracy'
    ) -> List[Tuple[str, float, float]]:
        """Create a leaderboard sorted by specified metric.
        
        Args:
            results: Results from compare_models
            metric: Metric to sort by
            
        Returns:
            List of (model_name, metric_value, std) tuples sorted by metric
        """
        leaderboard = []
        
        for model_name, metrics in results.items():
            if metric in metrics:
                leaderboard.append((
                    model_name, 
                    metrics[metric], 
                    metrics.get('cv_std', 0.0)
                ))
                
        # Sort by metric value (descending)
        leaderboard.sort(key=lambda x: x[1], reverse=True)
        
        return leaderboard
        
    def _cross_validate(self, model: Any, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Perform cross-validation.
        
        Args:
            model: Model to validate
            X: Features
            y: Targets
            
        Returns:
            Array of CV scores
        """
        try:
            scores = cross_val_score(model, X, y, cv=self.cv, scoring='accuracy')
            return scores
        except Exception as e:
            logger.warning(f"Cross-validation failed: {e}")
            return np.array([0.0])
            
    def _calculate_metrics(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray, 
        y_proba: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """Calculate comprehensive metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_proba: Predicted probabilities (optional)
            
        Returns:
            Dictionary of metrics
        """
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'f1': f1_score(y_true, y_pred, average='weighted', zero_division=0)
        }
        
        # Add probability-based metrics if available
        if y_proba is not None:
            try:
                # For binary classification, use probabilities for positive class
                if y_proba.shape[1] == 2:
                    metrics['roc_auc'] = roc_auc_score(y_true, y_proba[:, 1])
                    metrics['average_precision'] = average_precision_score(y_true, y_proba[:, 1])
                else:
                    metrics['roc_auc'] = roc_auc_score(y_true, y_proba, multi_class='ovr')
                    metrics['average_precision'] = average_precision_score(y_true, y_proba, average='weighted')
                    
                metrics['log_loss'] = log_loss(y_true, y_proba)
            except Exception as e:
                logger.warning(f"Probability metrics calculation failed: {e}")
                
        return metrics


def calculate_ensemble_diversity(
    predictions: Dict[str, np.ndarray],
    y_true: np.ndarray
) -> Dict[str, float]:
    """Calculate diversity metrics for ensemble predictions.
    
    Args:
        predictions: Dictionary of model_name -> predictions
        y_true: True labels
        
    Returns:
        Dictionary of diversity metrics
    """
    model_names = list(predictions.keys())
    n_models = len(model_names)
    
    if n_models < 2:
        return {'disagreement': 0.0, 'correlation': 1.0}
    
    # Calculate pairwise disagreement
    disagreements = []
    correlations = []
    
    for i in range(n_models):
        for j in range(i + 1, n_models):
            pred_i = predictions[model_names[i]]
            pred_j = predictions[model_names[j]]
            
            # Disagreement rate
            disagreement = np.mean(pred_i != pred_j)
            disagreements.append(disagreement)
            
            # Correlation
            correlation = np.corrcoef(pred_i, pred_j)[0, 1]
            if not np.isnan(correlation):
                correlations.append(correlation)
    
    return {
        'mean_disagreement': np.mean(disagreements),
        'std_disagreement': np.std(disagreements),
        'mean_correlation': np.mean(correlations) if correlations else 0.0,
        'std_correlation': np.std(correlations) if correlations else 0.0
    }


def calculate_feature_importance_consensus(
    models: Dict[str, Any],
    feature_names: Optional[List[str]] = None
) -> Dict[str, Any]:
    """Calculate consensus feature importance across models.
    
    Args:
        models: Dictionary of fitted models
        feature_names: Names of features (optional)
        
    Returns:
        Dictionary with consensus importance metrics
    """
    importances = {}
    
    for model_name, model in models.items():
        if hasattr(model, 'feature_importances_'):
            importances[model_name] = model.feature_importances_
        elif hasattr(model, 'coef_'):
            # For linear models, use absolute coefficients
            importances[model_name] = np.abs(model.coef_[0])
        else:
            logger.warning(f"Model {model_name} has no feature importance")
            continue
    
    if not importances:
        return {}
    
    # Calculate consensus metrics
    importance_matrix = np.array(list(importances.values()))
    
    # Mean importance across models
    mean_importance = np.mean(importance_matrix, axis=0)
    
    # Standard deviation (disagreement)
    std_importance = np.std(importance_matrix, axis=0)
    
    # Rank correlation
    ranks = np.argsort(importance_matrix, axis=1)
    rank_correlations = []
    
    for i in range(len(ranks)):
        for j in range(i + 1, len(ranks)):
            correlation = np.corrcoef(ranks[i], ranks[j])[0, 1]
            if not np.isnan(correlation):
                rank_correlations.append(correlation)
    
    result = {
        'mean_importance': mean_importance,
        'std_importance': std_importance,
        'mean_rank_correlation': np.mean(rank_correlations) if rank_correlations else 0.0,
        'std_rank_correlation': np.std(rank_correlations) if rank_correlations else 0.0
    }
    
    if feature_names:
        # Create feature importance dataframe
        import pandas as pd
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'mean_importance': mean_importance,
            'std_importance': std_importance
        }).sort_values('mean_importance', ascending=False)
        
        result['importance_dataframe'] = importance_df
    
    return result


def generate_evaluation_report(
    results: Dict[str, Dict[str, Any]],
    save_path: Optional[str] = None
) -> str:
    """Generate a comprehensive evaluation report.
    
    Args:
        results: Results from model evaluation
        save_path: Path to save report (optional)
        
    Returns:
        Formatted report string
    """
    report = "=" * 80 + "\n"
    report += "ENSEMBLE LEARNING EVALUATION REPORT\n"
    report += "=" * 80 + "\n\n"
    
    # Summary statistics
    report += "SUMMARY STATISTICS\n"
    report += "-" * 40 + "\n"
    
    metrics_to_report = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'log_loss']
    
    for metric in metrics_to_report:
        values = [results[model][metric] for model in results if metric in results[model]]
        if values:
            report += f"{metric.upper()}: {np.mean(values):.4f} ± {np.std(values):.4f}\n"
    
    report += "\n"
    
    # Individual model results
    report += "INDIVIDUAL MODEL RESULTS\n"
    report += "-" * 40 + "\n"
    
    for model_name, metrics in results.items():
        report += f"\n{model_name.upper()}:\n"
        for metric, value in metrics.items():
            if isinstance(value, (int, float)):
                report += f"  {metric}: {value:.4f}\n"
    
    report += "\n"
    
    # Leaderboard
    report += "LEADERBOARD (by Accuracy)\n"
    report += "-" * 40 + "\n"
    
    evaluator = EnsembleEvaluator()
    leaderboard = evaluator.create_leaderboard(results, 'accuracy')
    
    for i, (model_name, score, std) in enumerate(leaderboard, 1):
        report += f"{i:2d}. {model_name:<20} {score:.4f} ± {std:.4f}\n"
    
    report += "\n" + "=" * 80 + "\n"
    
    if save_path:
        with open(save_path, 'w') as f:
            f.write(report)
        logger.info(f"Report saved to {save_path}")
    
    return report
