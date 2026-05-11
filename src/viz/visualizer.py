"""Visualization utilities for ensemble learning results."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Optional, Tuple
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import logging

logger = logging.getLogger(__name__)

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class EnsembleVisualizer:
    """Visualizer for ensemble learning results."""
    
    def __init__(self, figsize: Tuple[int, int] = (12, 8)):
        """Initialize visualizer.
        
        Args:
            figsize: Default figure size
        """
        self.figsize = figsize
        
    def plot_model_comparison(
        self, 
        results: Dict[str, Dict[str, Any]], 
        metrics: List[str] = None,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """Plot comparison of models across multiple metrics.
        
        Args:
            results: Model evaluation results
            metrics: List of metrics to plot
            save_path: Path to save figure
            
        Returns:
            Matplotlib figure
        """
        if metrics is None:
            metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
        
        # Prepare data
        model_names = list(results.keys())
        metric_data = {metric: [] for metric in metrics}
        
        for model_name, model_results in results.items():
            for metric in metrics:
                value = model_results.get(metric, 0)
                metric_data[metric].append(value)
        
        # Create subplots
        n_metrics = len(metrics)
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, metric in enumerate(metrics):
            if i < len(axes):
                ax = axes[i]
                bars = ax.bar(model_names, metric_data[metric])
                ax.set_title(f'{metric.upper()} Comparison')
                ax.set_ylabel(metric.upper())
                ax.tick_params(axis='x', rotation=45)
                
                # Add value labels on bars
                for bar, value in zip(bars, metric_data[metric]):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                           f'{value:.3f}', ha='center', va='bottom')
        
        # Hide unused subplots
        for i in range(len(metrics), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Model comparison plot saved to {save_path}")
        
        return fig
        
    def plot_leaderboard(
        self, 
        leaderboard: List[Tuple[str, float, float]], 
        metric: str = "Accuracy",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """Plot leaderboard of models.
        
        Args:
            leaderboard: List of (model_name, score, std) tuples
            metric: Name of metric
            save_path: Path to save figure
            
        Returns:
            Matplotlib figure
        """
        if not leaderboard:
            logger.warning("Empty leaderboard provided")
            return None
        
        model_names, scores, stds = zip(*leaderboard)
        
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Create horizontal bar plot
        y_pos = np.arange(len(model_names))
        bars = ax.barh(y_pos, scores, xerr=stds, capsize=5)
        
        # Customize plot
        ax.set_yticks(y_pos)
        ax.set_yticklabels(model_names)
        ax.set_xlabel(f'{metric} Score')
        ax.set_title(f'Model Leaderboard - {metric}')
        
        # Add value labels
        for i, (score, std) in enumerate(zip(scores, stds)):
            ax.text(score + std + 0.01, i, f'{score:.3f} ± {std:.3f}', 
                   va='center', ha='left')
        
        # Invert y-axis to show best model at top
        ax.invert_yaxis()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Leaderboard plot saved to {save_path}")
        
        return fig
        
    def plot_cross_validation_scores(
        self, 
        results: Dict[str, Dict[str, Any]], 
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """Plot cross-validation scores distribution.
        
        Args:
            results: Model evaluation results
            save_path: Path to save figure
            
        Returns:
            Matplotlib figure
        """
        # Prepare data
        cv_data = []
        for model_name, model_results in results.items():
            if 'cv_scores' in model_results:
                scores = model_results['cv_scores']
                for score in scores:
                    cv_data.append({'Model': model_name, 'CV_Score': score})
        
        if not cv_data:
            logger.warning("No cross-validation scores found")
            return None
        
        df = pd.DataFrame(cv_data)
        
        # Create box plot
        fig, ax = plt.subplots(figsize=self.figsize)
        
        sns.boxplot(data=df, x='Model', y='CV_Score', ax=ax)
        ax.set_title('Cross-Validation Scores Distribution')
        ax.set_ylabel('CV Score')
        ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"CV scores plot saved to {save_path}")
        
        return fig
        
    def plot_training_time_comparison(
        self, 
        training_times: Dict[str, float], 
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """Plot training time comparison.
        
        Args:
            training_times: Dictionary of model_name -> training_time
            save_path: Path to save figure
            
        Returns:
            Matplotlib figure
        """
        if not training_times:
            logger.warning("No training times provided")
            return None
        
        model_names = list(training_times.keys())
        times = list(training_times.values())
        
        fig, ax = plt.subplots(figsize=self.figsize)
        
        bars = ax.bar(model_names, times)
        ax.set_title('Training Time Comparison')
        ax.set_ylabel('Training Time (seconds)')
        ax.tick_params(axis='x', rotation=45)
        
        # Add value labels
        for bar, time in zip(bars, times):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + max(times)*0.01,
                   f'{time:.2f}s', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Training time plot saved to {save_path}")
        
        return fig
        
    def plot_ensemble_diversity(
        self, 
        diversity_metrics: Dict[str, float], 
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """Plot ensemble diversity metrics.
        
        Args:
            diversity_metrics: Dictionary of diversity metrics
            save_path: Path to save figure
            
        Returns:
            Matplotlib figure
        """
        if not diversity_metrics:
            logger.warning("No diversity metrics provided")
            return None
        
        metrics = list(diversity_metrics.keys())
        values = list(diversity_metrics.values())
        
        fig, ax = plt.subplots(figsize=self.figsize)
        
        bars = ax.bar(metrics, values)
        ax.set_title('Ensemble Diversity Metrics')
        ax.set_ylabel('Value')
        ax.tick_params(axis='x', rotation=45)
        
        # Add value labels
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + max(values)*0.01,
                   f'{value:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Diversity plot saved to {save_path}")
        
        return fig
        
    def plot_feature_importance_consensus(
        self, 
        importance_data: Dict[str, Any], 
        top_n: int = 10,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """Plot consensus feature importance.
        
        Args:
            importance_data: Feature importance data
            top_n: Number of top features to show
            save_path: Path to save figure
            
        Returns:
            Matplotlib figure
        """
        if 'importance_dataframe' not in importance_data:
            logger.warning("No feature importance dataframe found")
            return None
        
        df = importance_data['importance_dataframe'].head(top_n)
        
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Create error bar plot
        x_pos = np.arange(len(df))
        bars = ax.bar(x_pos, df['mean_importance'], 
                     yerr=df['std_importance'], capsize=5)
        
        ax.set_xlabel('Features')
        ax.set_ylabel('Importance')
        ax.set_title(f'Top {top_n} Feature Importance Consensus')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(df['feature'], rotation=45, ha='right')
        
        # Add value labels
        for i, (mean_val, std_val) in enumerate(zip(df['mean_importance'], df['std_importance'])):
            ax.text(i, mean_val + std_val + max(df['mean_importance'])*0.01,
                   f'{mean_val:.3f} ± {std_val:.3f}', ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Feature importance plot saved to {save_path}")
        
        return fig
        
    def create_interactive_dashboard(
        self, 
        results: Dict[str, Dict[str, Any]], 
        training_times: Dict[str, float],
        save_path: Optional[str] = None
    ) -> go.Figure:
        """Create interactive dashboard with Plotly.
        
        Args:
            results: Model evaluation results
            training_times: Training times
            save_path: Path to save HTML file
            
        Returns:
            Plotly figure
        """
        # Prepare data
        model_names = list(results.keys())
        metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC', 'Training Time'],
            specs=[[{"type": "bar"}, {"type": "bar"}, {"type": "bar"}],
                   [{"type": "bar"}, {"type": "bar"}, {"type": "bar"}]]
        )
        
        # Add metric plots
        for i, metric in enumerate(metrics):
            row = i // 3 + 1
            col = i % 3 + 1
            
            values = [results[model].get(metric, 0) for model in model_names]
            
            fig.add_trace(
                go.Bar(x=model_names, y=values, name=metric.title()),
                row=row, col=col
            )
        
        # Add training time plot
        times = [training_times.get(model, 0) for model in model_names]
        fig.add_trace(
            go.Bar(x=model_names, y=times, name='Training Time'),
            row=2, col=3
        )
        
        # Update layout
        fig.update_layout(
            title_text="Ensemble Learning Results Dashboard",
            showlegend=False,
            height=800
        )
        
        if save_path:
            fig.write_html(save_path)
            logger.info(f"Interactive dashboard saved to {save_path}")
        
        return fig
        
    def plot_ablation_results(
        self, 
        ablation_results: Dict[str, Dict[str, Any]], 
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """Plot ablation study results.
        
        Args:
            ablation_results: Ablation study results
            save_path: Path to save figure
            
        Returns:
            Matplotlib figure
        """
        if 'results' not in ablation_results:
            logger.warning("No ablation results found")
            return None
        
        results = ablation_results['results']
        
        # Separate individual models and ensemble
        individual_models = []
        individual_scores = []
        ensemble_score = None
        
        for model_name, model_results in results.items():
            if model_name.startswith('individual_'):
                individual_models.append(model_name.replace('individual_', ''))
                individual_scores.append(model_results.get('accuracy', 0))
            elif model_name == 'full_ensemble':
                ensemble_score = model_results.get('accuracy', 0)
        
        if not individual_models or ensemble_score is None:
            logger.warning("Insufficient data for ablation plot")
            return None
        
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Plot individual models
        x_pos = np.arange(len(individual_models))
        bars1 = ax.bar(x_pos - 0.2, individual_scores, 0.4, label='Individual Models', alpha=0.7)
        
        # Plot ensemble
        bars2 = ax.bar(x_pos + 0.2, [ensemble_score] * len(individual_models), 0.4, 
                     label='Full Ensemble', alpha=0.7)
        
        ax.set_xlabel('Models')
        ax.set_ylabel('Accuracy')
        ax.set_title('Ablation Study: Individual vs Ensemble Performance')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(individual_models, rotation=45, ha='right')
        ax.legend()
        
        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{height:.3f}', ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Ablation plot saved to {save_path}")
        
        return fig


def create_summary_plots(
    experiment_results: Dict[str, Any],
    save_dir: str = "assets"
) -> None:
    """Create all summary plots for an experiment.
    
    Args:
        experiment_results: Complete experiment results
        save_dir: Directory to save plots
    """
    from pathlib import Path
    
    save_path = Path(save_dir)
    save_path.mkdir(exist_ok=True)
    
    visualizer = EnsembleVisualizer()
    
    # Extract data
    results = experiment_results.get('results', {})
    training_times = experiment_results.get('training_times', {})
    leaderboard = experiment_results.get('leaderboard', [])
    
    # Create plots
    try:
        # Model comparison
        visualizer.plot_model_comparison(
            results, 
            save_path=str(save_path / "model_comparison.png")
        )
        
        # Leaderboard
        if leaderboard:
            visualizer.plot_leaderboard(
                leaderboard, 
                save_path=str(save_path / "leaderboard.png")
            )
        
        # Training times
        if training_times:
            visualizer.plot_training_time_comparison(
                training_times, 
                save_path=str(save_path / "training_times.png")
            )
        
        # Interactive dashboard
        visualizer.create_interactive_dashboard(
            results, training_times,
            save_path=str(save_path / "dashboard.html")
        )
        
        logger.info(f"All summary plots saved to {save_path}")
        
    except Exception as e:
        logger.error(f"Error creating summary plots: {e}")
