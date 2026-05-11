"""Interactive Streamlit demo for stacking ensemble learning."""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.data.loader import DataLoader, get_available_datasets
from src.models.ensembles import get_ensemble_models
from src.train.trainer import EnsembleTrainer
from src.metrics.evaluator import EnsembleEvaluator
from src.viz.visualizer import EnsembleVisualizer

# Page configuration
st.set_page_config(
    page_title="Stacking Ensemble Learning Demo",
    page_icon="🧠",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        color: #1f77b4;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

def main():
    """Main demo function."""
    
    # Header
    st.markdown('<h1 class="main-header">🧠 Stacking Ensemble Learning Demo</h1>', unsafe_allow_html=True)
    
    # Safety disclaimer
    st.markdown("""
    <div class="warning-box">
    <h4>⚠️ Research Demo Disclaimer</h4>
    <p><strong>This is a research and educational demonstration only.</strong></p>
    <ul>
        <li>Not intended for production use or real-world decision making</li>
        <li>Results are for educational purposes and algorithm comparison</li>
        <li>No medical, financial, or critical decisions should be based on these results</li>
        <li>Always consult domain experts for real-world applications</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar configuration
    st.sidebar.header("🔧 Configuration")
    
    # Dataset selection
    dataset_options = get_available_datasets()
    selected_dataset = st.sidebar.selectbox(
        "Select Dataset",
        dataset_options,
        index=0,
        help="Choose a dataset for the ensemble learning experiment"
    )
    
    # Model selection
    st.sidebar.subheader("Model Selection")
    all_models = get_ensemble_models()
    model_options = list(all_models.keys())
    
    selected_models = st.sidebar.multiselect(
        "Select Models to Compare",
        model_options,
        default=model_options[:3],  # Default to first 3 models
        help="Choose which ensemble methods to compare"
    )
    
    # Random seed
    random_seed = st.sidebar.number_input(
        "Random Seed",
        min_value=0,
        max_value=10000,
        value=42,
        help="Set random seed for reproducibility"
    )
    
    # CV folds
    cv_folds = st.sidebar.slider(
        "Cross-Validation Folds",
        min_value=3,
        max_value=10,
        value=5,
        help="Number of folds for cross-validation"
    )
    
    # Run experiment button
    if st.sidebar.button("🚀 Run Experiment", type="primary"):
        run_experiment(selected_dataset, selected_models, random_seed, cv_folds)
    
    # Information section
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📚 About Stacking")
    st.sidebar.markdown("""
    Stacking is an ensemble learning technique that:
    - Combines multiple base models
    - Uses a meta-model to learn optimal combinations
    - Often achieves better performance than individual models
    - Requires careful cross-validation to avoid overfitting
    """)
    
    # Main content area
    if 'experiment_results' not in st.session_state:
        st.info("👈 Configure your experiment in the sidebar and click 'Run Experiment' to get started!")
        
        # Show dataset information
        st.subheader("📊 Available Datasets")
        data_loader = DataLoader()
        
        dataset_info = {
            "breast_cancer": "Breast cancer classification dataset with 30 features",
            "wine": "Wine quality classification with 13 features", 
            "iris": "Classic iris species classification with 4 features",
            "synthetic": "Synthetic binary classification dataset with 20 features"
        }
        
        for dataset, description in dataset_info.items():
            st.write(f"**{dataset.replace('_', ' ').title()}**: {description}")
        
        # Show model information
        st.subheader("🤖 Available Models")
        model_info = {
            "simple_stacking": "Basic stacking with CV-generated meta-features",
            "advanced_stacking": "Multi-level stacking with advanced base models",
            "blending": "Holdout-based blending ensemble",
            "sklearn_stacking": "Scikit-learn's built-in StackingClassifier",
            "voting": "Soft voting ensemble for comparison"
        }
        
        for model, description in model_info.items():
            st.write(f"**{model.replace('_', ' ').title()}**: {description}")
    
    else:
        display_results()


def run_experiment(dataset_name, model_names, random_seed, cv_folds):
    """Run the ensemble learning experiment."""
    
    with st.spinner("Running experiment..."):
        try:
            # Initialize trainer
            trainer = EnsembleTrainer(
                random_state=random_seed,
                cv_folds=cv_folds
            )
            
            # Run experiment
            results = trainer.run_experiment(
                dataset_name=dataset_name,
                models_to_test=model_names,
                save_results=False
            )
            
            # Store results in session state
            st.session_state.experiment_results = results
            st.session_state.dataset_name = dataset_name
            
            st.success("✅ Experiment completed successfully!")
            
        except Exception as e:
            st.error(f"❌ Error running experiment: {str(e)}")


def display_results():
    """Display experiment results."""
    
    results = st.session_state.experiment_results
    dataset_name = st.session_state.dataset_name
    
    # Results summary
    st.subheader(f"📈 Results Summary - {dataset_name.replace('_', ' ').title()}")
    
    # Key metrics cards
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Models Tested",
            len(results['results']),
            help="Number of ensemble models evaluated"
        )
    
    with col2:
        best_acc = results['summary']['best_accuracy']
        st.metric(
            "Best Accuracy",
            f"{best_acc:.4f}",
            help="Highest accuracy achieved by any model"
        )
    
    with col3:
        mean_acc = results['summary']['mean_accuracy']
        st.metric(
            "Mean Accuracy",
            f"{mean_acc:.4f}",
            help="Average accuracy across all models"
        )
    
    with col4:
        total_time = results['summary']['total_training_time']
        st.metric(
            "Total Training Time",
            f"{total_time:.2f}s",
            help="Total time to train all models"
        )
    
    # Leaderboard
    st.subheader("🏆 Model Leaderboard")
    
    if results['leaderboard']:
        leaderboard_df = pd.DataFrame(
            results['leaderboard'],
            columns=['Model', 'Accuracy', 'Std']
        )
        leaderboard_df.index = range(1, len(leaderboard_df) + 1)
        
        st.dataframe(
            leaderboard_df,
            use_container_width=True,
            hide_index=True
        )
    
    # Detailed metrics comparison
    st.subheader("📊 Detailed Metrics Comparison")
    
    # Prepare data for visualization
    model_names = list(results['results'].keys())
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    
    # Create comparison plot
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
        
        values = [results['results'][model].get(metric, 0) for model in model_names]
        
        fig.add_trace(
            go.Bar(x=model_names, y=values, name=metric.title()),
            row=row, col=col
        )
    
    # Add training time plot
    times = [results['training_times'].get(model, 0) for model in model_names]
    fig.add_trace(
        go.Bar(x=model_names, y=times, name='Training Time'),
        row=2, col=3
    )
    
    # Update layout
    fig.update_layout(
        title_text="Model Performance Comparison",
        showlegend=False,
        height=600
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Cross-validation scores
    st.subheader("🔄 Cross-Validation Scores")
    
    cv_data = []
    for model_name, model_results in results['results'].items():
        if 'cv_scores' in model_results:
            scores = model_results['cv_scores']
            for score in scores:
                cv_data.append({'Model': model_name, 'CV_Score': score})
    
    if cv_data:
        cv_df = pd.DataFrame(cv_data)
        
        fig_cv = px.box(
            cv_df, 
            x='Model', 
            y='CV_Score',
            title='Cross-Validation Scores Distribution'
        )
        fig_cv.update_xaxes(tickangle=45)
        
        st.plotly_chart(fig_cv, use_container_width=True)
    
    # Individual model details
    st.subheader("🔍 Individual Model Details")
    
    for model_name, model_results in results['results'].items():
        with st.expander(f"📋 {model_name.replace('_', ' ').title()}"):
            
            # Key metrics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Accuracy", f"{model_results.get('accuracy', 0):.4f}")
                st.metric("Precision", f"{model_results.get('precision', 0):.4f}")
            
            with col2:
                st.metric("Recall", f"{model_results.get('recall', 0):.4f}")
                st.metric("F1 Score", f"{model_results.get('f1', 0):.4f}")
            
            with col3:
                st.metric("ROC AUC", f"{model_results.get('roc_auc', 0):.4f}")
                st.metric("CV Mean", f"{model_results.get('cv_mean', 0):.4f}")
            
            # Cross-validation details
            if 'cv_scores' in model_results:
                st.write("**Cross-Validation Scores:**")
                cv_scores = model_results['cv_scores']
                st.write(f"Mean: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")
                st.write(f"Scores: {[f'{score:.4f}' for score in cv_scores]}")
    
    # Dataset information
    st.subheader("📊 Dataset Information")
    
    metadata = results.get('metadata', {})
    if metadata:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Samples", metadata.get('n_samples', 'N/A'))
            st.metric("Features", metadata.get('n_features', 'N/A'))
        
        with col2:
            st.metric("Classes", metadata.get('n_classes', 'N/A'))
            st.metric("Description", metadata.get('description', 'N/A'))
        
        with col3:
            if 'class_names' in metadata:
                st.write("**Class Names:**")
                for class_name in metadata['class_names']:
                    st.write(f"- {class_name}")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666;">
        <p><strong>Author:</strong> kryptologyst | <strong>GitHub:</strong> <a href="https://github.com/kryptologyst">https://github.com/kryptologyst</a></p>
        <p><em>This demo is for research and educational purposes only.</em></p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
