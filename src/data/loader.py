"""Data loading and preprocessing utilities for stacking ensemble learning."""

import numpy as np
import pandas as pd
from typing import Tuple, Optional, Union, List, Dict, Any
from sklearn.datasets import load_breast_cancer, load_wine, load_iris, make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.utils import shuffle
import logging

logger = logging.getLogger(__name__)


class DataLoader:
    """Data loader for ensemble learning experiments."""
    
    def __init__(self, random_state: int = 42):
        """Initialize data loader with random seed.
        
        Args:
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        
    def load_dataset(
        self, 
        dataset_name: str = "breast_cancer",
        test_size: float = 0.2,
        return_metadata: bool = False
    ) -> Union[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], 
               Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]]:
        """Load and preprocess dataset for ensemble learning.
        
        Args:
            dataset_name: Name of dataset to load ('breast_cancer', 'wine', 'iris', 'synthetic')
            test_size: Fraction of data to use for testing
            return_metadata: Whether to return dataset metadata
            
        Returns:
            X_train, X_test, y_train, y_test (and optionally metadata)
        """
        logger.info(f"Loading dataset: {dataset_name}")
        
        # Load dataset
        if dataset_name == "breast_cancer":
            data = load_breast_cancer()
            X, y = data.data, data.target
            metadata = {
                "n_samples": len(X),
                "n_features": X.shape[1],
                "n_classes": len(np.unique(y)),
                "class_names": data.target_names,
                "feature_names": data.feature_names,
                "description": "Breast cancer classification dataset"
            }
        elif dataset_name == "wine":
            data = load_wine()
            X, y = data.data, data.target
            metadata = {
                "n_samples": len(X),
                "n_features": X.shape[1],
                "n_classes": len(np.unique(y)),
                "class_names": data.target_names,
                "feature_names": data.feature_names,
                "description": "Wine classification dataset"
            }
        elif dataset_name == "iris":
            data = load_iris()
            X, y = data.data, data.target
            metadata = {
                "n_samples": len(X),
                "n_features": X.shape[1],
                "n_classes": len(np.unique(y)),
                "class_names": data.target_names,
                "feature_names": data.feature_names,
                "description": "Iris classification dataset"
            }
        elif dataset_name == "synthetic":
            X, y = make_classification(
                n_samples=1000,
                n_features=20,
                n_informative=15,
                n_redundant=5,
                n_classes=2,
                random_state=self.random_state
            )
            metadata = {
                "n_samples": len(X),
                "n_features": X.shape[1],
                "n_classes": len(np.unique(y)),
                "class_names": ["Class_0", "Class_1"],
                "feature_names": [f"feature_{i}" for i in range(X.shape[1])],
                "description": "Synthetic classification dataset"
            }
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        
        # Shuffle data
        X, y = shuffle(X, y, random_state=self.random_state)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        logger.info(f"Dataset loaded: {metadata['n_samples']} samples, {metadata['n_features']} features, {metadata['n_classes']} classes")
        logger.info(f"Train set: {X_train_scaled.shape[0]} samples, Test set: {X_test_scaled.shape[0]} samples")
        
        if return_metadata:
            return X_train_scaled, X_test_scaled, y_train, y_test, metadata
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def load_custom_data(
        self, 
        file_path: str,
        target_column: str,
        test_size: float = 0.2
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Load custom dataset from file.
        
        Args:
            file_path: Path to data file (CSV, Parquet, etc.)
            target_column: Name of target column
            test_size: Fraction of data to use for testing
            
        Returns:
            X_train, X_test, y_train, y_test
        """
        logger.info(f"Loading custom dataset from: {file_path}")
        
        # Load data
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif file_path.endswith('.parquet'):
            df = pd.read_parquet(file_path)
        else:
            raise ValueError("Unsupported file format. Use CSV or Parquet.")
        
        # Separate features and target
        X = df.drop(columns=[target_column]).values
        y = df[target_column].values
        
        # Encode labels if needed
        if not np.issubdtype(y.dtype, np.integer):
            y = self.label_encoder.fit_transform(y)
        
        # Shuffle and split
        X, y = shuffle(X, y, random_state=self.random_state)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        logger.info(f"Custom dataset loaded: {len(X)} samples, {X.shape[1]} features")
        return X_train_scaled, X_test_scaled, y_train, y_test


def set_random_seeds(seed: int = 42) -> None:
    """Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value
    """
    np.random.seed(seed)
    logger.info(f"Random seeds set to: {seed}")


def get_available_datasets() -> List[str]:
    """Get list of available datasets.
    
    Returns:
        List of available dataset names
    """
    return ["breast_cancer", "wine", "iris", "synthetic"]
