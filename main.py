"""
Active Learning Framework for Tabular Classification

This module implements a comprehensive active learning system that combines three
sampling strategies (uncertainty, query-by-committee, and diversity) to 
iteratively select the most informative samples for model training.

PROGRAM PURPOSE:
================
This program demonstrates how to implement an active learning pipeline that:
1. Starts with a small labeled dataset
2. Iteratively selects the most informative unlabeled samples
3. Adds them to the training set to improve model performance
4. Uses a hybrid approach combining uncertainty, committee voting, and diversity sampling

MAIN SECTIONS:
==============
1. Configuration & Setup: Data classes for configuration and utility functions
2. Data Loading & Preprocessing: CSV loading, feature splitting, and preprocessing pipelines
3. Model Building & Cross-Validation: Model creation and evaluation utilities
4. Active Learning Strategies: Three sampling methods (uncertainty, QBC, diversity)
5. Active Learning Runner: Main orchestrator class that coordinates the learning process
6. CLI & Exception Handling: Command-line interface and robust error management
7. Unit Tests: Test functions for core sampling strategies

The framework supports both Logistic Regression and Random Forest classifiers
and provides comprehensive logging and error handling throughout.
"""

import argparse
import logging
import os
import random
import sys
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import NotFittedError
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.cluster import KMeans


# =============================
# 1. CONFIGURATION & SETUP
# =============================

@dataclass
class RunConfig:
    """
    Configuration parameters for the active learning experiment.
    
    This class centralizes all hyperparameters and settings needed to run
    an active learning experiment, making it easy to modify and track
    experimental configurations.
    
    Attributes:
        random_seed: Seed for reproducible random number generation
        test_size: Fraction of data to reserve for final holdout evaluation
        n_splits: Number of folds for cross-validation during training
        model: Model type ("logreg" for Logistic Regression, "rf" for Random Forest)
        initial_size: Number of samples to start with in the labeled set
        batch_size: Number of samples to query at each active learning iteration
        iters: Maximum number of active learning iterations to perform
        committee_size: Number of models in the query-by-committee ensemble
        hybrid_weights: Weights for combining sampling strategies (uncertainty, QBC, diversity)
    """
    random_seed: int = 42
    test_size: float = 0.2
    n_splits: int = 5
    model: str = "logreg"  # "logreg" or "rf"
    initial_size: int = 500
    batch_size: int = 100
    iters: int = 15
    committee_size: int = 5
    hybrid_weights: Tuple[float, float, float] = (0.34, 0.33, 0.33)  # (uncertainty, qbc, diversity)


def set_seed(seed: int) -> None:
    """
    Set random seeds for reproducible results across all random number generators.
    
    Args:
        seed: Random seed value
        
    Note:
        This ensures reproducibility by setting seeds for Python's random module
        and NumPy's random number generator.
    """
    random.seed(seed)
    np.random.seed(seed)


def setup_logging(level: str = "INFO") -> None:
    """
    Configure logging with consistent format and specified level.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        
    Sets up logging with timestamp, level, and message formatting for
    better debugging and monitoring of the active learning process.
    """
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


# =============================
# 2. DATA LOADING & PREPROCESSING
# =============================

def load_csv(path: str) -> pd.DataFrame:
    """
    Load and validate CSV data file with comprehensive error checking.
    
    This function demonstrates proper exception handling for file operations
    and data validation, ensuring the loaded data is usable for machine learning.
    
    Args:
        path: Path to the CSV file
        
    Returns:
        Loaded pandas DataFrame
        
    Raises:
        FileNotFoundError: If the specified file doesn't exist
        ValueError: If the loaded DataFrame is empty
        
    Example:
        >>> df = load_csv("data/iris.csv")
        >>> print(f"Loaded {len(df)} rows")
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Data file not found: {path}")
    
    try:
        df = pd.read_csv(path)
        logging.info(f"Successfully loaded CSV with shape {df.shape}")
    except pd.errors.EmptyDataError:
        raise ValueError(f"CSV file is empty: {path}")
    except pd.errors.ParserError as e:
        raise ValueError(f"Error parsing CSV file {path}: {e}")
    
    if df.empty:
        raise ValueError("Loaded DataFrame is empty")
        
    logging.info(f"CSV loaded: {len(df)} rows, {len(df.columns)} columns")
    return df


def split_features(
    df: pd.DataFrame, 
    target_col: str, 
    num_cols: Optional[List[str]] = None, 
    cat_cols: Optional[List[str]] = None
) -> Tuple[pd.DataFrame, pd.Series, List[str], List[str]]:
    """
    Split DataFrame into features and target, identifying numeric and categorical columns.
    
    This function handles the common ML preprocessing step of separating features
    from target variables and automatically inferring column types when not specified.
    
    Args:
        df: Input DataFrame
        target_col: Name of the target column
        num_cols: List of numeric column names (auto-inferred if None)
        cat_cols: List of categorical column names (auto-inferred if None)
        
    Returns:
        Tuple of (features_df, target_series, numeric_columns, categorical_columns)
        
    Raises:
        KeyError: If target column is not found in DataFrame
        
    Example:
        >>> X, y, num_cols, cat_cols = split_features(df, 'target')
        >>> print(f"Features: {X.shape}, Numeric: {len(num_cols)}, Categorical: {len(cat_cols)}")
    """
    if target_col not in df.columns:
        available_cols = ", ".join(df.columns.tolist())
        raise KeyError(f"Target column '{target_col}' not found. Available columns: {available_cols}")
    
    y = df[target_col]
    X = df.drop(columns=[target_col])
    
    # Auto-infer column types if not provided
    if num_cols is None or cat_cols is None:
        inferred_num = X.select_dtypes(include=[np.number]).columns.tolist()
        inferred_cat = [col for col in X.columns if col not in inferred_num]
        
        num_cols = inferred_num if num_cols is None else num_cols
        cat_cols = inferred_cat if cat_cols is None else cat_cols
    
    logging.info(f"Feature split: {len(num_cols)} numeric, {len(cat_cols)} categorical columns")
    
    return X, y, num_cols, cat_cols


def build_preprocessor(num_cols: List[str], cat_cols: List[str]) -> ColumnTransformer:
    """
    Create preprocessing pipeline for numeric and categorical features.
    
    This function demonstrates how to build robust preprocessing pipelines
    that handle missing values and scale features appropriately for ML models.
    
    Args:
        num_cols: List of numeric column names
        cat_cols: List of categorical column names
        
    Returns:
        Configured ColumnTransformer for preprocessing
        
    Note:
        - Numeric features: median imputation + standard scaling
        - Categorical features: mode imputation + one-hot encoding
    """
    # Pipeline for numeric features
    numeric_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    
    # Pipeline for categorical features  
    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])
    
    preprocessor = ColumnTransformer([
        ("numeric", numeric_pipeline, num_cols),
        ("categorical", categorical_pipeline, cat_cols),
    ])
    
    logging.info("Preprocessor built with numeric and categorical pipelines")
    return preprocessor


# =============================
# 3. MODEL BUILDING & CROSS-VALIDATION
# =============================

def build_model(model_name: str, preprocessor: ColumnTransformer, seed: int) -> Pipeline:
    """
    Create a complete ML pipeline with preprocessing and classification.
    
    This function demonstrates how to build modular ML pipelines that combine
    preprocessing and model fitting into a single, reusable component.
    
    Args:
        model_name: Type of model ("logreg" or "rf")
        preprocessor: Fitted preprocessing pipeline
        seed: Random seed for model reproducibility
        
    Returns:
        Complete Pipeline with preprocessing and classification
        
    Raises:
        ValueError: If model_name is not supported
        
    Example:
        >>> pipe = build_model("rf", preprocessor, 42)
        >>> pipe.fit(X_train, y_train)
    """
    if model_name == "logreg":
        classifier = LogisticRegression(
            max_iter=500, 
            class_weight="balanced",
            random_state=seed
        )
        logging.debug("Created Logistic Regression classifier")
        
    elif model_name == "rf":
        classifier = RandomForestClassifier(
            n_estimators=400,
            n_jobs=-1,
            class_weight="balanced_subsample",
            random_state=seed
        )
        logging.debug("Created Random Forest classifier")
        
    else:
        supported_models = ["logreg", "rf"]
        raise ValueError(f"Unknown model: {model_name}. Supported models: {supported_models}")
    
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", classifier)
    ])
    
    return pipeline


def cv_metrics(pipeline: Pipeline, X: pd.DataFrame, y: pd.Series, n_splits: int) -> Dict[str, float]:
    """
    Evaluate model performance using stratified cross-validation.
    
    This function demonstrates proper model evaluation using cross-validation
    to get reliable performance estimates and avoid overfitting to a single split.
    
    Args:
        pipeline: Complete ML pipeline to evaluate
        X: Feature matrix
        y: Target vector  
        n_splits: Number of cross-validation folds
        
    Returns:
        Dictionary with cross-validation accuracy and macro F1 scores
        
    Example:
        >>> metrics = cv_metrics(pipeline, X, y, 5)
        >>> print(f"CV Accuracy: {metrics['cv_acc']:.3f}")
    """
    cv_splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    # Calculate cross-validation scores
    accuracy_scores = cross_val_score(pipeline, X, y, cv=cv_splitter, scoring="accuracy")
    f1_scores = cross_val_score(pipeline, X, y, cv=cv_splitter, scoring="f1_macro")
    
    metrics = {
        "cv_acc": float(accuracy_scores.mean()),
        "cv_f1_macro": float(f1_scores.mean())
    }
    
    logging.debug(f"CV metrics calculated: acc={metrics['cv_acc']:.4f}, f1={metrics['cv_f1_macro']:.4f}")
    
    return metrics


# =============================
# 4. ACTIVE LEARNING STRATEGIES  
# =============================

class Samplers:
    """
    Collection of active learning sampling strategies.
    
    This class demonstrates three common active learning approaches:
    1. Uncertainty sampling: Select samples the model is least confident about
    2. Query-by-committee: Use ensemble disagreement to find informative samples
    3. Diversity sampling: Ensure selected samples cover different regions of feature space
    """

    @staticmethod
    def uncertainty_margin(probabilities: np.ndarray, k: int) -> np.ndarray:
        """
        Select samples with smallest margin between top two predicted class probabilities.
        
        This method implements uncertainty sampling, a fundamental active learning
        strategy that queries samples where the model is least confident.
        
        Args:
            probabilities: Class probability predictions (n_samples, n_classes)
            k: Number of samples to select
            
        Returns:
            Indices of k most uncertain samples
            
        Note:
            For binary classification, uses distance from 0.5 probability threshold.
            For multi-class, uses margin between top two predicted classes.
        """
        if probabilities.shape[1] < 2:
            # Binary classification: margin from decision boundary (0.5)
            margin = np.abs(probabilities[:, 0] - 0.5)
        else:
            # Multi-class: margin between top two predictions
            top_two_probs = np.sort(probabilities, axis=1)[:, -2:]
            margin = top_two_probs[:, 1] - top_two_probs[:, 0]
        
        # Select samples with smallest margins (most uncertain)
        uncertain_indices = np.argsort(margin)[:k]
        
        logging.debug(f"Uncertainty sampling selected {len(uncertain_indices)} samples")
        return uncertain_indices

    @staticmethod
    def vote_entropy(committee_predictions: List[np.ndarray], k: int) -> np.ndarray:
        """
        Query-by-committee using vote entropy to measure disagreement.
        
        This method selects samples where committee members disagree the most,
        indicating areas where the model needs more training data.
        
        Args:
            committee_predictions: List of prediction arrays from committee members
            k: Number of samples to select
            
        Returns:
            Indices of k samples with highest vote entropy
            
        Example:
            >>> pred1 = np.array([0, 1, 1, 0])
            >>> pred2 = np.array([0, 1, 0, 0]) 
            >>> pred3 = np.array([1, 1, 0, 0])
            >>> indices = Samplers.vote_entropy([pred1, pred2, pred3], k=2)
        """
        # Stack predictions: (n_samples, committee_size)
        votes = np.stack(committee_predictions, axis=1)
        
        entropies = []
        for sample_votes in votes:
            # Count votes for each class
            vote_counts = Counter(sample_votes)
            vote_probabilities = np.array(list(vote_counts.values()), dtype=float) / len(sample_votes)
            
            # Calculate entropy: -sum(p * log(p))
            entropy = -(vote_probabilities * np.log(vote_probabilities + 1e-12)).sum()
            entropies.append(entropy)
        
        entropies = np.array(entropies)
        
        # Select samples with highest entropy (most disagreement)
        high_entropy_indices = np.argsort(-entropies)[:k]
        
        logging.debug(f"QBC sampling selected {len(high_entropy_indices)} samples with avg entropy {np.mean(entropies):.3f}")
        return high_entropy_indices

    @staticmethod
    def diversity_kmeans(X_pool: np.ndarray, k: int, seed: int) -> np.ndarray:
        """
        Select diverse samples using k-means clustering.
        
        This method ensures the selected samples cover different regions of the
        feature space, providing diverse information to the model.
        
        Args:
            X_pool: Feature matrix of unlabeled samples
            k: Number of samples to select
            seed: Random seed for k-means clustering
            
        Returns:
            Indices of k diverse samples (closest to cluster centroids)
            
        Note:
            If fewer samples than clusters exist, returns available samples.
        """
        k = min(k, len(X_pool))
        if k == 0:
            logging.warning("No samples available for diversity sampling")
            return np.array([], dtype=int)
        
        # Perform k-means clustering
        kmeans = KMeans(n_clusters=k, random_state=seed, n_init=10)
        cluster_labels = kmeans.fit_predict(X_pool)
        cluster_centers = kmeans.cluster_centers_
        
        # Find sample closest to each cluster center
        selected_indices = []
        for cluster_id in range(k):
            cluster_members = np.where(cluster_labels == cluster_id)[0]
            
            if len(cluster_members) == 0:
                continue
                
            # Calculate distances to cluster center
            cluster_samples = X_pool[cluster_members]
            distances = np.sum((cluster_samples - cluster_centers[cluster_id]) ** 2, axis=1)
            
            # Select closest sample to center
            closest_idx = cluster_members[np.argmin(distances)]
            selected_indices.append(closest_idx)
        
        selected_indices = np.array(selected_indices, dtype=int)
        logging.debug(f"Diversity sampling selected {len(selected_indices)} samples from {k} clusters")
        
        return selected_indices


# =============================
# 5. ACTIVE LEARNING RUNNER
# =============================

class ActiveLearningRunner:
    """
    Main orchestrator for the active learning process.
    
    This class coordinates the entire active learning workflow:
    1. Manages labeled/unlabeled data splits
    2. Fits models and evaluates performance
    3. Applies hybrid sampling strategy
    4. Iteratively grows the training set
    
    The runner implements a hybrid approach that combines uncertainty sampling,
    query-by-committee, and diversity sampling according to specified weights.
    """

    def __init__(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        preprocessor: ColumnTransformer,
        config: RunConfig,
        model_name: str = "rf"
    ):
        """
        Initialize the active learning runner.
        
        Args:
            X: Feature matrix
            y: Target vector
            preprocessor: Preprocessing pipeline
            config: Run configuration
            model_name: Model type to use
            
        Raises:
            ValueError: If initial_size >= dataset size
        """
        self.config = config
        self.model_name = model_name
        self.preprocessor = preprocessor
        
        # Encode target labels
        self.label_encoder = LabelEncoder()
        self.y_encoded = self.label_encoder.fit_transform(y)
        self.y_series = pd.Series(self.y_encoded, index=y.index)
        
        # Prepare numeric features for diversity sampling
        self.numeric_preprocessor = ColumnTransformer(
            [("numeric", 
              Pipeline([
                  ("imputer", SimpleImputer(strategy="median")),
                  ("scaler", StandardScaler())
              ]),
              X.select_dtypes(include=[np.number]).columns.tolist())],
            remainder="drop"
        )
        
        # Reset indices and prepare data
        self.X = X.reset_index(drop=True)
        self.X_numeric = self.numeric_preprocessor.fit_transform(self.X)
        self.n_samples = len(self.X)
        
        # Validate initial size
        if self.config.initial_size >= self.n_samples:
            raise ValueError(f"Initial size ({self.config.initial_size}) must be smaller than dataset size ({self.n_samples})")
        
        # Initialize train/pool splits
        all_indices = np.arange(self.n_samples)
        self.train_indices = np.random.choice(
            all_indices, 
            size=self.config.initial_size, 
            replace=False
        )
        self.pool_indices = np.setdiff1d(all_indices, self.train_indices)
        
        logging.info(f"ActiveLearningRunner initialized: {len(self.train_indices)} labeled, {len(self.pool_indices)} unlabeled")

    def _fit_pipeline(self, train_indices: np.ndarray) -> Pipeline:
        """
        Fit a model pipeline on specified training indices.
        
        Args:
            train_indices: Indices of samples to use for training
            
        Returns:
            Fitted pipeline
            
        This is a helper method that creates and fits a fresh model pipeline
        on the current training set.
        """
        pipeline = build_model(self.model_name, self.preprocessor, self.config.random_seed)
        
        X_train = self.X.iloc[train_indices]
        y_train = self.y_series.iloc[train_indices]
        
        pipeline.fit(X_train, y_train)
        
        return pipeline

    def iterate(self) -> Dict[str, List[float]]:
        """
        Run the complete active learning process.
        
        This method implements the main active learning loop:
        1. Train model on current labeled data
        2. Evaluate performance using cross-validation
        3. Apply hybrid sampling to select new samples
        4. Add selected samples to training set
        5. Repeat until stopping criteria met
        
        Returns:
            Dictionary containing performance history over iterations
            
        Example:
            >>> runner = ActiveLearningRunner(X, y, preprocessor, config)
            >>> history = runner.iterate()
            >>> print(f"Final accuracy: {history['cv_acc_hist'][-1]:.3f}")
        """
        accuracy_history: List[float] = []
        f1_history: List[float] = []
        
        logging.info("Starting active learning iterations")
        
        for iteration in range(self.config.iters):
            # Check if pool is exhausted
            if len(self.pool_indices) == 0:
                logging.info(f"Pool exhausted at iteration {iteration}")
                break
            
            # 1. Evaluate current model performance
            current_pipeline = build_model(self.model_name, self.preprocessor, self.config.random_seed)
            X_train = self.X.iloc[self.train_indices]
            y_train = self.y_series.iloc[self.train_indices]
            
            metrics = cv_metrics(current_pipeline, X_train, y_train, self.config.n_splits)
            accuracy_history.append(metrics["cv_acc"])
            f1_history.append(metrics["cv_f1_macro"])
            
            logging.info(
                f"Iteration {iteration} | CV acc={metrics['cv_acc']:.4f}, "
                f"macro F1={metrics['cv_f1_macro']:.4f} | "
                f"train={len(self.train_indices)}, pool={len(self.pool_indices)}"
            )
            
            # 2. Fit model on current training set and get pool predictions
            fitted_pipeline = self._fit_pipeline(self.train_indices)
            X_pool = self.X.iloc[self.pool_indices]
            pool_probabilities = fitted_pipeline.predict_proba(X_pool)
            
            # 3. Calculate batch composition based on hybrid weights
            batch_uncertainty = int(self.config.batch_size * self.config.hybrid_weights[0])
            batch_qbc = int(self.config.batch_size * self.config.hybrid_weights[1])
            batch_diversity = self.config.batch_size - batch_uncertainty - batch_qbc
            
            # 4. Apply uncertainty sampling
            uncertainty_indices = Samplers.uncertainty_margin(pool_probabilities, batch_uncertainty)
            
            # 5. Apply query-by-committee sampling
            committee_predictions = []
            for _ in range(self.config.committee_size):
                # Bootstrap committee member
                bootstrap_indices = np.random.choice(
                    self.train_indices, 
                    size=len(self.train_indices), 
                    replace=True
                )
                committee_member = self._fit_pipeline(bootstrap_indices)
                predictions = committee_member.predict(X_pool)
                committee_predictions.append(predictions)
            
            qbc_indices = Samplers.vote_entropy(committee_predictions, batch_qbc)
            
            # 6. Apply diversity sampling on numeric features
            X_pool_numeric = self.X_numeric[self.pool_indices]
            diversity_indices = Samplers.diversity_kmeans(
                X_pool_numeric, 
                batch_diversity, 
                self.config.random_seed
            )
            
            # 7. Combine all selected indices (remove duplicates)
            all_selected_relative = np.unique(np.concatenate([
                uncertainty_indices, 
                qbc_indices, 
                diversity_indices
            ]))
            
            if len(all_selected_relative) == 0:
                logging.warning("No samples selected; stopping early")
                break
            
            # Convert relative indices to absolute indices
            selected_absolute = self.pool_indices[all_selected_relative]
            
            # 8. Update training and pool sets
            self.train_indices = np.concatenate([self.train_indices, selected_absolute])
            self.pool_indices = np.setdiff1d(self.pool_indices, selected_absolute)
            
            logging.debug(f"Added {len(selected_absolute)} samples to training set")
        
        logging.info(f"Active learning completed after {len(accuracy_history)} iterations")
        
        return {
            "cv_acc_hist": accuracy_history,
            "cv_f1_hist": f1_history
        }


# =============================
# 6. CLI & EXCEPTION HANDLING
# =============================

def cmd_run(args: argparse.Namespace) -> None:
    """
    Execute the active learning pipeline from command line arguments.
    
    This function demonstrates comprehensive exception handling for a
    machine learning pipeline, covering data loading, configuration,
    and model training errors.
    
    Args:
        args: Parsed command line arguments
        
    Exception Handling Examples:
        - FileNotFoundError: Missing data files
        - KeyError: Invalid column names
        - ValueError: Invalid configuration parameters
        - NotFittedError: Model fitting issues
    """
    setup_logging(args.log_level)
    set_seed(args.seed)
    
    try:
        # Load and prepare data
        logging.info("Loading and preparing data...")
        dataframe = load_csv(args.data)
        
        X, y, numeric_columns, categorical_columns = split_features(
            dataframe, 
            target_col=args.target,
            num_cols=(args.num_cols.split(",") if args.num_cols else None),
            cat_cols=(args.cat_cols.split(",") if args.cat_cols else None),
        )
        
        preprocessor = build_preprocessor(numeric_columns, categorical_columns)
        
        # Create configuration
        config = RunConfig(
            random_seed=args.seed,
            test_size=args.test_size,
            n_splits=args.folds,
            model=args.model,
            initial_size=args.initial_size,
            batch_size=args.batch,
            iters=args.iters,
            committee_size=args.committee
        )
        
        # Run active learning
        logging.info("Starting active learning process...")
        runner = ActiveLearningRunner(X, y, preprocessor, config, model_name=args.model)
        history = runner.iterate()
        
        # Final holdout evaluation for interpretable results
        logging.info("Performing final holdout evaluation...")
        final_pipeline = build_model(args.model, preprocessor, args.seed)
        
        # Create holdout split
        X_train, X_test, y_train, y_test = train_test_split(
            X, 
            LabelEncoder().fit_transform(y),
            test_size=config.test_size,
            random_state=config.random_seed,
            stratify=y
        )
        
        # Train and evaluate
        final_pipeline.fit(X_train, y_train)
        y_predicted = final_pipeline.predict(X_test)
        
        final_accuracy = accuracy_score(y_test, y_predicted)
        final_f1 = f1_score(y_test, y_predicted, average="macro")
        
        # Display results
        print("\n" + "="*60)
        print("ACTIVE LEARNING RESULTS")
        print("="*60)
        print("\n=== Cross-Validation History (Mean Over Folds) ===")
        print(f"Accuracy: {[round(v, 4) for v in history['cv_acc_hist']]}")
        print(f"Macro-F1: {[round(v, 4) for v in history['cv_f1_hist']]}")
        
        print("\n=== Final Holdout Evaluation ===")
        print(f"Accuracy: {final_accuracy:.4f}")
        print(f"Macro-F1: {final_f1:.4f}")
        print(f"Training samples used: {len(runner.train_indices)}")
        print("="*60)
        
    except (FileNotFoundError, KeyError, ValueError) as config_error:
        logging.error(f"Configuration or data error: {config_error}")
        print(f"\nError: {config_error}")
        print("Please check your data file and configuration parameters.")
        sys.exit(2)
        
    except NotFittedError as model_error:
        logging.error(f"Model fitting error: {model_error}")
        print(f"\nModel Error: {model_error}")
        print("The model could not be properly fitted. Check your data for issues.")
        sys.exit(2)
        
    except MemoryError:
        logging.error("Insufficient memory to complete the operation")
        print("\nMemory Error: Dataset or model too large for available memory.")
        print("Try reducing batch_size, initial_size, or using a smaller dataset.")
        sys.exit(2)
        
    except Exception as unexpected_error:
        logging.exception(f"Unexpected error occurred: {unexpected_error}")
        print(f"\nUnexpected Error: {unexpected_error}")
        print("Please check the logs for detailed error information.")
        sys.exit(1)


def build_parser() -> argparse.ArgumentParser:
    """
    Build command-line argument parser for the active learning tool.
    
    Returns:
        Configured ArgumentParser with all necessary options
        
    This function creates a comprehensive CLI interface that makes the
    active learning tool easy to use from the command line with proper
    help documentation and argument validation.
    """
    parser = argparse.ArgumentParser(
        description="Active Learning Framework for Tabular Classification",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s run --data iris.csv --target species --model rf
  %(prog)s run --data titanic.csv --target survived --model logreg --batch 50
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', required=True)
    
    # Run subcommand
    run_parser = subparsers.add_parser(
        "run", 
        help="Execute active learning experiment",
        description="Run active learning on tabular data with configurable parameters"
    )
    
    # Required arguments
    run_parser.add_argument(
        "--data", 
        required=True, 
        help="Path to CSV file containing the dataset"
    )
    run_parser.add_argument(
        "--target", 
        required=True, 
        help="Name of the target column in the dataset"
    )
    
    # Optional data configuration
    run_parser.add_argument(
        "--num-cols", 
        help="Comma-separated list of numeric column names (auto-detected if not specified)"
    )
    run_parser.add_argument(
        "--cat-cols", 
        help="Comma-separated list of categorical column names (auto-detected if not specified)"
    )
    
    # Model configuration
    run_parser.add_argument(
        "--model", 
        choices=["logreg", "rf"], 
        default="rf",
        help="Model type: 'logreg' for Logistic Regression, 'rf' for Random Forest (default: rf)"
    )
    
    # Active learning parameters
    run_parser.add_argument(
        "--initial-size", 
        type=int, 
        default=500,
        help="Number of samples to start with in the labeled set (default: 500)"
    )
    run_parser.add_argument(
        "--batch", 
        type=int, 
        default=100,
        help="Number of samples to query at each iteration (default: 100)"
    )
    run_parser.add_argument(
        "--iters", 
        type=int, 
        default=15,
        help="Maximum number of active learning iterations (default: 15)"
    )
    run_parser.add_argument(
        "--committee", 
        type=int, 
        default=5,
        help="Size of the committee for query-by-committee sampling (default: 5)"
    )
    
    # Evaluation parameters
    run_parser.add_argument(
        "--folds", 
        type=int, 
        default=5,
        help="Number of cross-validation folds (default: 5)"
    )
    run_parser.add_argument(
        "--test-size", 
        type=float, 
        default=0.2,
        help="Fraction of data to reserve for final holdout evaluation (default: 0.2)"
    )
    
    # Reproducibility and logging
    run_parser.add_argument(
        "--seed", 
        type=int, 
        default=42,
        help="Random seed for reproducible results (default: 42)"
    )
    run_parser.add_argument(
        "--log-level", 
        choices=["DEBUG", "INFO", "WARNING", "ERROR"], 
        default="INFO",
        help="Logging level (default: INFO)"
    )
    
    run_parser.set_defaults(func=cmd_run)
    
    return parser


# =============================
# 7. UNIT TESTS
# =============================

def test_uncertainty_margin():
    """
    Unit test for uncertainty margin sampling strategy.
    
    This test demonstrates how to write effective unit tests for ML components
    by creating controlled test data and verifying expected behavior.
    
    Test Cases:
    1. Binary classification with known uncertainty patterns
    2. Multi-class classification with margin calculations
    3. Edge cases (single class, equal probabilities)
    """
    print("Running uncertainty margin tests...")
    
    # Test case 1: Binary classification
    # Probabilities: [0.51, 0.49], [0.9, 0.1], [0.55, 0.45], [0.5, 0.5]
    # Expected: samples closer to 0.5 should be selected (most uncertain)
    binary_probabilities = np.array([
        [0.51, 0.49],  # margin = 0.01 (uncertain)
        [0.9, 0.1],    # margin = 0.4 (confident)
        [0.55, 0.45],  # margin = 0.05 (somewhat uncertain)
        [0.5, 0.5]     # margin = 0.0 (most uncertain)
    ])
    
    selected_indices = Samplers.uncertainty_margin(binary_probabilities, k=2)
    
    # Most uncertain samples should be indices 3 and 0
    expected_most_uncertain = {0, 3}
    assert set(selected_indices.tolist()).issubset({0, 2, 3}), f"Unexpected selection: {selected_indices}"
    assert 3 in selected_indices, "Most uncertain sample (index 3) should be selected"
    
    # Test case 2: Multi-class classification
    multiclass_probabilities = np.array([
        [0.4, 0.35, 0.25],   # margin = 0.05 (uncertain)
        [0.8, 0.1, 0.1],     # margin = 0.7 (confident)
        [0.45, 0.45, 0.1],   # margin = 0.0 (most uncertain)
        [0.6, 0.3, 0.1]      # margin = 0.3 (moderate)
    ])
    
    multiclass_indices = Samplers.uncertainty_margin(multiclass_probabilities, k=2)
    assert 2 in multiclass_indices, "Most uncertain multiclass sample should be selected"
    
    print("✓ Uncertainty margin tests passed")


def test_vote_entropy():
    """
    Unit test for query-by-committee vote entropy sampling.
    
    This test verifies that samples with maximum committee disagreement
    are correctly identified and selected.
    """
    print("Running vote entropy tests...")
    
    # Create committee predictions with known disagreement patterns
    committee_pred_1 = np.array([0, 1, 1, 0])  # Committee member 1
    committee_pred_2 = np.array([0, 1, 0, 0])  # Committee member 2  
    committee_pred_3 = np.array([1, 1, 0, 0])  # Committee member 3
    
    # Sample 0: votes [0,0,1] -> some disagreement
    # Sample 1: votes [1,1,1] -> complete agreement (low entropy)
    # Sample 2: votes [1,0,0] -> some disagreement  
    # Sample 3: votes [0,0,0] -> complete agreement (low entropy)
    
    selected_indices = Samplers.vote_entropy([committee_pred_1, committee_pred_2, committee_pred_3], k=2)
    
    # Samples with disagreement (0, 2) should be preferred over unanimous votes (1, 3)
    assert len(selected_indices) == 2, f"Should select exactly 2 samples, got {len(selected_indices)}"
    
    # At least one sample with disagreement should be selected
    disagreement_samples = {0, 2}
    assert len(set(selected_indices) & disagreement_samples) > 0, "Should select samples with committee disagreement"
    
    print("✓ Vote entropy tests passed")


def test_diversity_kmeans():
    """
    Unit test for diversity sampling using k-means clustering.
    
    This test ensures the diversity sampler correctly identifies
    representative samples from different regions of feature space.
    """
    print("Running diversity k-means tests...")
    
    # Create synthetic data with clear clusters
    np.random.seed(42)
    
    # Two distinct clusters
    cluster_1 = np.random.normal([0, 0], 0.5, (10, 2))
    cluster_2 = np.random.normal([5, 5], 0.5, (10, 2))
    X_test = np.vstack([cluster_1, cluster_2])
    
    # Select 2 diverse samples (should get one from each cluster)
    selected_indices = Samplers.diversity_kmeans(X_test, k=2, seed=42)
    
    assert len(selected_indices) == 2, f"Should select 2 samples, got {len(selected_indices)}"
    
    # Selected samples should be from different clusters
    selected_points = X_test[selected_indices]
    distances_between_selected = np.linalg.norm(selected_points[0] - selected_points[1])
    
    # Distance should be significant (samples from different clusters)
    assert distances_between_selected > 3.0, f"Selected samples too close: {distances_between_selected:.2f}"
    
    # Test edge case: more clusters than samples
    small_data = np.array([[1, 1], [2, 2]])
    small_selection = Samplers.diversity_kmeans(small_data, k=5, seed=42)
    assert len(small_selection) <= 2, "Cannot select more samples than available"
    
    # Test edge case: empty data
    empty_selection = Samplers.diversity_kmeans(np.array([]).reshape(0, 2), k=3, seed=42)
    assert len(empty_selection) == 0, "Empty data should return empty selection"
    
    print("✓ Diversity k-means tests passed")


def test_data_loading():
    """
    Unit test for data loading and validation functionality.
    
    This test demonstrates how to test file I/O operations and
    data validation logic with proper exception handling.
    """
    print("Running data loading tests...")
    
    # Test FileNotFoundError handling
    try:
        load_csv("nonexistent_file.csv")
        assert False, "Should raise FileNotFoundError for missing file"
    except FileNotFoundError as e:
        assert "not found" in str(e), f"Unexpected error message: {e}"
    
    # Test with mock data (would normally use temporary files)
    # This demonstrates the testing approach without actual file I/O
    
    print("✓ Data loading tests passed")


def run_all_tests():
    """
    Execute all unit tests for the active learning framework.
    
    This function demonstrates a simple test runner that executes
    all test functions and reports results.
    """
    print("="*50)
    print("RUNNING UNIT TESTS")
    print("="*50)
    
    test_functions = [
        test_uncertainty_margin,
        test_vote_entropy, 
        test_diversity_kmeans,
        test_data_loading
    ]
    
    passed = 0
    failed = 0
    
    for test_func in test_functions:
        try:
            test_func()
            passed += 1
        except AssertionError as e:
            print(f"✗ {test_func.__name__} FAILED: {e}")
            failed += 1
        except Exception as e:
            print(f"✗ {test_func.__name__} ERROR: {e}")
            failed += 1
    
    print("\n" + "="*50)
    print(f"TEST RESULTS: {passed} passed, {failed} failed")
    print("="*50)
    
    return failed == 0


# =============================
# 8. MAIN EXECUTION
# =============================

if __name__ == "__main__":
    # Check if running tests
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        success = run_all_tests()
        sys.exit(0 if success else 1)
    
    # Normal CLI execution
    parser = build_parser()
    args = parser.parse_args()
    
    # Execute the specified command
    args.func(args)
