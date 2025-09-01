"""Data splitting utilities for train/test/validation splits.
Provides flexible and configurable data splitting for ML pipelines.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from mlpipe.core.interfaces import Preprocessor
from mlpipe.core.registry import register


@register("preprocessing.data_split")
class DataSplitter(Preprocessor):
    """Flexible data splitting for train/test/validation splits.

    Supports:
    - Simple train/test splits
    - Train/validation/test splits
    - Stratified splits for classification
    - Time series splits (ordered)
    - Random seed control for reproducibility

    Example usage:
        splitter = DataSplitter({
            'train_size': 0.7,
            'val_size': 0.15,
            'test_size': 0.15,
            'stratify': True,
            'random_state': 42
        })
        splits = splitter.fit_transform(X, y)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize data splitter with configuration."""
        if config is None:
            config = {}

        # Split sizes (must sum to 1.0)
        self.train_size = config.get("train_size", 0.8)
        self.val_size = config.get("val_size", 0.0)  # 0 means no validation set
        self.test_size = config.get("test_size", 0.2)

        # Split strategy
        self.stratify = config.get("stratify", False)  # Use stratified sampling
        self.shuffle = config.get("shuffle", True)  # Shuffle before splitting
        self.random_state = config.get("random_state", 42)  # For reproducibility

        # Time series support
        self.time_series = config.get("time_series", False)  # Ordered splits
        self.time_column = config.get("time_column", None)  # Column to sort by

        # Advanced options
        self.verbose = config.get("verbose", True)
        self.validate_splits = config.get("validate_splits", True)

        # Store config
        self.config = config

        # Validate configuration
        self._validate_config()

        # Store data info after fit
        self.n_samples = None
        self.n_features = None
        self.target_distribution = None

    def _validate_config(self):
        """Validate configuration parameters."""
        # Check sizes sum to 1.0 (with small tolerance)
        total_size = self.train_size + self.val_size + self.test_size
        if abs(total_size - 1.0) > 1e-6:
            raise ValueError(f"Split sizes must sum to 1.0, got {total_size}")

        # Check all sizes are positive
        if self.train_size <= 0:
            raise ValueError(f"train_size must be positive, got {self.train_size}")
        if self.test_size <= 0:
            raise ValueError(f"test_size must be positive, got {self.test_size}")
        if self.val_size < 0:
            raise ValueError(f"val_size must be non-negative, got {self.val_size}")

        # Time series validation
        if self.time_series and self.shuffle:
            if self.verbose:
                print("⚠️  Warning: time_series=True overrides shuffle=False")
            self.shuffle = False

    def _print_split_info(self, X: pd.DataFrame, y: pd.Series, splits: Dict[str, Tuple]):
        """Print information about the data splits."""
        if not self.verbose:
            return

        print("📊 Data Split Summary:")
        print("=" * 40)
        print(f"Total samples: {len(X):,}")
        print(f"Total features: {X.shape[1]:,}")

        for split_name, (X_split, y_split) in splits.items():
            size = len(X_split)
            percentage = (size / len(X)) * 100
            print(f"{split_name.title():<12}: {size:,} samples ({percentage:.1f}%)")

            # Show target distribution for classification
            if y_split is not None and hasattr(y_split, "value_counts"):
                unique_vals = y_split.nunique()
                if unique_vals <= 10:  # Show distribution for <= 10 classes
                    print(f"{'':>14} Target dist: {dict(y_split.value_counts())}")
        print()

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> DataSplitter:
        """Fit the splitter (mainly for storing data info).

        Args:
            X: Feature dataframe
            y: Target series (optional)

        Returns:
            Self for method chaining
        """
        self.n_samples = len(X)
        self.n_features = X.shape[1]

        if y is not None:
            # Store target distribution info
            if hasattr(y, "value_counts"):
                self.target_distribution = dict(y.value_counts())
            else:
                self.target_distribution = {"type": "continuous", "range": (y.min(), y.max())}

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """For consistency with Preprocessor interface, but splitting is done in fit_transform.
        """
        raise NotImplementedError(
            "DataSplitter doesn't support transform(). Use fit_transform() or split() instead."
        )

    def fit_transform(
        self, X: pd.DataFrame, y: Optional[pd.Series] = None
    ) -> Dict[str, Tuple[pd.DataFrame, pd.Series]]:
        """Fit and split the data in one step.

        Args:
            X: Feature dataframe
            y: Target series (optional)

        Returns:
            Dictionary with split data: {'train': (X_train, y_train), 'test': (X_test, y_test), ...}
        """
        self.fit(X, y)
        return self.split(X, y)

    def split(
        self, X: pd.DataFrame, y: Optional[pd.Series] = None
    ) -> Dict[str, Tuple[pd.DataFrame, pd.Series]]:
        """Split the data into train/test/validation sets.

        Args:
            X: Feature dataframe to split
            y: Target series to split (optional)

        Returns:
            Dictionary containing the splits:
            - Always includes: 'train', 'test'
            - Includes 'val' if val_size > 0
        """
        if len(X) == 0:
            raise ValueError("Cannot split empty dataset")

        # Handle time series splitting
        if self.time_series:
            return self._time_series_split(X, y)

        # Handle stratification
        stratify_target = None
        if self.stratify and y is not None:
            # Only stratify for classification (discrete targets)
            if hasattr(y, "nunique") and y.nunique() <= 100:  # Reasonable class limit
                stratify_target = y
            elif self.verbose:
                print(
                    "⚠️  Warning: stratify=True but target appears continuous. Using random split."
                )

        # Perform splits
        if self.val_size > 0:
            # Three-way split: train/val/test
            splits = self._three_way_split(X, y, stratify_target)
        else:
            # Two-way split: train/test
            splits = self._two_way_split(X, y, stratify_target)

        # Print split information
        self._print_split_info(X, y, splits)

        # Validate splits if requested
        if self.validate_splits:
            self._validate_splits(splits)

        return splits

    def _two_way_split(
        self, X: pd.DataFrame, y: Optional[pd.Series], stratify_target
    ) -> Dict[str, Tuple]:
        """Perform a two-way train/test split."""
        # Adjust test size to account for no validation set
        actual_test_size = self.test_size + self.val_size

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=actual_test_size,
            random_state=self.random_state,
            shuffle=self.shuffle,
            stratify=stratify_target,
        )

        return {"train": (X_train, y_train), "test": (X_test, y_test)}

    def _three_way_split(
        self, X: pd.DataFrame, y: Optional[pd.Series], stratify_target
    ) -> Dict[str, Tuple]:
        """Perform a three-way train/val/test split."""
        # First split: separate train+val from test
        temp_size = self.train_size + self.val_size  # Size of train+val combined

        X_temp, X_test, y_temp, y_test = train_test_split(
            X,
            y,
            test_size=self.test_size,
            random_state=self.random_state,
            shuffle=self.shuffle,
            stratify=stratify_target,
        )

        # Second split: separate train from val
        # Calculate relative sizes within the temp set
        val_relative_size = self.val_size / temp_size

        # Handle stratification for second split
        temp_stratify = None
        if stratify_target is not None:
            temp_stratify = y_temp

        X_train, X_val, y_train, y_val = train_test_split(
            X_temp,
            y_temp,
            test_size=val_relative_size,
            random_state=self.random_state,
            shuffle=self.shuffle,
            stratify=temp_stratify,
        )

        return {"train": (X_train, y_train), "val": (X_val, y_val), "test": (X_test, y_test)}

    def _time_series_split(self, X: pd.DataFrame, y: Optional[pd.Series]) -> Dict[str, Tuple]:
        """Perform ordered splits for time series data."""
        # Sort by time column if specified
        if self.time_column and self.time_column in X.columns:
            sort_idx = X[self.time_column].argsort()
            X = X.iloc[sort_idx]
            if y is not None:
                y = y.iloc[sort_idx]

        n_samples = len(X)

        # Calculate split indices
        train_end = int(n_samples * self.train_size)

        if self.val_size > 0:
            val_end = train_end + int(n_samples * self.val_size)

            X_train = X.iloc[:train_end]
            X_val = X.iloc[train_end:val_end]
            X_test = X.iloc[val_end:]

            splits = {"train": (X_train, None), "val": (X_val, None), "test": (X_test, None)}

            if y is not None:
                y_train = y.iloc[:train_end]
                y_val = y.iloc[train_end:val_end]
                y_test = y.iloc[val_end:]
                splits = {
                    "train": (X_train, y_train),
                    "val": (X_val, y_val),
                    "test": (X_test, y_test),
                }
        else:
            X_train = X.iloc[:train_end]
            X_test = X.iloc[train_end:]

            splits = {"train": (X_train, None), "test": (X_test, None)}

            if y is not None:
                y_train = y.iloc[:train_end]
                y_test = y.iloc[train_end:]
                splits = {"train": (X_train, y_train), "test": (X_test, y_test)}

        if self.verbose:
            print("⏰ Time series split: maintaining temporal order")

        return splits

    def _validate_splits(self, splits: Dict[str, Tuple]):
        """Validate that splits don't have overlapping indices."""
        all_indices = set()

        for split_name, (X_split, _) in splits.items():
            split_indices = set(X_split.index)

            # Check for overlap
            overlap = all_indices.intersection(split_indices)
            if overlap:
                raise ValueError(
                    f"Split '{split_name}' has overlapping indices: {list(overlap)[:5]}..."
                )

            all_indices.update(split_indices)

        if self.verbose:
            print("✅ Split validation passed: no overlapping indices")


# Convenience function for easy splitting
def split_data(
    X: pd.DataFrame,
    y: Optional[pd.Series] = None,
    train_size: float = 0.8,
    val_size: float = 0.0,
    test_size: float = 0.2,
    stratify: bool = False,
    random_state: int = 42,
    **kwargs,
) -> Dict[str, Tuple[pd.DataFrame, pd.Series]]:
    """Convenience function for quick data splitting.

    Args:
        X: Feature dataframe
        y: Target series (optional)
        train_size: Fraction for training set (default: 0.8)
        val_size: Fraction for validation set (default: 0.0)
        test_size: Fraction for test set (default: 0.2)
        stratify: Whether to use stratified sampling (default: False)
        random_state: Random seed for reproducibility (default: 42)
        **kwargs: Additional configuration options

    Returns:
        Dictionary with split data: {'train': (X_train, y_train), 'test': (X_test, y_test), ...}

    Example:
        splits = split_data(X, y, train_size=0.7, val_size=0.15, test_size=0.15, stratify=True)
        X_train, y_train = splits['train']
        X_val, y_val = splits['val']
        X_test, y_test = splits['test']
    """
    config = {
        "train_size": train_size,
        "val_size": val_size,
        "test_size": test_size,
        "stratify": stratify,
        "random_state": random_state,
        **kwargs,
    }

    splitter = DataSplitter(config)
    return splitter.fit_transform(X, y)
