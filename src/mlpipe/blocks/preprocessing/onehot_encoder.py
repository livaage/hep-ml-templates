"""One-hot encoding for categorical features.

This module provides functionality to encode categorical variables as one-hot vectors,
which is essential for many machine learning algorithms that cannot handle categorical
data directly.

Features:
- Automatic categorical column detection
- Handling of unknown categories
- Memory-efficient sparse output option
- Integration with existing pipelines
"""

from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

from mlpipe.core.interfaces import Preprocessor
from mlpipe.core.registry import register


@register("preprocessing.onehot_encoder")
class OneHotEncoderBlock(Preprocessor):
    """One-hot encoding for categorical features.

    Converts categorical variables into binary vectors, enabling their use
    with algorithms that require numerical input.

    Features:
    - Automatic detection of categorical columns
    - Handling of unknown categories during inference
    - Option to drop first category to avoid multicollinearity
    - Memory-efficient sparse output

    Example usage:
        encoder = OneHotEncoderBlock()
        encoder.build({
            'categorical_columns': ['category_A', 'category_B'],
            'drop_first': True,
            'handle_unknown': 'ignore'
        })
        encoder.fit(X_train)
        X_encoded = encoder.transform(X)
    """

    def __init__(self):
        super().__init__()
        self.encoder = None
        self.categorical_columns = None
        self.feature_names = None
        self.config = {}

    def build(self, config: Dict[str, Any]) -> None:
        """Configure the one-hot encoder."""
        default_config = {
            "categorical_columns": "auto",  # 'auto' to detect, or list of column names/indices
            "drop_first": False,  # Drop first category to avoid multicollinearity
            "handle_unknown": "ignore",  # 'error', 'ignore', or 'infrequent_if_exist'
            "sparse_output": False,  # Return sparse matrix (memory efficient)
            "dtype": np.float32,  # Output data type
            "max_categories": None,  # Limit number of categories per feature
            "min_frequency": None,  # Minimum frequency for a category to be included
            "verbose": True,
        }

        self.config = {**default_config, **config}

        if self.config["verbose"]:
            print("🎯 One-Hot Encoder Configuration:")
            print(f"   Categorical columns: {self.config['categorical_columns']}")
            print(f"   Drop first: {self.config['drop_first']}")
            print(f"   Handle unknown: {self.config['handle_unknown']}")
            print(f"   Sparse output: {self.config['sparse_output']}")

    def _detect_categorical_columns(
        self, X: Union[np.ndarray, pd.DataFrame]
    ) -> List[Union[int, str]]:
        """Automatically detect categorical columns."""
        if isinstance(X, pd.DataFrame):
            # For DataFrames, use dtype information
            categorical_cols = []

            for col in X.columns:
                if X[col].dtype == "object" or X[col].dtype.name == "category":
                    categorical_cols.append(col)
                elif X[col].dtype in ["int64", "int32"] and X[col].nunique() <= 20:
                    # Assume integer columns with few unique values are categorical
                    categorical_cols.append(col)

            if self.config["verbose"] and categorical_cols:
                print(f"🔍 Auto-detected categorical columns: {categorical_cols}")
                for col in categorical_cols:
                    unique_vals = X[col].nunique()
                    print(f"   {col}: {unique_vals} unique values")

            return categorical_cols

        else:
            # For numpy arrays, assume all columns with few unique values are categorical
            categorical_cols = []

            for i in range(X.shape[1]):
                unique_vals = len(np.unique(X[:, i]))
                if unique_vals <= 20:  # Heuristic for categorical
                    categorical_cols.append(i)

            if self.config["verbose"] and categorical_cols:
                print(f"🔍 Auto-detected categorical columns (indices): {categorical_cols}")

            return categorical_cols

    def _validate_columns(self, X: Union[np.ndarray, pd.DataFrame], columns: List) -> List:
        """Validate that specified columns exist in the data."""
        if isinstance(X, pd.DataFrame):
            missing_cols = [col for col in columns if col not in X.columns]
            if missing_cols:
                raise ValueError(f"Columns not found in DataFrame: {missing_cols}")
        else:
            # For numpy arrays, check indices
            max_idx = X.shape[1] - 1
            invalid_indices = [idx for idx in columns if isinstance(idx, int) and idx > max_idx]
            if invalid_indices:
                raise ValueError(f"Column indices out of range: {invalid_indices}")

        return columns

    def fit(
        self, X: Union[np.ndarray, pd.DataFrame], y: Optional[np.ndarray] = None
    ) -> "OneHotEncoderBlock":
        """Fit the one-hot encoder to the data."""
        # Determine categorical columns
        if self.config["categorical_columns"] == "auto":
            self.categorical_columns = self._detect_categorical_columns(X)
        else:
            self.categorical_columns = self._validate_columns(X, self.config["categorical_columns"])

        if not self.categorical_columns:
            if self.config["verbose"]:
                print("⚠️  No categorical columns found - encoder will pass through data unchanged")
            # Create a passthrough transformer
            self.encoder = ColumnTransformer([("passthrough", "passthrough", slice(None))])
            self.encoder.fit(X)
            return self

        # Configure the OneHotEncoder
        encoder_params = {
            "drop": "first" if self.config["drop_first"] else None,
            "handle_unknown": self.config["handle_unknown"],
            "sparse_output": self.config["sparse_output"],
            "dtype": self.config["dtype"],
        }

        # Add optional parameters if specified
        if self.config["max_categories"] is not None:
            encoder_params["max_categories"] = self.config["max_categories"]
        if self.config["min_frequency"] is not None:
            encoder_params["min_frequency"] = self.config["min_frequency"]

        onehot_encoder = OneHotEncoder(**encoder_params)

        if isinstance(X, pd.DataFrame):
            # For DataFrames, we'll use ColumnTransformer to handle both categorical and numerical columns
            numerical_columns = [col for col in X.columns if col not in self.categorical_columns]

            transformers = []
            if self.categorical_columns:
                transformers.append(("onehot", onehot_encoder, self.categorical_columns))
            if numerical_columns:
                transformers.append(("passthrough", "passthrough", numerical_columns))

            self.encoder = ColumnTransformer(transformers, remainder="drop")

        else:
            # For numpy arrays, create a simple encoder for categorical columns only
            self.encoder = onehot_encoder

        # Fit the encoder
        if isinstance(X, pd.DataFrame):
            self.encoder.fit(X)
        else:
            # For numpy arrays, select only categorical columns
            X_categorical = X[:, self.categorical_columns] if self.categorical_columns else X
            self.encoder.fit(X_categorical)

        # Store feature names for later use
        if hasattr(self.encoder, "get_feature_names_out"):
            try:
                self.feature_names = self.encoder.get_feature_names_out()
            except:
                self.feature_names = None

        if self.config["verbose"]:
            print("✅ One-hot encoder fitted:")
            if isinstance(X, pd.DataFrame):
                print(f"   Input features: {X.shape[1]}")
            else:
                print(f"   Input features: {X.shape[1]}")
            print(f"   Categorical features: {len(self.categorical_columns)}")

        return self

    def transform(self, X: Union[np.ndarray, pd.DataFrame]) -> Union[np.ndarray, pd.DataFrame]:
        """Transform data using the fitted encoder."""
        if self.encoder is None:
            raise ValueError("Encoder must be fitted before transform. Call fit() first.")

        if not self.categorical_columns:
            # No categorical columns - return unchanged
            return X

        if isinstance(X, pd.DataFrame):
            # Transform using ColumnTransformer
            X_transformed = self.encoder.transform(X)

            if self.config["sparse_output"]:
                return X_transformed  # Return sparse matrix
            else:
                if hasattr(X_transformed, "toarray"):
                    X_transformed = X_transformed.toarray()

                # Try to create DataFrame with feature names
                if self.feature_names is not None:
                    try:
                        return pd.DataFrame(
                            X_transformed, columns=self.feature_names, index=X.index
                        )
                    except:
                        pass

                return pd.DataFrame(X_transformed, index=X.index)

        else:
            # For numpy arrays
            X_categorical = X[:, self.categorical_columns]
            X_encoded = self.encoder.transform(X_categorical)

            if self.config["sparse_output"]:
                return X_encoded
            else:
                if hasattr(X_encoded, "toarray"):
                    X_encoded = X_encoded.toarray()

                # Combine with non-categorical columns if any
                numerical_columns = [
                    i for i in range(X.shape[1]) if i not in self.categorical_columns
                ]
                if numerical_columns:
                    X_numerical = X[:, numerical_columns]
                    return np.hstack([X_numerical, X_encoded])
                else:
                    return X_encoded

    def fit_transform(
        self, X: Union[np.ndarray, pd.DataFrame], y: Optional[np.ndarray] = None
    ) -> Union[np.ndarray, pd.DataFrame]:
        """Fit the encoder and transform the data in one step."""
        return self.fit(X, y).transform(X)

    def get_feature_names_out(self, input_features: Optional[List[str]] = None) -> np.ndarray:
        """Get output feature names for transformation."""
        if self.encoder is None:
            raise ValueError("Encoder must be fitted before getting feature names.")

        if hasattr(self.encoder, "get_feature_names_out"):
            return self.encoder.get_feature_names_out(input_features)
        else:
            # Fallback for older scikit-learn versions
            if self.feature_names is not None:
                return np.array(self.feature_names)
            else:
                # Generate generic names
                if isinstance(self.encoder, ColumnTransformer):
                    n_features = self.encoder.transform(
                        np.zeros((1, len(input_features or [])))
                    ).shape[1]
                else:
                    n_features = self.encoder.transform(
                        np.zeros((1, len(self.categorical_columns)))
                    ).shape[1]
                return np.array([f"feature_{i}" for i in range(n_features)])

    def inverse_transform(
        self, X: Union[np.ndarray, pd.DataFrame]
    ) -> Union[np.ndarray, pd.DataFrame]:
        """Inverse transform the encoded data back to original format."""
        if self.encoder is None:
            raise ValueError("Encoder must be fitted before inverse transform.")

        try:
            return self.encoder.inverse_transform(X)
        except AttributeError:
            raise NotImplementedError("Inverse transform not available for this configuration.")
