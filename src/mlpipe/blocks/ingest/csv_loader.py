from __future__ import annotations
from pathlib import Path
from typing import Tuple, Optional, Dict, Any
import pandas as pd
import warnings
from mlpipe.core.interfaces import DataIngestor
from mlpipe.core.registry import register


@register("ingest.csv")
class UniversalCSVLoader(DataIngestor):
    """
    Universal CSV data loader that's completely config-driven and beginner-friendly.

    Works with any CSV dataset by specifying metadata in config files.
    Automatically handles data validation, type inference, and preprocessing.

    Example usage via config:
        loader = UniversalCSVLoader(config)
        X, y = loader.load()
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize with configuration."""
        if config is None:
            config = {}

        # Core settings (required)
        self.file_path = config.get('file_path', '')
        self.target_column = config.get('target_column', 'target')

        # Dataset structure (auto-detected if not specified)
        self.has_header = config.get('has_header', True)
        self.column_names = config.get('column_names', None)  # List of column names
        self.separator = config.get('separator', ',')
        self.encoding = config.get('encoding', 'utf-8')

        # Data sampling (optional)
        self.nrows = config.get('nrows', None)  # Limit rows for testing
        self.skip_rows = config.get('skip_rows', None)  # Skip rows at start
        self.sample_fraction = config.get('sample_fraction', None)  # Random sample

        # Data preprocessing (optional)
        self.missing_values = config.get('missing_values', ['', 'NA', 'NaN', 'null', 'NULL'])
        self.drop_missing_threshold = config.get(
            'drop_missing_threshold', 0.5)  # Drop cols with >50% missing
        self.categorical_columns = config.get('categorical_columns', [])
        self.numerical_columns = config.get('numerical_columns', [])

        # Target processing
        # 'binary', 'multiclass', 'regression', 'auto'
        self.target_type = config.get('target_type', 'auto')
        self.positive_class_labels = config.get(
            'positive_class_labels', [1, '1', 'yes', 'true', 'signal'])

        # Advanced options
        self.validation = config.get('validation', True)  # Perform data validation
        self.verbose = config.get('verbose', True)

        # Store config for reference
        self.config = config

    def _validate_file_path(self) -> Path:
        """Validate and resolve file path."""
        if not self.file_path:
            raise ValueError("file_path is required in config")

        path = Path(self.file_path)

        # If relative path, assume it's relative to project root
        if not path.is_absolute():
            # Look for the file in common locations
            possible_paths = [
                Path.cwd() / self.file_path,
                Path.cwd() / "data" / self.file_path,
                Path(__file__).parent.parent.parent.parent / self.file_path,
                Path(__file__).parent.parent.parent.parent / "data" / self.file_path
            ]

            for p in possible_paths:
                if p.exists():
                    path = p
                    break
            else:
                if self.verbose:
                    print("⚠️  File not found in standard locations:")
                    for p in possible_paths:
                        print(f"   - {p}")
                raise FileNotFoundError(f"CSV file not found: {self.file_path}")

        if not path.exists():
            raise FileNotFoundError(f"CSV file not found: {path}")

        if self.verbose:
            print(f"📁 Loading CSV from: {path}")

        return path

    def _detect_separator(self, file_path: Path) -> str:
        """Auto-detect CSV separator if not specified."""
        if self.separator != ',':
            return self.separator

        # Read first few lines to detect separator
        with open(file_path, 'r', encoding=self.encoding) as f:
            sample_lines = [f.readline() for _ in range(min(5, sum(1 for _ in f)))]

        separators = [',', ';', '\t', '|']
        separator_scores = {}

        for sep in separators:
            scores = []
            for line in sample_lines:
                if line.strip():
                    scores.append(line.count(sep))
            if scores and all(s == scores[0] for s in scores) and scores[0] > 0:
                separator_scores[sep] = scores[0]

        if separator_scores:
            detected_sep = max(separator_scores, key=separator_scores.get)
            if detected_sep != ',' and self.verbose:
                print(f"🔍 Auto-detected separator: '{detected_sep}'")
            return detected_sep

        return ','

    def _load_raw_data(self, file_path: Path) -> pd.DataFrame:
        """Load raw CSV data with error handling."""
        separator = self._detect_separator(file_path)

        try:
            # Initial load to detect structure
            df_sample = pd.read_csv(
                file_path,
                sep=separator,
                encoding=self.encoding,
                nrows=100,  # Sample for structure detection
                na_values=self.missing_values
            )

            if self.verbose:
                print("📊 Dataset structure detected:")
                print(f"   - Shape (sample): {df_sample.shape}")
                print(f"   - Columns: {len(df_sample.columns)}")
                print(f"   - Has header: {self.has_header}")

            # Load full dataset
            load_kwargs = {
                'sep': separator,
                'encoding': self.encoding,
                'na_values': self.missing_values,
                'nrows': self.nrows,
                'skiprows': self.skip_rows
            }

            # Handle column names
            if not self.has_header:
                if self.column_names:
                    load_kwargs['names'] = self.column_names
                    load_kwargs['header'] = None
                else:
                    # Auto-generate column names
                    n_cols = len(df_sample.columns)
                    load_kwargs['names'] = [f'feature_{i}' for i in range(n_cols)]
                    load_kwargs['header'] = None
            elif self.column_names:
                # Override header names
                df = pd.read_csv(file_path, **load_kwargs)
                if len(df.columns) == len(self.column_names):
                    df.columns = self.column_names
                else:
                    warnings.warn(
                        f"Column name count mismatch: got {len(self.column_names)}, "
                        f"expected {len(df.columns)}")
                return df

            df = pd.read_csv(file_path, **load_kwargs)

            if self.verbose:
                print(f"✅ Data loaded successfully: {df.shape}")

            return df

        except Exception as e:
            raise RuntimeError(f"Failed to load CSV file: {e}")

    def _validate_target_column(self, df: pd.DataFrame) -> str:
        """Validate and locate target column."""
        if self.target_column not in df.columns:
            if self.verbose:
                print(f"⚠️  Target column '{self.target_column}' not found.")
                print(f"   Available columns: {list(df.columns)}")

            # Try common target column names
            common_targets = ['target', 'label', 'y', 'class', 'output']
            for target in common_targets:
                if target in df.columns:
                    if self.verbose:
                        print(f"🎯 Using '{target}' as target column")
                    return target

            # If still not found, use last column as common convention
            target_col = df.columns[-1]
            if self.verbose:
                print(f"🎯 Using last column '{target_col}' as target")
            return target_col

        return self.target_column

    def _preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess the loaded data."""
        if self.verbose:
            print("🔧 Preprocessing data...")

        original_shape = df.shape

        # Handle missing values
        missing_stats = df.isnull().sum()
        if missing_stats.sum() > 0:
            if self.verbose:
                print(f"   Missing values found: {missing_stats.sum()} total")

            # Drop columns with too many missing values
            high_missing_cols = missing_stats[missing_stats >
                                              len(df) * self.drop_missing_threshold].index
            if len(high_missing_cols) > 0:
                df = df.drop(columns=high_missing_cols)
                if self.verbose:
                    print(
                        f"   Dropped {len(high_missing_cols)} columns with "
                        f">{self.drop_missing_threshold*100}% missing")

        # Sample data if requested
        if self.sample_fraction and 0 < self.sample_fraction < 1:
            df = df.sample(frac=self.sample_fraction, random_state=42)
            if self.verbose:
                print(f"   Sampled {self.sample_fraction*100}% of data")

        if self.verbose and df.shape != original_shape:
            print(f"   Shape after preprocessing: {df.shape}")

        return df

    def _process_target(self, y: pd.Series) -> Tuple[pd.Series, Dict[str, Any]]:
        """Process target variable and determine task type."""
        target_info = {'original_type': str(y.dtype)}

        if self.target_type == 'auto':
            # Auto-detect task type
            unique_values = y.dropna().unique()
            n_unique = len(unique_values)

            if n_unique == 2:
                target_info['task_type'] = 'binary'
            elif n_unique <= 10 and y.dtype == 'object':
                target_info['task_type'] = 'multiclass'
            elif y.dtype in ['int64', 'float64'] and n_unique > 10:
                target_info['task_type'] = 'regression'
            else:
                target_info['task_type'] = 'multiclass'
        else:
            target_info['task_type'] = self.target_type

        # Process based on task type
        if target_info['task_type'] == 'binary':
            # Convert to binary (0/1)
            positive_mask = y.isin(self.positive_class_labels)
            y_processed = positive_mask.astype(int)
            target_info['classes'] = [0, 1]
            target_info['positive_class'] = 1
        elif target_info['task_type'] == 'multiclass':
            # Encode categorical labels
            if y.dtype == 'object':
                from sklearn.preprocessing import LabelEncoder
                le = LabelEncoder()
                y_processed = pd.Series(le.fit_transform(y.dropna()),
                                        index=y.dropna().index, name=y.name)
                target_info['classes'] = list(le.classes_)
                target_info['label_encoder'] = le
            else:
                y_processed = y.copy()
                target_info['classes'] = sorted(y.dropna().unique())
        else:  # regression
            y_processed = pd.to_numeric(y, errors='coerce')
            target_info['min_value'] = float(y_processed.min())
            target_info['max_value'] = float(y_processed.max())

        if self.verbose:
            print("🎯 Target processing:")
            print(f"   Task type: {target_info['task_type']}")
            if 'classes' in target_info:
                print(f"   Classes: {target_info['classes']}")
            print(f"   Target distribution: {y_processed.value_counts().head()}")

        return y_processed, target_info

    def load(self) -> Tuple[pd.DataFrame, pd.Series, Dict[str, Any]]:
        """
        Load and preprocess CSV data.

        Returns:
            X: Features dataframe
            y: Target series
            metadata: Dictionary with dataset information
        """
        if self.verbose:
            print("🚀 Universal CSV Loader")
            print("=" * 40)

        # Validate and load
        file_path = self._validate_file_path()
        df = self._load_raw_data(file_path)
        df = self._preprocess_data(df)

        # Separate features and target
        target_column = self._validate_target_column(df)
        y = df[target_column].copy()
        X = df.drop(columns=[target_column])

        # Process target
        y_processed, target_info = self._process_target(y)

        # Create metadata
        metadata = {
            'file_path': str(file_path),
            'shape': df.shape,
            'feature_columns': list(X.columns),
            'target_column': target_column,
            'target_info': target_info,
            'missing_values': X.isnull().sum().to_dict(),
            'data_types': X.dtypes.to_dict()
        }

        if self.verbose:
            print("✅ Loading complete:")
            print(f"   Features: {X.shape}")
            print(f"   Target: {y_processed.shape}")
            print(f"   Task: {target_info['task_type']}")

        return X, y_processed, metadata
