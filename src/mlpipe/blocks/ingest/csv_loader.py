from __future__ import annotations
from pathlib import Path
from typing import Tuple, List, Optional
import pandas as pd
from mlpipe.core.interfaces import DataIngestor
from mlpipe.core.registry import register

_HIGGS_COLS = [
    'label','lepton_pT','lepton_eta','lepton_phi','missing_energy_magnitude','missing_energy_phi',
    'jet_1_pt','jet_1_eta','jet_1_phi','jet_1_b_tag','jet_2_pt','jet_2_eta','jet_2_phi','jet_2_b_tag',
    'jet_3_pt','jet_3_eta','jet_3_phi','jet_3_b_tag','jet_4_pt','jet_4_eta','jet_4_phi','jet_4_b_tag',
    'm_jj','m_jjj','m_lv','m_jlv','m_bb','m_wbb','m_wwbb'
]

def _infer_names(n_cols: int, label_first: bool, label_name: str) -> List[str]:
    cols = [f"f{i}" for i in range(n_cols)]
    if label_first and n_cols >= 1:
        cols[0] = label_name
    return cols

@register("ingest.csv")
class CSVLoader(DataIngestor):
    """
    Config keys (all optional except `path` and `label`):
      path: str
      label: str                    # name to use for label column
      has_header: bool = False      # file already has a header row
      names: null | "higgs" | [...] # use preset or explicit names
      infer_names: bool = True      # if no header/names, auto-create f0..fN-1
      label_is_first_column: bool = True
      usecols: null | list[str]     # optional column subset
      nrows: null | int             # read a subset for quick tests
      compression: null | "gzip"    # pandas will auto-detect .gz; override if needed
    """
    def __init__(
        self,
        path: str,
        label: str,
        has_header: bool = False,
        names: Optional[object] = None,
        infer_names: bool = True,
        label_is_first_column: bool = True,
        usecols: Optional[list] = None,
        nrows: Optional[int] = None,
        compression: Optional[str] = None,
    ):
        self.path = Path(path)
        self.label = label
        self.has_header = has_header
        self.names = names
        self.infer_names = infer_names
        self.label_is_first_column = label_is_first_column
        self.usecols = usecols
        self.nrows = nrows
        self.compression = compression

    def load(self) -> Tuple[pd.DataFrame, pd.Series]:
        if self.has_header:
            df = pd.read_csv(self.path, usecols=self.usecols, nrows=self.nrows, compression=self.compression)
            
            # If names is specified (like "higgs" preset), override the column names
            if self.names == "higgs":
                # Use HIGGS preset column names
                expected_cols = len(_HIGGS_COLS)
                if len(df.columns) == expected_cols:
                    df.columns = _HIGGS_COLS
            elif isinstance(self.names, list):
                if len(df.columns) == len(self.names):
                    df.columns = self.names
        else:
            if self.names == "higgs":
                names = _HIGGS_COLS
            elif isinstance(self.names, list):
                names = self.names
            elif self.infer_names:
                # Peek first row to count columns
                peek = pd.read_csv(self.path, header=None, nrows=1, compression=self.compression)
                names = _infer_names(peek.shape[1], self.label_is_first_column, self.label)
            else:
                names = None  # let pandas assign numeric columns

            df = pd.read_csv(
                self.path, header=None if names else "infer", names=names,
                usecols=self.usecols, nrows=self.nrows, compression=self.compression
            )

        # ensure label column exists; if not but label_is_first_column, rename first col
        if self.label not in df.columns and self.label_is_first_column:
            first = df.columns[0]
            df = df.rename(columns={first: self.label})

        y = df[self.label]
        X = df.drop(columns=[self.label])
        return X, y
