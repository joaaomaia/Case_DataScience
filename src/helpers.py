"""Helper utilities for AuditTrail."""

from typing import Dict, Tuple, List
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp


def calculate_ks(dist1: Dict, dist2: Dict) -> float:
    """Return the KS statistic for two distributions."""
    v1 = list(dist1.values())
    v2 = list(dist2.values())
    if len(v1) > 1 and len(v2) > 1:
        return ks_2samp(v1, v2).statistic
    return 0.0

def calculate_psi(expected: Dict, actual: Dict, epsilon: float = 1e-6) -> float:
    """Calculate Population Stability Index between two distributions."""
    keys = set(expected.keys()).union(actual.keys())
    total_expected = sum(expected.values()) + epsilon
    total_actual = sum(actual.values()) + epsilon
    psi = 0.0
    for k in keys:
        e = expected.get(k, 0) / total_expected
        a = actual.get(k, 0) / total_actual
        if e > 0:
            psi += (e - a) * np.log((e + epsilon) / (a + epsilon))
    return round(psi, 4)


def search_dtypes(df: pd.DataFrame, target_col: str, limite_categorico: int) -> Tuple[List[str], List[str]]:
    """Automatically identify numeric and categorical columns."""
    id_patterns = ['client_id', '_id', 'id_', 'codigo', 'key']
    num_cols, cat_cols = [], []
    for col in df.columns:
        if col == target_col:
            continue
        s = df[col]
        if s.isnull().mean() > 0.9:
            continue
        if pd.api.types.is_numeric_dtype(s):
            if any(p in col.lower() for p in id_patterns):
                continue
            num_cols.append(col)
        elif s.dtype == 'object' or pd.api.types.is_string_dtype(s):
            if any(p in col.lower() for p in id_patterns):
                continue
            if s.nunique(dropna=True) <= limite_categorico:
                cat_cols.append(col)
        elif pd.api.types.is_bool_dtype(s):
            cat_cols.append(col)
    return num_cols, cat_cols