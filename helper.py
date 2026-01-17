# preprocess_helper.py
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Tuple
import inspect

FLOAT32_MAX = np.finfo(np.float32).max
FLOAT32_MIN = -FLOAT32_MAX

def _coerce_features_to_numeric(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """Converte le colonne feature a numeriche (coerce → NaN se non convertibile)."""
    df_num = df.copy()
    for c in feature_cols:
        df_num[c] = pd.to_numeric(df_num[c], errors="coerce")
    return df_num

def _drop_nonfinite_rows(df: pd.DataFrame, feature_cols: list[str], label_col: str) -> pd.DataFrame:
    """Sostituisce inf/-inf con NaN e rimuove righe con NaN nelle feature o nel label."""
    df2 = df.replace([np.inf, -np.inf], np.nan)  # inf/-inf -> NaN
    before = len(df2)
    df2 = df2.dropna(subset=feature_cols + [label_col])
    dropped = before - len(df2)
    if dropped > 0:
        print(f"[preprocess] Righe eliminate per non-finiti/NaN: {dropped} (rimangono {len(df2)})")
    return df2

def _cast_and_clip_float32(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """Clippa ai limiti float32 e casta a float32 (evita 'value too large for float32')."""
    df2 = df.copy()
    for c in feature_cols:
        s = df2[c].to_numpy(copy=False)
        mask = np.isfinite(s)
        s_clipped = np.clip(s[mask], FLOAT32_MIN, FLOAT32_MAX).astype(np.float32, copy=False)
        s_out = s.astype(np.float32, copy=True)
        s_out[mask] = s_clipped
        df2[c] = s_out
    return df2

def _apply_smote(
    X: np.ndarray,
    y: np.ndarray,
    *,
    sampling_strategy: str | float | dict = "auto",
    k_neighbors: int = 5,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    try:
        from imblearn.over_sampling import SMOTE
    except Exception as e:
        raise ImportError(
        ) from e

    # aggiunge n_jobs solo se supportato
    kwargs = dict(
        sampling_strategy=sampling_strategy,
        k_neighbors=k_neighbors,
        random_state=random_state,
    )
    if "n_jobs" in inspect.signature(SMOTE).parameters:
        kwargs["n_jobs"] = -1

    sm = SMOTE(**kwargs)
    X_res, y_res = sm.fit_resample(X, y)
    return X_res, y_res

def preprocess_dataframe(
    df: pd.DataFrame,
    label_col: str = "marker",
    *,
    apply_smote: bool = False,
    smote_sampling_strategy: str | float | dict = "auto",
    smote_k_neighbors: int = 5,
    random_state: int = 42,
    coerce_numeric: bool = True,
    cast_float32: bool = True,
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    if label_col not in df.columns:
        raise ValueError(f"Colonna label '{label_col}' non trovata nel DataFrame.")

    feature_cols = [c for c in df.columns if c != label_col]
    df_work = df.copy()

    if coerce_numeric:
        df_work = _coerce_features_to_numeric(df_work, feature_cols)

    df_work = _drop_nonfinite_rows(df_work, feature_cols, label_col)

    y = df_work[label_col].astype(int).to_numpy()
    X = df_work[feature_cols].to_numpy()

    if apply_smote:
        X, y = _apply_smote(
            X, y,
            sampling_strategy=smote_sampling_strategy,
            k_neighbors=smote_k_neighbors,
            random_state=random_state,
        )
        df_work = pd.DataFrame(X, columns=feature_cols)
        df_work[label_col] = y

    if cast_float32:
        X = np.clip(X, FLOAT32_MIN, FLOAT32_MAX).astype(np.float32, copy=False)

    return X, y, df_work

def load_and_preprocess_csv(
    path: str,
    label_col: str = "marker",
    **kwargs,
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """legge il CSV e applica preprocess_dataframe(**kwargs)."""
    df = pd.read_csv(path)
    return preprocess_dataframe(df, label_col=label_col, **kwargs)
