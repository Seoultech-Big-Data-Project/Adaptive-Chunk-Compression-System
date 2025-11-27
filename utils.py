# utils.py

import json
from pathlib import Path
from typing import List, Dict

import numpy as np
import pandas as pd

from config import (
    CODECS,
    CODEC_TO_ID,
    ID_TO_CODEC,
    SCORE_ALPHA,
    DATA_DIR,
    PREPROCESSED_DIR,
    RESULTS_DIR,
)


# ----------------------
# 파일 경로 유틸
# ----------------------
def data_csv_path(chunk_size: str) -> Path:
    """팀원이 만들어둔 피처 CSV 경로 (예: data/1MB.csv)."""
    return DATA_DIR / f"{chunk_size}.csv"


def compression_csv_path(chunk_size: str) -> Path:
    """코덱별 압축 결과 CSV 경로 (예: results/compression_1MB.csv)."""
    return RESULTS_DIR / f"compression_{chunk_size}.csv"


def preprocessed_csv_path(chunk_size: str) -> Path:
    """전처리 완료된 CSV 경로 (예: preprocessed/1MB.csv)."""
    return PREPROCESSED_DIR / f"{chunk_size}.csv"


def benchmark_csv_path(chunk_size: str) -> Path:
    """전략 비교 결과 저장 경로 (예: results/benchmark_1MB.csv)."""
    return RESULTS_DIR / f"benchmark_{chunk_size}.csv"


def benchmark_summary_path() -> Path:
    """전체 요약 결과 CSV 경로."""
    return RESULTS_DIR / "benchmark_summary.csv"


# ----------------------
# 스코어 계산 유틸
# ----------------------
def compute_ratio(original_bytes: np.ndarray, compressed_bytes: np.ndarray) -> np.ndarray:
    """
    압축률 계산.
    예시: ratio = original / compressed  (값이 클수록 더 잘 압축한 것)
    """
    return original_bytes / (compressed_bytes + 1e-9)


def compute_speed_mb_s(original_bytes: np.ndarray, encode_time_ms: np.ndarray) -> np.ndarray:
    """
    인코딩 스루풋 (MB/s).
    original_bytes를 MB 단위로 나눠서, encode_time_ms를 초로 나눔.
    """
    mb = original_bytes / (1024 * 1024)
    seconds = encode_time_ms / 1000.0
    return mb / (seconds + 1e-9)


def normalize_min_max(values: np.ndarray) -> np.ndarray:
    vmin = np.nanmin(values)
    vmax = np.nanmax(values)
    if vmax - vmin < 1e-12:
        return np.zeros_like(values)
    return (values - vmin) / (vmax - vmin)


def compute_combined_score(ratio: np.ndarray, speed: np.ndarray, alpha: float = SCORE_ALPHA) -> np.ndarray:
    """
    ratio와 speed를 0~1로 정규화한 뒤, 가중합으로 score 계산.
    score = alpha * ratio_norm + (1 - alpha) * speed_norm
    """
    ratio_norm = normalize_min_max(ratio)
    speed_norm = normalize_min_max(speed)
    return alpha * ratio_norm + (1.0 - alpha) * speed_norm


# ----------------------
# 피처 선택
# ----------------------
def get_feature_columns(df: pd.DataFrame) -> List[str]:
    """
    학습에 사용할 피처 컬럼 이름 리스트를 반환.
    - 원본 피처들만 사용하고, 코덱 성능에서 유도된 컬럼들은 제외.
    """
    exclude_prefixes = (
        "compressed_bytes_",
        "encode_time_ms_",
        "ratio_",
        "speed_",
        "ratio_norm_",
        "speed_norm_",
        "score_",
    )
    exclude_exact = {"chunk_id", "file_name", "chunk_size_bytes", "label_codec", "label_codec_id", "split"}

    feature_cols = []
    for col in df.columns:
        if col in exclude_exact:
            continue
        if any(col.startswith(p) for p in exclude_prefixes):
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            feature_cols.append(col)
    return feature_cols


# ----------------------
# JSON 저장 유틸
# ----------------------
def save_json(obj: Dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
