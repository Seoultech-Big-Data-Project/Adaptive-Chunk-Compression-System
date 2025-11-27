# pipeline/common.py
from pathlib import Path
import os
import random

import numpy as np

# =========================
# 경로 설정
# =========================

BASE_DIR = Path(__file__).resolve().parents[1]

RAW_DIR = BASE_DIR / "raw"
DATA_DIR = BASE_DIR / "data"
PREPROCESSED_DIR = BASE_DIR / "preprocessed"
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"

for d in [RAW_DIR, DATA_DIR, PREPROCESSED_DIR, MODELS_DIR, RESULTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# =========================
# 파이프라인 공통 설정
# =========================

# 사용하려는 청크 사이즈 파일 이름(확장자 제외)
# 필요시 여기만 수정하거나, os.listdir 기반으로 자동화해도 됨
CHUNK_SIZES = ["1MB", "2MB"]  # 예시

# 사용 코덱 및 각 코덱에 대응하는 컬럼명
CODECS = {
    "zstd": {"ratio_col": "zstd_ratio", "time_col": "zstd_time"},
    "lz4": {"ratio_col": "lz4_ratio", "time_col": "lz4_time"},
    "snappy": {"ratio_col": "snappy_ratio", "time_col": "snappy_time"},
}

TRAIN_RATIO = 0.8
N_SPLITS = 5
RANDOM_SEED = 42

# # XGBoost 기본 파라미터 (필요에 따라 수정)
# XGB_DEFAULT_PARAMS = {
#     "objective": "multi:softmax",
#     "tree_method": "gpu_hist",      # GPU
#     "predictor": "gpu_predictor",   # GPU
#     "eval_metric": "mlogloss",
#     "learning_rate": 0.1,
#     "max_depth": 6,
#     "n_estimators": 300,
# }

# pipeline/common.py

XGB_DEFAULT_PARAMS = {
    "objective": "multi:softmax",
    "tree_method": "hist",   # ✅ CPU용 (gpu_hist 대신 hist)
    "eval_metric": "mlogloss",
    "learning_rate": 0.1,
    "max_depth": 6,
    "n_estimators": 300,
}



def set_global_seed(seed: int = RANDOM_SEED) -> None:
    """numpy / random 시드 고정."""
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def log(msg: str) -> None:
    """간단한 로깅 함수."""
    print(f"[LOG] {msg}")
