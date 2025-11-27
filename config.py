# config.py

from pathlib import Path

# ----------------------
# 멀티프로세싱 설정
# ----------------------
MAX_WORKERS = 32

# ----------------------
# 경로 설정
# ----------------------
PROJECT_ROOT = Path(__file__).resolve().parent

RAW_DIR = PROJECT_ROOT / "raw"
DATA_DIR = PROJECT_ROOT / "data"
PREPROCESSED_DIR = PROJECT_ROOT / "preprocessed"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

# 디렉토리가 없으면 자동으로 생성하도록 해도 됨 (선택)
for _d in [PREPROCESSED_DIR, MODELS_DIR, RESULTS_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

# ----------------------
# 청크 사이즈 / 코덱 설정
# ----------------------
# 실제 사용 중인 청크 사이즈만 넣어두면 됨
CHUNK_SIZES = ["1MB", "2MB"]  # 필요시 추가

# Feature 추출용 청크 크기별 CSV 파일 매핑 (1/4/8/16 MB)
FEATURE_EXTRACT_CHUNK_SIZES = {
    1 * 1024 * 1024: DATA_DIR / "1MB.csv",
    4 * 1024 * 1024: DATA_DIR / "4MB.csv",
    8 * 1024 * 1024: DATA_DIR / "8MB.csv",
    16 * 1024 * 1024: DATA_DIR / "16MB.csv"
}

# 사용 코덱 리스트 (compression_* CSV, preprocess에서 모두 이 이름 기준)
CODECS = ["zstd", "lz4", "snappy"]

# 단일코덱 baseline으로 쓸 코덱 이름
BASELINE_CODEC = "zstd"

# 코덱 ↔ ID 매핑 (멀티클래스 레이블용)
CODEC_TO_ID = {c: i for i, c in enumerate(CODECS)}
ID_TO_CODEC = {i: c for c, i in CODEC_TO_ID.items()}

# ----------------------
# 레이블 설정
# ----------------------
LABEL_CONFIG = {
    "task_type": "multiclass",   # 향후 multilabel, regression 등 확장 가능
    "column": "label_codec"      # preprocess가 이 이름으로 라벨 컬럼 생성
}

# ----------------------
# 스코어 계산 설정 (압축률 vs 속도 가중치)
# ----------------------
SCORE_ALPHA = 0.5  # 1.0이면 압축률만, 0.0이면 속도만

# ----------------------
# 데이터 분할 비율
# ----------------------
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# ----------------------
# 랜덤 시드
# ----------------------
RANDOM_SEED = 42

# ----------------------
# XGBoost 설정 (GPU 사용)
# ----------------------
USE_XGBOOST_GPU = True

XGBOOST_PARAMS = {
    "n_estimators": 300,
    "max_depth": 8,
    "learning_rate": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "objective": "multi:softprob",  # 멀티클래스 확률 출력
    "eval_metric": "mlogloss",
}
