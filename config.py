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

# Feature 추출용 청크 크기별 CSV 파일 매핑 (1/4/8/16 MB)
FEATURE_EXTRACT_CHUNK_SIZES = {
    1 * 1024 * 1024: DATA_DIR / "1MB.csv",
    4 * 1024 * 1024: DATA_DIR / "4MB.csv",
    8 * 1024 * 1024: DATA_DIR / "8MB.csv",
    16 * 1024 * 1024: DATA_DIR / "16MB.csv"
}