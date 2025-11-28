# main.py
from __future__ import annotations

import sys
from pathlib import Path
import utils.file_processor
from pipeline.train_xgb import train_and_evaluate_xgb_for_chunk


def run_pipeline(chunk_size_mb: int = 1) -> None:
    #extract features 실행부분

    #----------

    #모델 트레이닝
    train_and_evaluate_xgb_for_chunk(chunk_size_mb)
    #----------

    #벤치마크 실행부분

    #----------


if __name__ == "__main__":
    # 사용법:
    #   python main.py           -> chunk_size_mb = 1MB
    #   python main.py 2         -> chunk_size_mb = 2MB
    #   python main.py 4         -> chunk_size_mb = 4MB
    if len(sys.argv) >= 2:
        size_mb = int(sys.argv[1])
    else:
        size_mb = 1

    run_pipeline(chunk_size_mb=size_mb)
