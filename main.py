# main.py
from __future__ import annotations

import sys
from pathlib import Path

from pipeline.common import get_chunk_dir
from pipeline.chunking import split_all_raw_to_chunks
from pipeline.features import build_and_save_datasets
from pipeline.train_xgb import train_and_evaluate_xgb_for_chunk
from pipeline.benchmark import run_benchmark


def run_pipeline(chunk_size_mb: int = 1) -> None:
    """
    전체 파이프라인:
      1) raw 데이터를 chunk/{chunk_size}MB/*.bin 으로 분할 (이미 있으면 스킵)
      2) 청크에서 통계 피쳐 + 코덱 성능 측정 -> full/train/val/test 생성
      3) XGBoost로 best_codec 예측 모델 학습 + 평가
      4) test set 기준으로
         - always_zstd / always_lz4 / always_snappy
         - oracle(best_codec)
         - xgb_pred_ideal / xgb_pred_real
         전략 벤치마크
    """
    print("=" * 60)
    print(f"[main] PIPELINE START (chunk_size = {chunk_size_mb}MB)")
    print("=" * 60)

    # --------------------------------------------------------
    # 2) 피쳐 + 라벨 + train/val/test 생성
    # --------------------------------------------------------
    print(f"[main] 2/4 features: 통계 피쳐 + cost 기반 best_codec 라벨 생성")
    build_and_save_datasets(chunk_size_mb)
    print(f"[main]    features 완료 (full/train/val/test 저장)")

    # --------------------------------------------------------
    # 3) XGBoost 학습 + 평가
    # --------------------------------------------------------
    print(f"[main] 3/4 train_xgb: XGBoost 모델 학습 및 평가")
    train_and_evaluate_xgb_for_chunk(chunk_size_mb)
    print(f"[main]    train_xgb 완료 (모델/메트릭 저장)")


if __name__ == "__main__":
    # 사용법:
    #   python main.py           -> chunk_size_mb = 1MB
    #   python main.py 2         -> chunk_size_mb = 2MB
    #   python main.py 4         -> chunk_size_mb = 4MB
    if len(sys.argv) >= 2:
        size_mb = int(sys.argv[1])
    else:
        size_mb = 4

    run_pipeline(chunk_size_mb=size_mb)
