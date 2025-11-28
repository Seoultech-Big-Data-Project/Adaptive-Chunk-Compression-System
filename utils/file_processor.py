"""
파일 처리 유틸리티
청크 분할 및 특성 추출
"""

import math
from pathlib import Path
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from config import MAX_WORKERS

from features import (
    shannon_entropy,
    byte_stats,
    run_length_stats,
    proportion_features,
    test_all_compressions,
)


# ===============================
# 단일 청크 feature 추출
# ===============================
def compute_chunk_features(
    data: bytes,
    chunk_idx: int,
    include_compression: bool = True,
):
    """
    단일 청크에 대한 feature 추출 (+ 선택적 압축 테스트)
    """
    size = len(data)

    # 기본값 초기화
    entropy = 0.0
    byte_std = 0.0
    byte_max = 0
    num_runs = 0
    run_mean = 0.0
    run_std = 0.0

    props = proportion_features(data)
    compression_results = {}

    if size > 0:
        byte_std, byte_max = byte_stats(data)
        entropy = shannon_entropy(data)
        num_runs, run_mean, run_std = run_length_stats(data)

        if include_compression:
            compression_results = test_all_compressions(data)

    row = {
        "chunk_index": chunk_idx,

        # 통계적 특성
        "entropy": entropy,
        "byte_std": byte_std,
        "byte_max": byte_max,

        # Run-length 통계
        "num_runs": num_runs,
        "run_mean": run_mean,
        "run_std": run_std,

        # 바이트 분포 비율
        **props,
    }

    if include_compression:
        row.update(compression_results)

    return row


# ===============================
# 멀티프로세싱용 래퍼
# ===============================
def _process_chunk_wrapper(args):
    """
    multiprocessing.Pool에서 사용되는 래퍼 함수
    """
    chunk, chunk_idx, include_compression = args
    return compute_chunk_features(
        data=chunk,
        chunk_idx=chunk_idx,
        include_compression=include_compression,
    )


# ===============================
# 파일 단위 처리
# ===============================
def process_file(
    file_path: Path,
    target_chunk_size: int,
    use_multiprocessing: bool = True,
    include_compression: bool = True,
):
    """
    파일을 청크 단위로 분할 후 feature 추출
    """
    file_size = file_path.stat().st_size

    print(f"      Reading file ({file_size / (1024 * 1024):.1f} MB)...")
    with file_path.open("rb") as f:
        data = f.read()

    num_chunks = (
        math.ceil(file_size / target_chunk_size) if file_size > 0 else 1
    )

    # 청크 인자 구성
    chunk_args = []
    for i in range(num_chunks):
        start = i * target_chunk_size
        end = min(start + target_chunk_size, file_size)
        chunk = data[start:end]
        chunk_args.append((chunk, i, include_compression))

    if use_multiprocessing and num_chunks > 1:
        num_workers = min(cpu_count(), MAX_WORKERS, num_chunks)
        print(f"      Processing {num_chunks} chunk(s) with {num_workers} workers...")

        with Pool(processes=num_workers) as pool:
            rows = list(
                tqdm(
                    pool.imap(_process_chunk_wrapper, chunk_args),
                    total=num_chunks,
                    desc="        Chunks",
                    leave=False,
                    unit="chunk",
                )
            )

    else:
        print(f"      Processing {num_chunks} chunk(s) (single process)...")
        rows = []
        for args in tqdm(
            chunk_args,
            desc="        Chunks",
            leave=False,
            unit="chunk",
        ):
            rows.append(_process_chunk_wrapper(args))

    return rows