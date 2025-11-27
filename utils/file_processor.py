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
    test_all_compressions
)


def compute_chunk_features(data: bytes, file_path: Path, chunk_idx: int, chunk_size_target: int):
    """
    단일 청크에 대한 모든 특성(feature) 추출 및 압축 테스트
    """
    size = len(data)

    # === 1. 통계적 특성 추출 ===
    if size == 0:
        mean, std, bmin, bmax = 0.0, 0.0, 0, 0
        entropy = 0.0
        num_runs, run_mean, run_std, run_max = 0, 0.0, 0.0, 0
        props = proportion_features(data)
        compression_results = {
            "lz4_size": 0, "lz4_time_ms": 0.0, "lz4_ratio": 1.0,
            "snappy_size": 0, "snappy_time_ms": 0.0, "snappy_ratio": 1.0,
            "zstd_size": 0, "zstd_time_ms": 0.0, "zstd_ratio": 1.0,
        }
    else:
        mean, std, bmin, bmax = byte_stats(data)              # 바이트 기본 통계
        entropy = shannon_entropy(data)                       # 엔트로피 (복잡도)
        num_runs, run_mean, run_std, run_max = run_length_stats(data)  # Run length 통계
        props = proportion_features(data)                     # 바이트 타입별 비율
        compression_results = test_all_compressions(data)     # 압축 성능 테스트

    # === 2. 모든 특성을 하나의 딕셔너리로 결합 ===
    row = {
        # 메타 정보
        # "file_path": str(file_path),              # 원본 파일 경로
        "chunk_index": chunk_idx,                 # 청크 순번
        "target_chunk_size": chunk_size_target,   # 목표 청크 크기
        "chunk_size": size,                       # 실제 청크 크기
        
        # 기본 통계
        "entropy": entropy,                       # 샤논 엔트로피
        "byte_mean": mean,                        # 바이트 평균
        "byte_std": std,                          # 바이트 표준편차
        "byte_min": bmin,                         # 바이트 최소값
        "byte_max": bmax,                         # 바이트 최대값
        
        # Run length 통계
        "num_runs": num_runs,                     # 총 run 개수
        "run_mean": run_mean,                     # 평균 run 길이
        "run_std": run_std,                       # run 길이 표준편차
        "run_max": run_max,                       # 최대 run 길이
        
        # 바이트 타입별 비율
        **props,
        
        # 압축 성능 결과
        **compression_results,
    }

    return row


def _process_chunk_wrapper(args):
    chunk, file_path, idx, target_size = args
    return compute_chunk_features(chunk, file_path, idx, target_size)


def process_file(file_path: Path, target_chunk_size: int, use_multiprocessing: bool = True):
    """
    특정 청크 크기로 파일을 분할하여 각 청크의 특성 추출
    """
    file_size = file_path.stat().st_size 

    print(f"      Reading file ({file_size / (1024*1024):.1f} MB)...")
    with file_path.open("rb") as f:
        data = f.read()

    num_chunks = math.ceil(file_size / target_chunk_size) if file_size > 0 else 1
    
    chunk_args = []
    for i in range(num_chunks):
        start = i * target_chunk_size          
        end = min(start + target_chunk_size, file_size)
        chunk = data[start:end]
        chunk_args.append((chunk, file_path, i, target_chunk_size))
    
    # 멀티 프로세싱 사용 여부에 따라 처리
    if use_multiprocessing and num_chunks > 1:
        num_workers = min(cpu_count(), MAX_WORKERS, num_chunks)
        print(f"      Processing {num_chunks} chunk(s) with {num_workers} workers...")
        
        with Pool(processes=num_workers) as pool:
            rows = list(tqdm(
                pool.imap(_process_chunk_wrapper, chunk_args),
                total=num_chunks,
                desc="        Chunks",
                leave=False,
                unit="chunk"
            ))
    else:
        # 단일 프로세스로 순차 처리
        print(f"      Processing {num_chunks} chunk(s) (single process)...")
        rows = []
        for args in tqdm(chunk_args, desc="        Chunks", leave=False, unit="chunk"):
            row = _process_chunk_wrapper(args)
            rows.append(row)

    return rows
