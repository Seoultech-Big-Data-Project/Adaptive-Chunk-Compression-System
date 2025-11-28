"""
압축 성능 테스트 함수들
LZ4, Snappy, ZSTD 압축 알고리즘 테스트
"""

import time
import lz4.frame
import snappy
import zstandard as zstd


def compress_with_lz4(data: bytes):
    """
    LZ4 압축 수행 및 성능 측정
    """
    t0 = time.perf_counter_ns()
    compressed = lz4.frame.compress(data)
    t1 = time.perf_counter_ns()
    
    size = len(compressed)
    time_ms = (t1 - t0) / 1e6
    ratio = size / len(data) if len(data) > 0 else 1.0
    
    return size, time_ms, ratio


def compress_with_snappy(data: bytes):
    """
    Snappy 압축 수행 및 성능 측정
    """
    t0 = time.perf_counter_ns()
    compressed = snappy.compress(data)
    t1 = time.perf_counter_ns()
    
    size = len(compressed)
    time_ms = (t1 - t0) / 1e6
    ratio = size / len(data) if len(data) > 0 else 1.0
    
    return size, time_ms, ratio


def compress_with_zstd(data: bytes):
    """
    ZSTD 압축 수행 및 성능 측정
    """
    compressor = zstd.ZstdCompressor()
    t0 = time.perf_counter_ns()
    compressed = compressor.compress(data)
    t1 = time.perf_counter_ns()
    
    size = len(compressed)
    time_ms = (t1 - t0) / 1e6
    ratio = size / len(data) if len(data) > 0 else 1.0
    
    return size, time_ms, ratio


def test_all_compressions(data: bytes):
    """
    모든 압축 알고리즘 테스트 후,
    cost = ratio * time_ms 계산하여
    가장 cost가 작은 코덱 이름만 반환
    """
    # 각각 크기, 시간, ratio 계산
    lz4_size, lz4_time, lz4_ratio = compress_with_lz4(data)
    snappy_size, snappy_time, snappy_ratio = compress_with_snappy(data)
    zstd_size, zstd_time, zstd_ratio = compress_with_zstd(data)

    # cost 계산: ratio * time_ms
    lz4_cost = lz4_ratio * lz4_time
    snappy_cost = snappy_ratio * snappy_time
    zstd_cost = zstd_ratio * zstd_time

    # 코덱별 cost dict
    cost_dict = {
        "lz4": lz4_cost,
        "snappy": snappy_cost,
        "zstd": zstd_cost
    }

    # 가장 cost 작은 코덱 선택
    best_codec = min(cost_dict, key=cost_dict.get)

    return {
        "best_cost": best_codec
    }

