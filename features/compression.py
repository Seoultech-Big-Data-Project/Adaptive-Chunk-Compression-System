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
    모든 압축 알고리즘 테스트 및 결과 반환
    """
    lz4_size, lz4_time, lz4_ratio = compress_with_lz4(data)
    snappy_size, snappy_time, snappy_ratio = compress_with_snappy(data)
    zstd_size, zstd_time, zstd_ratio = compress_with_zstd(data)
    
    return {
        "lz4_size": lz4_size,
        "lz4_time_ms": lz4_time,
        "lz4_ratio": lz4_ratio,
        "snappy_size": snappy_size,
        "snappy_time_ms": snappy_time,
        "snappy_ratio": snappy_ratio,
        "zstd_size": zstd_size,
        "zstd_time_ms": zstd_time,
        "zstd_ratio": zstd_ratio,
    }
