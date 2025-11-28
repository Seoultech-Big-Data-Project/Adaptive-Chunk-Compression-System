"""
압축 성능 테스트 함수들
LZ4, Snappy, ZSTD 압축 알고리즘 테스트
"""

import time
import lz4.frame
import snappy
import zstandard as zstd

def compress(data: bytes, method: str):
    """
    지정된 압축 알고리즘을 사용하여 압축 수행
    """
    if method == "lz4":
        return lz4.frame.compress(data)
    elif method == "snappy":
        return snappy.compress(data)
    elif method == "zstd":
        return zstd.ZstdCompressor().compress(data)
    else:
        raise ValueError(f"Unsupported compression method: {method}")
    
def compress_and_measure(data: bytes, method: str):
    """
    지정된 압축 알고리즘으로 압축을 수행 후,
    압축된 크기, 소요 시간(밀리초), 압축률을 반환
    """
    t0 = time.perf_counter_ns()
    compressed = compress(data, method)
    t1 = time.perf_counter_ns()
    
    size = len(compressed)
    time_ms = (t1 - t0) / 1e6
    ratio = size / len(data) if len(data) > 0 else 1.0
    
    return size, time_ms, ratio


def test_all_compressions(data: bytes):
    """
    모든 압축 알고리즘 테스트 및 결과 반환
    """
    lz4_size, lz4_time, lz4_ratio = compress_and_measure(data, "lz4")
    snappy_size, snappy_time, snappy_ratio = compress_and_measure(data, "snappy")
    zstd_size, zstd_time, zstd_ratio = compress_and_measure(data, "zstd")
    
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
