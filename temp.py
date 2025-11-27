#!/usr/bin/env python
"""
raw/ 아래의 바이너리 파일들을 읽어서
1MB, 2MB 청크 단위로 자르고,
각 청크에 대해 엔트로피 + 코덱별 압축률/시간을 계산해서

data/1MB.csv
data/2MB.csv

형태로 저장하는 임시 스크립트.

※ 의존 라이브러리
    pip install pandas zstandard lz4 python-snappy
"""

import math
import time
from pathlib import Path

import pandas as pd
import zstandard as zstd
import lz4.frame
import snappy


# =========================
# 경로 설정
# =========================

BASE_DIR = Path(__file__).resolve().parent
RAW_DIR = BASE_DIR / "raw"
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# 사용할 청크 사이즈 (바이트 단위) 및 파일 이름 레이블
CHUNK_SPECS = {
    "1MB": 1 * 1024 * 1024,
    "2MB": 2 * 1024 * 1024,
}


# =========================
# 유틸 함수
# =========================

def log(msg: str) -> None:
    print(f"[TEMP] {msg}")


def compute_entropy(chunk: bytes) -> float:
    """
    간단한 샤논 엔트로피 계산 (byte 분포 기준).
    """
    if not chunk:
        return 0.0

    # 0~255 빈도
    counts = [0] * 256
    for b in chunk:
        counts[b] += 1

    total = len(chunk)
    entropy = 0.0
    for c in counts:
        if c == 0:
            continue
        p = c / total
        entropy -= p * math.log2(p)
    return entropy


def compress_with_timing(codec: str, data: bytes) -> tuple[float, float]:
    """
    codec: "zstd" | "lz4" | "snappy"
    return: (ratio, time_ms)
        ratio = compressed_size / original_size
    """
    if not data:
        return 1.0, 0.0

    start = time.perf_counter()

    if codec == "zstd":
        cctx = zstd.ZstdCompressor()
        compressed = cctx.compress(data)
    elif codec == "lz4":
        compressed = lz4.frame.compress(data)
    elif codec == "snappy":
        compressed = snappy.compress(data)
    else:
        raise ValueError(f"Unknown codec: {codec}")

    elapsed_ms = (time.perf_counter() - start) * 1000.0

    ratio = len(compressed) / len(data)
    return float(ratio), float(elapsed_ms)


# =========================
# 메인 로직
# =========================

def process_chunk_size(label: str, chunk_bytes: int) -> None:
    """
    주어진 청크 크기(바이트)로 raw/ 아래 모든 파일을 자르고,
    각 청크에 대한 feature/압축 성능을 data/{label}.csv 로 저장.
    """
    log(f"{label} 처리 시작 (chunk={chunk_bytes} bytes)")

    rows = []

    # raw/ 아래의 모든 파일 순회
    raw_files = sorted(p for p in RAW_DIR.glob("**/*") if p.is_file())

    if not raw_files:
        log("raw/ 디렉토리에 파일이 없습니다. 스킵합니다.")
        return

    for path in raw_files:
        log(f"  파일 처리: {path.name}")
        with path.open("rb") as f:
            file_data = f.read()

        file_size = len(file_data)
        if file_size == 0:
            log(f"    빈 파일, 스킵: {path.name}")
            continue

        num_chunks = (file_size + chunk_bytes - 1) // chunk_bytes

        for idx in range(num_chunks):
            start = idx * chunk_bytes
            end = min(start + chunk_bytes, file_size)
            chunk = file_data[start:end]

            if not chunk:
                continue

            chunk_id = f"{path.name}_chunk{idx}"
            offset = start

            entropy = compute_entropy(chunk)

            # 각 코덱에 대해 압축률/시간 계산
            zstd_ratio, zstd_time = compress_with_timing("zstd", chunk)
            lz4_ratio, lz4_time = compress_with_timing("lz4", chunk)
            snappy_ratio, snappy_time = compress_with_timing("snappy", chunk)

            rows.append(
                {
                    "chunk_id": chunk_id,
                    "source_file": path.name,
                    "offset": offset,
                    "size": len(chunk),
                    "entropy": entropy,
                    "zstd_ratio": zstd_ratio,
                    "zstd_time": zstd_time,
                    "lz4_ratio": lz4_ratio,
                    "lz4_time": lz4_time,
                    "snappy_ratio": snappy_ratio,
                    "snappy_time": snappy_time,
                }
            )

    if not rows:
        log(f"{label}: 생성된 청크가 없습니다. CSV를 만들지 않습니다.")
        return

    df = pd.DataFrame(rows)
    out_path = DATA_DIR / f"{label}.csv"
    df.to_csv(out_path, index=False)
    log(f"{label} 완료 → {out_path} (rows={len(df)})")


def main():
    for label, size in CHUNK_SPECS.items():
        process_chunk_size(label, size)
    log("모든 청크 사이즈 처리 완료.")


if __name__ == "__main__":
    main()
