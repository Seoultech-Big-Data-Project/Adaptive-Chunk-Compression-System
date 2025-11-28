# pipeline/features.py
from pathlib import Path
import time

import numpy as np
import pandas as pd
import zstandard as zstd
import lz4.frame
import snappy

from .common import get_chunk_dir, DATA_DIR

CODECS = ["zstd", "lz4", "snappy"]


# =========================
# 통계 피쳐 계산
# =========================

def compute_entropy_bits_per_byte(data: bytes) -> float:
    """
    Shannon entropy (bits per byte)
    """
    if not data:
        return 0.0

    arr = np.frombuffer(data, dtype=np.uint8)
    counts = np.bincount(arr, minlength=256)
    total = counts.sum()
    if total == 0:
        return 0.0

    probs = counts / total
    probs = probs[probs > 0]
    entropy = -(probs * np.log2(probs)).sum()
    return float(entropy)


def compute_basic_stats(data: bytes) -> dict:
    """
    청크 하나에 대해 기본 통계 피쳐 계산
    """
    if not data:
        return {
            "orig_size": 0,
            "entropy_bits_per_byte": 0.0,
            "mean_byte": 0.0,
            "std_byte": 0.0,
            "zero_ratio": 0.0,
            "ascii_ratio": 0.0,
        }

    arr = np.frombuffer(data, dtype=np.uint8)
    orig_size = int(arr.size)
    entropy = compute_entropy_bits_per_byte(data)
    mean = float(arr.mean())
    std = float(arr.std())
    zero_ratio = float((arr == 0).mean())
    ascii_ratio = float(((arr >= 32) & (arr <= 126)).mean())

    return {
        "orig_size": orig_size,
        "entropy_bits_per_byte": entropy,
        "mean_byte": mean,
        "std_byte": std,
        "zero_ratio": zero_ratio,
        "ascii_ratio": ascii_ratio,
    }


# =========================
# 코덱 별 압축 피쳐 (학습 라벨 계산용)
# =========================

_zstd_cctx = zstd.ZstdCompressor(level=3)


def measure_zstd(data: bytes) -> dict:
    if not data:
        return {"zstd_ratio": 1.0, "zstd_time_ms": 0.0}

    start = time.perf_counter()
    compressed = _zstd_cctx.compress(data)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    ratio = len(compressed) / len(data)
    return {
        "zstd_ratio": float(ratio),
        "zstd_time_ms": float(elapsed_ms),
    }


def measure_lz4(data: bytes) -> dict:
    if not data:
        return {"lz4_ratio": 1.0, "lz4_time_ms": 0.0}

    start = time.perf_counter()
    compressed = lz4.frame.compress(data)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    ratio = len(compressed) / len(data)
    return {
        "lz4_ratio": float(ratio),
        "lz4_time_ms": float(elapsed_ms),
    }


def measure_snappy(data: bytes) -> dict:
    if not data:
        return {"snappy_ratio": 1.0, "snappy_time_ms": 0.0}

    start = time.perf_counter()
    compressed = snappy.compress(data)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    ratio = len(compressed) / len(data)
    return {
        "snappy_ratio": float(ratio),
        "snappy_time_ms": float(elapsed_ms),
    }


# =========================
# 청크 전체에 대해 피쳐 + 코덱정보 생성
# =========================

def build_raw_features_for_chunks(chunk_size_mb: int) -> pd.DataFrame:
    """
    chunk/{chunk_size_mb}MB 안의 모든 청크를 읽어서
    - 통계 피쳐 (entropy, mean, std, zero_ratio, ascii_ratio 등)
    - 코덱별 압축률 / 시간 (zstd_ratio, lz4_time_ms, ...)
    까지 포함한 DataFrame 생성

    이 DataFrame은 학습용 X에는 그대로 쓰지 않고,
    cost 기반 best 코덱 라벨을 만들기 위한 중간 결과용이다.
    """
    chunk_dir = get_chunk_dir(chunk_size_mb)
    files = sorted(
        chunk_dir.glob("*.bin"),
        key=lambda p: int(p.stem)  # 파일명이 "0.bin", "1.bin" 가정
    )

    if not files:
        raise FileNotFoundError(f"[features] 청크 파일이 없습니다: {chunk_dir}")

    print(f"[features] chunk_dir = {chunk_dir}")
    print(f"[features] num_chunks = {len(files)}")

    rows = []

    for i, path in enumerate(files):
        chunk_idx = int(path.stem)
        with path.open("rb") as f:
            data = f.read()

        # 통계 피쳐
        stats = compute_basic_stats(data)
        row = {
            "chunk_idx": chunk_idx,
            **stats,
        }

        # 코덱별 압축 피쳐 (라벨 계산용)
        row.update(measure_zstd(data))
        row.update(measure_lz4(data))
        row.update(measure_snappy(data))

        rows.append(row)

        if (i + 1) % 100 == 0:
            print(f"  - processed {i + 1}/{len(files)} chunks")

    df = pd.DataFrame(rows).sort_values("chunk_idx").reset_index(drop=True)
    return df


# =========================
# cost 기반 best 코덱 라벨 추가
# =========================

def add_best_codec_label_by_cost(df: pd.DataFrame) -> pd.DataFrame:
    """
    각 코덱에 대해 cost = ratio * time_ms 를 계산하고,
    cost가 가장 작은 코덱을 best_codec 라벨로 추가한다.
    또한 코덱별 비율을 콘솔에 출력한다.
    """
    for codec in CODECS:
        r_col = f"{codec}_ratio"
        t_col = f"{codec}_time_ms"
        if r_col not in df.columns or t_col not in df.columns:
            raise ValueError(f"필수 컬럼 없음: {r_col}, {t_col}")
        df[f"{codec}_cost"] = df[r_col] * df[t_col]

    cost_cols = [f"{c}_cost" for c in CODECS]
    df["best_codec"] = df[cost_cols].idxmin(axis=1).str.split("_").str[0]

    # 비율 출력
    total = len(df)
    print(f"\n[features] cost 기준 best codec 분포 (총 청크 수 = {total})")
    counts = df["best_codec"].value_counts()
    for codec in CODECS:
        cnt = int(counts.get(codec, 0))
        pct = (cnt / total * 100.0) if total > 0 else 0.0
        print(f"  - {codec:7s}: {cnt:6d} 개 ({pct:5.1f} %)")

    return df


# =========================
# train / val / test 분할
# =========================

def split_train_val_test(
    df: pd.DataFrame,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
):
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6

    n = len(df)
    rng = np.random.default_rng(seed)
    indices = rng.permutation(n)

    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    n_test = n - n_train - n_val

    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]

    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df = df.iloc[val_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)

    return train_df, val_df, test_df


# =========================
# 메인: full + train/val/test 생성
# =========================

def build_and_save_datasets(chunk_size_mb: int) -> None:
    """
    1) 청크별 통계 피쳐 + 코덱별 압축 정보 계산
    2) cost 기준 best_codec 라벨 추가
    3) data/{chunk_size}MB/full.csv 로 전체 저장 (디버깅/분석용)
       - 통계 피쳐 + 코덱별 ratio/time + cost + best_codec 모두 포함
    4) 학습용 DataFrame은
       - 통계 피쳐 + chunk_idx + best_codec 만 남기고
       - train / val / test 로 나눠서
         data/{chunk_size}MB/train.csv, val.csv, test.csv 로 저장
    """
    # 1. 전체 피쳐 (통계 + 코덱정보)
    df_full = build_raw_features_for_chunks(chunk_size_mb)

    # 2. cost 기준 best_codec 라벨 추가 + 비율 출력
    df_full = add_best_codec_label_by_cost(df_full)

    # 3. full 저장 (코덱정보까지 포함, 분석용)
    target_dir = DATA_DIR / f"{chunk_size_mb}MB"
    target_dir.mkdir(parents=True, exist_ok=True)

    full_path = target_dir / "full.csv"
    df_full.to_csv(full_path, index=False)
    print(f"\n[features] full dataset saved to: {full_path}")

    # 4. 학습용 DataFrame: 통계 피쳐 + chunk_idx + best_codec만 사용
    drop_cols = []
    for col in df_full.columns:
        if col.endswith("_ratio") or col.endswith("_time_ms") or col.endswith("_cost"):
            drop_cols.append(col)

    df_ml = df_full.drop(columns=drop_cols)

    # (원하면 여기서 chunk_idx도 드롭 가능 -> 나중에 결정)
    # df_ml = df_ml.drop(columns=["chunk_idx"])

    train_df, val_df, test_df = split_train_val_test(df_ml)

    train_path = target_dir / "train.csv"
    val_path = target_dir / "val.csv"
    test_path = target_dir / "test.csv"

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"[features] train saved to: {train_path} (n={len(train_df)})")
    print(f"[features] val   saved to: {val_path} (n={len(val_df)})")
    print(f"[features] test  saved to: {test_path} (n={len(test_df)})")


if __name__ == "__main__":
    # 예: python -m pipeline.features       -> 1MB 기준
    #     python -m pipeline.features 2     -> 2MB 기준
    import sys

    if len(sys.argv) >= 2:
        size_mb = int(sys.argv[1])
    else:
        size_mb = 1

    build_and_save_datasets(chunk_size_mb=size_mb)
