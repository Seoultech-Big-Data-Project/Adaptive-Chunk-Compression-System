# benchmark.py
from __future__ import annotations

import json
import sys
import time
from typing import Dict, Any
from multiprocessing import Pool, cpu_count
from config import MAX_WORKERS

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from pipeline.common import DATA_DIR, MODELS_DIR, RESULTS_DIR, get_chunk_dir
from pipeline.features import compute_basic_stats
from features.compression import compress


CODECS = ["zstd", "lz4", "snappy"]


def load_data_for_benchmark(chunk_size_mb: int):
    data_dir = DATA_DIR / f"{chunk_size_mb}MB"
    test_path = data_dir / "test.csv"
    full_path = data_dir / "full.csv"

    print(f"[benchmark] loading test: {test_path}")
    print(f"[benchmark] loading full: {full_path}")

    test_df = pd.read_csv(test_path)
    full_df = pd.read_csv(full_path)

    if "chunk_idx" not in test_df.columns or "chunk_idx" not in full_df.columns:
        raise ValueError("test.csv와 full.csv 모두 chunk_idx 컬럼이 필요합니다.")

    df = pd.merge(
        test_df,
        full_df,
        on="chunk_idx",
        suffixes=("", "_full"),
    )
    return df


def load_model_and_meta(chunk_size_mb: int):
    model_path = MODELS_DIR / f"{chunk_size_mb}MB_xgb.json"
    metrics_path = RESULTS_DIR / f"{chunk_size_mb}MB_xgb_metrics.json"

    print(f"[benchmark] loading model:   {model_path}")
    print(f"[benchmark] loading metrics: {metrics_path}")

    if not model_path.exists():
        raise FileNotFoundError(f"XGBoost 모델 파일이 없습니다: {model_path}")
    if not metrics_path.exists():
        raise FileNotFoundError(f"메트릭 파일이 없습니다: {metrics_path}")

    model = XGBClassifier()
    model.load_model(model_path)

    with metrics_path.open("r", encoding="utf-8") as f:
        meta = json.load(f)

    classes = meta["classes"]
    feature_columns = meta["feature_columns"]

    return model, classes, feature_columns


def compute_strategy_metrics(df: pd.DataFrame, codec_choices: pd.Series) -> Dict[str, Any]:
    if len(codec_choices) != len(df):
        raise ValueError("codec_choices 길이가 df와 다릅니다.")

    total_orig_bytes = float(df["orig_size"].sum())
    total_comp_bytes = 0.0
    total_time_ms = 0.0

    for codec in CODECS:
        mask = (codec_choices == codec)
        if not mask.any():
            continue

        comp_bytes = (df.loc[mask, f"{codec}_ratio"] * df.loc[mask, "orig_size"]).sum()
        time_ms = df.loc[mask, f"{codec}_time_ms"].sum()

        total_comp_bytes += float(comp_bytes)
        total_time_ms += float(time_ms)

    if total_orig_bytes <= 0:
        overall_ratio = 1.0
        throughput_mb_s = 0.0
    else:
        overall_ratio = total_comp_bytes / total_orig_bytes
        total_time_sec = total_time_ms / 1000.0
        if total_time_sec > 0:
            throughput_mb_s = (total_orig_bytes / (1024 * 1024)) / total_time_sec
        else:
            throughput_mb_s = float("inf")

    return {
        "total_orig_bytes": total_orig_bytes,
        "total_comp_bytes": total_comp_bytes,
        "overall_ratio": overall_ratio,
        "total_time_ms": total_time_ms,
        "throughput_mb_s": throughput_mb_s,
    }


def measure_feature_and_predict_overhead(
    chunk_size_mb: int,
    df: pd.DataFrame,
    model: XGBClassifier,
    feature_columns: list[str],
    classes: list[str],
) -> Dict[str, Any]:
    chunk_dir = get_chunk_dir(chunk_size_mb)
    chunk_indices = df["chunk_idx"].tolist()

    # 1) 피쳐 추출 시간
    print("[benchmark] measuring feature extraction overhead (compute_basic_stats on test chunks)...")
    t0 = time.perf_counter()
    for idx in chunk_indices:
        path = chunk_dir / f"{idx}.bin"
        data = path.read_bytes()
        _ = compute_basic_stats(data)
    t1 = time.perf_counter()
    feature_time_ms = (t1 - t0) * 1000.0

    # 2) 예측 시간
    print("[benchmark] measuring model prediction overhead (XGBoost.predict on test set)...")
    X_test = df[feature_columns]
    t2 = time.perf_counter()
    y_pred_enc = model.predict(X_test)
    t3 = time.perf_counter()
    predict_time_ms = (t3 - t2) * 1000.0

    class_arr = np.array(classes)
    pred_codecs = pd.Series(class_arr[y_pred_enc], index=df.index)

    return {
        "feature_time_ms": feature_time_ms,
        "predict_time_ms": predict_time_ms,
        "pred_codecs": pred_codecs,
    }


def _compress_chunk_task(args):
    chunk_path, codec = args
    data = chunk_path.read_bytes()
    comp_size, comp_time, _ = compress(data, codec)
    return len(data), comp_size, comp_time


def measure_real_compression(
    chunk_size_mb: int,
    df: pd.DataFrame,
    pred_codecs: pd.Series,
    use_multiprocessing: bool = True,
) -> Dict[str, float]:
    """
    예측된 코덱으로 실제 압축 수행 및 측정
    """
    chunk_dir = get_chunk_dir(chunk_size_mb)
    chunk_indices = df["chunk_idx"].tolist()
    
    print("[benchmark] performing REAL compression with predicted codecs...")
    
    tasks = []
    for idx, codec in zip(chunk_indices, pred_codecs):
        path = chunk_dir / f"{idx}.bin"
        tasks.append((path, codec))
    
    total_orig_bytes = 0.0
    total_comp_bytes = 0.0
    total_comp_time_ms = 0.0
    
    if use_multiprocessing and len(tasks) > 1:
        num_workers = min(cpu_count(), MAX_WORKERS, len(tasks))
        print(f"[benchmark] using {num_workers} workers for parallel compression...")
        
        with Pool(processes=num_workers) as pool:
            results = pool.map(_compress_chunk_task, tasks)
        
        for orig_size, comp_size, comp_time in results:
            total_orig_bytes += orig_size
            total_comp_bytes += comp_size
            total_comp_time_ms += comp_time
    else:
        print("[benchmark] using single process for compression...")
        for path, codec in tasks:
            data = path.read_bytes()
            comp_size, comp_time, _ = compress(data, codec)
            
            total_orig_bytes += len(data)
            total_comp_bytes += comp_size
            total_comp_time_ms += comp_time
    
    overall_ratio = total_comp_bytes / total_orig_bytes if total_orig_bytes > 0 else 1.0
    total_time_sec = total_comp_time_ms / 1000.0
    throughput_mb_s = (total_orig_bytes / (1024 * 1024)) / total_time_sec if total_time_sec > 0 else float("inf")
    
    return {
        "total_orig_bytes": total_orig_bytes,
        "total_comp_bytes": total_comp_bytes,
        "overall_ratio": overall_ratio,
        "total_time_ms": total_comp_time_ms,
        "throughput_mb_s": throughput_mb_s,
    }



def run_benchmark(chunk_size_mb: int = 1) -> None:
    # 1. 데이터
    df = load_data_for_benchmark(chunk_size_mb)
    total_chunks = len(df)
    total_orig_mb = df["orig_size"].sum() / (1024 * 1024)
    print(f"[benchmark] num test chunks = {total_chunks}, total_orig ≈ {total_orig_mb:.2f} MB")

    # 2. 모델 + 메타
    model, classes, feature_columns = load_model_and_meta(chunk_size_mb)

    # 3. always_zstd
    always_zstd_choices = pd.Series(["zstd"] * len(df), index=df.index)
    metrics_always_zstd = compute_strategy_metrics(df, always_zstd_choices)

    # 4. always_lz4
    always_lz4_choices = pd.Series(["lz4"] * len(df), index=df.index)
    metrics_always_lz4 = compute_strategy_metrics(df, always_lz4_choices)

    # 5. always_snappy 
    always_snappy_choices = pd.Series(["snappy"] * len(df), index=df.index)
    metrics_always_snappy = compute_strategy_metrics(df, always_snappy_choices)

    # 6. oracle
    if "best_codec" not in df.columns:
        raise ValueError("df에 best_codec 컬럼이 없습니다. features.py에서 라벨이 제대로 생성되었는지 확인하십시오.")
    oracle_choices = df["best_codec"].astype(str)
    metrics_oracle = compute_strategy_metrics(df, oracle_choices)

    # 7. XGBoost 예측 (ideal)
    X_test = df[feature_columns]
    y_pred_enc = model.predict(X_test)
    class_arr = np.array(classes)
    xgb_pred_codecs = pd.Series(class_arr[y_pred_enc], index=df.index)
    metrics_xgb_pred = compute_strategy_metrics(df, xgb_pred_codecs)

    # 8. 오버헤드 측정 (feature extraction + prediction)
    overhead = measure_feature_and_predict_overhead(
        chunk_size_mb=chunk_size_mb,
        df=df,
        model=model,
        feature_columns=feature_columns,
        classes=classes,
    )
    feature_time_ms = overhead["feature_time_ms"]
    predict_time_ms = overhead["predict_time_ms"]
    pred_codecs_for_real = overhead["pred_codecs"]
    
    # 9. 실전 시나리오(real): 실제 압축 수행
    metrics_real_compression = measure_real_compression(
        chunk_size_mb=chunk_size_mb,
        df=df,
        pred_codecs=pred_codecs_for_real,
    )

    metrics_xgb_pred_real = metrics_real_compression.copy()
    metrics_xgb_pred_real["total_time_ms"] += feature_time_ms + predict_time_ms
    
    total_time_sec = metrics_xgb_pred_real["total_time_ms"] / 1000.0
    total_orig_bytes = metrics_xgb_pred_real["total_orig_bytes"]
    if total_time_sec > 0:
        metrics_xgb_pred_real["throughput_mb_s"] = (total_orig_bytes / (1024 * 1024)) / total_time_sec
    else:
        metrics_xgb_pred_real["throughput_mb_s"] = float("inf")

    # 10. 결과 모으기
    strategies = {
        "always_zstd": metrics_always_zstd,
        "always_lz4": metrics_always_lz4,
        "always_snappy": metrics_always_snappy,
        "oracle": metrics_oracle,
        "xgb_pred_ideal": metrics_xgb_pred,
        "xgb_pred_real": metrics_xgb_pred_real,
    }

    # 11. 콘솔 표 출력
    print("\n[benchmark] 결과 (test set)")
    print(f"{'strategy':15s} | {'ratio':>8s} | {'time_ms':>10s} | {'throughput_MB/s':>15s}")
    print("-" * 60)
    for name, m in strategies.items():
        ratio = m["overall_ratio"]
        time_ms = m["total_time_ms"]
        thr = m["throughput_mb_s"]
        print(f"{name:15s} | {ratio:8.4f} | {time_ms:10.1f} | {thr:15.2f}")

    # print("\n[benchmark] overhead:")
    # print(f"  - feature extraction time (ms): {feature_time_ms:.1f}")
    # print(f"  - model prediction time (ms):   {predict_time_ms:.1f}")
    # print(f"  - overhead per chunk (ms):      {(feature_time_ms + predict_time_ms) / total_chunks:.4f}")

    # 12. JSON 저장
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f"{chunk_size_mb}MB_xgb_benchmark.json"

    output = {
        "chunk_size_mb": chunk_size_mb,
        "num_test_chunks": total_chunks,
        "total_orig_bytes": float(df["orig_size"].sum()),
        "overhead": {
            "feature_time_ms": feature_time_ms,
            "predict_time_ms": predict_time_ms,
            "overhead_per_chunk_ms": (feature_time_ms + predict_time_ms) / total_chunks,
        },
        "strategies": strategies,
    }

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\n[benchmark] benchmark results saved to: {out_path}")


if __name__ == "__main__":
    if len(sys.argv) >= 2:
        size_mb = int(sys.argv[1])
    else:
        size_mb = 1

    run_benchmark(chunk_size_mb=size_mb)
