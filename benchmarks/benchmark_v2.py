"""
벤치마크 측정
/data_benchmark 디렉토리의 파일들을 대상으로, 각 청크 크기별로 다음 항목들을 측정하여 출력 및 저장
  - 항상 zstd/lz4/snappy 사용 시의 압축 성능
  - 오라클(최적 코덱 선택) 압축 성능
  - 학습된 XGBoost 모델의 예측 기반 압축 성능
    1. 각 파일별로 청크 단위로 feature 추출 및 청크 저장
    2. 학습된 XGBoost 모델로 각 청크의 코덱 예측
    3. 예측된 코덱으로 실제 압축 수행
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Dict, Any
from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

# 프로젝트 루트를 sys.path에 추가
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import MAX_WORKERS, TEST_CODECS, DATA_BENCHMARK_DIR, MODELS_DIR, RESULTS_DIR
from pipeline.features import compute_basic_stats, add_best_codec_label_by_cost
from features.compression import compress


def list_benchmark_files() -> list[Path]:
    """data_benchmark 디렉토리의 모든 파일 리스트"""
    if not DATA_BENCHMARK_DIR.exists():
        raise FileNotFoundError(f"data_benchmark 디렉토리가 없습니다: {DATA_BENCHMARK_DIR}")
    
    files = [p for p in DATA_BENCHMARK_DIR.iterdir() if p.is_file()]
    files.sort()
    return files


def load_model_and_meta(chunk_size_mb: int):
    """학습된 XGBoost 모델과 메타데이터 로드"""
    model_path = MODELS_DIR / f"{chunk_size_mb}MB_xgb.json"
    metrics_path = RESULTS_DIR / f"{chunk_size_mb}MB_xgb_metrics.json"

    print(f"[benchmark_v2] loading model:   {model_path}")
    print(f"[benchmark_v2] loading metrics: {metrics_path}")

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


def split_file_to_chunks(file_path: Path, chunk_size_bytes: int) -> list[bytes]:
    """파일을 청크로 분할"""
    chunks = []
    with file_path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size_bytes)
            if not chunk:
                break
            chunks.append(chunk)
    return chunks


def _extract_and_predict_task(args):
    """멀티프로세싱용: 청크별 특성 추출"""
    chunk, chunk_idx = args
    features = compute_basic_stats(chunk)
    return chunk_idx, features, len(chunk)


def extract_features_parallel(chunks: list[bytes], use_multiprocessing: bool = True) -> tuple[list[dict], list[int]]:
    """모든 청크의 특성 추출 (병렬)"""
    if use_multiprocessing and len(chunks) > 1:
        num_workers = min(cpu_count(), MAX_WORKERS, len(chunks))
        print(f"[benchmark_v2] extracting features with {num_workers} workers...")
        
        tasks = [(chunk, idx) for idx, chunk in enumerate(chunks)]
        with Pool(processes=num_workers) as pool:
            results = pool.map(_extract_and_predict_task, tasks)
        
        results.sort(key=lambda x: x[0])
        features_list = [r[1] for r in results]
        sizes = [r[2] for r in results]
    else:
        print(f"[benchmark_v2] extracting features (single process)...")
        features_list = []
        sizes = []
        for chunk in chunks:
            features = compute_basic_stats(chunk)
            features_list.append(features)
            sizes.append(len(chunk))
    
    return features_list, sizes


def predict_codecs(model, features_list: list[dict], feature_columns: list[str], classes: list[str]) -> list[str]:
    """XGBoost 모델로 코덱 예측 (배치)"""
    # features_list를 DataFrame으로 변환
    df = pd.DataFrame(features_list)
    X = df[feature_columns]
    
    y_pred_enc = model.predict(X)
    class_arr = np.array(classes)
    pred_codecs = class_arr[y_pred_enc].tolist()
    
    return pred_codecs


def _compress_task(args):
    """멀티프로세싱용: 청크 압축"""
    chunk, codec = args
    comp_size, comp_time, comp_ratio = compress(chunk, codec)
    return len(chunk), comp_size, comp_time, comp_ratio


def compress_chunks_parallel(chunks: list[bytes], codecs: list[str], use_multiprocessing: bool = True) -> Dict[str, Any]:
    """예측된 코덱으로 실제 압축 수행 (병렬)"""
    if use_multiprocessing and len(chunks) > 1:
        num_workers = min(cpu_count(), MAX_WORKERS, len(chunks))
        print(f"[benchmark_v2] compressing with {num_workers} workers...")
        
        tasks = list(zip(chunks, codecs))
        with Pool(processes=num_workers) as pool:
            results = pool.map(_compress_task, tasks)
    else:
        print(f"[benchmark_v2] compressing (single process)...")
        results = []
        for chunk, codec in zip(chunks, codecs):
            result = _compress_task((chunk, codec))
            results.append(result)
    
    total_orig = sum(r[0] for r in results)
    total_comp = sum(r[1] for r in results)
    total_time = sum(r[2] for r in results)
    
    overall_ratio = total_comp / total_orig if total_orig > 0 else 1.0
    
    return {
        "total_orig_bytes": total_orig,
        "total_comp_bytes": total_comp,
        "overall_ratio": overall_ratio,
        "total_time_ms": total_time,
    }


def compress_with_single_codec(chunks: list[bytes], codec: str, use_multiprocessing: bool = True) -> Dict[str, Any]:
    """모든 청크를 단일 코덱으로 압축"""
    codecs = [codec] * len(chunks)
    return compress_chunks_parallel(chunks, codecs, use_multiprocessing)


def compress_with_oracle(chunks: list[bytes], use_multiprocessing: bool = True) -> Dict[str, Any]:
    """오라클: 각 청크마다 cost (ratio * time_ms) 기준 최적 코덱 선택 후 압축"""
    print(f"[benchmark_v2] finding best codec for each chunk (oracle based on cost)...")
    
    # 모든 코덱으로 압축하여 DataFrame 생성
    rows = []
    for i, chunk in enumerate(chunks):
        row = {"chunk_idx": i}
        for codec in TEST_CODECS:
            size, time_ms, ratio = compress(chunk, codec)
            row[f"{codec}_ratio"] = ratio
            row[f"{codec}_time_ms"] = time_ms
        rows.append(row)
    
    df = pd.DataFrame(rows)
    
    df = add_best_codec_label_by_cost(df)
    best_codecs = df["best_codec"].tolist()
    
    # best codec으로 재압축
    return compress_chunks_parallel(chunks, best_codecs, use_multiprocessing)


def benchmark_file(file_path: Path, chunk_size_mb: int, model, classes: list[str], feature_columns: list[str]) -> Dict[str, Any]:
    """단일 파일에 대한 벤치마크"""
    print(f"\n[benchmark_v2] Processing file: {file_path.name}")
    chunk_size_bytes = chunk_size_mb * 1024 * 1024
    
    # 1. 파일을 청크로 분할
    print(f"[benchmark_v2] splitting file into {chunk_size_mb}MB chunks...")
    t_start = time.perf_counter()
    chunks = split_file_to_chunks(file_path, chunk_size_bytes)
    t_split = (time.perf_counter() - t_start) * 1000.0
    print(f"[benchmark_v2] split into {len(chunks)} chunks ({t_split:.1f} ms)")
    
    if not chunks:
        print(f"[benchmark_v2] WARNING: no chunks for {file_path.name}")
        return None
    
    # 2. Feature 추출 (병렬)
    print(f"[benchmark_v2] extracting features...")
    t_start = time.perf_counter()
    features_list, sizes = extract_features_parallel(chunks, use_multiprocessing=True)
    t_extract = (time.perf_counter() - t_start) * 1000.0
    print(f"[benchmark_v2] feature extraction done ({t_extract:.1f} ms)")
    
    # 3. 코덱 예측 (배치)
    print(f"[benchmark_v2] predicting codecs with XGBoost...")
    t_start = time.perf_counter()
    pred_codecs = predict_codecs(model, features_list, feature_columns, classes)
    t_predict = (time.perf_counter() - t_start) * 1000.0
    print(f"[benchmark_v2] prediction done ({t_predict:.1f} ms)")
    
    # 4. 압축 성능 측정
    print(f"[benchmark_v2] measuring compression performance...")
    
    # 4-1. Always ZSTD
    print(f"[benchmark_v2]   - always_zstd...")
    metrics_zstd = compress_with_single_codec(chunks, "zstd", use_multiprocessing=True)
    
    # 4-2. Always LZ4
    print(f"[benchmark_v2]   - always_lz4...")
    metrics_lz4 = compress_with_single_codec(chunks, "lz4", use_multiprocessing=True)
    
    # 4-3. Always Snappy
    print(f"[benchmark_v2]   - always_snappy...")
    metrics_snappy = compress_with_single_codec(chunks, "snappy", use_multiprocessing=True)
    
    # 4-4. Oracle
    print(f"[benchmark_v2]   - oracle...")
    metrics_oracle = compress_with_oracle(chunks, use_multiprocessing=True)
    
    # 4-5. XGBoost 예측 기반 (ideal - 오버헤드 제외)
    print(f"[benchmark_v2]   - xgb_pred_ideal...")
    metrics_xgb_ideal = compress_chunks_parallel(chunks, pred_codecs, use_multiprocessing=True)
    
    # 4-6. XGBoost 예측 기반 (real - 오버헤드 포함)
    metrics_xgb_real = metrics_xgb_ideal.copy()
    metrics_xgb_real["total_time_ms"] += t_extract + t_predict
    
    # Throughput 계산
    def calc_throughput(orig_bytes, time_ms):
        return (orig_bytes / (1024 * 1024)) / (time_ms / 1000.0) if time_ms > 0 else float("inf")
    
    total_orig = sum(sizes)
    metrics_zstd["throughput_mb_s"] = calc_throughput(total_orig, metrics_zstd["total_time_ms"])
    metrics_lz4["throughput_mb_s"] = calc_throughput(total_orig, metrics_lz4["total_time_ms"])
    metrics_snappy["throughput_mb_s"] = calc_throughput(total_orig, metrics_snappy["total_time_ms"])
    metrics_oracle["throughput_mb_s"] = calc_throughput(total_orig, metrics_oracle["total_time_ms"])
    metrics_xgb_ideal["throughput_mb_s"] = calc_throughput(total_orig, metrics_xgb_ideal["total_time_ms"])
    metrics_xgb_real["throughput_mb_s"] = calc_throughput(total_orig, metrics_xgb_real["total_time_ms"])
    
    return {
        "file": file_path.name,
        "file_size_bytes": sum(sizes),
        "num_chunks": len(chunks),
        "chunk_size_mb": chunk_size_mb,
        "overhead": {
            "split_time_ms": t_split,
            "feature_time_ms": t_extract,
            "predict_time_ms": t_predict,
            "overhead_per_chunk_ms": (t_extract + t_predict) / len(chunks),
        },
        "strategies": {
            "always_zstd": metrics_zstd,
            "always_lz4": metrics_lz4,
            "always_snappy": metrics_snappy,
            "oracle": metrics_oracle,
            "xgb_pred_ideal": metrics_xgb_ideal,
            "xgb_pred_real": metrics_xgb_real,
        }
    }


def run_benchmark(chunk_size_mb: int = 1) -> None:
    """벤치마크 실행"""
    print("="*70)
    print("BENCHMARK V2 - data_benchmark 파일 기반 실시간 압축 벤치마크")
    print("="*70)
    
    # 1. 벤치마크 파일 목록
    files = list_benchmark_files()
    if not files:
        print(f"[benchmark_v2] ERROR: data_benchmark 디렉토리에 파일이 없습니다.")
        return
    
    print(f"[benchmark_v2] Found {len(files)} file(s) in data_benchmark/")
    for f in files:
        print(f"  - {f.name} ({f.stat().st_size / (1024*1024):.2f} MB)")
    
    # 2. 모델 로드
    model, classes, feature_columns = load_model_and_meta(chunk_size_mb)
    
    # 3. 각 파일별로 벤치마크 수행
    all_results = []
    for file_path in files:
        result = benchmark_file(file_path, chunk_size_mb, model, classes, feature_columns)
        if result:
            all_results.append(result)
    
    # 4. 결과 출력
    print("\n" + "="*70)
    print(f"[benchmark_v2] 벤치마크 결과 요약 (chunk_size={chunk_size_mb}MB)")
    print("="*70)
    
    for result in all_results:
        print(f"\n파일: {result['file']} ({result['file_size_bytes']/(1024*1024):.2f} MB, {result['num_chunks']} chunks)")
        print(f"{'strategy':16s} | {'ratio':>8s} | {'time_ms':>10s} | {'throughput_MB/s':>16s}")
        print("-" * 70)
        
        for name, m in result['strategies'].items():
            ratio = m["overall_ratio"]
            time_ms = m["total_time_ms"]
            thr = m["throughput_mb_s"]
            print(f"{name:16s} | {ratio:8.4f} | {time_ms:10.1f} | {thr:16.2f}")
        
        overhead = result['overhead']
        print(f"\n오버헤드:")
        print(f"  - split time:           {overhead['split_time_ms']:.1f} ms")
        print(f"  - feature extraction:   {overhead['feature_time_ms']:.1f} ms")
        print(f"  - model prediction:     {overhead['predict_time_ms']:.1f} ms")
        print(f"  - overhead per chunk:   {overhead['overhead_per_chunk_ms']:.4f} ms")
    
    # # 5. JSON 저장
    # RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    # out_path = RESULTS_DIR / f"{chunk_size_mb}MB_benchmark_v2.json"
    
    # output = {
    #     "version": "v2_realtime_compression",
    #     "chunk_size_mb": chunk_size_mb,
    #     "max_workers": MAX_WORKERS,
    #     "files": all_results,
    # }
    
    # with out_path.open("w", encoding="utf-8") as f:
    #     json.dump(output, f, ensure_ascii=False, indent=2)
    
    # print(f"\n[benchmark_v2] 결과 저장: {out_path}")
    # print("="*70)


if __name__ == "__main__":
    if len(sys.argv) >= 2:
        size_mb = int(sys.argv[1])
    else:
        size_mb = 1

    run_benchmark(chunk_size_mb=size_mb)