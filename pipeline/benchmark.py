# pipeline/benchmark.py
import pickle

import numpy as np
import pandas as pd
import xgboost as xgb

from .common import (
    PREPROCESSED_DIR,
    MODELS_DIR,
    RESULTS_DIR,
    CHUNK_SIZES,
    CODECS,
    TRAIN_RATIO,
    RANDOM_SEED,
    log,
    set_global_seed,
)


def _cast_object_to_category(df: pd.DataFrame) -> pd.DataFrame:
    """
    object 타입 컬럼들을 category로 변환.
    label 컬럼은 제외 (라벨 인코딩 따로 함).
    """
    obj_cols = df.select_dtypes(include=["object"]).columns.tolist()
    for col in obj_cols:
        if col == "label":
            continue
        df[col] = df[col].astype("category")
    return df


def _load_model_and_encoder(chunk_size: str):
    """
    models/{chunk_size}.xgb 와
    models/{chunk_size}_label_encoder.pkl 을 로드.
    """
    model_path = MODELS_DIR / f"{chunk_size}.xgb"
    le_path = MODELS_DIR / f"{chunk_size}_label_encoder.pkl"

    model = xgb.XGBClassifier()
    model.load_model(model_path)

    with open(le_path, "rb") as f:
        le = pickle.load(f)

    return model, le


def _get_test_df(chunk_size: str) -> pd.DataFrame:
    """
    preprocessed/{chunk_size}.csv 를 읽어서
    TRAIN_RATIO 기준으로 뒤 20%를 test로 사용.
    object → category 변환도 여기서 처리.
    """
    path = PREPROCESSED_DIR / f"{chunk_size}.csv"
    df = pd.read_csv(path)

    # trainer와 동일하게 문자열을 카테고리로 변환
    df = _cast_object_to_category(df)

    n = len(df)
    split_idx = int(n * TRAIN_RATIO)
    test_df = df.iloc[split_idx:].reset_index(drop=True)
    log(f"[{chunk_size}] 테스트(벤치마크) 데이터 수: {len(test_df)} / 전체 {n}")
    return test_df


def _compute_metrics_for_codec(df: pd.DataFrame, codec_name: str):
    """
    단일 코덱 시나리오:
    - 해당 codec의 ratio/time 컬럼 평균을 구함.
    """
    cols = CODECS[codec_name]
    ratio_col = cols["ratio_col"]
    time_col = cols["time_col"]

    avg_ratio = df[ratio_col].mean()
    avg_time = df[time_col].mean()
    return float(avg_ratio), float(avg_time)


def _compute_oracle_metrics(df: pd.DataFrame):
    """
    Oracle 시나리오:
    - 각 행의 label(best_codec)에 따라 해당 codec의 ratio/time을 선택해서 평균 계산.
    """
    ratios = []
    times = []

    for _, row in df.iterrows():
        codec = row["label"]
        cols = CODECS[codec]
        ratios.append(row[cols["ratio_col"]])
        times.append(row[cols["time_col"]])

    return float(np.mean(ratios)), float(np.mean(times))


def _compute_model_based_metrics(df: pd.DataFrame, chunk_size: str):
    """
    Model-based 시나리오:
    - XGBoost 모델을 이용해 각 행의 codec을 예측하고,
    - 예측된 codec의 ratio/time으로 평균 계산.
    """
    model, le = _load_model_and_encoder(chunk_size)

    # trainer와 동일하게: label 제외 전부 feature로 사용 (숫자 + 카테고리)
    feature_cols = [c for c in df.columns if c != "label"]

    if not feature_cols:
        raise ValueError(f"[{chunk_size}] 벤치마크용 feature가 없습니다.")

    # DataFrame 그대로 넘겨야 category dtype이 유지됨
    X_test = df[feature_cols]

    # 예측
    y_pred = model.predict(X_test)
    pred_labels = le.inverse_transform(y_pred)

    ratios = []
    times = []
    for i, codec in enumerate(pred_labels):
        row = df.iloc[i]
        cols = CODECS[codec]
        ratios.append(row[cols["ratio_col"]])
        times.append(row[cols["time_col"]])

    return float(np.mean(ratios)), float(np.mean(times))


def benchmark_chunk(chunk_size: str) -> None:
    """
    하나의 chunk_size에 대해:
      - single codec (zstd/lz4/snappy)
      - oracle (label 기반)
      - model-based (XGBoost 예측)
    를 모두 계산하고,
    results/benchmark_{chunk_size}.csv 로 저장.
    """
    set_global_seed(RANDOM_SEED)

    df_test = _get_test_df(chunk_size)

    results = []

    # 1) Single codec 시나리오
    for codec in CODECS.keys():
        avg_ratio, avg_time = _compute_metrics_for_codec(df_test, codec)
        results.append(
            {
                "chunk_size": chunk_size,
                "scenario": "single",
                "codec": codec,
                "avg_ratio": avg_ratio,
                "avg_time": avg_time,
            }
        )

    # 2) Oracle 시나리오
    oracle_ratio, oracle_time = _compute_oracle_metrics(df_test)
    results.append(
        {
            "chunk_size": chunk_size,
            "scenario": "oracle",
            "codec": "mixed",  # 청크마다 codec이 다름
            "avg_ratio": oracle_ratio,
            "avg_time": oracle_time,
        }
    )

    # 3) Model-based 시나리오
    model_ratio, model_time = _compute_model_based_metrics(df_test, chunk_size)
    results.append(
        {
            "chunk_size": chunk_size,
            "scenario": "model_based",
            "codec": "mixed",  # 청크마다 codec이 다름
            "avg_ratio": model_ratio,
            "avg_time": model_time,
        }
    )

    # 결과 저장
    out_df = pd.DataFrame(results)
    out_path = RESULTS_DIR / f"benchmark_{chunk_size}.csv"
    out_df.to_csv(out_path, index=False)

    log(f"[{chunk_size}] 벤치마크 완료 → {out_path}")


def run_all_benchmarks() -> None:
    for cs in CHUNK_SIZES:
        benchmark_chunk(cs)


if __name__ == "__main__":
    # 모듈 단독 실행 테스트용
    run_all_benchmarks()
