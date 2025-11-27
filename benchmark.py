# benchmark.py

import argparse

import joblib
import numpy as np
import pandas as pd

from config import (
    CHUNK_SIZES,
    CODECS,
    BASELINE_CODEC,
    MODELS_DIR,
    SCORE_ALPHA,
)
from utils import (
    preprocessed_csv_path,
    get_feature_columns,
    benchmark_csv_path,
    benchmark_summary_path,
)
from config import ID_TO_CODEC


def compute_strategy_metrics(df_test: pd.DataFrame, codec_series: pd.Series):
    """
    각 row마다 사용할 코덱(codec_series)에 따라,
    ratio_{codec}, speed_{codec}를 읽어와 평균 및 가중 score 계산.
    """
    ratios = []
    speeds = []

    for idx, codec in codec_series.items():
        r_col = f"ratio_{codec}"
        s_col = f"speed_{codec}"
        if r_col not in df_test.columns or s_col not in df_test.columns:
            raise ValueError(f"컬럼 {r_col} 또는 {s_col}이 존재하지 않습니다. preprocess 단계를 확인하세요.")
        ratios.append(df_test.loc[idx, r_col])
        speeds.append(df_test.loc[idx, s_col])

    ratios = np.array(ratios, dtype=float)
    speeds = np.array(speeds, dtype=float)

    # 공통 스코어 계산
    # (여기서는 각 전략 내에서만 min-max 정규화)
    def norm(x):
        x = np.asarray(x)
        if np.nanmax(x) - np.nanmin(x) < 1e-12:
            return np.zeros_like(x)
        return (x - np.nanmin(x)) / (np.nanmax(x) - np.nanmin(x))

    ratio_norm = norm(ratios)
    speed_norm = norm(speeds)
    scores = SCORE_ALPHA * ratio_norm + (1.0 - SCORE_ALPHA) * speed_norm

    return {
        "avg_ratio": float(np.nanmean(ratios)),
        "avg_speed": float(np.nanmean(speeds)),
        "avg_score": float(np.nanmean(scores)),
    }


def benchmark_for_chunk_size(chunk_size: str):
    print(f"[benchmark] chunk_size = {chunk_size}")

    csv_path = preprocessed_csv_path(chunk_size)
    print(f"  - loading preprocessed: {csv_path}")
    df = pd.read_csv(csv_path)

    if "split" not in df.columns:
        raise ValueError("preprocessed CSV에 split 컬럼이 없습니다. 먼저 train_model.py를 실행하세요.")

    df_test = df[df["split"] == "test"].copy()
    if df_test.empty:
        raise ValueError("test(split=='test') 데이터가 없습니다.")

    # -------------------------
    # 모델 로드 & ML 예측
    # -------------------------
    model_path = MODELS_DIR / chunk_size / "model.pkl"
    print(f"  - loading model: {model_path}")
    model = joblib.load(model_path)

    feature_cols = get_feature_columns(df_test)
    X_test = df_test[feature_cols].to_numpy()
    y_pred = model.predict(X_test)  # label_codec_id 기준

    ml_codec_series = pd.Series([ID_TO_CODEC[int(i)] for i in y_pred], index=df_test.index)

    # -------------------------
    # 각 전략별 사용할 codec 시리즈 준비
    # -------------------------
    # oracle: preprocess에서 만든 label_codec 사용
    if "label_codec" not in df_test.columns:
        raise ValueError("preprocessed CSV에 label_codec 컬럼이 없습니다.")
    oracle_codec_series = df_test["label_codec"]

    # baseline: config.BASELINE_CODEC 고정
    baseline_codec_series = pd.Series(BASELINE_CODEC, index=df_test.index)

    # -------------------------
    # 전략별 메트릭 계산
    # -------------------------
    print("  - computing metrics for baseline/oracle/ML...")

    baseline_metrics = compute_strategy_metrics(df_test, baseline_codec_series)
    oracle_metrics = compute_strategy_metrics(df_test, oracle_codec_series)
    ml_metrics = compute_strategy_metrics(df_test, ml_codec_series)

    # 상대적인 oracle 대비 비율 등 계산
    def oracle_gap(metric_value, oracle_value):
        if oracle_value == 0:
            return 0.0
        return float(100.0 * (metric_value / oracle_value - 1.0))

    # per-chunk 상세 결과 저장 (옵션)
    out_path = benchmark_csv_path(chunk_size)
    print(f"  - saving per-strategy summary: {out_path}")

    rows = []
    for name, m in [
        ("baseline", baseline_metrics),
        ("ml_recommend", ml_metrics),
        ("oracle", oracle_metrics),
    ]:
        rows.append(
            {
                "chunk_size": chunk_size,
                "strategy": name,
                "avg_ratio": m["avg_ratio"],
                "avg_speed": m["avg_speed"],
                "avg_score": m["avg_score"],
                "ratio_vs_oracle_%": oracle_gap(m["avg_ratio"], oracle_metrics["avg_ratio"]),
                "speed_vs_oracle_%": oracle_gap(m["avg_speed"], oracle_metrics["avg_speed"]),
                "score_vs_oracle_%": oracle_gap(m["avg_score"], oracle_metrics["avg_score"]),
            }
        )

    df_out = pd.DataFrame(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(out_path, index=False)

    # 전체 summary 파일에 append
    summary_path = benchmark_summary_path()
    if summary_path.exists():
        df_summary = pd.read_csv(summary_path)
        df_summary = pd.concat([df_summary, df_out], ignore_index=True)
    else:
        df_summary = df_out
    df_summary.to_csv(summary_path, index=False)
    print(f"  - updated summary: {summary_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--chunk-size",
        type=str,
        default=None,
        help="특정 청크 사이즈만 벤치마크하고 싶을 때 (예: 1MB). 지정하지 않으면 CHUNK_SIZES 전체 수행.",
    )
    args = parser.parse_args()

    if args.chunk_size:
        benchmark_for_chunk_size(args.chunk_size)
    else:
        for cs in CHUNK_SIZES:
            benchmark_for_chunk_size(cs)


if __name__ == "__main__":
    main()
