# preprocess.py

"""
data/{chunk_size}.csv        : 팀원이 만든 피처 + 위치 정보
results/compression_{chunk_size}.csv : 각 청크 × 코덱 압축 결과

두 CSV를 합쳐서:
- 코덱별 ratio/speed/score 계산
- best codec을 label로 지정
- preprocessed/{chunk_size}.csv 로 저장
"""

import argparse

import numpy as np
import pandas as pd

from config import (
    CHUNK_SIZES,
    CODECS,
    CODEC_TO_ID,
    SCORE_ALPHA,
)
from utils import (
    data_csv_path,
    compression_csv_path,
    preprocessed_csv_path,
    compute_ratio,
    compute_speed_mb_s,
    compute_combined_score,
)


def preprocess_one_chunk_size(chunk_size: str) -> None:
    print(f"[preprocess] chunk_size = {chunk_size}")

    features_path = data_csv_path(chunk_size)
    compression_path = compression_csv_path(chunk_size)
    output_path = preprocessed_csv_path(chunk_size)

    print(f"  - loading features:   {features_path}")
    df_feat = pd.read_csv(features_path)

    print(f"  - loading compression: {compression_path}")
    df_comp = pd.read_csv(compression_path)

    # 기대하는 컬럼 구조:
    # df_feat:   chunk_id, file_name, start_offset, chunk_size_bytes, <feature1>, <feature2>, ...
    # df_comp:   chunk_id, codec, compressed_bytes, encode_time_ms, [decode_time_ms, ...]
    required_feat_cols = {"chunk_id", "chunk_size_bytes"}
    missing = required_feat_cols - set(df_feat.columns)
    if missing:
        raise ValueError(f"features CSV({features_path})에 {missing} 컬럼이 필요합니다.")

    required_comp_cols = {"chunk_id", "codec", "compressed_bytes", "encode_time_ms"}
    missing = required_comp_cols - set(df_comp.columns)
    if missing:
        raise ValueError(f"compression CSV({compression_path})에 {missing} 컬럼이 필요합니다.")

    # 피처는 chunk_id 기준으로 인덱스 설정
    df_feat = df_feat.set_index("chunk_id")

    # compression 결과를 metric별로 pivot해서 wide 형태로 합치기
    merged = df_feat.copy()

    for metric in ["compressed_bytes", "encode_time_ms"]:
        pivot = df_comp.pivot(index="chunk_id", columns="codec", values=metric)
        # 우리가 사용하는 CODECS만 필터링 (혹시 CSV에 다른 코덱이 있어도 무시)
        pivot = pivot[[c for c in CODECS if c in pivot.columns]]
        pivot.columns = [f"{metric}_{c}" for c in pivot.columns]
        merged = merged.join(pivot, how="left")

    # ratio, speed 계산
    original_bytes = merged["chunk_size_bytes"].to_numpy()

    ratio_dict = {}
    speed_dict = {}

    for codec in CODECS:
        cb_col = f"compressed_bytes_{codec}"
        et_col = f"encode_time_ms_{codec}"
        if cb_col not in merged.columns or et_col not in merged.columns:
            print(f"  [warn] {codec}에 대한 압축 결과 컬럼이 부족합니다. (건너뜀)")
            continue

        compressed_bytes = merged[cb_col].to_numpy()
        encode_time_ms = merged[et_col].to_numpy()

        ratio = compute_ratio(original_bytes, compressed_bytes)
        speed = compute_speed_mb_s(original_bytes, encode_time_ms)

        merged[f"ratio_{codec}"] = ratio
        merged[f"speed_{codec}"] = speed

        ratio_dict[codec] = ratio
        speed_dict[codec] = speed

    # score 계산 (ratio/speed 전체 범위 기준 min-max 정규화)
    # 모든 코덱의 ratio, speed를 한꺼번에 모아서 정규화 범위 계산
    all_ratio_values = np.concatenate([v for v in ratio_dict.values()]) if ratio_dict else np.array([])
    all_speed_values = np.concatenate([v for v in speed_dict.values()]) if speed_dict else np.array([])

    if all_ratio_values.size == 0 or all_speed_values.size == 0:
        raise ValueError("ratio 또는 speed 값이 비어 있습니다. compression CSV를 확인하세요.")

    ratio_min, ratio_max = np.nanmin(all_ratio_values), np.nanmax(all_ratio_values)
    speed_min, speed_max = np.nanmin(all_speed_values), np.nanmax(all_speed_values)

    # 정규화 함수 직접 사용 (utils.normalize_min_max 써도 되지만 여기선 범위 재사용)
    def norm(values, vmin, vmax):
        values = np.asarray(values)
        if vmax - vmin < 1e-12:
            return np.zeros_like(values)
        return (values - vmin) / (vmax - vmin)

    score_cols = []
    for codec in CODECS:
        r_col = f"ratio_{codec}"
        s_col = f"speed_{codec}"
        if r_col not in merged.columns or s_col not in merged.columns:
            continue

        r = merged[r_col].to_numpy()
        s = merged[s_col].to_numpy()

        r_norm = norm(r, ratio_min, ratio_max)
        s_norm = norm(s, speed_min, speed_max)
        score = SCORE_ALPHA * r_norm + (1.0 - SCORE_ALPHA) * s_norm

        merged[f"ratio_norm_{codec}"] = r_norm
        merged[f"speed_norm_{codec}"] = s_norm
        score_col = f"score_{codec}"
        merged[score_col] = score
        score_cols.append(score_col)

    if not score_cols:
        raise ValueError("score 컬럼이 하나도 생성되지 않았습니다. CODECS 설정과 compression CSV를 확인하세요.")

    # 각 청크별로 score가 가장 높은 코덱을 label로 지정
    score_df = merged[score_cols]
    best_score_col = score_df.idxmax(axis=1)  # 예: "score_zstd"
    best_codec = best_score_col.str.replace("score_", "", regex=False)

    merged["label_codec"] = best_codec
    merged["label_codec_id"] = merged["label_codec"].map(CODEC_TO_ID)

    # 저장용으로 index를 다시 컬럼으로
    merged.reset_index(inplace=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"  - saving preprocessed: {output_path}")
    merged.to_csv(output_path, index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--chunk-size",
        type=str,
        default=None,
        help="특정 청크 사이즈만 전처리하고 싶을 때 (예: 1MB). 지정하지 않으면 CHUNK_SIZES 전체 수행.",
    )
    args = parser.parse_args()

    if args.chunk_size:
        preprocess_one_chunk_size(args.chunk_size)
    else:
        for cs in CHUNK_SIZES:
            preprocess_one_chunk_size(cs)


if __name__ == "__main__":
    main()
