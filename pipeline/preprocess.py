# pipeline/preprocess.py
import pandas as pd
import numpy as np

from .common import (
    DATA_DIR,
    PREPROCESSED_DIR,
    CHUNK_SIZES,
    CODECS,
    set_global_seed,
    log,
    RANDOM_SEED,
)


def choose_best_codec(row: pd.Series) -> str:
    """
    한 행(한 청크)에 대해 '최적 코덱'을 고르는 함수.
    지금은 예시로:
        - ratio: 작을수록 좋음 (압축 크기 / 원본 크기라고 가정)
        - time: 작을수록 좋음
    두 값을 단순 합으로 score = ratio + alpha * time 계산해서 최소인 코덱 선택.
    alpha 등은 상황에 맞게 조절하면 됨.
    """
    alpha = 0.01  # 시간 가중치 (필요에 따라 바꾸기)
    best_codec = None
    best_score = np.inf

    for codec_name, cols in CODECS.items():
        ratio = row[cols["ratio_col"]]
        time_ = row[cols["time_col"]]
        score = float(ratio) + alpha * float(time_)
        if score < best_score:
            best_score = score
            best_codec = codec_name

    return best_codec


def preprocess_chunk(chunk_size: str) -> None:
    """
    data/{chunk_size}.csv 를 읽어서 label(best_codec)을 생성하고,
    preprocessed/{chunk_size}.csv 로 저장.
    """
    set_global_seed(RANDOM_SEED)

    input_path = DATA_DIR / f"{chunk_size}.csv"
    output_path = PREPROCESSED_DIR / f"{chunk_size}.csv"

    log(f"[{chunk_size}] 전처리 시작: {input_path}")

    df = pd.read_csv(input_path)

    # label 생성
    df["label"] = df.apply(choose_best_codec, axis=1)

    # 필요하다면 여기서 feature 엔지니어링/정규화/필터링 추가
    # 예: df["entropy_log"] = np.log1p(df["entropy"])

    df.to_csv(output_path, index=False)
    log(f"[{chunk_size}] 전처리 완료 → {output_path}")


def run_all_preprocess() -> None:
    for cs in CHUNK_SIZES:
        preprocess_chunk(cs)


if __name__ == "__main__":
    # 모듈 단독 실행 테스트용
    run_all_preprocess()
