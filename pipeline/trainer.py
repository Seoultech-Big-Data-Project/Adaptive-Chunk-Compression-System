# pipeline/trainer.py
import pickle

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

import xgboost as xgb

from .common import (
    PREPROCESSED_DIR,
    MODELS_DIR,
    CHUNK_SIZES,
    TRAIN_RATIO,
    N_SPLITS,
    RANDOM_SEED,
    XGB_DEFAULT_PARAMS,
    log,
    set_global_seed,
)


def train_for_chunk(chunk_size: str) -> None:
    """
    preprocessed/{chunk_size}.csv 를 읽고
      - 앞 80%: train+val
      - 5-fold CV (fold마다 fresh 모델 생성)
      - 전체 train+val로 최종 모델 학습
      - models/{chunk_size}.xgb 로 저장
      - label encoder는 models/{chunk_size}_label_encoder.pkl 로 저장
    """
    set_global_seed(RANDOM_SEED)

    input_path = PREPROCESSED_DIR / f"{chunk_size}.csv"
    df = pd.read_csv(input_path)

    n = len(df)
    split_idx = int(n * TRAIN_RATIO)
    train_val_df = df.iloc[:split_idx].reset_index(drop=True)

    log(f"[{chunk_size}] 학습 데이터 수: {len(train_val_df)} / 전체 {n}")

    # =========================
    # Label 인코딩
    # =========================
    if "label" not in train_val_df.columns:
        raise ValueError(f"[{chunk_size}] 'label' 컬럼을 찾을 수 없습니다. 전처리를 확인하세요.")

    le = LabelEncoder()
    y = le.fit_transform(train_val_df["label"].values)

    # =========================
    # feature 선택
    #  - 숫자형 컬럼 중에서 label + 코덱 결과 컬럼 제외
    # =========================
    NON_FEATURE_COLS = [
        "label",
        "lz4_size", "lz4_time_ms", "lz4_ratio",
        "snappy_size", "snappy_time_ms", "snappy_ratio",
        "zstd_size", "zstd_time_ms", "zstd_ratio",
    ]

    numeric_cols = train_val_df.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [c for c in numeric_cols if c not in NON_FEATURE_COLS]

    if not feature_cols:
        raise ValueError(f"[{chunk_size}] 사용할 feature가 없습니다.")

    log(f"[{chunk_size}] feature cols: {feature_cols}")

    X = train_val_df[feature_cols]

    # =========================
    # XGBoost 파라미터 준비
    # =========================
    params = XGB_DEFAULT_PARAMS.copy()
    params["num_class"] = len(le.classes_)

    # =========================
    # Stratified K-Fold
    # =========================
    skf = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_SEED,
    )

    cv_accuracies = []

    log(f"[{chunk_size}] {N_SPLITS}-Fold CV 시작")

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), start=1):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # ⚠️ Fold마다 fresh 모델 생성 (여기서가 핵심 수정 포인트)
        model = xgb.XGBClassifier(
            **params,
            random_state=RANDOM_SEED,
            use_label_encoder=False,
        )

        model.fit(X_train, y_train)

        y_pred = model.predict(X_val)
        acc = accuracy_score(y_val, y_pred)
        cv_accuracies.append(acc)

        log(f"[{chunk_size}] Fold {fold} ACC = {acc:.4f}")

    log(
        f"[{chunk_size}] CV 평균 ACC = {np.mean(cv_accuracies):.4f}, "
        f"표준편차 = {np.std(cv_accuracies):.4f}"
    )

    # =========================
    # 전체 train_val로 최종 모델 학습
    # =========================
    log(f"[{chunk_size}] 전체 train_val 데이터로 최종 모델 학습")

    final_model = xgb.XGBClassifier(
        **params,
        random_state=RANDOM_SEED,
        use_label_encoder=False,
    )
    final_model.fit(X, y)

    # 모델 및 LabelEncoder 저장
    model_path = MODELS_DIR / f"{chunk_size}.xgb"
    final_model.save_model(model_path)
    log(f"[{chunk_size}] 모델 저장: {model_path}")

    le_path = MODELS_DIR / f"{chunk_size}_label_encoder.pkl"
    with open(le_path, "wb") as f:
        pickle.dump(le, f)
    log(f"[{chunk_size}] LabelEncoder 저장: {le_path}")


def run_all_train() -> None:
    for cs in CHUNK_SIZES:
        train_for_chunk(cs)


if __name__ == "__main__":
    # 모듈 단독 실행 테스트용
    run_all_train()
