# train_model.py

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split

from config import (
    CHUNK_SIZES,
    LABEL_CONFIG,
    TRAIN_RATIO,
    VAL_RATIO,
    TEST_RATIO,
    RANDOM_SEED,
    MODELS_DIR,
    CODECS,
    USE_XGBOOST_GPU,
    XGBOOST_PARAMS,
)
from utils import preprocessed_csv_path, get_feature_columns, save_json

try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None


def get_model_for_num_classes(num_classes: int):
    """XGBoost 기반 멀티클래스 분류 모델 생성."""
    if XGBClassifier is None:
        raise ImportError("xgboost가 설치되어 있지 않습니다. `pip install xgboost` 후 다시 시도하세요.")

    params = XGBOOST_PARAMS.copy()
    params["num_class"] = num_classes

    if USE_XGBOOST_GPU:
        params["tree_method"] = "gpu_hist"
        params["predictor"] = "gpu_predictor"
    else:
        params["tree_method"] = "hist"

    model = XGBClassifier(**params)
    return model


def train_for_chunk_size(chunk_size: str) -> None:
    print(f"[train] chunk_size = {chunk_size}")
    csv_path = preprocessed_csv_path(chunk_size)
    print(f"  - loading preprocessed: {csv_path}")
    df = pd.read_csv(csv_path)

    label_col = LABEL_CONFIG["column"]
    if label_col not in df.columns:
        raise ValueError(f"preprocessed CSV({csv_path})에 '{label_col}' 컬럼이 없습니다.")

    if "label_codec_id" in df.columns:
        y = df["label_codec_id"].to_numpy()
    else:
        # 혹시 label_codec_id가 없으면, label_col을 직접 카테고리 인덱스로 변환
        classes = sorted(df[label_col].unique())
        class_to_id = {c: i for i, c in enumerate(classes)}
        y = df[label_col].map(class_to_id).to_numpy()

    feature_cols = get_feature_columns(df)
    print(f"  - feature columns ({len(feature_cols)}): {feature_cols}")
    X = df[feature_cols].to_numpy()

    # -------------------------
    # train / val / test split
    # -------------------------
    if not np.isclose(TRAIN_RATIO + VAL_RATIO + TEST_RATIO, 1.0):
        raise ValueError("TRAIN_RATIO + VAL_RATIO + TEST_RATIO != 1.0 입니다. config.py를 확인하세요.")

    # 먼저 train+val vs test
    X_temp, X_test, y_temp, y_test, idx_temp, idx_test = train_test_split(
        X,
        y,
        df.index.to_numpy(),
        test_size=TEST_RATIO,
        random_state=RANDOM_SEED,
        stratify=y,
    )

    # 그 다음 train vs val
    val_ratio_adjusted = VAL_RATIO / (TRAIN_RATIO + VAL_RATIO)
    X_train, X_val, y_train, y_val, idx_train, idx_val = train_test_split(
        X_temp,
        y_temp,
        idx_temp,
        test_size=val_ratio_adjusted,
        random_state=RANDOM_SEED,
        stratify=y_temp,
    )

    print(f"  - split sizes: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")

    # -------------------------
    # 모델 학습 (XGBoost)
    # -------------------------
    num_classes = len(CODECS)
    model = get_model_for_num_classes(num_classes)

    print("  - training model...")
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    # -------------------------
    # 성능 측정
    # -------------------------
    def eval_split(name, X_split, y_split):
        y_pred = model.predict(X_split)
        acc = accuracy_score(y_split, y_pred)
        prec, rec, f1, _ = precision_recall_fscore_support(
            y_split, y_pred, average="macro", zero_division=0
        )
        print(f"    [{name}] acc={acc:.4f}, prec={prec:.4f}, rec={rec:.4f}, f1={f1:.4f}")
        return {
            "accuracy": float(acc),
            "precision_macro": float(prec),
            "recall_macro": float(rec),
            "f1_macro": float(f1),
        }

    print("  - evaluating...")
    metrics_train = eval_split("train", X_train, y_train)
    metrics_val = eval_split("val", X_val, y_val)
    metrics_test = eval_split("test", X_test, y_test)

    # -------------------------
    # split 정보 df에 기록 & 저장
    # -------------------------
    df["split"] = "train"
    df.loc[idx_val, "split"] = "val"
    df.loc[idx_test, "split"] = "test"

    print(f"  - saving updated preprocessed with split: {csv_path}")
    df.to_csv(csv_path, index=False)

    # -------------------------
    # 모델 및 metric 저장
    # -------------------------
    model_dir = MODELS_DIR / chunk_size
    model_dir.mkdir(parents=True, exist_ok=True)

    model_path = model_dir / "model.pkl"
    print(f"  - saving model: {model_path}")
    joblib.dump(model, model_path)

    metrics_path = model_dir / "metrics.json"
    metrics_obj = {
        "chunk_size": chunk_size,
        "n_train": int(len(X_train)),
        "n_val": int(len(X_val)),
        "n_test": int(len(X_test)),
        "train": metrics_train,
        "val": metrics_val,
        "test": metrics_test,
        "classes": CODECS,
    }
    print(f"  - saving metrics: {metrics_path}")
    save_json(metrics_obj, metrics_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--chunk-size",
        type=str,
        default=None,
        help="특정 청크 사이즈만 학습하고 싶을 때 (예: 1MB). 지정하지 않으면 CHUNK_SIZES 전체 수행.",
    )
    args = parser.parse_args()

    if args.chunk_size:
        train_for_chunk_size(args.chunk_size)
    else:
        for cs in CHUNK_SIZES:
            train_for_chunk_size(cs)


if __name__ == "__main__":
    main()
