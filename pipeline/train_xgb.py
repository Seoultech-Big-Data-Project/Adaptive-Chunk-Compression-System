# pipeline/train_xgb.py
from __future__ import annotations

import json
import sys
from typing import Tuple

import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
)
from sklearn.preprocessing import LabelEncoder

from .common import DATA_DIR, MODELS_DIR, RESULTS_DIR


FEATURE_DROP_COLS = ["best_codec", "chunk_idx"]  # chunk_idx도 학습에서는 버림
LABEL_COL = "best_codec"
CODECS = ["zstd", "lz4", "snappy"]  # 기대 클래스 순서 (편의용)


def load_split(chunk_size_mb: int) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    data/{chunk_size}MB/train.csv, val.csv, test.csv 로부터 데이터셋을 로드한다.
    """
    base_dir = DATA_DIR / f"{chunk_size_mb}MB"

    train_path = base_dir / "train.csv"
    val_path = base_dir / "val.csv"
    test_path = base_dir / "test.csv"

    print(f"[train_xgb] loading train: {train_path}")
    print(f"[train_xgb] loading val:   {val_path}")
    print(f"[train_xgb] loading test:  {test_path}")

    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)

    return train_df, val_df, test_df


def split_X_y(df: pd.DataFrame):
    """
    DataFrame에서 X (피쳐), y (라벨), feature_names를 분리.
    """
    if LABEL_COL not in df.columns:
        raise ValueError(f"라벨 컬럼({LABEL_COL})이 없습니다: {df.columns}")

    y = df[LABEL_COL].astype(str)

    # 학습에 사용하지 않을 컬럼 제거
    feature_cols = [c for c in df.columns if c not in FEATURE_DROP_COLS]
    X = df[feature_cols]

    return X, y, feature_cols


def train_xgb_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    num_classes: int,
) -> Tuple[XGBClassifier, LabelEncoder, dict]:
    """
    XGBoost 멀티클래스 분류 모델 학습.
    - train으로 학습, val로 early stopping
    - LabelEncoder로 문자열 라벨을 정수로 변환
    - 학습 로그(예: best_iteration 등)를 info로 반환
    """
    # 문자열 라벨 -> 정수 라벨
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_val_enc = le.transform(y_val)

    # XGBoost 모델 설정 (필요하면 파라미터 조정)
    model = XGBClassifier(
        objective="multi:softprob",
        num_class=num_classes,
        tree_method="hist",       # CPU에서도 빠르게
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric="mlogloss",
    )

    eval_set = [(X_train, y_train_enc), (X_val, y_val_enc)]

    print("[train_xgb] training XGBoost model...")
    model.fit(
        X_train,
        y_train_enc,
        eval_set=eval_set,
        verbose=False,
    )

    info = {
        "best_iteration": int(model.best_iteration) if hasattr(model, "best_iteration") else None,
        "classes": list(le.classes_),
    }

    return model, le, info


def evaluate_model(
    model: XGBClassifier,
    le: LabelEncoder,
    X: pd.DataFrame,
    y_true: pd.Series,
) -> dict:
    """
    모델을 X, y_true에 대해 평가하고
    accuracy, per-class precision/recall/f1, confusion matrix를 dict로 반환.
    """
    y_true_enc = le.transform(y_true.astype(str))
    y_pred_enc = model.predict(X)

    acc = float(accuracy_score(y_true_enc, y_pred_enc))

    # average=None -> 클래스별
    prec, rec, f1, support = precision_recall_fscore_support(
        y_true_enc, y_pred_enc, labels=range(len(le.classes_)), zero_division=0
    )

    cm = confusion_matrix(y_true_enc, y_pred_enc, labels=range(len(le.classes_)))

    # dict로 정리
    metrics = {
        "accuracy": acc,
        "per_class": {},
        "confusion_matrix": cm.tolist(),
        "support_total": int(support.sum()),
    }

    for idx, cls_name in enumerate(le.classes_):
        metrics["per_class"][cls_name] = {
            "precision": float(prec[idx]),
            "recall": float(rec[idx]),
            "f1": float(f1[idx]),
            "support": int(support[idx]),
        }

    return metrics


def train_and_evaluate_xgb_for_chunk(chunk_size_mb: int = 1) -> None:
    """
    전체 파이프라인:
      1) train/val/test CSV 로드
      2) X, y 분리 + LabelEncoder
      3) XGBoost 학습 (train + val 사용, val은 early stopping)
      4) train/val/test 평가
      5) 모델 저장 (models/{chunk_size}MB_xgb.json)
      6) 지표 저장 (results/{chunk_size}MB_xgb_metrics.json)
    """
    # 1. 데이터 로드
    train_df, val_df, test_df = load_split(chunk_size_mb)

    # 2. X, y 분리
    X_train, y_train, feature_cols = split_X_y(train_df)
    X_val, y_val, _ = split_X_y(val_df)
    X_test, y_test, _ = split_X_y(test_df)

    print(f"[train_xgb] feature columns: {feature_cols}")
    print(f"[train_xgb] train size: {len(train_df)}, val size: {len(val_df)}, test size: {len(test_df)}")

    # 3. 모델 학습
    num_classes = len(np.unique(y_train))
    model, le, train_info = train_xgb_model(X_train, y_train, X_val, y_val, num_classes)

    # 4. 평가 (train/val/test)
    print("\n[train_xgb] evaluating on train set...")
    train_metrics = evaluate_model(model, le, X_train, y_train)
    print(f"  - train accuracy: {train_metrics['accuracy']:.4f}")

    print("[train_xgb] evaluating on val set...")
    val_metrics = evaluate_model(model, le, X_val, y_val)
    print(f"  - val accuracy:   {val_metrics['accuracy']:.4f}")

    print("[train_xgb] evaluating on test set...")
    test_metrics = evaluate_model(model, le, X_test, y_test)
    print(f"  - test accuracy:  {test_metrics['accuracy']:.4f}")

    # 클래스별 지표도 콘솔에 보기 좋게 출력
    def print_per_class(title: str, metrics: dict):
        print(f"\n[{title}] per-class metrics")
        for cls_name, m in metrics["per_class"].items():
            print(
                f"  - {cls_name:7s} | "
                f"precision={m['precision']:.3f}, "
                f"recall={m['recall']:.3f}, "
                f"f1={m['f1']:.3f}, "
                f"support={m['support']}"
            )

    print_per_class("train", train_metrics)
    print_per_class("val", val_metrics)
    print_per_class("test", test_metrics)

    # 5. 모델 저장
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / f"{chunk_size_mb}MB_xgb.json"
    model.save_model(model_path)
    print(f"\n[train_xgb] model saved to: {model_path}")

    # 6. 지표 저장 (JSON)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    metrics_path = RESULTS_DIR / f"{chunk_size_mb}MB_xgb_metrics.json"

    all_metrics = {
        "chunk_size_mb": chunk_size_mb,
        "classes": list(le.classes_),
        "feature_columns": feature_cols,
        "train_info": train_info,
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
    }

    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(all_metrics, f, ensure_ascii=False, indent=2)

    print(f"[train_xgb] metrics saved to: {metrics_path}")


if __name__ == "__main__":
    # 사용법:
    #   python -m pipeline.train_xgb        -> 1MB 기준
    #   python -m pipeline.train_xgb 2      -> 2MB 기준
    if len(sys.argv) >= 2:
        size_mb = int(sys.argv[1])
    else:
        size_mb = 1

    train_and_evaluate_xgb_for_chunk(chunk_size_mb=size_mb)
