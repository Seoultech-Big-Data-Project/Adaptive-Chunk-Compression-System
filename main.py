# main.py

import argparse

from config import CHUNK_SIZES
from preprocess import preprocess_one_chunk_size
from train_model import train_for_chunk_size
from benchmark import benchmark_for_chunk_size


def run_pipeline_for_chunk_size(chunk_size: str, do_preprocess: bool, do_train: bool, do_benchmark: bool):
    print(f"\n=== PIPELINE START: chunk_size = {chunk_size} ===")

    if do_preprocess:
        print(f"[main] preprocess 단계 실행 (chunk_size={chunk_size})")
        preprocess_one_chunk_size(chunk_size)
    else:
        print(f"[main] preprocess 단계 건너뜀 (chunk_size={chunk_size})")

    if do_train:
        print(f"[main] train 단계 실행 (chunk_size={chunk_size})")
        train_for_chunk_size(chunk_size)
    else:
        print(f"[main] train 단계 건너뜀 (chunk_size={chunk_size})")

    if do_benchmark:
        print(f"[main] benchmark 단계 실행 (chunk_size={chunk_size})")
        benchmark_for_chunk_size(chunk_size)
    else:
        print(f"[main] benchmark 단계 건너뜀 (chunk_size={chunk_size})")

    print(f"=== PIPELINE END: chunk_size = {chunk_size} ===\n")


def main():
    parser = argparse.ArgumentParser(
        description="압축 코덱 추천 전체 파이프라인 실행 스크립트"
    )
    parser.add_argument(
        "--chunk-size",
        type=str,
        default=None,
        help="특정 청크 사이즈만 실행하고 싶을 때 (예: 1MB). 지정하지 않으면 config.CHUNK_SIZES 전체 실행.",
    )
    parser.add_argument(
        "--no-preprocess",
        action="store_true",
        help="preprocess 단계 건너뛰기 (이미 preprocessed/*.csv 있는 경우)",
    )
    parser.add_argument(
        "--no-train",
        action="store_true",
        help="train 단계 건너뛰기 (이미 models/*/model.pkl 있는 경우)",
    )
    parser.add_argument(
        "--no-benchmark",
        action="store_true",
        help="benchmark 단계 건너뛰기",
    )

    args = parser.parse_args()

    do_preprocess = not args.no_preprocess
    do_train = not args.no_train
    do_benchmark = not args.no_benchmark

    if args.chunk_size:
        # 특정 청크 사이즈만 실행
        run_pipeline_for_chunk_size(args.chunk_size, do_preprocess, do_train, do_benchmark)
    else:
        # config.CHUNK_SIZES 전체에 대해 실행
        for cs in CHUNK_SIZES:
            run_pipeline_for_chunk_size(cs, do_preprocess, do_train, do_benchmark)


if __name__ == "__main__":
    main()
