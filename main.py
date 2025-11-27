# main.py
import argparse

from pipeline.preprocess import run_all_preprocess
from pipeline.trainer import run_all_train
from pipeline.benchmark import run_all_benchmarks
from pipeline.common import log


def main():
    parser = argparse.ArgumentParser(description="Compression codec selection pipeline")
    parser.add_argument(
        "--stage",
        choices=["preprocess", "train", "benchmark", "all"],
        default="all",
        help="실행할 단계 선택",
    )
    args = parser.parse_args()

    if args.stage in ("preprocess", "all"):
        log("==== [1/3] Preprocess 단계 시작 ====")
        run_all_preprocess()

    if args.stage in ("train", "all"):
        log("==== [2/3] Train 단계 시작 ====")
        run_all_train()

    if args.stage in ("benchmark", "all"):
        log("==== [3/3] Benchmark 단계 시작 ====")
        run_all_benchmarks()

    log("==== 파이프라인 실행 완료 ====")


if __name__ == "__main__":
    main()
