# pipeline/chunking.py
import sys
from pathlib import Path

from .common import RAW_DIR, get_chunk_dir


def list_raw_files() -> list[Path]:
    """
    raw/ 디렉토리 안의 모든 '파일' 리스트 반환.
    """
    files = [p for p in RAW_DIR.iterdir() if p.is_file()]
    files.sort()
    return files


def split_all_raw_to_chunks(chunk_size_mb: int) -> None:
    """
    raw/ 안에 있는 모든 파일을 chunk_size_mb 단위로 잘라서
    chunk/{chunk_size_mb}MB/ 밑에 0.bin, 1.bin, 2.bin ... 형태로 저장.
    """
    chunk_size_bytes = chunk_size_mb * 1024 * 1024
    chunk_dir = get_chunk_dir(chunk_size_mb)

    raw_files = list_raw_files()
    if not raw_files:
        print(f"[chunking] raw 디렉토리에 파일이 없음: {RAW_DIR}")
        return

    print(f"[chunking] RAW_DIR = {RAW_DIR}")
    print(f"[chunking] 대상 파일 = {[p.name for p in raw_files]}")
    print(f"[chunking] chunk_size = {chunk_size_mb}MB ({chunk_size_bytes} bytes)")
    print(f"[chunking] output_dir = {chunk_dir}")

    global_idx = 0  # 모든 파일이 공유하는 연속 인덱스

    for file_path in raw_files:
        file_size = file_path.stat().st_size
        print(f"  [file] {file_path.name} ({file_size} bytes) -> 청크 분할 중...")

        with file_path.open("rb") as f:
            while True:
                data = f.read(chunk_size_bytes)
                if not data:
                    break

                # 파일 이름을 global_idx로 설정: 0.bin, 1.bin, ...
                out_path = chunk_dir / f"{global_idx}.bin"
                with out_path.open("wb") as out_f:
                    out_f.write(data)

                global_idx += 1
                if global_idx % 100 == 0:
                    print(f"    - 현재까지 생성된 청크: {global_idx}개")

    print(f"[chunking] 완료. 총 생성된 청크 수 = {global_idx}")
    print(f"[chunking] 저장 위치: {chunk_dir}")


if __name__ == "__main__":
    # 사용법:
    #   python -m pipeline.chunking
    #   python -m pipeline.chunking 2   # 2MB 기준
    if len(sys.argv) >= 2:
        size_mb = int(sys.argv[1])
    else:
        size_mb = 1

    split_all_raw_to_chunks(chunk_size_mb=size_mb)
