"""
feature 추출 스크립트

추출하는 Feature:
  1. 기본 통계: 엔트로피, 바이트 평균/표준편차/최소/최대
  2. 바이트 분포: 0x00, 0xFF, ASCII, 제어문자 비율
  3. 텍스트/인코딩 패턴: 줄바꿈, 공백, JSON 괄호/구두점, base64, hex, non-ASCII 문자 비율
  4. 압축 성능: LZ4, Snappy, ZSTD의 압축 크기/시간/압축률
"""

import csv
from tqdm import tqdm

from config import RAW_DIR, FEATURE_EXTRACT_CHUNK_SIZES, DATA_DIR
from utils import process_file


def main():
    """각 청크 크기별로 별도의 CSV 파일을 생성 (data/1MB.csv, data/4MB.csv, ...)"""
    
    DATA_DIR.mkdir(exist_ok=True)
    
    file_list = [path for path in RAW_DIR.rglob("*") if path.is_file()]
    
    if not file_list:
        print("No data found in raw/.")
        return
    
    print(f"Found {len(file_list)} file(s) in {RAW_DIR}\n")
    
    # 각 청크 크기별로 처리
    for target_chunk_size, csv_path in tqdm(FEATURE_EXTRACT_CHUNK_SIZES.items(), desc="Overall Progress", unit="chunk_size"):
        chunk_size_mb = target_chunk_size // (1024*1024)
        print(f"\n{'='*60}")
        print(f"Processing chunk size: {chunk_size_mb}MB")
        print(f"{'='*60}")
        
        all_rows = []
        
        # 각 파일을 해당 청크 크기로 처리
        for file_path in tqdm(file_list, desc=f"  {chunk_size_mb}MB chunks", unit="file"):
            rows = process_file(file_path, target_chunk_size)
            all_rows.extend(rows)
        
        if not all_rows:
            print(f"  No chunks extracted for {chunk_size_mb}MB")
            continue
        
        # CSV 파일에 저장
        file_exists = csv_path.exists()
        
        fieldnames = list(all_rows[0].keys())
        mode = "a" if file_exists else "w"
        
        with csv_path.open(mode, newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)   

            if not file_exists:
                writer.writeheader()

            writer.writerows(all_rows)
        
        action = "Appended to" if file_exists else "Created"
        print(f"\n  ✓ {action} {csv_path}")
        print(f"    Total rows: {len(all_rows)}\n")


if __name__ == "__main__":
    print("="*60)
    print("Adaptive Chunk Compression System - Feature Extraction")
    print("="*60)
    main()
    print("\n" + "="*60)
    print("Feature extraction completed!")
    print("="*60)