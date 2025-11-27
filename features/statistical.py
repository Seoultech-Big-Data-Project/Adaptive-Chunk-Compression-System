"""
통계적 특성 추출 함수들
-샤논 엔트로피, 바이트 통계, Run Length 통계
"""

import math


def shannon_entropy(data: bytes) -> float:
    """
    샤논 엔트로피 계산(데이터 무작위성/복잡도 측정)
    
    - 값이 높을수록 (최대 8.0): 데이터가 무작위적이고 압축이 어려움
    - 값이 낮을수록 (최소 0.0): 데이터가 반복적이고 압축이 쉬움
    """
    if not data:
        return 0.0
    
    # 각 바이트 값(0-255)의 출현 횟수 카운트
    counts = [0] * 256
    for b in data:
        counts[b] += 1
    
    # 샤논 엔트로피 공식: H = -Σ(p * log2(p))
    length = len(data)
    ent = 0.0
    for c in counts:
        if c == 0:
            continue
        p = c / length  # 확률 계산
        ent -= p * math.log2(p)
    
    return ent


def byte_stats(data: bytes):
    """
    바이트 값의 기본 통계량 계산

    - 평균: 바이트 값의 중심 경향성
    - 표준편차: 바이트 값의 분산 정도
    - 최소/최대: 바이트 값의 범위
    """
    if not data:
        return 0.0, 0.0, 0, 0
    
    length = len(data)
    s = sum(data)  # 모든 바이트 값의 합
    mean = s / length  # 평균 계산
    
    # 분산 = Σ(x - 평균)² / n
    var = sum((b - mean) ** 2 for b in data) / length
    std = math.sqrt(var)  # 표준편차 = √분산
    
    return mean, std, min(data), max(data)


def run_length_stats(data: bytes):
    """
    연속된 동일 바이트의 길이(Run Length) 통계 계산

    - num_runs가 적을수록: 데이터가 반복적 (압축 효율 높음)
    - run_mean이 클수록: 긴 반복 구간 존재
    - run_max: 최대 연속 반복 길이
    """
    if not data:
        return 0, 0.0, 0.0, 0

    # 연속된 동일 바이트의 길이를 저장
    runs = []
    prev = data[0]
    current_len = 1
    
    for b in data[1:]:
        if b == prev:  # 같은 바이트가 계속되면
            current_len += 1
        else:  # 다른 바이트가 나오면
            runs.append(current_len)
            prev = b
            current_len = 1
    runs.append(current_len)  # 마지막 run 추가

    # run 길이에 대한 통계 계산
    num_runs = len(runs)
    mean = sum(runs) / num_runs
    var = sum((r - mean) ** 2 for r in runs) / num_runs
    std = math.sqrt(var)
    max_run = max(runs)
    
    return num_runs, mean, std, max_run
