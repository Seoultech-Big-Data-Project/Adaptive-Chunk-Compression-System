import math

# =========================================
# 1. Shannon Entropy (미세 최적화)
# =========================================
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
    mv = memoryview(data)  # bytes 직접 순회보다 약간 빠름
    for b in mv:
        counts[b] += 1

    length = float(len(mv))
    inv_len = 1.0 / length

    ent = 0.0
    for c in counts:
        if c:
            p = c * inv_len
            ent -= p * math.log2(p)

    return ent


# =========================================
# 2. byte_stats: byte_std, byte_max만
# =========================================
def byte_stats(data: bytes):
    """
    바이트 값의 기본 통계량 중
    - 표준편차(byte_std)
    - 최대값(byte_max)
    만 계산 (byte_mean, byte_min은 버렸음)

    반환: (byte_std, byte_max)
    """
    if not data:
        return 0.0, 0  # std, max

    length = len(data)

    # 한 번의 루프로 합/제곱합/최대값 계산
    s = 0
    ssq = 0.0
    max_b = 0

    mv = memoryview(data)
    for b in mv:
        s += b
        ssq += b * b
        if b > max_b:
            max_b = b

    mean = s / length
    # Var(X) = E[X^2] - (E[X])^2
    var = ssq / length - mean * mean
    if var < 0.0:  # 부동소수점 오차 방어
        var = 0.0
    std = math.sqrt(var)

    return std, max_b   # (byte_std, byte_max)


# =========================================
# 3. run_length_stats: num_runs, run_mean, run_std만
# =========================================
def run_length_stats(data: bytes):
    """
    연속된 동일 바이트의 길이(Run Length) 통계 계산

    - num_runs: 연속 구간 개수
    - run_mean: 평균 run 길이
    - run_std: run 길이 표준편차

    run_max는 상관계수 낮아서 제거했으므로 계산 안 함.
    반환: (num_runs, run_mean, run_std)
    """
    if not data:
        return 0, 0.0, 0.0

    mv = memoryview(data)

    prev = mv[0]
    current_len = 1

    num_runs = 0
    sum_len = 0
    sum_sq_len = 0

    # runs 리스트를 만들지 않고, 합/제곱합만 누적
    for b in mv[1:]:
        if b == prev:
            current_len += 1
        else:
            num_runs += 1
            sum_len += current_len
            sum_sq_len += current_len * current_len
            prev = b
            current_len = 1

    # 마지막 run 반영
    num_runs += 1
    sum_len += current_len
    sum_sq_len += current_len * current_len

    mean = sum_len / num_runs
    var = sum_sq_len / num_runs - mean * mean
    if var < 0.0:
        var = 0.0
    std = math.sqrt(var)

    return num_runs, mean, std
