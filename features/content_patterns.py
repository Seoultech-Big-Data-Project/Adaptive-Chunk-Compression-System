import numpy as np

# (선택) 전역에 미리 ord 값 캐시해두면 아주 약간 더 이득
O_A, O_Z = ord('A'), ord('Z')
O_a, O_z = ord('a'), ord('z')
O_0, O_9 = ord('0'), ord('9')
O_BRACE = [ord('{'), ord('}'), ord('['), ord(']')]
O_PUNCT = [ord(':'), ord(','), ord('"')]
O_PLUS, O_SLASH, O_EQ = ord('+'), ord('/'), ord('=')
O_F, O_f = ord('F'), ord('f')


def proportion_features(data: bytes):
    """
    바이트 타입별 비율(proportion) 특성 추출 (bincount 기반 최적화 버전)
    """
    length = len(data)
    if length == 0:
        return {
            "prop_newline": 0.0,
            "prop_space": 0.0,
            "prop_json_brace": 0.0,
            "prop_json_punct": 0.0,
            "prop_base64_charset": 0.0,
            "prop_hex_charset": 0.0,
        }

    # bytes → uint8 배열 (복사 X)
    arr = np.frombuffer(data, dtype=np.uint8)

    # 0~255 각 값이 몇 번 나왔는지 한 번에 카운트
    # → 여기서만 전체 배열 한 번 훑음
    counts = np.bincount(arr, minlength=256)

    # 단일 값
    newline = counts[0x0A]
    space   = counts[0x20]

    # JSON 괄호: { } [ ]
    json_brace = counts[O_BRACE].sum()

    # JSON 구두점: : , "
    json_punct = counts[O_PUNCT].sum()

    # Base64: A-Z, a-z, 0-9, +, /, =
    base64_ch = (
        counts[O_A:O_Z + 1].sum() +
        counts[O_a:O_z + 1].sum() +
        counts[O_0:O_9 + 1].sum() +
        counts[[O_PLUS, O_SLASH, O_EQ]].sum()
    )

    # Hex: 0-9, A-F, a-f
    hex_ch = (
        counts[O_0:O_9 + 1].sum() +
        counts[O_A:O_F + 1].sum() +
        counts[O_a:O_f + 1].sum()
    )

    length_f = float(length)
    return {
        "prop_newline": newline / length_f,
        "prop_space": space / length_f,
        "prop_json_brace": json_brace / length_f,
        "prop_json_punct": json_punct / length_f,
        "prop_base64_charset": base64_ch / length_f,
        "prop_hex_charset": hex_ch / length_f,
    }
