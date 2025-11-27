"""
콘텐츠 패턴 특성 추출
바이트 타입별 비율 계산
"""


def proportion_features(data: bytes):
    """
    바이트 타입별 비율(proportion) 특성 추출
    
    추출하는 비율:
    - prop_zero: 0x00 바이트 비율 (null 바이트, 바이너리 데이터에 많음)
    - prop_ff: 0xFF 바이트 비율 (초기화되지 않은 메모리 등)
    - prop_ascii: 출력 가능한 ASCII 문자 비율 (0x20-0x7E, 텍스트 데이터)
    - prop_control: 제어 문자 비율 (0x00-0x1F, 0x7F)
    - prop_newline: 줄바꿈 문자 비율 (텍스트 파일)
    - prop_space: 공백 문자 비율
    - prop_json_brace: JSON 괄호 비율 ({, }, [, ], JSON 데이터)
    - prop_json_punct: JSON 구두점 비율 (:, ,, ", JSON 데이터)
    - prop_base64_charset: Base64 문자 비율 (A-Z, a-z, 0-9, +, /, =)
    - prop_hex_charset: 16진수 문자 비율 (0-9, A-F, a-f)
    - prop_printable_non_ascii: ASCII 범위 밖의 출력 가능 문자 비율 (0x80-0xFF)
    """
    length = len(data)
    if length == 0:
        return {
            "prop_zero": 0.0,
            "prop_ff": 0.0,
            "prop_ascii": 0.0,
            "prop_control": 0.0,
            "prop_newline": 0.0,
            "prop_space": 0.0,
            "prop_json_brace": 0.0,
            "prop_json_punct": 0.0,
            "prop_base64_charset": 0.0,
            "prop_hex_charset": 0.0,
            "prop_printable_non_ascii": 0.0,
        }

    zero = ff = ascii_ = control = 0
    newline = space = 0
    json_brace = json_punct = 0
    base64_ch = hex_ch = printable_non_ascii = 0

    # 모든 바이트를 순회하며 각 특성에 해당하는지 검사
    for b in data:
        # === 특수 바이트 값 ===
        if b == 0x00:  # NULL 바이트
            zero += 1
        if b == 0xFF:  # 최대값 바이트
            ff += 1

        # === ASCII 문자 범위 ===
        if 0x20 <= b <= 0x7E:  # 출력 가능한 ASCII (공백~틸드)
            ascii_ += 1

        if (0x00 <= b <= 0x1F) or b == 0x7F:  # 제어 문자
            control += 1

        # === 텍스트/JSON 특성 ===
        if b == 0x0A:  # 줄바꿈 (\n)
            newline += 1
        if b == 0x20:  # 공백 (space)
            space += 1
        if b in (ord('{'), ord('}'), ord('['), ord(']')):  # JSON 배열/객체 괄호
            json_brace += 1
        if b in (ord(':'), ord(','), ord('"')):  # JSON 구두점
            json_punct += 1

        # === Base64 인코딩 문자 집합 ===
        # Base64는 A-Z, a-z, 0-9, +, /, = 만 사용
        if (
            (ord('A') <= b <= ord('Z')) or
            (ord('a') <= b <= ord('z')) or
            (ord('0') <= b <= ord('9')) or
            b in (ord('+'), ord('/'), ord('='))
        ):
            base64_ch += 1

        # === 16진수(Hexadecimal) 문자 집합 ===
        # Hex는 0-9, A-F, a-f 만 사용
        if (
            (ord('0') <= b <= ord('9')) or
            (ord('A') <= b <= ord('F')) or
            (ord('a') <= b <= ord('f'))
        ):
            hex_ch += 1

        # === Non-ASCII 출력 가능 문자 ===
        # 0x80-0xFF 범위 (확장 ASCII, UTF-8 멀티바이트 등)
        if 0x80 <= b <= 0xFF:
            printable_non_ascii += 1

    length_f = float(length)
    return {
        "prop_zero": zero / length_f,                          # NULL 바이트 비율
        "prop_ff": ff / length_f,                              # 0xFF 바이트 비율
        "prop_ascii": ascii_ / length_f,                       # ASCII 문자 비율
        "prop_control": control / length_f,                    # 제어 문자 비율
        "prop_newline": newline / length_f,                    # 줄바꿈 비율
        "prop_space": space / length_f,                        # 공백 비율
        "prop_json_brace": json_brace / length_f,              # JSON 괄호 비율
        "prop_json_punct": json_punct / length_f,              # JSON 구두점 비율
        "prop_base64_charset": base64_ch / length_f,           # Base64 문자 비율
        "prop_hex_charset": hex_ch / length_f,                 # Hex 문자 비율
        "prop_printable_non_ascii": printable_non_ascii / length_f,  # Non-ASCII 비율
    }
