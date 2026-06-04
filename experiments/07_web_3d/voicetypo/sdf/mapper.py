"""모음 공간 위치 → SDF 글리프 모핑 매핑.

blend_weights에서 상위 2개 모음을 뽑아
src/dst/t를 계산하여 기존 SDF 엔진에 전달.
"""


def weights_to_morph(weights, glyph_data):
    """소프트 가중치 → (src_data, dst_data, t).

    Args:
        weights: dict[str, float] — 모음별 가중치 (합=1)
        glyph_data: dict[str, GlyphData] — 모음별 글리프 데이터

    Returns:
        (src_data, dst_data, t) — _blend_glyphs()에 바로 전달 가능.
        weights가 비어있으면 None 반환.
    """
    if not weights:
        return None

    # glyph_data에 있는 모음만 필터
    valid = {k: v for k, v in weights.items() if k in glyph_data}
    if not valid:
        return None

    top2 = sorted(valid.items(), key=lambda x: x[1], reverse=True)[:2]

    if len(top2) == 1:
        return (glyph_data[top2[0][0]], glyph_data[top2[0][0]], 0.0)

    src_name, src_w = top2[0]
    dst_name, dst_w = top2[1]
    t = dst_w / (src_w + dst_w + 1e-9)

    return (glyph_data[src_name], glyph_data[dst_name], t)
