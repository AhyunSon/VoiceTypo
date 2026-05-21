"""한글 글리프 SDF 모핑 — 태그 기반 획 매칭, translate 모핑

핵심 규칙:
- 자음(초성): SDF 선형 블렌딩
- 모음(중성): (orientation, length_class) 태그로 공통 획 판단
  - 공통 획(같은 태그): SDF shift + union 으로 translate 모핑
  - 동일 태그 2개: 왼쪽=base, 오른쪽=extra → base에서 분기/흡수
  - 비공통 획: base_main 방향으로 이동하며 흡수 / base_main에서 출발하며 분리

SDF 값 lerp 사용 안 함 (cho 제외). 각 획을 개체 단위로 shift/warp.

키 1-7: 목표 모음 선택, ESC: 종료
"""

import os, sys, math, re, time
import numpy as np
from collections import defaultdict
from scipy.ndimage import distance_transform_edt
import xml.etree.ElementTree as ET
from typing import Dict, List, Tuple, Optional

from PySide6.QtCore import Qt, QTimer, QRectF
from PySide6.QtGui import QPainter, QPainterPath, QColor, QImage, QPen, QFont
from PySide6.QtWidgets import QApplication, QWidget

# ── Types ──
Point = Tuple[float, float]
Contour = List[Point]
Shape = List[Contour]

# ── Parameters ──
GRID = 512
MORPH_SEC = 0.45
FPS = 60

AA_WIDTH = 1.5
ABSORB_K = 15.0         # 흡수/방출 시 SDF shrink 오프셋
ABSORB_DELAY = 0.3      # 흡수 시작 시점 (s 기준, 0~1)
EMIT_END = 0.7          # 방출 완료 시점 (s 기준, 0~1)

# ── 사전 계산 좌표 그리드 (warp 용) ──
_YY, _XX = np.mgrid[0:GRID, 0:GRID].astype(np.float32)

VOWELS = ["아", "이", "우", "에", "오", "으", "어"]


# ═══════════════════════════════════════════════════
#  Geometry
# ═══════════════════════════════════════════════════
def _lerp(a, b, t):
    return a + (b - a) * t

def _lerp_pt(p, q, t):
    return (_lerp(p[0], q[0], t), _lerp(p[1], q[1], t))

def _dist(p, q):
    return math.hypot(p[0] - q[0], p[1] - q[1])

def _poly_area(c):
    n = len(c)
    if n < 3:
        return 0.0
    return 0.5 * sum(
        c[i][0] * c[(i + 1) % n][1] - c[(i + 1) % n][0] * c[i][1]
        for i in range(n)
    )

def _centroid(c):
    n = len(c)
    if n == 0:
        return (0.0, 0.0)
    return (sum(p[0] for p in c) / n, sum(p[1] for p in c) / n)

def _bbox(c):
    xs = [p[0] for p in c]
    ys = [p[1] for p in c]
    return (min(xs), min(ys), max(xs), max(ys))

def _point_in_poly(pt, poly):
    x, y = pt
    inside = False
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        if (y1 > y) != (y2 > y):
            if x1 + (y - y1) * (x2 - x1) / (y2 - y1 + 1e-12) > x:
                inside = not inside
    return inside

def _ensure_ccw(c, want_ccw):
    if (_poly_area(c) > 0) != want_ccw:
        return list(reversed(c))
    return c

def _normalize_shape(shape):
    shape = [c for c in shape if len(c) >= 3 and abs(_poly_area(c)) > 1e-9]
    if not shape:
        return []
    areas = [_poly_area(c) for c in shape]
    oi = max(range(len(shape)), key=lambda i: abs(areas[i]))
    return [_ensure_ccw(c, i == oi) for i, c in enumerate(shape)]

def _shape_bbox(shape):
    xs, ys = [], []
    for c in shape:
        for x, y in c:
            xs.append(x)
            ys.append(y)
    if not xs:
        return (0, 0, 0, 0)
    return (min(xs), min(ys), max(xs), max(ys))

def _shape_centroid(shape):
    pts = [p for c in shape for p in c]
    if not pts:
        return (0.0, 0.0)
    return (sum(p[0] for p in pts) / len(pts),
            sum(p[1] for p in pts) / len(pts))


# ═══════════════════════════════════════════════════
#  SVG path parsing
# ═══════════════════════════════════════════════════
_cmd_re = re.compile(r'([MmLlHhVvCcZz])|([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)')

def _tokenize(d):
    for m in _cmd_re.finditer(d.replace(',', ' ')):
        if m.group(1):
            yield m.group(1)
        else:
            yield float(m.group(2))

def _cubic(p0, p1, p2, p3, t):
    a = _lerp_pt(p0, p1, t)
    b = _lerp_pt(p1, p2, t)
    c = _lerp_pt(p2, p3, t)
    d = _lerp_pt(a, b, t)
    e = _lerp_pt(b, c, t)
    return _lerp_pt(d, e, t)

def _parse_svg_path(d, steps=18):
    tokens = list(_tokenize(d))
    i = 0
    cur = (0.0, 0.0)
    start = None
    current: Contour = []
    contours: Shape = []
    last = None

    def close():
        nonlocal current, start
        if len(current) >= 3:
            if _dist(current[0], current[-1]) < 1e-6:
                current.pop()
            contours.append(current)
        current = []
        start = None

    while i < len(tokens):
        tok = tokens[i]
        if isinstance(tok, str):
            cmd = tok
            i += 1
        else:
            if not last:
                raise ValueError("no cmd")
            cmd = last
        last = cmd

        if cmd in "Mm":
            if current:
                close()
            x, y = tokens[i], tokens[i + 1]
            i += 2
            cur = (cur[0] + x, cur[1] + y) if cmd == "m" else (x, y)
            start = cur
            current.append(cur)
            while i + 1 < len(tokens) and not isinstance(tokens[i], str):
                x, y = tokens[i], tokens[i + 1]
                i += 2
                cur = (cur[0] + x, cur[1] + y) if cmd == "m" else (x, y)
                current.append(cur)
        elif cmd in "Ll":
            while i + 1 < len(tokens) and not isinstance(tokens[i], str):
                x, y = tokens[i], tokens[i + 1]
                i += 2
                cur = (cur[0] + x, cur[1] + y) if cmd == "l" else (x, y)
                current.append(cur)
        elif cmd in "Hh":
            while i < len(tokens) and not isinstance(tokens[i], str):
                x = tokens[i]
                i += 1
                cur = (cur[0] + x, cur[1]) if cmd == "h" else (x, cur[1])
                current.append(cur)
        elif cmd in "Vv":
            while i < len(tokens) and not isinstance(tokens[i], str):
                y = tokens[i]
                i += 1
                cur = (cur[0], cur[1] + y) if cmd == "v" else (cur[0], y)
                current.append(cur)
        elif cmd in "Cc":
            while i + 5 < len(tokens) and not isinstance(tokens[i], str):
                x1, y1, x2, y2, x, y = tokens[i:i + 6]
                i += 6
                if cmd == "c":
                    p1 = (cur[0] + x1, cur[1] + y1)
                    p2 = (cur[0] + x2, cur[1] + y2)
                    p3 = (cur[0] + x, cur[1] + y)
                else:
                    p1, p2, p3 = (x1, y1), (x2, y2), (x, y)
                for s in range(1, steps + 1):
                    current.append(_cubic(cur, p1, p2, p3, s / steps))
                cur = p3
        elif cmd in "Zz":
            if start:
                current.append(start)
            close()

    if current:
        close()
    return contours


# ═══════════════════════════════════════════════════
#  SVG 로드 — 초성/중성 분리, 개별 획 반환
# ═══════════════════════════════════════════════════
def _load_svg_bunri(path, steps=18):
    tree = ET.parse(path)
    root = tree.getroot()
    ns_m = re.match(r"\{(.*)\}", root.tag)
    ns = ns_m.group(1) if ns_m else None

    def fa(tag):
        if ns:
            return root.findall(f".//{{{ns}}}{tag}")
        return root.findall(f".//{tag}")

    path_data = []
    for p in fa("path"):
        d = p.get("d")
        if not d:
            continue
        fr = p.get("fill-rule", "nonzero")
        cs = _parse_svg_path(d, steps)
        cs = _normalize_shape(cs)
        if cs:
            path_data.append((fr, cs))

    if not path_data:
        return [], []

    cho_idx = None
    for idx, (fr, cs) in enumerate(path_data):
        if fr == "evenodd":
            cho_idx = idx
            break

    if cho_idx is None:
        cands = []
        for idx, (fr, cs) in enumerate(path_data):
            if len(cs) < 2:
                continue
            for i in range(len(cs)):
                bb = _bbox(cs[i])
                for j in range(len(cs)):
                    if i == j:
                        continue
                    c = _centroid(cs[j])
                    if (bb[0] <= c[0] <= bb[2] and bb[1] <= c[1] <= bb[3]
                            and _point_in_poly(c, cs[i])):
                        cands.append((idx, abs(_poly_area(cs[i]))))
                        break
        if cands:
            cands.sort(key=lambda x: -x[1])
            cho_idx = cands[0][0]
        else:
            cho_idx = max(range(len(path_data)),
                         key=lambda k: len(path_data[k][1]))

    cho = _normalize_shape(path_data[cho_idx][1])
    jung_strokes = []
    for k, (fr, cs) in enumerate(path_data):
        if k != cho_idx and cs:
            jung_strokes.append(cs)

    return cho, jung_strokes


# ═══════════════════════════════════════════════════
#  획 태깅: (orientation, length_class)
# ═══════════════════════════════════════════════════
ORIENT_RATIO = 1.2   # h > w * ORIENT_RATIO 이면 V, 아니면 H (정사각형은 H)
MAIN_RATIO = 0.5     # skeleton_len >= max * MAIN_RATIO 이면 long, 아니면 short

def _tag_strokes(strokes: List[Shape]) -> List[Tuple[str, str]]:
    """각 획에 (V/H, long/short) 태그를 부여."""
    if not strokes:
        return []

    skels = []
    for s in strokes:
        x0, y0, x1, y1 = _shape_bbox(s)
        skels.append(max(y1 - y0, x1 - x0))

    max_skel = max(skels)
    threshold = max_skel * MAIN_RATIO

    tags = []
    for s, sk in zip(strokes, skels):
        x0, y0, x1, y1 = _shape_bbox(s)
        w, h = x1 - x0, y1 - y0
        orientation = 'V' if h > w * ORIENT_RATIO else 'H'
        length_class = 'long' if sk >= threshold else 'short'
        tags.append((orientation, length_class))

    return tags


# ═══════════════════════════════════════════════════
#  (orientation, length_class) 기반 획 매칭
# ═══════════════════════════════════════════════════
def _match_strokes(src_strokes, dst_strokes):
    """(orientation, length_class) 태그로 매칭.
    동일 태그 복수 시 centroid x 순서로 페어링 (왼쪽=base 우선).
    Returns: (matched_pairs, unmatched_src, unmatched_dst)
    """
    src_by_tag = defaultdict(list)
    for s in src_strokes:
        src_by_tag[s.tag].append(s)

    dst_by_tag = defaultdict(list)
    for d in dst_strokes:
        dst_by_tag[d.tag].append(d)

    all_tags = set(src_by_tag) | set(dst_by_tag)

    matched = []
    unmatched_src = []
    unmatched_dst = []

    for tag in all_tags:
        ss = sorted(src_by_tag.get(tag, []), key=lambda s: s.cx)
        ds = sorted(dst_by_tag.get(tag, []), key=lambda d: d.cx)
        n = min(len(ss), len(ds))
        for i in range(n):
            matched.append((ss[i], ds[i]))
        unmatched_src.extend(ss[n:])
        unmatched_dst.extend(ds[n:])

    return matched, unmatched_src, unmatched_dst


# ═══════════════════════════════════════════════════
#  래스터화 & SDF
# ═══════════════════════════════════════════════════
def _rasterize(contours, xform, w, h, fill_rule=Qt.FillRule.OddEvenFill):
    img = QImage(w, h, QImage.Format.Format_Grayscale8)
    img.fill(0)
    if not contours:
        return np.zeros((h, w), dtype=bool)

    path = QPainterPath()
    path.setFillRule(fill_rule)
    for cont in contours:
        if len(cont) < 3:
            continue
        tx, ty = xform(cont[0])
        path.moveTo(tx, ty)
        for p in cont[1:]:
            tx, ty = xform(p)
            path.lineTo(tx, ty)
        path.closeSubpath()

    painter = QPainter(img)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)
    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(QColor(255, 255, 255))
    painter.drawPath(path)
    painter.end()

    bpl = img.bytesPerLine()
    buf = bytes(img.constBits())
    arr = np.frombuffer(buf, dtype=np.uint8).reshape(h, bpl)[:, :w].copy()
    return arr > 128


def _sdf_from_mask(mask):
    if not mask.any():
        return np.full(mask.shape, 9999.0, dtype=np.float32)
    if mask.all():
        return np.full(mask.shape, -9999.0, dtype=np.float32)
    out = distance_transform_edt(~mask)
    inn = distance_transform_edt(mask)
    return (out - inn).astype(np.float32)


# ═══════════════════════════════════════════════════
#  데이터 클래스
# ═══════════════════════════════════════════════════
class StrokeInfo:
    __slots__ = ('sdf', 'tag', 'cx', 'cy', 'hw', 'hh')

    def __init__(self, sdf, tag, cx, cy):
        self.sdf = sdf       # np.ndarray (GRID x GRID)
        self.tag = tag        # 'V' or 'H'
        # bbox center & half-dims from SDF (grid 좌표)
        mask = sdf < 0
        ys_i, xs_i = np.where(mask)
        if len(xs_i) > 0:
            x0, x1 = float(xs_i.min()), float(xs_i.max())
            y0, y1 = float(ys_i.min()), float(ys_i.max())
            self.cx = (x0 + x1) / 2.0
            self.cy = (y0 + y1) / 2.0
            self.hw = (x1 - x0) / 2.0
            self.hh = (y1 - y0) / 2.0
        else:
            self.cx = cx
            self.cy = cy
            self.hw = 1.0
            self.hh = 1.0


class GlyphData:
    __slots__ = ('cho_sdf', 'strokes', 'anchor_cx', 'anchor_cy')

    def __init__(self, cho_sdf, strokes, anchor_cx, anchor_cy):
        self.cho_sdf = cho_sdf
        self.strokes = strokes        # List[StrokeInfo]
        self.anchor_cx = anchor_cx    # base_main centroid x (흡수/방출 기준점)
        self.anchor_cy = anchor_cy


# ═══════════════════════════════════════════════════
#  SDF 블렌딩 — warp(이동 + 스케일) 기반
# ═══════════════════════════════════════════════════
def _smoothstep(t):
    t = max(0.0, min(1.0, t))
    return t * t * (3.0 - 2.0 * t)


def _shift_sdf(sdf, dx, dy):
    sx, sy = int(round(dx)), int(round(dy))
    if sx == 0 and sy == 0:
        return sdf
    h, w = sdf.shape
    result = np.full_like(sdf, 9999.0)
    src_y0, src_y1 = max(0, -sy), min(h, h - sy)
    dst_y0, dst_y1 = max(0, sy), min(h, h + sy)
    src_x0, src_x1 = max(0, -sx), min(w, w - sx)
    dst_x0, dst_x1 = max(0, sx), min(w, w + sx)
    if dst_y1 > dst_y0 and dst_x1 > dst_x0:
        result[dst_y0:dst_y1, dst_x0:dst_x1] = sdf[src_y0:src_y1, src_x0:src_x1]
    return result


def _warp_sdf(sdf, scx, scy, shw, shh, tcx, tcy, thw, thh):
    """SDF를 src bbox → target bbox 로 affine warp.

    획이 이동하면서 길이/폭이 연속적으로 변하도록.
    """
    h, w = sdf.shape
    sx = shw / max(thw, 0.5)
    sy = shh / max(thh, 0.5)
    xx_s = (_XX[:h, :w] - tcx) * sx + scx
    yy_s = (_YY[:h, :w] - tcy) * sy + scy
    xi = np.clip(np.round(xx_s).astype(np.int32), 0, w - 1)
    yi = np.clip(np.round(yy_s).astype(np.int32), 0, h - 1)
    return sdf[yi, xi]


def _blend_glyphs(src: GlyphData, dst: GlyphData, t: float) -> np.ndarray:
    """방향 기반 매칭 + warp 모핑.

    - 공통 획(같은 방향): bbox warp (이동 + 크기 연속 변화, 사라짐 없음)
    - 비공통 획: anchor 방향으로 이동하며 shrink (흡수/방출)
    """
    s = _smoothstep(t)

    # ── 양 끝 정확한 정지 상태 ──
    if s <= 0.0:
        return _resting_sdf(src)
    if s >= 1.0:
        return _resting_sdf(dst)

    # ── Layer 1: 초성 (SDF lerp) ──
    result = (1 - s) * src.cho_sdf + s * dst.cho_sdf

    # ── 방향 매칭 ──
    matched, um_src, um_dst = _match_strokes(src.strokes, dst.strokes)

    # ── 앵커: 매칭된 획 중 가장 큰 획의 centroid ──
    src_ax, src_ay = src.anchor_cx, src.anchor_cy
    dst_ax, dst_ay = dst.anchor_cx, dst.anchor_cy
    if matched:
        best = max(matched, key=lambda p: (p[0].sdf < 0).sum())
        src_ax, src_ay = best[0].cx, best[0].cy
        dst_ax, dst_ay = best[1].cx, best[1].cy

    # ── Layer 2: 공통 획 — warp (이동 + 크기 변화) ──
    #   src/dst 모두 동일 중간 bbox로 warp 후 SDF lerp.
    #   → 획이 사라지지 않고 위치·길이가 연속적으로 변함.
    for ss, ds in matched:
        icx = ss.cx + s * (ds.cx - ss.cx)
        icy = ss.cy + s * (ds.cy - ss.cy)
        ihw = ss.hw + s * (ds.hw - ss.hw)
        ihh = ss.hh + s * (ds.hh - ss.hh)
        w_src = _warp_sdf(ss.sdf, ss.cx, ss.cy, ss.hw, ss.hh,
                          icx, icy, ihw, ihh)
        w_dst = _warp_sdf(ds.sdf, ds.cx, ds.cy, ds.hw, ds.hh,
                          icx, icy, ihw, ihh)
        merged = (1 - s) * w_src + s * w_dst
        result = np.minimum(result, merged)

    # ── 메인 이동 벡터 (비공통 획 ride-along 용) ──
    main_dx = dst_ax - src_ax
    main_dy = dst_ay - src_ay

    # ── 공통 태그 유무에 따라 분기 ──
    has_match = len(matched) > 0

    # ── Layer 3: 비공통 src 획 — 이동 + fade out (+ 매칭 있으면 단축 압축) ──
    for stroke in um_src:
        rel_x = stroke.cx - src_ax
        rel_y = stroke.cy - src_ay
        absorb_t = _smoothstep(max(0.0, (s - ABSORB_DELAY) / (1.0 - ABSORB_DELAY)))
        target_cx = (src_ax + s * main_dx) + (1.0 - absorb_t) * rel_x
        target_cy = (src_ay + s * main_dy) + (1.0 - absorb_t) * rel_y
        if has_match:
            # 매칭 있는 쌍: 앵커 방향 한 축만 압축
            dir_x = src_ax - stroke.cx
            dir_y = src_ay - stroke.cy
            squeeze = max(1.0 - absorb_t, 0.01)
            if abs(dir_x) >= abs(dir_y):
                thw = stroke.hw * squeeze
                thh = stroke.hh
            else:
                thw = stroke.hw
                thh = stroke.hh * squeeze
            warped = _warp_sdf(stroke.sdf, stroke.cx, stroke.cy,
                               stroke.hw, stroke.hh,
                               target_cx, target_cy, thw, thh)
        else:
            # 공통 태그 없는 쌍: 이동만
            warped = _shift_sdf(stroke.sdf,
                                target_cx - stroke.cx,
                                target_cy - stroke.cy)
        stroke_sdf = warped
        # weighted union: fade = 1→0 으로 기여도 감소
        fade = 1.0 - absorb_t
        delta = np.minimum(0.0, stroke_sdf - result)
        result = result + fade * delta

    # ── Layer 4: 비공통 dst 획 — fade in + 이동 (+ 매칭 있으면 단축 팽창) ──
    for stroke in um_dst:
        rel_x = stroke.cx - dst_ax
        rel_y = stroke.cy - dst_ay
        emit_t = _smoothstep(min(1.0, s / EMIT_END))
        target_cx = (dst_ax - (1.0 - s) * main_dx) + emit_t * rel_x
        target_cy = (dst_ay - (1.0 - s) * main_dy) + emit_t * rel_y
        if has_match:
            # 매칭 있는 쌍: 앵커 방향 한 축에서 팽창
            dir_x = dst_ax - stroke.cx
            dir_y = dst_ay - stroke.cy
            squeeze = max(emit_t, 0.01)
            if abs(dir_x) >= abs(dir_y):
                thw = stroke.hw * squeeze
                thh = stroke.hh
            else:
                thw = stroke.hw
                thh = stroke.hh * squeeze
            warped = _warp_sdf(stroke.sdf, stroke.cx, stroke.cy,
                               stroke.hw, stroke.hh,
                               target_cx, target_cy, thw, thh)
        else:
            # 공통 태그 없는 쌍: 이동만
            warped = _shift_sdf(stroke.sdf,
                                target_cx - stroke.cx,
                                target_cy - stroke.cy)
        stroke_sdf = warped
        # weighted union: fade = 0→1 로 기여도 증가
        fade = emit_t
        delta = np.minimum(0.0, stroke_sdf - result)
        result = result + fade * delta

    return result


def _resting_sdf(glyph: GlyphData) -> np.ndarray:
    sdf = glyph.cho_sdf.copy()
    for s in glyph.strokes:
        sdf = np.minimum(sdf, s.sdf)
    return sdf


def _sdf_to_qimage(sdf):
    alpha = np.clip(0.5 - sdf / AA_WIDTH, 0.0, 1.0)
    pixels = (255.0 * (1.0 - alpha)).astype(np.uint8)
    pixels = np.ascontiguousarray(pixels)
    h, w = pixels.shape
    img = QImage(pixels.data, w, h, w, QImage.Format.Format_Grayscale8)
    return img.copy()


# ═══════════════════════════════════════════════════
#  Widget
# ═══════════════════════════════════════════════════
class MorphWidget(QWidget):
    def __init__(self, svg_dir: str):
        super().__init__()
        self.setWindowTitle("한글 모음 SDF 모핑 (태그 기반 translate)")
        self.resize(720, 720)

        # ── SVG 파싱 ──
        raw: Dict[str, Tuple[Shape, List[Shape]]] = {}
        for ch in VOWELS:
            cho, jung_strokes = _load_svg_bunri(f"{svg_dir}/{ch}_그룹_분리.svg")
            raw[ch] = (cho, jung_strokes)

        # ── 글로벌 바운딩 박스 ──
        xs, ys = [], []
        for cho, jung_strokes in raw.values():
            for c in cho:
                for x, y in c:
                    xs.append(x); ys.append(y)
            for stroke in jung_strokes:
                for c in stroke:
                    for x, y in c:
                        xs.append(x); ys.append(y)
        gx0, gy0, gx1, gy1 = min(xs), min(ys), max(xs), max(ys)
        gw, gh = gx1 - gx0, gy1 - gy0
        margin = 30
        scale = min((GRID - 2 * margin) / max(gw, 1e-6),
                    (GRID - 2 * margin) / max(gh, 1e-6))
        ox = (GRID - gw * scale) / 2
        oy = (GRID - gh * scale) / 2

        def xform(p):
            return ((p[0] - gx0) * scale + ox,
                    (p[1] - gy0) * scale + oy)

        # ── GlyphData 구축 ──
        self._data: Dict[str, GlyphData] = {}
        for ch in VOWELS:
            cho, jung_strokes = raw[ch]

            # 초성 SDF
            cho_sdf = _sdf_from_mask(
                _rasterize(cho, xform, GRID, GRID, Qt.FillRule.OddEvenFill))

            # 중성: 태깅 + SDF
            tags = _tag_strokes(jung_strokes)
            stroke_infos = []
            for shape, tag in zip(jung_strokes, tags):
                sdf = _sdf_from_mask(
                    _rasterize(shape, xform, GRID, GRID, Qt.FillRule.WindingFill))
                cx, cy = xform(_shape_centroid(shape))
                stroke_infos.append(StrokeInfo(sdf, tag, cx, cy))

            # 앵커: 가장 큰 획(면적 기준)의 centroid (없으면 cho centroid)
            cho_cx, cho_cy = xform(_shape_centroid(cho))
            if stroke_infos:
                biggest = max(stroke_infos,
                              key=lambda si: (si.sdf < 0).sum())
                anchor_cx, anchor_cy = biggest.cx, biggest.cy
            else:
                anchor_cx, anchor_cy = cho_cx, cho_cy

            self._data[ch] = GlyphData(cho_sdf, stroke_infos,
                                       anchor_cx, anchor_cy)

            tag_str = ", ".join(f"({t[0]},{t[1]})" for t in tags)
            print(f"  {ch}: [{tag_str}]", flush=True)

        # ── 애니메이션 상태 ──
        self._cur = "아"
        self._tgt: Optional[str] = None
        self._t = 0.0
        self._t0 = 0.0
        self._animating = False

        self._qimg = _sdf_to_qimage(_resting_sdf(self._data["아"]))

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.setInterval(1000 // FPS)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    def keyPressEvent(self, e):
        k = e.key()
        if k == Qt.Key.Key_Escape:
            self.close()
            return
        idx = k - Qt.Key.Key_1
        if 0 <= idx < len(VOWELS):
            tgt = VOWELS[idx]
            if tgt == self._cur and not self._animating:
                return
            if self._animating and self._tgt:
                self._cur = self._tgt
            self._tgt = tgt
            self._t0 = time.monotonic()
            self._animating = True
            self._timer.start()
        super().keyPressEvent(e)

    def _tick(self):
        if not self._animating:
            self._timer.stop()
            return

        elapsed = time.monotonic() - self._t0
        self._t = min(elapsed / MORPH_SEC, 1.0)

        sdf = _blend_glyphs(self._data[self._cur],
                            self._data[self._tgt], self._t)
        self._qimg = _sdf_to_qimage(sdf)

        if self._t >= 1.0:
            self._animating = False
            self._cur = self._tgt
            self._tgt = None
            self._timer.stop()

        self.update()

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        w, h = self.width(), self.height()
        p.fillRect(0, 0, w, h, QColor(255, 255, 255))

        side = min(w, h) - 60
        x0 = (w - side) / 2
        y0 = (h - side) / 2 + 10
        p.drawImage(QRectF(x0, y0, side, side),
                    self._qimg,
                    QRectF(0, 0, GRID, GRID))

        p.setPen(QPen(QColor(80, 80, 80), 1))
        p.setFont(QFont("Consolas", 11))

        if self._animating and self._tgt:
            hud = f"{self._cur} → {self._tgt}   t={self._t:.2f}"
        else:
            hud = f"current: {self._cur}   (1-7: morph, ESC: quit)"
        p.drawText(QRectF(16, 8, w - 32, 24),
                   Qt.AlignmentFlag.AlignLeft, hud)

        legend = "  ".join(f"{i + 1}:{v}" for i, v in enumerate(VOWELS))
        p.drawText(QRectF(16, h - 32, w - 32, 24),
                   Qt.AlignmentFlag.AlignCenter, legend)
        p.end()


# ═══════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════
def main():
    svg_dir = os.path.dirname(os.path.abspath(__file__))

    app = QApplication(sys.argv)
    try:
        w = MorphWidget(svg_dir)
    except Exception as ex:
        print(f"Error: {ex}")
        import traceback
        traceback.print_exc()
        print("\nExpected files:")
        for v in VOWELS:
            print(f"  {svg_dir}/{v}_그룹_분리.svg")
        sys.exit(1)

    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
