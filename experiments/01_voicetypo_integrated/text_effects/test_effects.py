"""텍스트 효과 프로토타입 — 슬라이더로 조절하며 시각 테스트

SDF 글리프 위에 다음 효과를 실시간 적용:
  - 피치 → 세로/가로 스케일 변형
  - 볼륨 → 전체 크기
  - 피치 → 색상 (높은음=빨강/주황, 낮은음=보라/파랑)
  - 비브라토 → 떨림 (노이즈 기반 변위)
  - 투명도

조작:
  슬라이더로 각 파라미터 조절
  키 1-7: 모음 전환 (SDF 모핑)
  ESC: 종료
"""

import sys, os, math, time
import numpy as np
from scipy.ndimage import distance_transform_edt

from PySide6.QtCore import Qt, QTimer, QRectF, QPointF
from PySide6.QtGui import (QPainter, QColor, QImage, QPen, QFont,
                            QLinearGradient, QRadialGradient, QBrush)
from PySide6.QtWidgets import (QApplication, QWidget, QVBoxLayout,
                                QHBoxLayout, QSlider, QLabel, QGroupBox,
                                QSplitter, QFrame)

# ── text_morphing 모듈 import ──
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from text_morphing.glyph_morph_sdf_final_v2 import (
    VOWELS, GRID, MORPH_SEC, AA_WIDTH,
    _load_svg_bunri, _normalize_shape, _shape_bbox, _shape_centroid,
    _tag_strokes, _sdf_from_mask, _rasterize, _resting_sdf,
    _blend_glyphs, GlyphData, StrokeInfo,
)


# ═══════════════════════════════════════════════════
#  SDF → 컬러 RGBA 이미지 (기존 흑백 대체)
# ═══════════════════════════════════════════════════
def sdf_to_rgba(sdf, color_top, color_bot, opacity=1.0):
    """SDF를 컬러 RGBA QImage로 변환.
    color_top/bot: (r, g, b) 튜플, 세로 그라데이션.
    """
    h, w = sdf.shape
    alpha = np.clip(0.5 - sdf / AA_WIDTH, 0.0, 1.0) * opacity

    # 세로 그라데이션 t: 0(상단) → 1(하단)
    t = np.linspace(0, 1, h, dtype=np.float32)[:, None]
    r = (color_top[0] * (1 - t) + color_bot[0] * t).astype(np.float32)
    g = (color_top[1] * (1 - t) + color_bot[1] * t).astype(np.float32)
    b = (color_top[2] * (1 - t) + color_bot[2] * t).astype(np.float32)

    # RGBA 배열 (premultiplied alpha)
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[:, :, 0] = np.clip(r * alpha, 0, 255).astype(np.uint8)
    rgba[:, :, 1] = np.clip(g * alpha, 0, 255).astype(np.uint8)
    rgba[:, :, 2] = np.clip(b * alpha, 0, 255).astype(np.uint8)
    rgba[:, :, 3] = np.clip(alpha * 255, 0, 255).astype(np.uint8)

    rgba = np.ascontiguousarray(rgba)
    img = QImage(rgba.data, w, h, w * 4,
                 QImage.Format.Format_RGBA8888_Premultiplied)
    return img.copy()


# ═══════════════════════════════════════════════════
#  비브라토: SDF에 노이즈 변위 적용
# ═══════════════════════════════════════════════════
_YY_GRID, _XX_GRID = np.mgrid[0:GRID, 0:GRID].astype(np.float32)

JOINT_PROXIMITY = 12.0  # 두 획 경계가 이 거리 이내면 접합부로 판정 (픽셀)
JOINT_RADIUS = 12.0     # 접합부로부터 이 거리까지 전환 (픽셀)


def compute_joint_mask(glyph):
    """접합부(두 획이 만나는 곳) 근처로부터의 거리 맵.
    0 = 접합부, 값이 클수록 접합부에서 멀리 떨어짐."""
    sdfs = [glyph.cho_sdf] + [s.sdf for s in glyph.strokes]
    if len(sdfs) < 2:
        return np.full_like(sdfs[0], 9999.0, dtype=np.float32)
    # 각 획의 "경계 근처" 마스크 (|sdf| < threshold)
    near_masks = [(np.abs(s) < JOINT_PROXIMITY) for s in sdfs]
    # 겹치는 곳 = 2개 이상 획의 경계가 동시에 가까운 픽셀
    near_sum = near_masks[0].astype(np.int32)
    for m in near_masks[1:]:
        near_sum += m.astype(np.int32)
    junction = near_sum >= 2
    if not junction.any():
        return np.full_like(sdfs[0], 9999.0, dtype=np.float32)
    # 접합부로부터의 유클리드 거리
    return distance_transform_edt(~junction).astype(np.float32)


def apply_vibrato_to_sdf(sdf, amount, phase, joint_dist=None):
    """SDF 값에 고주파 노이즈를 더해 외곽선이 바르르 떨리는 효과.
    amount: 떨림 크기 (0이면 효과 없음)
    phase: 떨림 위상 (속도가 이미 반영된 누적값)
    joint_dist: 접합부로부터의 거리 맵 (가까울수록 떨림 억제)
    """
    if amount < 0.5:
        return sdf

    h, w = sdf.shape
    tv = phase

    # 고주파 노이즈 — 여러 sin 중첩으로 불규칙한 미세 떨림
    noise = (
        np.sin(_XX_GRID[:h, :w] * 0.11 + _YY_GRID[:h, :w] * 0.07
               + tv * 14.0) * 0.30
        + np.sin(_YY_GRID[:h, :w] * 0.13 + _XX_GRID[:h, :w] * 0.05
                 + tv * 19.0) * 0.30
        + np.sin((_XX_GRID[:h, :w] + _YY_GRID[:h, :w]) * 0.09
                 + tv * 23.0) * 0.25
        + np.sin(_XX_GRID[:h, :w] * 0.17 - _YY_GRID[:h, :w] * 0.12
                 + tv * 11.0) * 0.15
    )

    perturbation = noise * amount * 0.15

    # 접합부: 안쪽 방향(틈 벌어짐)만 억제, 바깥 방향(팽창)은 허용
    # perturbation > 0 → SDF 감소 → 획 팽창 (안전)
    # perturbation < 0 → SDF 증가 → 획 수축 → 틈 발생 (위험)
    if joint_dist is not None:
        t = np.clip(joint_dist / JOINT_RADIUS, 0.0, 1.0)
        allow = t * t * (3.0 - 2.0 * t)  # smoothstep: 0(접합부) ~ 1(먼곳)
        # 수축 방향(perturbation < 0)만 접합부에서 억제
        shrink_mask = perturbation < 0
        perturbation = np.where(shrink_mask, perturbation * allow, perturbation)

    return sdf - perturbation


# ═══════════════════════════════════════════════════
#  색상 계산 (HTML 원본 포팅)
# ═══════════════════════════════════════════════════
def lerp_color(c1, c2, t):
    return tuple(int(c1[i] + (c2[i] - c1[i]) * t) for i in range(3))


COLOR_WHITE  = (255, 255, 255)
COLOR_RED    = (233, 69, 96)
COLOR_ORANGE = (243, 148, 34)
COLOR_PURPLE = (138, 43, 226)
COLOR_BLUE   = (33, 150, 243)


def compute_colors(pitch_ratio):
    """pitch_ratio: -1(최저) ~ 0(기준) ~ +1(최고)
    Returns: (color_top, color_bot, glow_color)
    """
    t = abs(pitch_ratio)
    if pitch_ratio >= 0:
        c_top = lerp_color(COLOR_WHITE, COLOR_RED, t)
        c_bot = lerp_color(COLOR_WHITE, COLOR_ORANGE, t)
        glow = lerp_color(COLOR_WHITE, COLOR_RED, t)
    else:
        c_top = lerp_color(COLOR_WHITE, COLOR_PURPLE, t)
        c_bot = lerp_color(COLOR_WHITE, COLOR_BLUE, t)
        glow = lerp_color(COLOR_WHITE, COLOR_BLUE, t)
        dim = 1 - t * 0.5
        c_top = tuple(int(v * dim) for v in c_top)
        c_bot = tuple(int(v * dim) for v in c_bot)
        glow = tuple(int(v * dim) for v in glow)
    return c_top, c_bot, glow


# ═══════════════════════════════════════════════════
#  캔버스 위젯
# ═══════════════════════════════════════════════════
class EffectCanvas(QWidget):
    def __init__(self, glyph_data):
        super().__init__()
        self._data = glyph_data
        self._cur = "아"
        self._tgt = None
        self._t = 0.0
        self._t0 = 0.0
        self._animating = False
        self._sdf = _resting_sdf(self._data["아"])
        self._joint_dists = {ch: compute_joint_mask(d) for ch, d in glyph_data.items()}
        self._joint_dist = self._joint_dists["아"]
        self._time_val = 0.0
        self._vibrato_phase = 0.0
        self._last_tick = time.monotonic()

        # 효과 파라미터 (슬라이더에서 설정)
        self.pitch_ratio = 0.0      # -1 ~ +1
        self.volume_scale = 1.0     # 0.2 ~ 3.5
        self.vibrato_amount = 0.0   # 0 ~ 40 (떨림 크기)
        self.vibrato_speed = 0.5    # 0 ~ 1 (떨림 속도)
        self.opacity = 1.0          # 0 ~ 1
        self.glow_strength = 0.0    # 0 ~ 1

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.setInterval(1000 // 60)
        self._timer.start()

    def trigger_morph(self, vowel):
        if vowel == self._cur and not self._animating:
            return
        if self._animating and self._tgt:
            self._cur = self._tgt
        self._tgt = vowel
        self._t0 = time.monotonic()
        self._animating = True

    def _tick(self):
        now = time.monotonic()
        dt = now - self._last_tick
        self._last_tick = now
        self._time_val += dt * 3.0
        self._vibrato_phase += dt * self.vibrato_speed * 3.0

        if self._animating and self._tgt:
            elapsed = now - self._t0
            self._t = min(elapsed / MORPH_SEC, 1.0)
            self._sdf = _blend_glyphs(self._data[self._cur],
                                       self._data[self._tgt], self._t)
            # joint_sdf도 보간 (접합부 마스크)
            s = self._t
            self._joint_dist = ((1 - s) * self._joint_dists[self._cur]
                               + s * self._joint_dists[self._tgt])
            if self._t >= 1.0:
                self._animating = False
                self._cur = self._tgt
                self._tgt = None
                self._sdf = _resting_sdf(self._data[self._cur])
                self._joint_dist = self._joint_dists[self._cur]

        self.update()

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        w, h = self.width(), self.height()

        # 배경: 검정
        p.fillRect(0, 0, w, h, QColor(0, 0, 0))

        # ── 비브라토 변위 적용 ──
        sdf = apply_vibrato_to_sdf(self._sdf, self.vibrato_amount,
                                    self._vibrato_phase, self._joint_dist)

        # ── 색상 계산 ──
        c_top, c_bot, glow = compute_colors(self.pitch_ratio)

        # ── SDF → 컬러 RGBA ──
        qimg = sdf_to_rgba(sdf, c_top, c_bot, self.opacity)

        # ── 스케일 계산 (피치 → 비율, 볼륨 → 크기) ──
        sx = 1 - self.pitch_ratio * 0.65
        sy = 1 + self.pitch_ratio * 0.9
        area = sx * sy
        norm = 1.0 / math.sqrt(max(area, 0.01))
        sx *= norm
        sy *= norm

        # ── 글로우 (배경에 큰 블러 원) ──
        if self.glow_strength > 0.01:
            cx, cy = w / 2, h / 2
            radius = min(w, h) * 0.3 * (self.volume_scale / 3.5)
            grad = QRadialGradient(QPointF(cx, cy), radius)
            gc = QColor(glow[0], glow[1], glow[2],
                        int(80 * self.glow_strength))
            grad.setColorAt(0, gc)
            grad.setColorAt(1, QColor(0, 0, 0, 0))
            p.setBrush(QBrush(grad))
            p.setPen(Qt.PenStyle.NoPen)
            p.drawEllipse(QPointF(cx, cy), radius, radius)

        # ── 이미지 그리기 (볼륨 최대 = 화면 최대) ──
        margin = 40
        max_w, max_h = w - margin * 2, h - margin * 2
        # 현재 비율(sx, sy)에서 화면에 딱 맞는 크기 계산
        fit = min(max_w / sx, max_h / sy)
        # 볼륨이 최대(3.5)일 때 화면 가득, 최소(0.2)일 때 작게
        scale = fit * (self.volume_scale / 3.5)
        img_w = sx * scale
        img_h = sy * scale
        x0 = (w - img_w) / 2
        y0 = (h - img_h) / 2

        p.drawImage(QRectF(x0, y0, img_w, img_h),
                     qimg, QRectF(0, 0, GRID, GRID))

        # ── HUD ──
        p.setPen(QPen(QColor(100, 100, 100), 1))
        p.setFont(QFont("Consolas", 10))
        if self._animating and self._tgt:
            hud = f"{self._cur} -> {self._tgt}  t={self._t:.2f}"
        else:
            hud = f"vowel: {self._cur}  (1-7: morph)"
        p.drawText(QRectF(12, 8, w - 24, 20),
                    Qt.AlignmentFlag.AlignLeft, hud)

        p.end()


# ═══════════════════════════════════════════════════
#  슬라이더 패널
# ═══════════════════════════════════════════════════
def make_slider(label_text, min_val, max_val, default, callback,
                parent=None, fmt="{:.2f}", scale=100):
    """슬라이더 + 라벨 생성. 값은 float, 내부는 int 슬라이더."""
    box = QHBoxLayout()
    label = QLabel(f"{label_text}: {fmt.format(default)}")
    label.setFixedWidth(200)
    label.setStyleSheet("color: #ccc; font-size: 12px;")

    slider = QSlider(Qt.Orientation.Horizontal)
    slider.setMinimum(int(min_val * scale))
    slider.setMaximum(int(max_val * scale))
    slider.setValue(int(default * scale))
    slider.setStyleSheet("""
        QSlider::groove:horizontal {
            background: #333; height: 6px; border-radius: 3px;
        }
        QSlider::handle:horizontal {
            background: #e94560; width: 16px; height: 16px;
            margin: -5px 0; border-radius: 8px;
        }
    """)

    def on_change(v):
        fv = v / scale
        label.setText(f"{label_text}: {fmt.format(fv)}")
        callback(fv)

    slider.valueChanged.connect(on_change)
    box.addWidget(label)
    box.addWidget(slider)
    return box


# ═══════════════════════════════════════════════════
#  메인 윈도우
# ═══════════════════════════════════════════════════
class EffectTestWindow(QWidget):
    def __init__(self, glyph_data):
        super().__init__()
        self.setWindowTitle("텍스트 효과 프로토타입")
        self.resize(1000, 750)
        self.setStyleSheet("background: #1a1a1a;")

        self.canvas = EffectCanvas(glyph_data)
        self.canvas.setMinimumSize(500, 500)

        # ── 슬라이더 패널 ──
        panel = QVBoxLayout()
        panel.setSpacing(8)

        title = QLabel("효과 파라미터")
        title.setStyleSheet("color: #e94560; font-size: 14px; "
                            "font-weight: bold; padding: 5px;")
        panel.addWidget(title)

        panel.addLayout(make_slider(
            "피치 (pitch ratio)", -1.0, 1.0, 0.0,
            lambda v: setattr(self.canvas, 'pitch_ratio', v)))

        panel.addLayout(make_slider(
            "볼륨 (volume scale)", 0.2, 3.5, 1.0,
            lambda v: setattr(self.canvas, 'volume_scale', v)))

        panel.addLayout(make_slider(
            "비브라토 크기 (amount)", 0.0, 40.0, 0.0,
            lambda v: setattr(self.canvas, 'vibrato_amount', v)))

        panel.addLayout(make_slider(
            "비브라토 속도 (speed)", 0.0, 1.0, 0.5,
            lambda v: setattr(self.canvas, 'vibrato_speed', v)))

        panel.addLayout(make_slider(
            "투명도 (opacity)", 0.0, 1.0, 1.0,
            lambda v: setattr(self.canvas, 'opacity', v)))

        panel.addLayout(make_slider(
            "글로우 (glow)", 0.0, 1.0, 0.0,
            lambda v: setattr(self.canvas, 'glow_strength', v)))

        # ── 설명 ──
        desc = QLabel(
            "키 1-7: 모음 전환 (아/이/우/에/오/으/어)\n"
            "ESC: 종료\n\n"
            "피치 +: 세로 늘어남 + 빨강/주황\n"
            "피치 -: 가로 늘어남 + 보라/파랑 + 어두워짐\n"
            "볼륨: 전체 크기\n"
            "비브라토 크기: 떨림 진폭\n"
            "비브라토 속도: 떨림 빈도\n"
            "투명도: 글자 투명도\n"
            "글로우: 배경 빛 번짐"
        )
        desc.setStyleSheet("color: #666; font-size: 11px; padding: 10px;")
        desc.setWordWrap(True)
        panel.addWidget(desc)

        panel.addStretch()

        # ── 레이아웃 ──
        panel_widget = QFrame()
        panel_widget.setLayout(panel)
        panel_widget.setFixedWidth(320)
        panel_widget.setStyleSheet("QFrame { background: #111; }")

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.canvas, 1)
        layout.addWidget(panel_widget)

        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    def keyPressEvent(self, e):
        k = e.key()
        if k == Qt.Key.Key_Escape:
            self.close()
            return
        idx = k - Qt.Key.Key_1
        if 0 <= idx < len(VOWELS):
            self.canvas.trigger_morph(VOWELS[idx])
        super().keyPressEvent(e)


# ═══════════════════════════════════════════════════
#  GlyphData 로드 (text_morphing에서 가져옴)
# ═══════════════════════════════════════════════════
def load_glyph_data():
    svg_dir = os.path.join(os.path.dirname(__file__), '..', 'text_morphing')
    svg_dir = os.path.abspath(svg_dir)

    from PySide6.QtCore import Qt as QtConst
    raw = {}
    for ch in VOWELS:
        cho, jung_strokes = _load_svg_bunri(f"{svg_dir}/{ch}_그룹_분리.svg")
        raw[ch] = (cho, jung_strokes)

    # 글로벌 바운딩 박스
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

    data = {}
    for ch in VOWELS:
        cho, jung_strokes = raw[ch]
        cho_sdf = _sdf_from_mask(
            _rasterize(cho, xform, GRID, GRID, QtConst.FillRule.OddEvenFill))
        tags = _tag_strokes(jung_strokes)
        stroke_infos = []
        for shape, tag in zip(jung_strokes, tags):
            sdf = _sdf_from_mask(
                _rasterize(shape, xform, GRID, GRID,
                           QtConst.FillRule.WindingFill))
            cx, cy = xform(_shape_centroid(shape))
            stroke_infos.append(StrokeInfo(sdf, tag, cx, cy))

        cho_cx, cho_cy = xform(_shape_centroid(cho))
        if stroke_infos:
            biggest = max(stroke_infos, key=lambda si: (si.sdf < 0).sum())
            anchor_cx, anchor_cy = biggest.cx, biggest.cy
        else:
            anchor_cx, anchor_cy = cho_cx, cho_cy

        data[ch] = GlyphData(cho_sdf, stroke_infos, anchor_cx, anchor_cy)

    return data


def main():
    app = QApplication(sys.argv)
    print("Loading glyphs...", flush=True)
    glyph_data = load_glyph_data()
    print("Done. Starting UI.", flush=True)

    win = EffectTestWindow(glyph_data)
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
