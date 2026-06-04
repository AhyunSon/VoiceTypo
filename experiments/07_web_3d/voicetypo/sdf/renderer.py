"""SDF 렌더러 — 기존 엔진에 VowelTracker 연결.

기존 SDF 엔진(glyph_morph_sdf_final_v2.py)을 수정 없이 사용.
타이머 기반 self._t를 VowelTracker의 blend_weights에서 계산된 t로 교체.
"""

import os
import sys
import time
import math
import numpy as np

# 기존 엔진 import
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from text_morphing.glyph_morph_sdf_final_v2 import (
    VOWELS, GRID, AA_WIDTH,
    _resting_sdf, _blend_glyphs, GlyphData, StrokeInfo,
)
from text_effects.test_effects import (
    load_glyph_data, compute_joint_mask, apply_vibrato_to_sdf,
    sdf_to_rgba, compute_colors,
)


class SDFRenderer:
    """VowelTracker 출력으로 SDF 글리프를 렌더링.

    기존 엔진의 _blend_glyphs를 사용하되,
    모핑 제어를 VowelTracker의 weights_to_morph 결과로 대체.
    """

    def __init__(self, glyph_data=None):
        if glyph_data is None:
            glyph_data = load_glyph_data()
        self._data = glyph_data
        self._joint_dists = {ch: compute_joint_mask(d)
                             for ch, d in glyph_data.items()}

        # 현재 상태
        self._cur = '아'
        self._sdf = _resting_sdf(self._data['아'])
        self._joint_dist = self._joint_dists['아']

        # 비브라토
        self._vibrato_phase = 0.0
        self._last_time = time.monotonic()

    @property
    def glyph_data(self):
        return self._data

    @property
    def current_vowel(self):
        return self._cur

    def update_from_morph(self, src_data, dst_data, t,
                          src_name=None, dst_name=None):
        """weights_to_morph 결과로 SDF 업데이트.

        Args:
            src_data, dst_data: GlyphData
            t: 모핑 진행도 0~1
            src_name, dst_name: 모음 이름 (joint_dist 보간용)
        """
        if t <= 0.0:
            self._sdf = _resting_sdf(src_data)
        elif t >= 1.0:
            self._sdf = _resting_sdf(dst_data)
        else:
            self._sdf = _blend_glyphs(src_data, dst_data, t)

        # joint dist 보간
        if src_name and dst_name:
            sd = self._joint_dists.get(src_name)
            dd = self._joint_dists.get(dst_name)
            if sd is not None and dd is not None:
                self._joint_dist = (1 - t) * sd + t * dd

    def update_from_weights(self, weights):
        """소프트 가중치에서 직접 SDF 업데이트.

        Args:
            weights: dict[str, float] — 모음별 가중치
        """
        if not weights:
            return

        valid = {k: v for k, v in weights.items() if k in self._data}
        if not valid:
            return

        top2 = sorted(valid.items(), key=lambda x: x[1], reverse=True)[:2]

        if len(top2) == 1:
            name = top2[0][0]
            self._sdf = _resting_sdf(self._data[name])
            self._joint_dist = self._joint_dists[name]
            self._cur = name
            return

        src_name, src_w = top2[0]
        dst_name, dst_w = top2[1]
        t = dst_w / (src_w + dst_w + 1e-9)

        self.update_from_morph(
            self._data[src_name], self._data[dst_name], t,
            src_name, dst_name)
        self._cur = src_name

    def render(self, pitch_ratio=0.0, opacity=1.0,
               vib_amount=0.0, vib_speed=0.0, dt=None):
        """현재 SDF를 RGBA QImage로 렌더링.

        Args:
            pitch_ratio: -1~+1 (색상·스케일)
            opacity: 0~1
            vib_amount: 비브라토 크기
            vib_speed: 비브라토 속도
            dt: 시간 간격 (None이면 자동 계산)

        Returns:
            (qimg, glow_color, sx, sy) 튜플
        """
        now = time.monotonic()
        if dt is None:
            dt = now - self._last_time
        self._last_time = now

        self._vibrato_phase += dt * vib_speed * 3.0

        # 비브라토 적용
        if vib_amount > 0.5:
            sdf = apply_vibrato_to_sdf(
                self._sdf, vib_amount, self._vibrato_phase, self._joint_dist)
        else:
            sdf = self._sdf

        # 색상
        c_top, c_bot, glow_color = compute_colors(pitch_ratio)

        # SDF → RGBA
        qimg = sdf_to_rgba(sdf, c_top, c_bot, opacity)

        # 스케일
        sx = 1 - pitch_ratio * 0.65
        sy = 1 + pitch_ratio * 0.9
        area = sx * sy
        norm = 1.0 / math.sqrt(max(area, 0.01))
        sx *= norm
        sy *= norm

        return (qimg, glow_color, sx, sy)
