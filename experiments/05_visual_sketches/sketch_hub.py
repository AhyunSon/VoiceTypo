"""
sketch_hub.py — 5개 시각 스케치를 '버튼 하나로 전환'하는 단일 창 데모

원본 sketch_01~05 를 그대로 두고, 그 그리기 로직을 '장면(Scene)' 으로 옮겨
하나의 pygame 창 + 상단 버튼 바에서 왔다갔다 볼 수 있게 묶은 통합 런처.

조작:
  상단 버튼 클릭  또는  숫자키 1~5 : 장면 전환
  C  : (현재 장면이 지원하면) 화면 비움 — 5번 'Messa'
  ESC: 종료

마이크(VoiceListener)는 한 번만 열어 모든 장면이 공유한다.

실행:  python sketch_hub.py
"""

import sys, math, colorsys
from pathlib import Path

import numpy as np
import pygame

sys.path.insert(0, str(Path(__file__).resolve().parent))
from voice_input import VoiceListener, VOWELS

# ── 창 레이아웃 ──────────────────────────────────────────────
WIN_W, WIN_H = 1280, 820
BAR_H = 58
STAGE = pygame.Rect(0, BAR_H, WIN_W, WIN_H - BAR_H)

KFONT = "applesdgothicneo,applegothic,malgungothic,arialunicode"


def lerp(a, b, t):
    return a + (b - a) * t


def clamp01(x):
    return max(0.0, min(1.0, x))


# ════════════════════════════════════════════════════════════
#  장면 1 — 매핑 어휘집 (Levin/Rozin)
# ════════════════════════════════════════════════════════════
class Scene01:
    name = "1 · 매핑 어휘집"
    fps = 60
    clears = False

    def __init__(self, w, h):
        self.w, self.h = w, h
        self.BG = (12, 14, 22)
        self.font = pygame.font.SysFont(KFONT, 16)
        self.s = dict(rms=0.0, f0=160.0, f1=500.0, f2=1500.0, jit=0.0)
        self.rot = 0.0
        self.N = 80
        self.phases = np.random.default_rng(7).uniform(0, math.tau, self.N)

    def on_key(self, key):
        pass

    def render(self, screen, sig):
        w, h, s = self.w, self.h, self.s
        if sig and sig.voiced:
            a = 0.25
            s["rms"] = lerp(s["rms"], min(1.0, sig.rms * 18), a)
            if sig.f0 > 0:  s["f0"] = lerp(s["f0"], sig.f0, a)
            if sig.f1 > 0:  s["f1"] = lerp(s["f1"], sig.f1, a)
            if sig.f2 > 0:  s["f2"] = lerp(s["f2"], sig.f2, a)
            s["jit"] = lerp(s["jit"], sig.jitter, a)
        else:
            s["rms"] = lerp(s["rms"], 0.0, 0.08)

        f0n = clamp01((s["f0"] - 80) / 240)
        f1n = clamp01((s["f1"] - 250) / 750)
        f2n = clamp01((s["f2"] - 600) / 2200)

        hue = 0.66 * (1 - f0n)
        rgb = tuple(int(c * 255) for c in colorsys.hsv_to_rgb(hue, 0.7, 1.0))
        base_r = 40 + 260 * s["rms"]
        aspect_x = lerp(0.6, 1.5, f2n)
        aspect_y = lerp(1.5, 0.6, f2n)
        cy = lerp(h * 0.32, h * 0.68, f1n)
        cx = w * 0.5
        self.rot += (0.3 + f0n * 2.5) * 0.02
        wob = s["jit"] * base_r * 0.6

        screen.fill(self.BG)
        t_now = pygame.time.get_ticks() * 0.004
        pts = []
        for i in range(self.N):
            ang = self.rot + i / self.N * math.tau
            noise = math.sin(self.phases[i] + t_now * 2 + ang * 3) * wob
            r = base_r + noise
            x = cx + math.cos(ang) * r * aspect_x
            y = cy + math.sin(ang) * r * aspect_y
            pts.append((x, y))
        if s["rms"] > 0.01:
            glow = tuple(int(c * 0.35) for c in rgb)
            big = [(cx + (x - cx) * 1.15, cy + (y - cy) * 1.15) for x, y in pts]
            pygame.draw.polygon(screen, glow, big)
            pygame.draw.polygon(screen, rgb, pts)
            pygame.draw.circle(screen, (255, 255, 255), (int(cx), int(cy)), 4)

        lines = [
            "매핑 어휘집  (Levin/Rozin)",
            f"f0  {s['f0']:5.0f}Hz  → 색조·회전   ({'고음' if f0n>0.5 else '저음'})",
            f"rms {s['rms']:4.2f}     → 크기",
            f"f2  {s['f2']:5.0f}Hz  → 가로세로비  ({'전설/납작' if f2n>0.5 else '후설/길쭉'})",
            f"f1  {s['f1']:5.0f}Hz  → 세로위치    ({'열림/아래' if f1n>0.5 else '닫힘/위'})",
            f"jit {s['jit']:4.2f}     → 가장자리 거칠기",
        ]
        for i, ln in enumerate(lines):
            screen.blit(self.font.render(ln, True, (190, 200, 220)), (16, 12 + i * 22))


# ════════════════════════════════════════════════════════════
#  장면 2 — 글자꼴 Visual Mapping (Lieberman/Reas)  [본체]
# ════════════════════════════════════════════════════════════
class Scene02:
    name = "2 · 글자꼴"
    fps = 60
    clears = False

    GLYPHS = {
        "아": ("ㅏ", (255, 90, 90)),  "어": ("ㅓ", (200, 105, 240)),
        "오": ("ㅗ", (90, 230, 130)), "우": ("ㅜ", (80, 200, 240)),
        "으": ("ㅡ", (110, 145, 240)), "이": ("ㅣ", (240, 240, 90)),
        "에": ("ㅔ", (255, 175, 60)),
    }
    DRIFT = {"아": (0, -0.5), "어": (-0.6, 0), "오": (0, -0.35), "우": (0, 0.55),
             "으": (0, 0.1), "이": (0, 0), "에": (0.6, 0)}
    BOX = 320

    def __init__(self, w, h):
        self.w, self.h = w, h
        self.BG = (10, 10, 16)
        self.cx, self.cy = w // 2, h // 2 - 30
        self.big = pygame.font.SysFont(KFONT, 200, bold=True)
        self.ui = pygame.font.SysFont(KFONT, 16)
        self.glyph_surf = {v: self.big.render(j, True, col)
                           for v, (j, col) in self.GLYPHS.items()}
        self.w_ = {v: 0.0 for v in VOWELS}
        self.st = dict(f1=500.0, f2=1500.0, f0=160.0, rms=0.0)

    def on_key(self, key):
        pass

    def render(self, screen, sig):
        w, h, BOX = self.w, self.h, self.BOX
        ww, st = self.w_, self.st
        voiced = bool(sig and sig.voiced)
        target = sig.vowel_weights if (voiced and sig.vowel_weights) else {v: 0 for v in VOWELS}
        for v in VOWELS:
            ww[v] = lerp(ww[v], target.get(v, 0.0), 0.22)
        a = 0.25
        if voiced:
            if sig.f1 > 0: st["f1"] = lerp(st["f1"], sig.f1, a)
            if sig.f2 > 0: st["f2"] = lerp(st["f2"], sig.f2, a)
            if sig.f0 > 0: st["f0"] = lerp(st["f0"], sig.f0, a)
            st["rms"] = lerp(st["rms"], min(1.0, sig.rms * 16), a)
        else:
            st["rms"] = lerp(st["rms"], 0.0, 0.08)

        f1n = clamp01((st["f1"] - 250) / 750)
        f2n = clamp01((st["f2"] - 600) / 2200)
        f0n = clamp01((st["f0"] - 80) / 240)

        canvas = pygame.Surface((BOX, BOX), pygame.SRCALPHA)
        for v in sorted(VOWELS, key=lambda v: ww[v]):
            wt = ww[v]
            if wt < 0.04:
                continue
            s2 = self.glyph_surf[v].copy()
            s2.set_alpha(int(255 * min(1.0, wt * 1.5)))
            dx, dy = self.DRIFT[v]
            rect = s2.get_rect(center=(BOX // 2 + dx * 40 * wt, BOX // 2 + dy * 40 * wt))
            canvas.blit(s2, rect)

        screen.fill(self.BG)
        if st["rms"] > 0.01:
            size = 0.55 + 1.25 * st["rms"]
            sx = lerp(0.55, 1.7, f2n) * size
            sy = lerp(0.55, 1.7, f1n) * size
            scaled = pygame.transform.smoothscale(
                canvas, (max(1, int(BOX * sx)), max(1, int(BOX * sy))))
            angle = (f0n - 0.5) * 80.0
            rotated = pygame.transform.rotate(scaled, angle)
            screen.blit(rotated, rotated.get_rect(center=(self.cx, self.cy)))

        rows = [
            ("F1  글자높이", f"{st['f1']:5.0f}Hz", f1n),
            ("F2  글자폭  ", f"{st['f2']:5.0f}Hz", f2n),
            ("Pitch 회전  ", f"{st['f0']:5.0f}Hz", f0n),
            ("Vol  크기   ", f"{st['rms']:4.2f}", st["rms"]),
        ]
        for i, (lab, val, n) in enumerate(rows):
            y = 16 + i * 22
            screen.blit(self.ui.render(f"{lab}  {val}", True, (190, 200, 220)), (16, y))
            pygame.draw.rect(screen, (60, 70, 90), (200, y + 4, 120, 10), 1)
            pygame.draw.rect(screen, (120, 200, 255), (200, y + 4, int(120 * clamp01(n)), 10))

        bw = w // len(VOWELS)
        for i, v in enumerate(VOWELS):
            j, col = self.GLYPHS[v]
            x = i * bw
            hgt = int(ww[v] * 110)
            pygame.draw.rect(screen, col, (x + 14, h - 28 - hgt, bw - 28, hgt))
            screen.blit(self.ui.render(v, True, col), (x + bw // 2 - 8, h - 24))

        dom = max(VOWELS, key=lambda v: ww[v])
        screen.blit(self.ui.render(
            f"형태(우세): {dom} {ww[dom]:.2f}   ·   '아우/오이' 이어 발음 → 글자꼴 모핑",
            True, (170, 180, 205)), (340, 16))


# ════════════════════════════════════════════════════════════
#  장면 3 — 목소리 거울 (Rozin)
# ════════════════════════════════════════════════════════════
class Scene03:
    name = "3 · 거울"
    fps = 45
    clears = False
    GX, GY = 24, 18
    MARGIN = 40

    def __init__(self, w, h):
        self.w, self.h = w, h
        self.BG = (8, 8, 12)
        self.ui = pygame.font.SysFont(KFONT, 16)
        self.px, self.py = 0.5, 0.5
        self.rms_s, self.f0n_s = 0.0, 0.5
        cw = (w - 2 * self.MARGIN) / self.GX
        ch = (h - 2 * self.MARGIN) / self.GY
        self.cell = min(cw, ch)

    def on_key(self, key):
        pass

    def render(self, screen, sig):
        w, h, M, GX, GY = self.w, self.h, self.MARGIN, self.GX, self.GY
        cell = self.cell
        if sig and sig.voiced:
            f2n = clamp01((sig.f2 - 600) / 2200) if sig.f2 > 0 else self.px
            f1n = clamp01((sig.f1 - 250) / 750) if sig.f1 > 0 else self.py
            f0n = clamp01((sig.f0 - 80) / 240) if sig.f0 > 0 else self.f0n_s
            self.px = lerp(self.px, f2n, 0.25)
            self.py = lerp(self.py, f1n, 0.25)
            self.f0n_s = lerp(self.f0n_s, f0n, 0.2)
            self.rms_s = lerp(self.rms_s, min(1.0, sig.rms * 16), 0.25)
        else:
            self.rms_s = lerp(self.rms_s, 0.0, 0.08)

        screen.fill(self.BG)
        cx = M + self.px * (w - 2 * M)
        cy = M + self.py * (h - 2 * M)
        hue = 0.66 * (1 - self.f0n_s)
        sigma = 2.2 + 2.0 * self.rms_s

        for gy in range(GY):
            for gx in range(GX):
                tx = M + (gx + 0.5) * (w - 2 * M) / GX
                ty = M + (gy + 0.5) * (h - 2 * M) / GY
                d = math.hypot((tx - cx) / cell, (ty - cy) / cell)
                b = math.exp(-(d * d) / (2 * sigma * sigma)) * (0.15 + self.rms_s)
                if b < 0.02:
                    pygame.draw.rect(screen, (22, 24, 32),
                                     (tx - cell * 0.32, ty - cell * 0.32, cell * 0.64, cell * 0.64))
                    continue
                val = clamp01(b)
                rgb = tuple(int(c * 255) for c in colorsys.hsv_to_rgb(hue, 0.6, val))
                ang = (d * 30 - pygame.time.get_ticks() * 0.05) % 360
                size = cell * (0.3 + 0.45 * val)
                surf = pygame.Surface((size * 1.6, size * 1.6), pygame.SRCALPHA)
                pygame.draw.rect(surf, rgb, (size * 0.3, size * 0.3, size, size), border_radius=3)
                surf = pygame.transform.rotate(surf, ang)
                screen.blit(surf, surf.get_rect(center=(tx, ty)))

        screen.blit(self.ui.render(
            "목소리 거울 (Rozin)  —  F2→가로, F1→세로, Volume→밝기, Pitch→색",
            True, (170, 180, 205)), (16, 12))


# ════════════════════════════════════════════════════════════
#  장면 4 — 키네틱 타이포 (Maeda)
# ════════════════════════════════════════════════════════════
class Scene04:
    name = "4 · 키네틱"
    fps = 60
    clears = False
    N = 600
    JAMO = {"아": "ㅏ", "어": "ㅓ", "오": "ㅗ", "우": "ㅜ", "으": "ㅡ", "이": "ㅣ", "에": "ㅔ"}

    def __init__(self, w, h):
        self.w, self.h = w, h
        self.cx, self.cy = w // 2, h // 2
        self.BG = (8, 9, 14)
        self.big = pygame.font.SysFont(KFONT, 380, bold=True)
        self.ui = pygame.font.SysFont(KFONT, 16)
        self.targets = {v: self._glyph_points(self.big, self.JAMO[v], self.N) for v in VOWELS}
        self.rng = np.random.default_rng(0)
        self.p = np.column_stack([self.rng.uniform(0, w, self.N),
                                  self.rng.uniform(0, h, self.N)])
        self.vel = np.zeros((self.N, 2))
        self.cur = "아"
        self.rms_s, self.jit_s, self.f0n_s = 0.0, 0.0, 0.5

    def _glyph_points(self, font, ch, n, scale=1.0):
        surf = font.render(ch, True, (255, 255, 255))
        alpha = pygame.surfarray.array_alpha(surf)
        xs, ys = np.where(alpha > 128)
        if len(xs) == 0:
            return np.zeros((n, 2))
        idx = np.random.default_rng(1).integers(0, len(xs), n)
        gw, gh = surf.get_size()
        pts = np.stack([xs[idx] - gw / 2, ys[idx] - gh / 2], axis=1) * scale
        return pts + np.array([self.cx, self.cy])

    def on_key(self, key):
        pass

    def render(self, screen, sig):
        if sig and sig.voiced:
            if sig.vowel:
                self.cur = sig.vowel
            self.rms_s = lerp(self.rms_s, min(1.0, sig.rms * 16), 0.2)
            self.jit_s = lerp(self.jit_s, sig.jitter, 0.2)
            if sig.f0 > 0:
                self.f0n_s = lerp(self.f0n_s, max(0, min(1, (sig.f0 - 80) / 240)), 0.2)
        else:
            self.rms_s = lerp(self.rms_s, 0.0, 0.06)
            self.jit_s = lerp(self.jit_s, 0.0, 0.1)

        tgt = self.targets[self.cur]
        center = np.array([self.cx, self.cy])
        spring = (tgt - self.p) * 0.06
        out = (self.p - center)
        out = out / (np.linalg.norm(out, axis=1, keepdims=True) + 1e-6)
        scatter = out * (self.rms_s * 6.0)
        shake = self.rng.normal(0, self.jit_s * 6.0, (self.N, 2))
        self.vel = self.vel * 0.82 + spring + scatter * 0.3 + shake * 0.3
        self.p = self.p + self.vel

        screen.fill(self.BG)
        hue = 0.66 * (1 - self.f0n_s)
        rgb = tuple(int(c * 255) for c in colorsys.hsv_to_rgb(hue, 0.55, 1.0))
        size = 2 + int(3 * self.rms_s)
        for i in range(self.N):
            pygame.draw.circle(screen, rgb, (int(self.p[i, 0]), int(self.p[i, 1])), size)

        screen.blit(self.ui.render(
            f"키네틱 타이포 (Maeda)  —  현재 모음: {self.cur}   ·   조용하면 모이고 크게 말하면 흩어짐",
            True, (170, 180, 205)), (16, 12))


# ════════════════════════════════════════════════════════════
#  장면 5 — 우리 Messa di Voce (Levin/Lieberman)   [C: 비움]
# ════════════════════════════════════════════════════════════
class Scene05:
    name = "5 · 흐름(Messa)"
    fps = 60
    clears = True
    VOWEL_COLOR = {
        "아": (255, 90, 90), "어": (200, 105, 240), "오": (90, 230, 130),
        "우": (80, 200, 240), "으": (110, 145, 240), "이": (240, 240, 90), "에": (255, 175, 60),
    }
    JAMO = {"아": "ㅏ", "어": "ㅓ", "오": "ㅗ", "우": "ㅜ", "으": "ㅡ", "이": "ㅣ", "에": "ㅔ"}

    def __init__(self, w, h):
        self.w, self.h = w, h
        self.BG = (6, 7, 11)
        self.SRC = (int(w * 0.18), h // 2)
        self.glyph_font = pygame.font.SysFont(KFONT, 30, bold=True)
        self.ui = pygame.font.SysFont(KFONT, 16)
        self.rng = np.random.default_rng(0)
        self.parts = []
        self.spawn_acc = 0.0
        self.glyph_cache = {}

    def _glyph(self, v, col):
        key = (v, col)
        if key not in self.glyph_cache:
            self.glyph_cache[key] = self.glyph_font.render(self.JAMO[v], True, col)
        return self.glyph_cache[key]

    def on_key(self, key):
        if key == pygame.K_c:
            self.parts.clear()

    def render(self, screen, sig):
        w, h, SRC, rng = self.w, self.h, self.SRC, self.rng
        parts = self.parts
        voiced = bool(sig and sig.voiced and sig.rms * 16 > 0.04)

        if voiced:
            rms = clamp01(sig.rms * 16)
            f0n = clamp01((sig.f0 - 80) / 240) if sig.f0 > 0 else 0.5
            f2n = clamp01((sig.f2 - 600) / 2200) if sig.f2 > 0 else 0.5
            dom = sig.vowel or "아"
            self.spawn_acc += 0.5 + rms * 4.0
            while self.spawn_acc >= 1.0:
                self.spawn_acc -= 1.0
                base = self.VOWEL_COLOR.get(dom, (220, 220, 220))
                hue_shift = (f0n - 0.5) * 0.1
                hsv = colorsys.rgb_to_hsv(*[c / 255 for c in base])
                col = tuple(int(c * 255) for c in colorsys.hsv_to_rgb(
                    (hsv[0] + hue_shift) % 1.0, hsv[1], hsv[2]))
                ang = (f2n - 0.5) * 1.2 + rng.normal(0, 0.25)
                spd = 1.5 + rms * 3.0
                parts.append(dict(
                    x=float(SRC[0]), y=float(SRC[1] + rng.normal(0, 8)),
                    vx=math.cos(ang) * spd + 1.2,
                    vy=math.sin(ang) * spd - (f0n - 0.5) * 4.0,
                    r=4 + rms * 34, col=col, life=1.0,
                    jit=sig.jitter, vowel=dom, stamp=(rms > 0.5),
                ))

        for p in parts:
            p["x"] += p["vx"]
            p["y"] += p["vy"] + math.sin(p["x"] * 0.02) * 0.4
            p["vy"] += 0.012
            p["x"] += rng.normal(0, p["jit"] * 2.0)
            p["life"] -= 0.004
        parts[:] = [p for p in parts if p["life"] > 0 and -50 < p["x"] < w + 50][-1200:]

        screen.fill(self.BG)
        for p in parts:
            a = clamp01(p["life"])
            r = max(1, int(p["r"] * (0.5 + 0.5 * a)))
            surf = pygame.Surface((r * 2, r * 2), pygame.SRCALPHA)
            pygame.draw.circle(surf, (*p["col"], int(120 * a)), (r, r), r)
            pygame.draw.circle(surf, (*p["col"], int(220 * a)), (r, r), max(1, r // 2))
            screen.blit(surf, (p["x"] - r, p["y"] - r))
            if p["stamp"] and a > 0.4:
                g = self._glyph(p["vowel"], p["col"])
                g2 = g.copy(); g2.set_alpha(int(200 * a))
                screen.blit(g2, g2.get_rect(center=(p["x"], p["y"])))

        pulse = 6 + (clamp01(sig.rms * 16) * 18 if voiced else 0)
        pygame.draw.circle(screen, (255, 255, 255), SRC, int(pulse), 2)

        dom = (sig.vowel if (sig and sig.voiced) else "") or "—"
        for i, ln in enumerate([
            "우리 Messa di Voce  —  말하면 형태가 태어난다",
            f"형태 {len(parts)}개 · 우세 모음 {dom}  ·  C: 비움",
        ]):
            screen.blit(self.ui.render(ln, True, (170, 180, 205)), (16, 12 + i * 20))


# ════════════════════════════════════════════════════════════
#  장면 6 — 생성형 손글씨 asemic (Akten 개념)   [C: 비움]
# ════════════════════════════════════════════════════════════
class Scene06:
    name = "6 · 손글씨"
    fps = 60
    clears = True
    PAPER = (250, 248, 242)
    INK = (20, 22, 30)
    # 모음별 획 제스처 (세로 진폭, 주파수) — '손버릇'
    GESTURE = {"아": (1.0, 1.0), "어": (0.7, 1.4), "오": (0.5, 0.7), "우": (0.4, 0.5),
               "으": (0.3, 2.2), "이": (0.9, 3.0), "에": (0.75, 1.8)}

    def __init__(self, w, h):
        self.w, self.h = w, h
        self.ui = pygame.font.SysFont(KFONT, 16)
        self.canvas = pygame.Surface((w, h))
        self.canvas.fill(self.PAPER)
        self.penx, self.peny = 60.0, h / 2
        self.prev = (self.penx, self.peny)
        self.phase = 0.0
        self.baseline, self.curv, self.thick = h / 2, 0.0, 2.0

    def on_key(self, key):
        if key == pygame.K_c:
            self.canvas.fill(self.PAPER)
            self.penx, self.peny = 60.0, self.h / 2
            self.prev = (self.penx, self.peny)

    def render(self, screen, sig):
        w, h = self.w, self.h
        if sig and sig.voiced and sig.rms * 16 > 0.03:
            f1n = clamp01((sig.f1 - 250) / 750) if sig.f1 > 0 else 0.5
            f2n = clamp01((sig.f2 - 600) / 2200) if sig.f2 > 0 else 0.5
            f0n = clamp01((sig.f0 - 80) / 240) if sig.f0 > 0 else 0.5
            amp, freq = self.GESTURE.get(sig.vowel, (0.6, 1.0))
            self.baseline = lerp(self.baseline, lerp(h * 0.25, h * 0.75, f1n), 0.15)
            self.curv = lerp(self.curv, (f2n - 0.5) * 2, 0.2)
            self.thick = lerp(self.thick, 1.5 + f0n * 6, 0.2)

            self.phase += 0.25 * freq
            self.penx += 2.2 + clamp01(sig.rms * 16) * 2.0
            self.peny = self.baseline + math.sin(self.phase) * (40 * amp) + self.curv * 30
            pygame.draw.line(self.canvas, self.INK, self.prev,
                             (self.penx, self.peny), max(1, int(self.thick)))
            self.prev = (self.penx, self.peny)
            if self.penx > w - 60:                              # 줄바꿈
                self.penx = 60.0
                self.baseline = min(self.baseline + 90, h - 60)
                self.prev = (self.penx, self.baseline)
                self.peny = self.baseline
        else:
            self.prev = (self.penx, self.peny)

        screen.blit(self.canvas, (0, 0))
        pygame.draw.circle(screen, (210, 60, 60), (int(self.penx), int(self.peny)), 4)
        screen.blit(self.ui.render(
            "생성형 손글씨(개념) — 말하면 획이 생성됨 · 모음=손버릇, F1=높이, F2=곡률, Pitch=굵기 · C 비움",
            True, (90, 90, 100)), (14, 12))


# ════════════════════════════════════════════════════════════
#  허브 (버튼 바 + 장면 전환)
# ════════════════════════════════════════════════════════════
SCENE_CLASSES = [Scene01, Scene02, Scene03, Scene04, Scene05, Scene06]


def main():
    pygame.init()
    screen = pygame.display.set_mode((WIN_W, WIN_H))
    pygame.display.set_caption("VoiceTypo — 시각 스케치 허브 (1~5 전환)")
    clock = pygame.time.Clock()
    bar_font = pygame.font.SysFont(KFONT, 16, bold=True)
    hint_font = pygame.font.SysFont(KFONT, 13)

    stage = screen.subsurface(STAGE)
    sw, sh = stage.get_size()

    # 마이크 한 번만 열어 모든 장면이 공유
    lis = VoiceListener().start()

    # 장면 인스턴스 (상태 보존: 전환해도 각자 상태 유지)
    scenes = [cls(sw, sh) for cls in SCENE_CLASSES]
    active = 1   # 기본: 글자꼴(본체)

    # 버튼 배치
    n = len(scenes)
    bw, bh, gap, x0, y0 = 138, 38, 6, 12, 10
    btn_rects = [pygame.Rect(x0 + i * (bw + gap), y0, bw, bh) for i in range(n)]

    def switch(i):
        nonlocal active
        if 0 <= i < n:
            active = i

    running = True
    while running:
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                running = False
            elif e.type == pygame.KEYDOWN:
                if e.key == pygame.K_ESCAPE:
                    running = False
                elif pygame.K_1 <= e.key <= pygame.K_6:
                    switch(e.key - pygame.K_1)
                else:
                    scenes[active].on_key(e.key)   # C 등 장면 키
            elif e.type == pygame.MOUSEBUTTONDOWN and e.button == 1:
                for i, r in enumerate(btn_rects):
                    if r.collidepoint(e.pos):
                        switch(i)

        sig = lis.latest()

        # 장면 그리기 (스테이지 영역)
        scenes[active].render(stage, sig)

        # 상단 버튼 바
        pygame.draw.rect(screen, (20, 22, 30), (0, 0, WIN_W, BAR_H))
        pygame.draw.line(screen, (50, 55, 70), (0, BAR_H - 1), (WIN_W, BAR_H - 1))
        for i, (r, sc) in enumerate(zip(btn_rects, scenes)):
            on = (i == active)
            fill = (70, 120, 200) if on else (38, 42, 54)
            pygame.draw.rect(screen, fill, r, border_radius=8)
            pygame.draw.rect(screen, (120, 160, 220) if on else (70, 76, 92), r, 1, border_radius=8)
            txt = bar_font.render(sc.name, True, (255, 255, 255) if on else (180, 188, 205))
            screen.blit(txt, txt.get_rect(center=r.center))

        hint = hint_font.render(
            "버튼 / 숫자키 1~6 전환  ·  C 비움(5·6)  ·  ESC 종료",
            True, (150, 158, 178))
        screen.blit(hint, (x0 + n * (bw + gap) + 14, y0 + 12))

        pygame.display.flip()
        clock.tick(scenes[active].fps)

    lis.stop()
    pygame.quit()


if __name__ == "__main__":
    main()
