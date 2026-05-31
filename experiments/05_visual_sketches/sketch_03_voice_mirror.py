"""
sketch_03_voice_mirror.py — Daniel Rozin "거울(Mirror)" 1:1 매핑

아이디어 (스터디로그 +Rozin):
  Rozin 의 Wooden Mirror 처럼, 입력을 '물리적 타일 밭'에 1:1 로 반사한다.
  목소리의 위치(F2→가로, F1→세로)가 타일 밭의 한 점을 비추고,
  그 점 주변 타일이 밝아지고 기울며 '목소리의 반사상'을 만든다.

매핑:
  F2 → 반사점 가로위치 / F1 → 세로위치 / Volume → 전체 밝기 / Pitch → 타일 색조

실행:  python sketch_03_voice_mirror.py   (ESC 종료)
"""

import sys, math, colorsys
from pathlib import Path

import numpy as np
import pygame

sys.path.insert(0, str(Path(__file__).resolve().parent))
from voice_input import VoiceListener

W, H = 960, 720
BG = (8, 8, 12)
GX, GY = 24, 18          # 타일 격자
MARGIN = 40


def lerp(a, b, t):
    return a + (b - a) * t


def clamp01(x):
    return max(0.0, min(1.0, x))


def main():
    pygame.init()
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption("sketch 03 — 목소리 거울 (Rozin)")
    clock = pygame.time.Clock()
    ui = pygame.font.SysFont("applesdgothicneo,applegothic,malgungothic,arialunicode", 16)

    lis = VoiceListener().start()
    px, py = 0.5, 0.5
    rms_s, f0n_s = 0.0, 0.5

    cw = (W - 2 * MARGIN) / GX
    ch = (H - 2 * MARGIN) / GY
    cell = min(cw, ch)

    running = True
    while running:
        for e in pygame.event.get():
            if e.type == pygame.QUIT or (e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE):
                running = False

        sig = lis.latest()
        if sig and sig.voiced:
            f2n = clamp01((sig.f2 - 600) / 2200) if sig.f2 > 0 else px
            f1n = clamp01((sig.f1 - 250) / 750) if sig.f1 > 0 else py
            f0n = clamp01((sig.f0 - 80) / 240) if sig.f0 > 0 else f0n_s
            px = lerp(px, f2n, 0.25)
            py = lerp(py, f1n, 0.25)
            f0n_s = lerp(f0n_s, f0n, 0.2)
            rms_s = lerp(rms_s, min(1.0, sig.rms * 16), 0.25)
        else:
            rms_s = lerp(rms_s, 0.0, 0.08)

        screen.fill(BG)
        # 반사점 (화면 좌표)
        cx = MARGIN + px * (W - 2 * MARGIN)
        cy = MARGIN + py * (H - 2 * MARGIN)
        hue = 0.66 * (1 - f0n_s)
        sigma = 2.2 + 2.0 * rms_s     # 반사 번짐 (음량 클수록 넓게)

        for gy in range(GY):
            for gx in range(GX):
                tx = MARGIN + (gx + 0.5) * (W - 2 * MARGIN) / GX
                ty = MARGIN + (gy + 0.5) * (H - 2 * MARGIN) / GY
                d = math.hypot((tx - cx) / cell, (ty - cy) / cell)
                b = math.exp(-(d * d) / (2 * sigma * sigma)) * (0.15 + rms_s)
                if b < 0.02:
                    # 어두운 바탕 타일
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

        for i, ln in enumerate([
            "목소리 거울 (Rozin)  —  F2→가로, F1→세로, Volume→밝기, Pitch→색",
            "발음을 바꾸면 반사점이 타일 밭 위를 이동.  ESC 종료",
        ]):
            screen.blit(ui.render(ln, True, (170, 180, 205)), (16, 12 + i * 20))

        pygame.display.flip()
        clock.tick(45)

    lis.stop()
    pygame.quit()


if __name__ == "__main__":
    main()
