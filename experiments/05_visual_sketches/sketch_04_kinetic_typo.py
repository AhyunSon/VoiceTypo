"""
sketch_04_kinetic_typo.py — John Maeda "반응형/키네틱 타이포그래피"

아이디어 (스터디로그 +Maeda):
  글자가 살아 움직인다. 현재 모음 자모를 '입자 구름'으로 그려서,
  조용하면 글자로 모이고, 음량이 크면 흩어지고, 떨림(jitter)이면 진동한다.
  모음이 바뀌면 입자들이 새 글자꼴로 흘러가 재배치된다(모핑).

매핑:
  모음 → 글자(입자 목표 위치) / Volume → 흩어짐 / jitter → 진동 / Pitch → 색

실행:  python sketch_04_kinetic_typo.py   (ESC 종료)
"""

import sys, colorsys
from pathlib import Path

import numpy as np
import pygame

sys.path.insert(0, str(Path(__file__).resolve().parent))
from voice_input import VoiceListener, VOWELS

W, H = 960, 720
BG = (8, 9, 14)
CX, CY = W // 2, H // 2
N = 600                  # 입자 수
JAMO = {"아": "ㅏ", "어": "ㅓ", "오": "ㅗ", "우": "ㅜ", "으": "ㅡ", "이": "ㅣ", "에": "ㅔ"}


def glyph_points(font, ch, n, scale=1.0):
    """자모를 렌더 → 채워진 픽셀 중 n개 샘플 → 화면 중심 기준 좌표."""
    surf = font.render(ch, True, (255, 255, 255))
    alpha = pygame.surfarray.array_alpha(surf)        # (w, h)
    xs, ys = np.where(alpha > 128)
    if len(xs) == 0:
        return np.zeros((n, 2))
    idx = np.random.default_rng(1).integers(0, len(xs), n)
    gw, gh = surf.get_size()
    pts = np.stack([xs[idx] - gw / 2, ys[idx] - gh / 2], axis=1) * scale
    return pts + np.array([CX, CY])


def main():
    pygame.init()
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption("sketch 04 — 키네틱 타이포 (Maeda)")
    clock = pygame.time.Clock()
    big = pygame.font.SysFont("applesdgothicneo,applegothic,malgungothic,arialunicode", 380, bold=True)
    ui = pygame.font.SysFont("applesdgothicneo,applegothic,malgungothic,arialunicode", 16)

    targets = {v: glyph_points(big, JAMO[v], N) for v in VOWELS}

    lis = VoiceListener().start()
    rng = np.random.default_rng(0)
    p = np.column_stack([rng.uniform(0, W, N), rng.uniform(0, H, N)])  # 위치
    vel = np.zeros((N, 2))
    cur = "아"
    rms_s, jit_s, f0n_s = 0.0, 0.0, 0.5

    running = True
    while running:
        for e in pygame.event.get():
            if e.type == pygame.QUIT or (e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE):
                running = False

        sig = lis.latest()
        if sig and sig.voiced:
            if sig.vowel:
                cur = sig.vowel
            rms_s = lerp(rms_s, min(1.0, sig.rms * 16), 0.2)
            jit_s = lerp(jit_s, sig.jit, 0.2)
            if sig.f0 > 0:
                f0n_s = lerp(f0n_s, max(0, min(1, (sig.f0 - 80) / 240)), 0.2)
        else:
            rms_s = lerp(rms_s, 0.0, 0.06)
            jit_s = lerp(jit_s, 0.0, 0.1)

        tgt = targets[cur]
        # 스프링 → 글자꼴로 모임
        spring = (tgt - p) * 0.06
        # 음량 → 바깥으로 흩어지는 힘
        out = (p - np.array([CX, CY]))
        out = out / (np.linalg.norm(out, axis=1, keepdims=True) + 1e-6)
        scatter = out * (rms_s * 6.0)
        # jitter → 무작위 진동
        shake = rng.normal(0, jit_s * 6.0, (N, 2))
        vel = vel * 0.82 + spring + scatter * 0.3 + shake * 0.3
        p = p + vel

        screen.fill(BG)
        hue = 0.66 * (1 - f0n_s)
        rgb = tuple(int(c * 255) for c in colorsys.hsv_to_rgb(hue, 0.55, 1.0))
        size = 2 + int(3 * rms_s)
        for i in range(N):
            pygame.draw.circle(screen, rgb, (int(p[i, 0]), int(p[i, 1])), size)

        for i, ln in enumerate([
            f"키네틱 타이포 (Maeda)  —  현재 모음: {cur}",
            "조용하면 글자로 모임 · 크게 말하면 흩어짐 · 떨면 진동 · 모음 바꾸면 모핑.  ESC",
        ]):
            screen.blit(ui.render(ln, True, (170, 180, 205)), (16, 12 + i * 20))

        pygame.display.flip()
        clock.tick(60)

    lis.stop()
    pygame.quit()


def lerp(a, b, t):
    return a + (b - a) * t


if __name__ == "__main__":
    main()
