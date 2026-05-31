"""
asemic_strokes.py — Memo Akten "생성형 글자" 레이어 (개념 프로토타입)

⚠️ 이건 '개념 스탠드인'입니다.
  Akten 의 진짜 레이어는 학습형(Graves 2013 RMDN / LSTM 생성형 손글씨)이고 GPU 학습이 필요.
  집에선 못 돌리므로, 같은 아이디어("목소리가 획을 생성한다")를 **절차적 규칙**으로 흉내냄.
  → 나중에 연구실에서 이 입력(모음열/포먼트)을 학습형 생성기로 교체하면 본 레이어가 됨.

하는 일:
  말하는 동안 펜이 화면을 가로질러 흐르며 손글씨 같은 '의미 없는(asemic)' 획을 그린다.
  펜의 높낮이·곡률이 F1/F2/Pitch 로 조절되고, 모음마다 고유한 획 제스처가 주입된다.

매핑:
  F1 → 펜 세로위치 / F2 → 곡률 / Pitch → 굵기 / 모음 → 획 제스처 / Volume → 진하기

실행:  python ai_generative/asemic_strokes.py   (ESC 종료 · C: 비움)
"""

import sys, math
from pathlib import Path

import numpy as np
import pygame

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # 상위(voice_input) 접근
from voice_input import VoiceListener

W, H = 1280, 600
BG = (250, 248, 242)
INK = (20, 22, 30)
# 모음별 획 제스처 (세로 진폭, 주파수) — '손버릇'
GESTURE = {"아": (1.0, 1.0), "어": (0.7, 1.4), "오": (0.5, 0.7), "우": (0.4, 0.5),
           "으": (0.3, 2.2), "이": (0.9, 3.0), "에": (0.75, 1.8)}


def clamp01(x):
    return max(0.0, min(1.0, x))


def lerp(a, b, t):
    return a + (b - a) * t


def main():
    pygame.init()
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption("asemic strokes — 생성형 손글씨 (Akten 개념)")
    canvas = pygame.Surface((W, H))
    canvas.fill(BG)
    clock = pygame.time.Clock()
    ui = pygame.font.SysFont("applesdgothicneo,applegothic,malgungothic,arialunicode", 16)

    lis = VoiceListener().start()
    penx, peny = 60.0, H / 2
    prev = (penx, peny)
    phase = 0.0
    baseline, curv, thick = H / 2, 0.0, 2.0

    running = True
    while running:
        for e in pygame.event.get():
            if e.type == pygame.QUIT or (e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE):
                running = False
            elif e.type == pygame.KEYDOWN and e.key == pygame.K_c:
                canvas.fill(BG); penx, peny = 60.0, H / 2; prev = (penx, peny)

        sig = lis.latest()
        if sig and sig.voiced and sig.rms * 16 > 0.03:
            f1n = clamp01((sig.f1 - 250) / 750) if sig.f1 > 0 else 0.5
            f2n = clamp01((sig.f2 - 600) / 2200) if sig.f2 > 0 else 0.5
            f0n = clamp01((sig.f0 - 80) / 240) if sig.f0 > 0 else 0.5
            amp, freq = GESTURE.get(sig.vowel, (0.6, 1.0))
            baseline = lerp(baseline, lerp(H * 0.25, H * 0.75, f1n), 0.15)
            curv = lerp(curv, (f2n - 0.5) * 2, 0.2)
            thick = lerp(thick, 1.5 + f0n * 6, 0.2)

            # 펜 전진 + 모음 제스처에 따른 상하 진동
            phase += 0.25 * freq
            penx += 2.2 + clamp01(sig.rms * 16) * 2.0
            peny = baseline + math.sin(phase) * (40 * amp) + curv * 30
            ink = tuple(int(c) for c in INK)
            pygame.draw.line(canvas, ink, prev, (penx, peny), max(1, int(thick)))
            prev = (penx, peny)
            if penx > W - 60:                                # 줄바꿈
                penx = 60.0; baseline = min(baseline + 90, H - 60)
                prev = (penx, baseline); peny = baseline
        else:
            prev = (penx, peny)

        screen.blit(canvas, (0, 0))
        pygame.draw.circle(screen, (210, 60, 60), (int(penx), int(peny)), 4)
        screen.blit(ui.render(
            "생성형 손글씨(개념) — 말하면 획이 생성됨 · 모음=손버릇, F1=높이, F2=곡률, Pitch=굵기 · C 비움 · ESC",
            True, (90, 90, 100)), (14, 12))
        pygame.display.flip()
        clock.tick(60)

    lis.stop()
    pygame.quit()


if __name__ == "__main__":
    main()
