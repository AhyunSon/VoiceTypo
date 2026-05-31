"""
sketch_01_mapping_vocabulary.py — Levin/Rozin "매핑 어휘집"

아이디어 (스터디로그 1·+Rozin):
  음성 파라미터를 시각 요소로 1:1 매핑하는 "어휘집"을 형태 하나로 보여준다.
  화면 중앙의 '목소리 생명체'(노이즈 다각형)가 매핑 표대로 변형된다.

매핑 어휘집 (← 이 표가 실험의 핵심. 마음껏 바꿔보세요):
  f0 (음높이)  → 색조(hue) + 회전 속도
  rms (음량)   → 크기
  f2 (전/후설) → 가로/세로 비율 (납작 ↔ 길쭉)
  f1 (개/폐)   → 세로 위치 (위 ↔ 아래)
  jitter (떨림)→ 가장자리 노이즈(거칠기)

실행:  python sketch_01_mapping_vocabulary.py   (ESC 종료)
"""

import sys, math, colorsys
from pathlib import Path

import numpy as np
import pygame

sys.path.insert(0, str(Path(__file__).resolve().parent))
from voice_input import VoiceListener

W, H = 1100, 750
BG = (12, 14, 22)


def lerp(a, b, t):
    return a + (b - a) * t


def main():
    pygame.init()
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption("sketch 01 — 매핑 어휘집 (Levin/Rozin)")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("applesdgothicneo,applegothic,malgungothic,arialunicode", 16)

    lis = VoiceListener().start()

    # 부드럽게 따라가는 상태값 (EMA)
    s = dict(rms=0.0, f0=160.0, f1=500.0, f2=1500.0, jit=0.0)
    rot = 0.0
    N = 80  # 다각형 꼭짓점 수
    phases = np.random.default_rng(7).uniform(0, math.tau, N)  # 고정 노이즈 위상

    running = True
    while running:
        for e in pygame.event.get():
            if e.type == pygame.QUIT or (e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE):
                running = False

        sig = lis.latest()
        if sig and sig.voiced:
            a = 0.25
            s["rms"] = lerp(s["rms"], min(1.0, sig.rms * 18), a)
            if sig.f0 > 0:  s["f0"] = lerp(s["f0"], sig.f0, a)
            if sig.f1 > 0:  s["f1"] = lerp(s["f1"], sig.f1, a)
            if sig.f2 > 0:  s["f2"] = lerp(s["f2"], sig.f2, a)
            s["jit"] = lerp(s["jit"], sig.jit, a)
        else:
            s["rms"] = lerp(s["rms"], 0.0, 0.08)

        # ── 매핑 적용 ──
        f0n = max(0.0, min(1.0, (s["f0"] - 80) / 240))      # 80~320Hz → 0~1
        f1n = max(0.0, min(1.0, (s["f1"] - 250) / 750))     # 250~1000Hz → 0~1
        f2n = max(0.0, min(1.0, (s["f2"] - 600) / 2200))    # 600~2800Hz → 0~1

        hue = 0.66 * (1 - f0n)                               # 저음=파랑, 고음=빨강
        rgb = tuple(int(c * 255) for c in colorsys.hsv_to_rgb(hue, 0.7, 1.0))
        base_r = 40 + 260 * s["rms"]                         # 음량 → 크기
        aspect_x = lerp(0.6, 1.5, f2n)                       # f2 → 가로 비율
        aspect_y = lerp(1.5, 0.6, f2n)                       # (납작 ↔ 길쭉)
        cy = lerp(H * 0.32, H * 0.68, f1n)                   # f1 → 세로 위치
        cx = W * 0.5
        rot += (0.3 + f0n * 2.5) * 0.02                      # f0 → 회전 속도
        wob = s["jit"] * base_r * 0.6                        # jitter → 가장자리 거칠기

        # ── 그리기 ──
        screen.fill(BG)
        t_now = pygame.time.get_ticks() * 0.004
        pts = []
        for i in range(N):
            ang = rot + i / N * math.tau
            noise = math.sin(phases[i] + t_now * 2 + ang * 3) * wob
            r = base_r + noise
            x = cx + math.cos(ang) * r * aspect_x
            y = cy + math.sin(ang) * r * aspect_y
            pts.append((x, y))
        if s["rms"] > 0.01:
            # 외광 (옅은 큰 형태)
            glow = tuple(int(c * 0.35) for c in rgb)
            big = [(cx + (x - cx) * 1.15, cy + (y - cy) * 1.15) for x, y in pts]
            pygame.draw.polygon(screen, glow, big)
            pygame.draw.polygon(screen, rgb, pts)
            pygame.draw.circle(screen, (255, 255, 255), (int(cx), int(cy)), 4)

        # 어휘집 + 현재값 표시
        lines = [
            "매핑 어휘집  (Levin/Rozin)",
            f"f0  {s['f0']:5.0f}Hz  → 색조·회전   ({'고음' if f0n>0.5 else '저음'})",
            f"rms {s['rms']:4.2f}     → 크기",
            f"f2  {s['f2']:5.0f}Hz  → 가로세로비  ({'전설/납작' if f2n>0.5 else '후설/길쭉'})",
            f"f1  {s['f1']:5.0f}Hz  → 세로위치    ({'열림/아래' if f1n>0.5 else '닫힘/위'})",
            f"jit {s['jit']:4.2f}     → 가장자리 거칠기",
            "",
            "ESC 종료 · 표는 코드에서 자유롭게 변경",
        ]
        for i, ln in enumerate(lines):
            screen.blit(font.render(ln, True, (190, 200, 220)), (16, 14 + i * 22))

        pygame.display.flip()
        clock.tick(60)

    lis.stop()
    pygame.quit()


if __name__ == "__main__":
    main()
