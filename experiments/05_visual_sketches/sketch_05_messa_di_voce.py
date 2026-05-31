"""
sketch_05_messa_di_voce.py — 우리 버전 "Messa di Voce" (Levin/Lieberman 2003 헌정)

⚠️ 제목은 헌정 표기일 뿐, 원작 재현이 아니라 '영감받은 우리 작품'입니다 (REFERENCES.md 참조).

원작의 핵심: 말·외침·노래가 실시간으로 '보이는 형태'가 되어 입에서 뿜어져 나온다.
우리 버전: 발화하는 동안 입(소스 점)에서 형태가 태어나, 그 순간의 목소리를 담고 흘러간다.
  → 말이 멈추면 형태도 멈추고 천천히 사라짐. 계속 말하면 목소리의 '흐름'이 그려짐.

매핑 (우리 파이프라인 통합):
  Volume → 형태 생성량·크기 / Pitch → 색 + 떠오름/가라앉음
  F2     → 흘러가는 방향 폭 / 모음 → 형태의 색조와 자모 각인 / jitter → 흔들림

실행:  python sketch_05_messa_di_voce.py   (ESC 종료 · C: 화면 비움)
"""

import sys, colorsys, math
from pathlib import Path

import numpy as np
import pygame

sys.path.insert(0, str(Path(__file__).resolve().parent))
from voice_input import VoiceListener, VOWELS

W, H = 1280, 720
BG = (6, 7, 11)
SRC = (int(W * 0.18), H // 2)        # '입' 소스 점
VOWEL_COLOR = {
    "아": (255, 90, 90), "어": (200, 105, 240), "오": (90, 230, 130),
    "우": (80, 200, 240), "으": (110, 145, 240), "이": (240, 240, 90), "에": (255, 175, 60),
}
JAMO = {"아": "ㅏ", "어": "ㅓ", "오": "ㅗ", "우": "ㅜ", "으": "ㅡ", "이": "ㅣ", "에": "ㅔ"}


def clamp01(x):
    return max(0.0, min(1.0, x))


def main():
    pygame.init()
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption("우리 Messa di Voce — 말이 보이는 형태가 된다")
    clock = pygame.time.Clock()
    glyph_font = pygame.font.SysFont("applesdgothicneo,applegothic,malgungothic,arialunicode", 30, bold=True)
    ui = pygame.font.SysFont("applesdgothicneo,applegothic,malgungothic,arialunicode", 16)

    lis = VoiceListener().start()
    rng = np.random.default_rng(0)
    parts = []                 # 떠다니는 형태들
    spawn_acc = 0.0
    glyph_cache = {}

    def glyph(v, col):
        key = (v, col)
        if key not in glyph_cache:
            glyph_cache[key] = glyph_font.render(JAMO[v], True, col)
        return glyph_cache[key]

    running = True
    while running:
        for e in pygame.event.get():
            if e.type == pygame.QUIT or (e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE):
                running = False
            elif e.type == pygame.KEYDOWN and e.key == pygame.K_c:
                parts.clear()

        sig = lis.latest()
        voiced = bool(sig and sig.voiced and sig.rms * 16 > 0.04)

        # ── 발화 중이면 형태 생성 (Volume → 생성량) ──
        if voiced:
            rms = clamp01(sig.rms * 16)
            f0n = clamp01((sig.f0 - 80) / 240) if sig.f0 > 0 else 0.5
            f2n = clamp01((sig.f2 - 600) / 2200) if sig.f2 > 0 else 0.5
            dom = sig.vowel or "아"
            spawn_acc += 0.5 + rms * 4.0
            while spawn_acc >= 1.0:
                spawn_acc -= 1.0
                base = VOWEL_COLOR.get(dom, (220, 220, 220))
                hue_shift = (f0n - 0.5) * 0.1
                hsv = colorsys.rgb_to_hsv(*[c / 255 for c in base])
                col = tuple(int(c * 255) for c in colorsys.hsv_to_rgb(
                    (hsv[0] + hue_shift) % 1.0, hsv[1], hsv[2]))
                ang = (f2n - 0.5) * 1.2 + rng.normal(0, 0.25)
                spd = 1.5 + rms * 3.0
                parts.append(dict(
                    x=float(SRC[0]), y=float(SRC[1] + rng.normal(0, 8)),
                    vx=math.cos(ang) * spd + 1.2,
                    vy=math.sin(ang) * spd - (f0n - 0.5) * 4.0,   # Pitch → 떠오름/가라앉음
                    r=4 + rms * 34, col=col, life=1.0,
                    jit=sig.jitter, vowel=dom, stamp=(rms > 0.5),
                ))

        # ── 갱신 ──
        for p in parts:
            p["x"] += p["vx"]
            p["y"] += p["vy"] + math.sin(p["x"] * 0.02) * 0.4
            p["vy"] += 0.012                                # 약한 부유 중력
            p["x"] += rng.normal(0, p["jit"] * 2.0)         # jitter → 흔들림
            p["life"] -= 0.004
        parts[:] = [p for p in parts if p["life"] > 0 and -50 < p["x"] < W + 50][-1200:]

        # ── 그리기 ──
        screen.fill(BG)
        for p in parts:
            a = clamp01(p["life"])
            r = max(1, int(p["r"] * (0.5 + 0.5 * a)))
            col = tuple(int(c * a) for c in p["col"])
            surf = pygame.Surface((r * 2, r * 2), pygame.SRCALPHA)
            pygame.draw.circle(surf, (*p["col"], int(120 * a)), (r, r), r)
            pygame.draw.circle(surf, (*p["col"], int(220 * a)), (r, r), max(1, r // 2))
            screen.blit(surf, (p["x"] - r, p["y"] - r))
            if p["stamp"] and a > 0.4:                       # 큰 소리엔 자모 각인
                g = glyph(p["vowel"], p["col"])
                g2 = g.copy(); g2.set_alpha(int(200 * a))
                screen.blit(g2, g2.get_rect(center=(p["x"], p["y"])))

        # 소스(입) 표시
        pulse = 6 + (clamp01(sig.rms * 16) * 18 if voiced else 0)
        pygame.draw.circle(screen, (255, 255, 255), SRC, int(pulse), 2)

        dom = (sig.vowel if (sig and sig.voiced) else "") or "—"
        for i, ln in enumerate([
            "우리 Messa di Voce  —  말하면 형태가 태어난다",
            f"형태 {len(parts)}개 · 우세 모음 {dom}  ·  계속 말하면 흐름이 그려짐",
            "C: 비움 · ESC 종료",
        ]):
            screen.blit(ui.render(ln, True, (170, 180, 205)), (16, 12 + i * 20))

        pygame.display.flip()
        clock.tick(60)

    lis.stop()
    pygame.quit()


if __name__ == "__main__":
    main()
