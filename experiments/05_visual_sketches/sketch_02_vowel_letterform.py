"""
sketch_02_vowel_letterform.py — 우리 파이프라인의 Visual Mapping 레이어

레퍼런스 총정리 문서의 매핑을 그대로 구현:
    Mic Input → Audio Analysis → [Visual Mapping] → Realtime Rendering

우리 매핑 (이 표가 핵심 — 자유롭게 수정):
    F1     → 글자 높이   (낮은 F1=닫힘=납작 / 높은 F1=열림=세로로 큼)
    F2     → 글자 폭     (낮은 F2=후설=좁음 / 높은 F2=전설=넓음)
    Pitch  → 회전        (음높이에 따라 글자 기울기)
    Volume → 크기        (음량에 따라 전체 스케일)
    모음   → 형태        (vowel_weights 로 7개 자모를 연속 블렌딩)

멘토: 개념=Levin+Lieberman / 구현 루프=Lieberman / 형태 규칙=Reas

실행:  python sketch_02_vowel_letterform.py   (ESC 종료)
"""

import sys
from pathlib import Path

import pygame

sys.path.insert(0, str(Path(__file__).resolve().parent))
from voice_input import VoiceListener, VOWELS

W, H = 1000, 760
BG = (10, 10, 16)
CX, CY = W // 2, H // 2 - 30

# 모음 → (자모, 색). "형태"는 글리프 정체성 + 약한 방향 오프셋으로 표현.
GLYPHS = {
    "아": ("ㅏ", (255, 90, 90)),  "어": ("ㅓ", (200, 105, 240)),
    "오": ("ㅗ", (90, 230, 130)), "우": ("ㅜ", (80, 200, 240)),
    "으": ("ㅡ", (110, 145, 240)), "이": ("ㅣ", (240, 240, 90)),
    "에": ("ㅔ", (255, 175, 60)),
}
# 모음별 약한 방향감 (형태가 어디로 쏠리는지) — dx,dy 비율
DRIFT = {"아": (0, -0.5), "어": (-0.6, 0), "오": (0, -0.35), "우": (0, 0.55),
         "으": (0, 0.1), "이": (0, 0), "에": (0.6, 0)}

BOX = 320  # 글리프 합성 캔버스 한 변


def lerp(a, b, t):
    return a + (b - a) * t


def clamp01(x):
    return max(0.0, min(1.0, x))


def main():
    pygame.init()
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption("sketch 02 — Visual Mapping (F1·F2·Pitch·Volume·모음)")
    clock = pygame.time.Clock()
    big = pygame.font.SysFont("applesdgothicneo,applegothic,malgungothic,arialunicode", 200, bold=True)
    ui = pygame.font.SysFont("applesdgothicneo,applegothic,malgungothic,arialunicode", 16)

    glyph_surf = {v: big.render(j, True, col) for v, (j, col) in GLYPHS.items()}

    lis = VoiceListener().start()
    w = {v: 0.0 for v in VOWELS}
    st = dict(f1=500.0, f2=1500.0, f0=160.0, rms=0.0)

    running = True
    while running:
        for e in pygame.event.get():
            if e.type == pygame.QUIT or (e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE):
                running = False

        sig = lis.latest()
        voiced = bool(sig and sig.voiced)
        target = sig.vowel_weights if (voiced and sig.vowel_weights) else {v: 0 for v in VOWELS}
        for v in VOWELS:
            w[v] = lerp(w[v], target.get(v, 0.0), 0.22)
        a = 0.25
        if voiced:
            if sig.f1 > 0: st["f1"] = lerp(st["f1"], sig.f1, a)
            if sig.f2 > 0: st["f2"] = lerp(st["f2"], sig.f2, a)
            if sig.f0 > 0: st["f0"] = lerp(st["f0"], sig.f0, a)
            st["rms"] = lerp(st["rms"], min(1.0, sig.rms * 16), a)
        else:
            st["rms"] = lerp(st["rms"], 0.0, 0.08)

        # ── 매핑값 정규화 ──
        f1n = clamp01((st["f1"] - 250) / 750)    # 글자 높이
        f2n = clamp01((st["f2"] - 600) / 2200)   # 글자 폭
        f0n = clamp01((st["f0"] - 80) / 240)     # 회전

        # ── 1) 모음 → 형태: 가중치로 자모 합성 (offscreen) ──
        canvas = pygame.Surface((BOX, BOX), pygame.SRCALPHA)
        for v in sorted(VOWELS, key=lambda v: w[v]):
            wt = w[v]
            if wt < 0.04:
                continue
            g = glyph_surf[v]
            s2 = g.copy()
            s2.set_alpha(int(255 * min(1.0, wt * 1.5)))
            dx, dy = DRIFT[v]
            rect = s2.get_rect(center=(BOX // 2 + dx * 40 * wt, BOX // 2 + dy * 40 * wt))
            canvas.blit(s2, rect)

        screen.fill(BG)
        if st["rms"] > 0.01:
            # ── 2) 전역 변형: F2→폭, F1→높이, Volume→크기, Pitch→회전 ──
            size = 0.55 + 1.25 * st["rms"]
            sx = lerp(0.55, 1.7, f2n) * size
            sy = lerp(0.55, 1.7, f1n) * size
            scaled = pygame.transform.smoothscale(
                canvas, (max(1, int(BOX * sx)), max(1, int(BOX * sy))))
            angle = (f0n - 0.5) * 80.0   # -40°~+40°
            rotated = pygame.transform.rotate(scaled, angle)
            screen.blit(rotated, rotated.get_rect(center=(CX, CY)))

        # ── 매핑표 + 현재값 (HUD) ──
        rows = [
            ("F1  글자높이", f"{st['f1']:5.0f}Hz", f1n),
            ("F2  글자폭  ", f"{st['f2']:5.0f}Hz", f2n),
            ("Pitch 회전  ", f"{st['f0']:5.0f}Hz", f0n),
            ("Vol  크기   ", f"{st['rms']:4.2f}", st["rms"]),
        ]
        for i, (lab, val, n) in enumerate(rows):
            y = 16 + i * 22
            screen.blit(ui.render(f"{lab}  {val}", True, (190, 200, 220)), (16, y))
            pygame.draw.rect(screen, (60, 70, 90), (200, y + 4, 120, 10), 1)
            pygame.draw.rect(screen, (120, 200, 255), (200, y + 4, int(120 * clamp01(n)), 10))

        # 7모음 가중치 막대
        bw = W // len(VOWELS)
        for i, v in enumerate(VOWELS):
            j, col = GLYPHS[v]
            x = i * bw
            hgt = int(w[v] * 110)
            pygame.draw.rect(screen, col, (x + 14, H - 28 - hgt, bw - 28, hgt))
            screen.blit(ui.render(v, True, col), (x + bw // 2 - 8, H - 24))

        dom = max(VOWELS, key=lambda v: w[v])
        screen.blit(ui.render(
            f"형태(우세): {dom} {w[dom]:.2f}   ·   '아우/오이' 이어 발음 → 글자꼴 모핑   ·   ESC",
            True, (170, 180, 205)), (340, 16))

        pygame.display.flip()
        clock.tick(60)

    lis.stop()
    pygame.quit()


if __name__ == "__main__":
    main()
