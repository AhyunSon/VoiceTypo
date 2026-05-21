"""
morphing_demo.py — 모음 공간 모핑 시각화 데모

작품 본질: F1/F2 라이브 좌표 → 모음 공간에서 점 자연 이동
("아우" 발음 시 점이 [아] → [우] 자연 보간)

- 마이크 → 300ms 청크 → Praat Burg → F1/F2 추출
- Bark 변환 → 화면 좌표 매핑
- EMA 스무딩 → 부드러운 모핑
- 학계 모음 reference 위치 표시 (학계 평균값)

실행:
  python morphing_demo.py

ESC 또는 창 닫기 → 종료
"""

import sys
import os
import threading
import queue
import time
from pathlib import Path

import numpy as np
import sounddevice as sd
import parselmouth
import pygame

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import SAMPLE_RATE


# ══════════════════════════════════════════
# 한국어 모음 학계 평균 (Yoon 2015 여성 기준)
# ══════════════════════════════════════════

VOWEL_REFS_FEMALE = {
    "아": (978, 1397),
    "에": (548, 2125),
    "이": (352, 2787),
    "오": (487,  840),
    "우": (367,  660),
    "으": (435, 1404),
    "어": (671, 1212),
}
VOWEL_REFS_MALE = {
    "아": (831, 1145),
    "에": (466, 1743),
    "이": (299, 2285),
    "오": (414,  689),
    "우": (312,  541),
    "으": (370, 1151),
    "어": (570,  994),
}

VOWEL_COLORS = {
    "아": (255,  90,  90),
    "에": (255, 175,  60),
    "이": (240, 240,  85),
    "오": ( 90, 230, 130),
    "우": ( 80, 220, 240),
    "으": (110, 145, 240),
    "어": (200, 105, 240),
}


# ══════════════════════════════════════════
# Bark + 좌표 매핑
# ══════════════════════════════════════════

def bark(f):
    f = np.asarray(f, dtype=float)
    return 13.0 * np.arctan(0.00076 * f) + 3.5 * np.arctan((f / 7500.0) ** 2)


# 화면 매핑 범위 (Bark)
F1_MIN_BARK = bark(150.0)   # 위 (닫힘)
F1_MAX_BARK = bark(1200.0)  # 아래 (열림)
F2_MIN_BARK = bark(450.0)   # 오른쪽 (뒤)
F2_MAX_BARK = bark(3300.0)  # 왼쪽 (앞)


def f_to_screen(f1, f2, screen_w, screen_h, margin=80):
    """F1/F2 (Hz) → 화면 좌표 (px). Phonetic chart 표준 (F2 inverted, F1 down)."""
    b1 = bark(f1)
    b2 = bark(f2)

    inner_w = screen_w - 2 * margin
    inner_h = screen_h - 2 * margin

    # F1: 위(닫힘) → 아래(열림)
    y_norm = (b1 - F1_MIN_BARK) / (F1_MAX_BARK - F1_MIN_BARK)
    y = margin + y_norm * inner_h

    # F2: 왼(앞) → 오른(뒤)  ← phonetic chart 표준
    x_norm = 1.0 - (b2 - F2_MIN_BARK) / (F2_MAX_BARK - F2_MIN_BARK)
    x = margin + x_norm * inner_w

    return int(x), int(y)


# ══════════════════════════════════════════
# 포먼트 추출
# ══════════════════════════════════════════

def extract_f1_f2(audio: np.ndarray) -> tuple:
    """단일 청크에서 F1/F2 추출."""
    audio = audio - np.mean(audio)
    rms = float(np.sqrt(np.mean(audio**2)))
    if rms < 0.005:
        return None, None, rms

    try:
        snd = parselmouth.Sound(audio.astype(np.float64),
                                sampling_frequency=float(SAMPLE_RATE))
        fmt = snd.to_formant_burg(
            time_step=None, max_number_of_formants=5,
            maximum_formant=5500, window_length=0.025, pre_emphasis_from=50,
        )
        dur = audio.shape[0] / SAMPLE_RATE
        # 청크 중앙 시점 측정
        t = dur / 2
        f1 = fmt.get_value_at_time(1, t)
        f2 = fmt.get_value_at_time(2, t)
        f1 = float(f1) if (f1 is not None
                           and not np.isnan(f1)
                           and 100 < f1 < 1500) else None
        f2 = float(f2) if (f2 is not None
                           and not np.isnan(f2)
                           and 200 < f2 < 4000) else None
        return f1, f2, rms
    except Exception:
        return None, None, rms


# ══════════════════════════════════════════
# 분석 thread
# ══════════════════════════════════════════

class AnalysisThread(threading.Thread):
    """오디오 큐 → 300ms 청크 → 포먼트 추출 → 결과 큐."""

    def __init__(self, audio_q, result_q, chunk_sec=0.3, hop_sec=0.1):
        super().__init__(daemon=True)
        self.audio_q = audio_q
        self.result_q = result_q
        self.chunk_samples = int(chunk_sec * SAMPLE_RATE)
        self.hop_samples = int(hop_sec * SAMPLE_RATE)
        self.running = True

    def run(self):
        buffer = np.zeros(0, dtype=np.float32)
        while self.running:
            try:
                block = self.audio_q.get(timeout=0.1)
            except queue.Empty:
                continue
            buffer = np.concatenate([buffer, block])

            while len(buffer) >= self.chunk_samples:
                chunk = buffer[:self.chunk_samples]
                buffer = buffer[self.hop_samples:]
                f1, f2, rms = extract_f1_f2(chunk)
                self.result_q.put((f1, f2, rms, time.time()))


# ══════════════════════════════════════════
# 메인 (Pygame)
# ══════════════════════════════════════════

W, H = 1200, 800
EMA_ALPHA = 0.4
TRAIL_MAX = 40
BG_COLOR = (15, 18, 28)


def render_grid(screen, font_small):
    """모음 공간 격자 + 축 라벨."""
    color_grid = (40, 45, 60)
    # F2 격자 (Hz: 500, 1000, 1500, 2000, 2500, 3000)
    for f2 in [500, 1000, 1500, 2000, 2500, 3000]:
        x, _ = f_to_screen(500, f2, W, H)
        pygame.draw.line(screen, color_grid, (x, 50), (x, H - 50), 1)
        label = font_small.render(f"F2={f2}", True, (90, 100, 130))
        screen.blit(label, (x - 25, 30))
    # F1 격자
    for f1 in [200, 400, 600, 800, 1000]:
        _, y = f_to_screen(f1, 1500, W, H)
        pygame.draw.line(screen, color_grid, (50, y), (W - 50, y), 1)
        label = font_small.render(f"F1={f1}", True, (90, 100, 130))
        screen.blit(label, (5, y - 8))


def render_vowel_refs(screen, refs_dict, font_big):
    """학계 모음 reference 점."""
    for v, (f1, f2) in refs_dict.items():
        x, y = f_to_screen(f1, f2, W, H)
        color = VOWEL_COLORS[v]
        # outline circle
        pygame.draw.circle(screen, color, (x, y), 22, width=3)
        # inner dim
        pygame.draw.circle(screen, (color[0]//4, color[1]//4, color[2]//4),
                           (x, y), 18)
        # label
        label = font_big.render(v, True, color)
        screen.blit(label, (x - label.get_width()//2,
                            y - label.get_height()//2))


def render_trail(screen, trail):
    """라이브 점 궤적 (페이드 아웃)."""
    n = len(trail)
    if n < 2:
        return
    for i in range(n - 1):
        f1a, f2a = trail[i]
        f1b, f2b = trail[i + 1]
        xa, ya = f_to_screen(f1a, f2a, W, H)
        xb, yb = f_to_screen(f1b, f2b, W, H)
        alpha = (i + 1) / n
        c = int(255 * alpha)
        pygame.draw.line(screen,
                         (c, int(c * 0.7), c),
                         (xa, ya), (xb, yb), max(1, int(3 * alpha)))


def render_live(screen, current, rms):
    """라이브 점 (RMS 비례 크기)."""
    if current is None:
        return
    f1, f2 = current
    x, y = f_to_screen(f1, f2, W, H)
    radius = int(15 + 30 * min(1.0, rms * 20))
    # 외광
    for r in range(radius, 0, -3):
        alpha = 1 - r / radius
        c = (int(255 * alpha), int(180 * alpha), int(255 * alpha))
        pygame.draw.circle(screen, c, (x, y), r, width=2)
    # 핵심
    pygame.draw.circle(screen, (255, 230, 255), (x, y), 6)


def main():
    pygame.init()
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption("모음 공간 모핑 데모 — F1/F2 라이브")
    clock = pygame.time.Clock()

    # 폰트 (한글)
    try:
        font_big = pygame.font.SysFont("malgungothic", 32, bold=True)
        font_small = pygame.font.SysFont("malgungothic", 16)
        font_info = pygame.font.SysFont("malgungothic", 14)
    except Exception:
        font_big = pygame.font.Font(None, 32)
        font_small = pygame.font.Font(None, 16)
        font_info = pygame.font.Font(None, 14)

    # 큐
    audio_q = queue.Queue(maxsize=30)
    result_q = queue.Queue(maxsize=20)

    # 오디오 콜백
    def audio_cb(indata, frames, time_info, status):
        if status:
            pass
        try:
            audio_q.put_nowait(indata[:, 0].copy().astype(np.float32))
        except queue.Full:
            pass

    block = int(SAMPLE_RATE * 0.05)  # 50ms blocks
    stream = sd.InputStream(
        samplerate=SAMPLE_RATE, channels=1, dtype="float32",
        callback=audio_cb, blocksize=block,
    )

    analyzer = AnalysisThread(audio_q, result_q)
    print("녹음 시작 — 화면에 마이크 입력이 점으로 표시됩니다 (ESC: 종료)")
    stream.start()
    analyzer.start()

    # 상태
    current = None       # EMA 보간된 (f1, f2)
    trail = []
    last_rms = 0.0
    refs_mode = "female"  # 'female' / 'male' (M 키로 전환)

    running = True
    while running:
        # 이벤트
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_m:
                    refs_mode = "male" if refs_mode == "female" else "female"
                    print(f"  refs: {refs_mode}")
                elif event.key == pygame.K_c:
                    trail.clear()
                    current = None

        # 결과 큐 처리
        latest = None
        while True:
            try:
                f1, f2, rms, ts = result_q.get_nowait()
                last_rms = rms
                if f1 is not None and f2 is not None:
                    latest = (f1, f2)
            except queue.Empty:
                break

        if latest is not None:
            if current is None:
                current = latest
            else:
                # EMA 모핑
                current = (
                    (1 - EMA_ALPHA) * current[0] + EMA_ALPHA * latest[0],
                    (1 - EMA_ALPHA) * current[1] + EMA_ALPHA * latest[1],
                )
            trail.append(current)
            if len(trail) > TRAIL_MAX:
                trail.pop(0)

        # 그리기
        screen.fill(BG_COLOR)
        render_grid(screen, font_info)
        refs = VOWEL_REFS_FEMALE if refs_mode == "female" else VOWEL_REFS_MALE
        render_vowel_refs(screen, refs, font_big)
        render_trail(screen, trail)
        render_live(screen, current, last_rms)

        # 정보
        info_lines = [
            f"refs: {refs_mode}  (M: 전환)",
            f"trail: {len(trail)}/{TRAIL_MAX}  (C: 지움)",
        ]
        if current:
            f1, f2 = current
            info_lines.append(f"F1={f1:.0f}  F2={f2:.0f}")
            info_lines.append(f"Bark: ({bark(f1):.1f}, {bark(f2):.1f})")
        info_lines.append(f"RMS: {last_rms:.4f}")

        for i, line in enumerate(info_lines):
            txt = font_info.render(line, True, (200, 210, 230))
            screen.blit(txt, (15, 60 + i * 18))

        # 안내
        guide = font_small.render(
            "마이크에 발음. '아우' 처럼 연속 발화 → 점이 자연스럽게 이동. "
            "ESC 종료.",
            True, (180, 190, 220)
        )
        screen.blit(guide, (15, H - 30))

        pygame.display.flip()
        clock.tick(30)

    stream.stop()
    analyzer.running = False
    pygame.quit()


if __name__ == "__main__":
    main()
