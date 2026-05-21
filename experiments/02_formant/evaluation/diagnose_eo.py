"""
evaluation/diagnose_eo.py — 어 H1 vs H2 진단

본인 어 wav 5개의 FFT 스펙트럼 + cepstral envelope 시각화.
600Hz 근처 진짜 envelope peak 있는지 확인.

H1: 본인 어 F1 이 실제로 ~300Hz (학계 629 보다 낮음). 알고리즘 해결 불가.
H2: 본인 어 진짜 F1 은 ~600Hz, Praat 가 F0 harmonic 에 빠져 못 찾음.
    Cepstral smoothing 으로 복원 가능.

판정 기준:
    5개 파일 모두 ~600Hz 영역에 cepstral peak 없음 → H1 확정
    5개 중 4+ 개에서 ~600Hz cepstral peak → H2 가능
"""

import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import numpy as np
from scipy.io import wavfile
from scipy.signal import find_peaks
import parselmouth
import pyworld as pw

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = ["Malgun Gothic", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import SAMPLE_RATE


# ══════════════════════════════════════════
# 상수
# ══════════════════════════════════════════
EO_FILES = [f"어_{i:02d}.wav" for i in range(1, 6)]
DATASET  = Path(__file__).resolve().parent / "dataset"
RESULTS  = Path(__file__).resolve().parent / "results"

# 학계 어 F1/F2 (하영우·오재혁 2017, 여성)
ACADEMIC_F1 = 629
ACADEMIC_F2 = 950

# 600Hz 영역 검출 범위
F1_PEAK_RANGE = (500, 800)


# ══════════════════════════════════════════
# 분석
# ══════════════════════════════════════════

def load_wav(path: Path) -> np.ndarray:
    sr, data = wavfile.read(str(path))
    if data.dtype == np.int16:
        data = data.astype(np.float32) / 32768.0
    elif data.dtype != np.float32:
        data = data.astype(np.float32)
    if data.ndim > 1:
        data = data[:, 0]
    return data


def compute_f0(audio: np.ndarray) -> float | None:
    x = (audio - np.mean(audio)).astype(np.float64)
    f0_arr, t_arr = pw.dio(
        x, float(SAMPLE_RATE),
        f0_floor=50.0, f0_ceil=500.0, frame_period=10.0,
    )
    f0_arr = pw.stonemask(x, f0_arr, t_arr, float(SAMPLE_RATE))
    voiced = f0_arr[f0_arr > 0]
    return float(np.mean(voiced)) if len(voiced) else None


def fft_spectrum(audio: np.ndarray, sr: int):
    """Hamming 윈도우 + FFT magnitude (dB)."""
    n = len(audio)
    win = audio * np.hamming(n)
    fft = np.fft.rfft(win)
    mag = np.abs(fft)
    log_mag = 20 * np.log10(np.maximum(mag, 1e-12))
    freqs = np.fft.rfftfreq(n, 1.0 / sr)
    return freqs, log_mag


def cepstral_envelope(audio: np.ndarray, sr: int,
                      lifter_quef_ms: float = 4.5):
    """
    Cepstral liftering 으로 spectral envelope 추출.

    1) Hamming 윈도우 + 전체 FFT
    2) log magnitude
    3) IFFT → cepstrum
    4) lifter_quef_ms 이상의 quefrency 제거 (harmonic 제거)
    5) FFT → smoothed log magnitude

    lifter_quef_ms 4.5 ms ≈ F0 222Hz 미만 quefrency 보존
    → harmonic spacing (F0 ~240Hz, period ~4.17ms) 평탄화
    """
    n = len(audio)
    win = audio * np.hamming(n)
    fft_full = np.fft.fft(win)
    log_mag_full = np.log(np.abs(fft_full) + 1e-12)
    cepstrum = np.real(np.fft.ifft(log_mag_full))

    lifter_idx = max(2, int(lifter_quef_ms * 1e-3 * sr))
    cepstrum_lifted = cepstrum.copy()
    cepstrum_lifted[lifter_idx:n - lifter_idx] = 0

    smoothed_log_full = np.real(np.fft.fft(cepstrum_lifted))
    freqs = np.fft.rfftfreq(n, 1.0 / sr)
    return freqs, smoothed_log_full[:len(freqs)]


def praat_formants(audio: np.ndarray, sr: int):
    """Production 과 동일한 Praat 설정 (window 30ms, ceiling 5500)."""
    audio_dc = (audio - np.mean(audio)).astype(np.float64)
    snd = parselmouth.Sound(audio_dc, sampling_frequency=float(sr))
    fmt = snd.to_formant_burg(
        time_step=None,
        max_number_of_formants=5,
        maximum_formant=5500,
        window_length=0.030,
        pre_emphasis_from=50,
    )
    t = audio.shape[0] / sr / 2

    def _g(n):
        v = fmt.get_value_at_time(n, t)
        return None if (v is None or np.isnan(v)) else float(v)

    return _g(1), _g(2), _g(3)


def find_envelope_peaks(freqs: np.ndarray, env_db: np.ndarray,
                        max_freq: float = 3000.0,
                        prominence: float = 1.5):
    """Cepstral envelope 에서 peak 찾기."""
    mask = freqs <= max_freq
    peaks_idx, _ = find_peaks(env_db[mask],
                               distance=20, prominence=prominence)
    return freqs[mask][peaks_idx]


# ══════════════════════════════════════════
# 메인
# ══════════════════════════════════════════

def main():
    RESULTS.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(5, 1, figsize=(13, 18))
    summary = []

    for i, fname in enumerate(EO_FILES):
        path = DATASET / fname
        if not path.exists():
            print(f"⚠ {path} 없음")
            continue

        audio = load_wav(path)

        # 중앙 1.0초 (안정 구간) 사용
        n = len(audio)
        center_n = int(SAMPLE_RATE * 1.0)
        start = max(0, (n - center_n) // 2)
        win = audio[start:start + center_n]
        win = win - np.mean(win)

        # F0
        f0 = compute_f0(win)

        # FFT spectrum (raw)
        freqs, raw_db = fft_spectrum(win, SAMPLE_RATE)
        raw_db_norm = raw_db - np.max(raw_db)

        # Cepstral envelope
        ce_freqs, ce_db = cepstral_envelope(win, SAMPLE_RATE,
                                             lifter_quef_ms=4.5)
        ce_db_norm = ce_db - np.max(ce_db)

        # Praat formants
        praat_f1, praat_f2, praat_f3 = praat_formants(audio, SAMPLE_RATE)

        # Cepstral envelope peaks
        peaks = find_envelope_peaks(ce_freqs, ce_db_norm, max_freq=3000.0)

        # ── Plot ──
        ax = axes[i]
        max_idx = np.searchsorted(freqs, 3000)

        ax.plot(freqs[:max_idx], raw_db_norm[:max_idx],
                color="lightgray", linewidth=0.6, label="Raw FFT")
        ax.plot(ce_freqs[:max_idx], ce_db_norm[:max_idx],
                color="#2266CC", linewidth=2.2, label="Cepstral envelope")

        # F0 mark
        if f0:
            ax.axvline(f0, color="orange", linewidth=1.0, alpha=0.7,
                       label=f"F0={f0:.0f}Hz")
            for h in (2, 3, 4):
                ax.axvline(f0 * h, color="orange", linewidth=0.5,
                           alpha=0.3, linestyle=":")

        # Cepstral peaks
        ymin, ymax = -60, 5
        for pf in peaks:
            ax.axvline(pf, color="green", linestyle="-", alpha=0.4, linewidth=1.0)
            ax.annotate(f"{pf:.0f}", xy=(pf, 2), color="green",
                        fontsize=9, ha="center")

        # Praat F1, F2 (red)
        if praat_f1:
            ax.axvline(praat_f1, color="red", linestyle="--", linewidth=1.8,
                       label=f"Praat F1={praat_f1:.0f}")
        if praat_f2:
            ax.axvline(praat_f2, color="red", linestyle=":", linewidth=1.8,
                       label=f"Praat F2={praat_f2:.0f}")

        # Academic 어 (black)
        ax.axvline(ACADEMIC_F1, color="black", linestyle="-.", linewidth=1.5,
                   alpha=0.7, label=f"학계 어 F1={ACADEMIC_F1}")
        ax.axvline(ACADEMIC_F2, color="black", linestyle=":", linewidth=1.2,
                   alpha=0.6, label=f"학계 어 F2={ACADEMIC_F2}")

        # Highlight F1 검출 영역 (500-800Hz)
        ax.axvspan(F1_PEAK_RANGE[0], F1_PEAK_RANGE[1],
                   color="yellow", alpha=0.12)

        ax.set_xlim(0, 3000)
        ax.set_ylim(ymin, ymax)
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Magnitude (dB, normalized)")

        peak_in_f1_zone = [p for p in peaks
                           if F1_PEAK_RANGE[0] <= p <= F1_PEAK_RANGE[1]]
        f1_zone_str = (f" peak@{peak_in_f1_zone[0]:.0f}"
                        if peak_in_f1_zone else " (없음)")
        f1_str = f"{praat_f1:.0f}" if praat_f1 else "—"
        title = (f"{fname}  F0={f0:.0f}  "
                 f"Praat F1={f1_str}  "
                 f"500-800Hz envelope{f1_zone_str}")
        ax.set_title(title, fontsize=11)
        ax.legend(loc="upper right", fontsize=7, ncol=2)
        ax.grid(True, alpha=0.3)

        summary.append(dict(
            file=fname,
            f0=f0,
            praat_f1=praat_f1, praat_f2=praat_f2, praat_f3=praat_f3,
            cepstral_peaks=peaks.tolist(),
            peak_in_f1_zone=peak_in_f1_zone,
        ))

    fig.suptitle("어 wav 진단 — Cepstral envelope 에서 600Hz 영역 peak 존재 확인 (H1 vs H2)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    out_png = RESULTS / "diagnose_eo_spectrum.png"
    fig.savefig(out_png, dpi=120)
    plt.close(fig)

    # ── 텍스트 보고서 + 콘솔 ──
    print("=" * 75)
    print("어 진단 결과")
    print("=" * 75)
    for s in summary:
        peaks_str = ", ".join(f"{p:.0f}" for p in s["cepstral_peaks"])
        print(f"  {s['file']}: "
              f"F0={s['f0']:.0f}  Praat F1={s['praat_f1']:.0f}  "
              f"F2={s['praat_f2']:.0f}")
        print(f"    Cepstral envelope peaks (Hz, ≤3000): [{peaks_str}]")
        f1_zone = s["peak_in_f1_zone"]
        if f1_zone:
            print(f"    500-800Hz 영역 peak: ✓ {f1_zone[0]:.0f}Hz")
        else:
            print(f"    500-800Hz 영역 peak: ✗ 없음")
        print()

    # 판정
    n_with_peak = sum(1 for s in summary if s["peak_in_f1_zone"])
    print("=" * 75)
    print("판정")
    print("=" * 75)
    print(f"  500-800Hz envelope peak 있는 파일: {n_with_peak}/5")
    if n_with_peak >= 4:
        verdict = ("**H2 가능성 높음** — 진짜 F1 이 600Hz 영역에 존재. "
                   "Praat 가 못 찾을 뿐. Cepstral smoothing → Praat 통합 시도 가치 있음.")
    elif n_with_peak <= 1:
        verdict = ("**H1 확정** — 본인 어의 F1 이 실제로 학계 평균(629Hz)보다 "
                   "낮은 영역에 있음. 알고리즘 수정으로 해결 불가. "
                   "다중 화자 데이터 또는 작품 설계 조정 필요.")
    else:
        verdict = (f"**불명확** — {n_with_peak}/5 만 peak. 추가 진단 또는 "
                   "cepstral 시도 후 판단.")
    print(f"  {verdict}")

    # 보고서 작성
    write_md_report(summary, n_with_peak, verdict, out_png)

    print()
    print("산출물:")
    print(f"  - {out_png}")
    print(f"  - {RESULTS / 'diagnose_eo.md'}")


def write_md_report(summary, n_with_peak, verdict, png_path):
    L = ["# 어 진단 보고서 — H1 vs H2", ""]
    L.append("## 가설")
    L.append("")
    L.append("- **H1**: 본인 어의 진짜 F1 이 학계 평균(629Hz)보다 낮음 (~300Hz)")
    L.append("  → 알고리즘 수정으로 해결 불가")
    L.append("- **H2**: 진짜 F1 은 ~600Hz, Praat LPC 가 F0 harmonic 에 빠져 못 찾음")
    L.append("  → Cepstral smoothing 으로 복원 가능")
    L.append("")
    L.append("## 방법")
    L.append("")
    L.append("어 wav 5개 (어_01~05) 의 중앙 1초 구간에 대해:")
    L.append("1. Hamming 윈도우 + FFT → raw spectrum")
    L.append("2. Cepstral liftering (lifter quefrency 4.5 ms) → smoothed envelope")
    L.append("3. envelope 의 peak 검출 (500-800Hz 영역 = 학계 어 F1 영역)")
    L.append("4. Praat 의 F1/F2 와 비교")
    L.append("")
    L.append(f"![어 spectrum]({png_path.name})")
    L.append("")
    L.append("## 결과")
    L.append("")
    L.append("| 파일 | F0 | Praat F1 | Praat F2 | Cepstral peaks ≤3000Hz | 500-800Hz peak |")
    L.append("|---|---:|---:|---:|---|---|")
    for s in summary:
        peaks = ", ".join(f"{p:.0f}" for p in s["cepstral_peaks"])
        zone = (f"✓ {s['peak_in_f1_zone'][0]:.0f}Hz"
                if s["peak_in_f1_zone"] else "✗ 없음")
        L.append(f"| {s['file']} | {s['f0']:.0f} | {s['praat_f1']:.0f} "
                 f"| {s['praat_f2']:.0f} | {peaks} | {zone} |")
    L.append("")
    L.append(f"**500-800Hz peak 검출**: {n_with_peak}/5 파일")
    L.append("")
    L.append("## 판정")
    L.append("")
    L.append(verdict)
    L.append("")
    L.append("## 다음 단계")
    L.append("")
    if n_with_peak <= 1:
        L.append("**H1 확정**:")
        L.append("- Cepstral smoothing 시도 무의미 (peak 자체가 없음)")
        L.append("- 본인 어는 학계 평균과 다른 발음 패턴 — 작품/연구 의의로 기록 가치")
        L.append("- 작품 진행 옵션:")
        L.append("  1. 잘 되는 모음 (이/으) 위주 + 인식 신뢰도를 시각 효과로 흡수")
        L.append("  2. 다중 화자 데이터 수집 → ML 분류기 학습")
        L.append("  3. 본 한계를 \"학계 평균 _REFS 의 화자별 한계\" 연구 챕터로")
    elif n_with_peak >= 4:
        L.append("**H2 가능성 높음**:")
        L.append("- Phase 4 (Cepstral smoothing) 코드 구현 진행")
        L.append("- formant_engine.py 의 Praat 호출 전에 cepstral pre-smoothing 추가")
        L.append("- 라이브 테스트로 어 인식 개선 확인")
    else:
        L.append("**불명확**:")
        L.append("- 추가 진단 또는 cepstral 시도 후 판단")
    L.append("")
    (RESULTS / "diagnose_eo.md").write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    main()
