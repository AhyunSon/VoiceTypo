"""
evaluation/diagnose_back_vowels.py — 오/우 H1 vs H2 진단

오 / 우 wav 파일의 cepstral envelope 에서 진짜 F2 peak 위치 확인.

가설:
  H1: 본인 오/우 F2 가 학계와 멀음 (발음 차이) → 알고리즘 해결 불가
  H2: 진짜 F2 는 600~900Hz 영역, Praat 가 F0 harmonic (720/960Hz) 에
      빠져 잘못된 F2 측정 → Cepstral smoothing 으로 해결 가능

진단:
  Praat F2 가 학계와 다른 케이스에서 cepstral envelope 가 학계 F2 영역에
  명확한 peak 보이면 H2 (Cepstral 도움됨).
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
DATASET = Path(__file__).resolve().parent / "dataset"
RESULTS = Path(__file__).resolve().parent / "results"

# 학계 평균 (하영우·오재혁 2017, 여성)
ACADEMIC_REFS = {
    "오": dict(F1=363, F2=642),
    "우": dict(F1=332, F2=832),
}

# F2 검출 영역 (학계 ± 200Hz)
F2_ZONES = {
    "오": (450, 900),    # 학계 642 중심
    "우": (650, 1050),   # 학계 832 중심
}


# ══════════════════════════════════════════
# 분석 함수 (diagnose_eo 와 동일)
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
    f0_arr, t_arr = pw.dio(x, float(SAMPLE_RATE),
                            f0_floor=50.0, f0_ceil=500.0,
                            frame_period=10.0)
    f0_arr = pw.stonemask(x, f0_arr, t_arr, float(SAMPLE_RATE))
    voiced = f0_arr[f0_arr > 0]
    return float(np.mean(voiced)) if len(voiced) else None


def fft_spectrum(audio: np.ndarray, sr: int):
    n = len(audio)
    win = audio * np.hamming(n)
    fft = np.fft.rfft(win)
    log_mag = 20 * np.log10(np.maximum(np.abs(fft), 1e-12))
    return np.fft.rfftfreq(n, 1.0 / sr), log_mag


def cepstral_envelope(audio: np.ndarray, sr: int,
                      lifter_quef_ms: float = 4.5):
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
    audio_dc = (audio - np.mean(audio)).astype(np.float64)
    snd = parselmouth.Sound(audio_dc, sampling_frequency=float(sr))
    fmt = snd.to_formant_burg(
        time_step=None, max_number_of_formants=5,
        maximum_formant=5500, window_length=0.030, pre_emphasis_from=50,
    )
    t = audio.shape[0] / sr / 2

    def _g(n):
        v = fmt.get_value_at_time(n, t)
        return None if (v is None or np.isnan(v)) else float(v)
    return _g(1), _g(2), _g(3)


def find_envelope_peaks(freqs: np.ndarray, env_db: np.ndarray,
                        max_freq: float = 3000.0):
    mask = freqs <= max_freq
    peaks_idx, _ = find_peaks(env_db[mask], distance=20, prominence=1.5)
    return freqs[mask][peaks_idx]


# ══════════════════════════════════════════
# 메인
# ══════════════════════════════════════════

def analyze_vowel(vowel: str, axes_col: list, summary: list):
    files = [f"{vowel}_{i:02d}.wav" for i in range(1, 6)]
    refs = ACADEMIC_REFS[vowel]
    f2_zone = F2_ZONES[vowel]

    for i, fname in enumerate(files):
        path = DATASET / fname
        if not path.exists():
            continue
        audio = load_wav(path)
        n = len(audio)
        center_n = int(SAMPLE_RATE * 1.0)
        start = max(0, (n - center_n) // 2)
        win = audio[start:start + center_n] - np.mean(audio[start:start + center_n])

        f0 = compute_f0(win)
        freqs, raw_db = fft_spectrum(win, SAMPLE_RATE)
        ce_freqs, ce_db = cepstral_envelope(win, SAMPLE_RATE, 4.5)
        raw_db_norm = raw_db - np.max(raw_db)
        ce_db_norm = ce_db - np.max(ce_db)
        praat_f1, praat_f2, praat_f3 = praat_formants(audio, SAMPLE_RATE)
        peaks = find_envelope_peaks(ce_freqs, ce_db_norm, 3000)
        peaks_in_zone = [p for p in peaks if f2_zone[0] <= p <= f2_zone[1]]

        ax = axes_col[i]
        max_idx = np.searchsorted(freqs, 3000)
        ax.plot(freqs[:max_idx], raw_db_norm[:max_idx],
                color="lightgray", linewidth=0.6, label="Raw FFT")
        ax.plot(ce_freqs[:max_idx], ce_db_norm[:max_idx],
                color="#2266CC", linewidth=2.0, label="Cepstral env")

        # F0 markers
        if f0:
            for h in (1, 2, 3, 4):
                ax.axvline(f0 * h, color="orange",
                           linewidth=(1.0 if h == 1 else 0.5),
                           linestyle=("-" if h == 1 else ":"),
                           alpha=(0.7 if h == 1 else 0.3))

        # Cepstral peaks
        for pf in peaks:
            in_zone = f2_zone[0] <= pf <= f2_zone[1]
            color = "darkgreen" if in_zone else "green"
            ax.axvline(pf, color=color, linestyle="-",
                       alpha=(0.7 if in_zone else 0.3),
                       linewidth=(1.5 if in_zone else 0.8))
            ax.annotate(f"{pf:.0f}", xy=(pf, 2),
                        color=color, fontsize=8, ha="center")

        # Praat
        if praat_f1:
            ax.axvline(praat_f1, color="red", linestyle="--", linewidth=1.5)
        if praat_f2:
            ax.axvline(praat_f2, color="red", linestyle=":", linewidth=2.0)
            ax.annotate(f"P-F2={praat_f2:.0f}", xy=(praat_f2, -10),
                        color="red", fontsize=8)

        # Academic
        ax.axvline(refs["F1"], color="black", linestyle="-.",
                   linewidth=1.0, alpha=0.6)
        ax.axvline(refs["F2"], color="black", linestyle=":",
                   linewidth=1.5, alpha=0.7)

        # F2 검출 영역 highlight
        ax.axvspan(f2_zone[0], f2_zone[1], color="yellow", alpha=0.10)

        ax.set_xlim(0, 3000)
        ax.set_ylim(-60, 5)
        zone_str = (f"✓ {peaks_in_zone[0]:.0f}"
                     if peaks_in_zone else "✗")
        praat_correct = (praat_f2 is not None
                         and f2_zone[0] <= praat_f2 <= f2_zone[1])
        f2_str = f"{praat_f2:.0f}" if praat_f2 else "—"
        f0_str = f"{f0:.0f}" if f0 else "—"
        ax.set_title(
            f"{fname}  F0={f0_str}  Praat F2={f2_str} "
            f"({'✓' if praat_correct else '✗'})  "
            f"Cepstral zone peak {zone_str}",
            fontsize=10,
        )
        if i == 0:
            ax.legend(loc="upper right", fontsize=7)
        ax.grid(True, alpha=0.3)

        summary.append(dict(
            file=fname, vowel=vowel, f0=f0,
            praat_f1=praat_f1, praat_f2=praat_f2, praat_f3=praat_f3,
            cepstral_peaks=peaks.tolist(),
            peaks_in_f2_zone=peaks_in_zone,
            praat_f2_correct=praat_correct,
        ))


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(5, 2, figsize=(20, 18))

    summary = []
    analyze_vowel("오", axes[:, 0], summary)
    analyze_vowel("우", axes[:, 1], summary)

    fig.suptitle("오 / 우 spectrum 진단 — Cepstral envelope vs Praat F2",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    out_png = RESULTS / "diagnose_back_vowels.png"
    fig.savefig(out_png, dpi=120)
    plt.close(fig)

    # 콘솔 + md
    print("=" * 75)
    print("오 / 우 진단 결과")
    print("=" * 75)

    for vowel in ("오", "우"):
        zone = F2_ZONES[vowel]
        ref = ACADEMIC_REFS[vowel]
        print(f"\n[{vowel}] 학계 F2={ref['F2']}, 검출 영역={zone}")
        items = [s for s in summary if s["vowel"] == vowel]
        for s in items:
            peaks_str = ", ".join(f"{p:.0f}" for p in s["cepstral_peaks"])
            zone_str = ("✓ " + ", ".join(f"{p:.0f}"
                                          for p in s["peaks_in_f2_zone"])
                         if s["peaks_in_f2_zone"] else "✗ 없음")
            praat_mark = "✓" if s["praat_f2_correct"] else "✗"
            f2_str = f"{s['praat_f2']:.0f}" if s["praat_f2"] else "—"
            f0_str = f"{s['f0']:.0f}" if s["f0"] else "—"
            print(f"  {s['file']}: F0={f0_str}  "
                  f"Praat F2={f2_str} {praat_mark}  "
                  f"Cepstral peaks: [{peaks_str}]  "
                  f"F2 영역 envelope peak: {zone_str}")

    # 판정
    print("\n" + "=" * 75)
    print("판정")
    print("=" * 75)
    verdicts = {}
    for vowel in ("오", "우"):
        items = [s for s in summary if s["vowel"] == vowel]
        n_zone = sum(1 for s in items if s["peaks_in_f2_zone"])
        n_praat_correct = sum(1 for s in items if s["praat_f2_correct"])
        n_recoverable = sum(
            1 for s in items
            if s["peaks_in_f2_zone"] and not s["praat_f2_correct"]
        )
        print(f"\n[{vowel}]")
        print(f"  Praat F2 정확 (영역 안): {n_praat_correct}/5")
        print(f"  Cepstral envelope 에 영역 peak 있음: {n_zone}/5")
        print(f"  Cepstral 에 있으나 Praat 가 놓친 케이스: {n_recoverable}/5")
        verdicts[vowel] = (n_zone, n_praat_correct, n_recoverable)

        if n_recoverable >= 3:
            print(f"  → **H2 강함** — Cepstral 통합 시 +{n_recoverable}/5 개선 가능")
        elif n_recoverable >= 1:
            print(f"  → **부분 H2** — Cepstral 통합 시 +{n_recoverable}/5 개선 가능")
        else:
            print(f"  → **H1 가까움** — Cepstral 도움 안 됨")

    # md 보고서
    write_md_report(summary, verdicts, out_png)

    print()
    print("산출물:")
    print(f"  - {out_png}")
    print(f"  - {RESULTS / 'diagnose_back_vowels.md'}")


def write_md_report(summary, verdicts, png_path):
    L = ["# 오 / 우 진단 보고서", ""]
    L.append("## 가설")
    L.append("- **H1**: 본인 오/우 F2 가 학계와 멀음 (발음 차이) → 알고리즘 해결 불가")
    L.append("- **H2**: 진짜 F2 는 학계 영역, Praat 가 F0 harmonic 에 빠져 못 찾음")
    L.append("  → Cepstral smoothing 으로 해결 가능")
    L.append("")
    L.append(f"![spectrum]({png_path.name})")
    L.append("")
    L.append("## 결과 — 오")
    L.append("")
    L.append(f"학계 F2 = {ACADEMIC_REFS['오']['F2']}, 검출 영역 = {F2_ZONES['오']}")
    L.append("")
    L.append("| 파일 | F0 | Praat F2 | F2 정확? | Cepstral peaks | F2 영역 peak |")
    L.append("|---|---:|---:|:---:|---|---|")
    for s in [x for x in summary if x["vowel"] == "오"]:
        peaks = ", ".join(f"{p:.0f}" for p in s["cepstral_peaks"])
        zone = ("✓ " + ", ".join(f"{p:.0f}" for p in s["peaks_in_f2_zone"])
                if s["peaks_in_f2_zone"] else "✗")
        ok = "✓" if s["praat_f2_correct"] else "✗"
        f2v = f"{s['praat_f2']:.0f}" if s["praat_f2"] else "—"
        f0v = f"{s['f0']:.0f}" if s["f0"] else "—"
        L.append(f"| {s['file']} | {f0v} | {f2v} "
                 f"| {ok} | {peaks} | {zone} |")
    L.append("")

    L.append("## 결과 — 우")
    L.append("")
    L.append(f"학계 F2 = {ACADEMIC_REFS['우']['F2']}, 검출 영역 = {F2_ZONES['우']}")
    L.append("")
    L.append("| 파일 | F0 | Praat F2 | F2 정확? | Cepstral peaks | F2 영역 peak |")
    L.append("|---|---:|---:|:---:|---|---|")
    for s in [x for x in summary if x["vowel"] == "우"]:
        peaks = ", ".join(f"{p:.0f}" for p in s["cepstral_peaks"])
        zone = ("✓ " + ", ".join(f"{p:.0f}" for p in s["peaks_in_f2_zone"])
                if s["peaks_in_f2_zone"] else "✗")
        ok = "✓" if s["praat_f2_correct"] else "✗"
        f2v = f"{s['praat_f2']:.0f}" if s["praat_f2"] else "—"
        f0v = f"{s['f0']:.0f}" if s["f0"] else "—"
        L.append(f"| {s['file']} | {f0v} | {f2v} "
                 f"| {ok} | {peaks} | {zone} |")
    L.append("")

    L.append("## 판정")
    L.append("")
    L.append("| 모음 | Praat 정확 | Cepstral 영역 peak | 회수 가능 (Praat 놓침) |")
    L.append("|---|---|---|---|")
    for vowel in ("오", "우"):
        n_zone, n_praat_correct, n_recoverable = verdicts[vowel]
        L.append(f"| {vowel} | {n_praat_correct}/5 | {n_zone}/5 | {n_recoverable}/5 |")
    L.append("")

    total_recoverable = sum(v[2] for v in verdicts.values())
    L.append("## 다음 단계")
    L.append("")
    if total_recoverable >= 4:
        L.append(f"**Cepstral smoothing 구현 가치 있음** — 총 {total_recoverable}/10 회수 가능")
        L.append("- 어 1/5 + 오/우 추가 → 전체 어/오/우 인식률 의미 있는 개선")
    elif total_recoverable >= 2:
        L.append(f"**Cepstral 부분적 가치** — 총 {total_recoverable}/10 회수 가능")
        L.append("- 효과 제한적이나 노력 대비 합리적")
    else:
        L.append(f"**Cepstral ROI 낮음** — 총 {total_recoverable}/10 만 회수 가능")
        L.append("- H1 우세. 다른 방향 모색 필요")
    L.append("")

    (RESULTS / "diagnose_back_vowels.md").write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    main()
