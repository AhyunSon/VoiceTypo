"""
evaluation/evaluate.py — VoiceTypo baseline 평가 스크립트

evaluation/dataset/ 의 wav 파일을 현재 시스템 파이프라인에 통과시켜
모음 인식 baseline 정확도를 측정한다.

기존 코드는 일절 수정하지 않으며, 다음만 import 한다:
  config, formant_engine, vowel_classifier, wav2vec_classifier

산출물(evaluation/results/):
  - accuracy_report.md
  - confusion_matrix.png
  - vowel_space.png
  - raw_distribution.csv

실행:
  cd /c/Users/admin/Desktop/realtime_formant
  python -m evaluation.evaluate
"""

import sys
import time
import csv
import threading
import traceback
from pathlib import Path
from collections import defaultdict

# Windows 콘솔(cp949) 한글/유니코드 출력 안전화
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import numpy as np
from scipy.io import wavfile

import matplotlib
matplotlib.use("Agg")              # 비대화형 백엔드 (창 안 띄움)
import matplotlib.pyplot as plt

# Windows 한글 폰트
plt.rcParams["font.family"] = ["Malgun Gothic", "AppleGothic",
                                "NanumGothic", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False

# 프로젝트 루트
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import SAMPLE_RATE, ANALYSIS_WIN_SEC
from formant_engine import FormantEngine
from vowel_classifier import classify_vowel, _REFS as VC_REFS
from wav2vec_classifier import Wav2VecVowelClassifier


# ══════════════════════════════════════════
# 상수
# ══════════════════════════════════════════
VOWELS      = ["아", "에", "이", "오", "우", "으", "어"]
DATASET_DIR = Path(__file__).resolve().parent / "dataset"
RESULTS_DIR = Path(__file__).resolve().parent / "results"
GENDER      = "female"                  # 하드코딩 (재원님 목소리)
CEILINGS    = [3500, 4800, 5200]        # speaker_tracker female 기본값
WIN_SAMPLES = int(SAMPLE_RATE * ANALYSIS_WIN_SEC)   # 13230 = 300 ms
WV_CONF_USE = 0.15                      # ui_window 의 wav2vec 채택 임계
AGR_BOOST   = 0.4                       # ui_window 의 agreement 부스트 임계

VOWEL_COLORS = {
    "아": "#FF4444", "에": "#FFAA22", "이": "#FFFF44",
    "오": "#44FF88", "우": "#44DDFF", "으": "#4488FF",
    "어": "#CC55FF",
}


# ══════════════════════════════════════════
# 데이터 로드
# ══════════════════════════════════════════

def load_wav(path: Path) -> np.ndarray:
    sr, data = wavfile.read(str(path))
    if sr != SAMPLE_RATE:
        raise ValueError(f"sample rate {sr} != {SAMPLE_RATE}")
    if data.dtype == np.int16:
        data = data.astype(np.float32) / 32768.0
    elif data.dtype == np.int32:
        data = data.astype(np.float32) / 2147483648.0
    elif data.dtype != np.float32:
        data = data.astype(np.float32)
    if data.ndim > 1:
        data = data[:, 0]
    return data


def extract_middle_window(audio: np.ndarray) -> np.ndarray:
    """Steady-state — 중앙 ANALYSIS_WIN_SEC(300ms) 절단"""
    n = len(audio)
    if n <= WIN_SAMPLES:
        return np.pad(audio, (0, WIN_SAMPLES - n))
    start = (n - WIN_SAMPLES) // 2
    return audio[start:start + WIN_SAMPLES]


def collect_files() -> list:
    """[(true_vowel, take, path), ...] 정렬"""
    items = []
    for f in sorted(DATASET_DIR.glob("*.wav")):
        stem = f.stem
        if "_" not in stem:
            continue
        v, _, t = stem.partition("_")
        if v in VOWELS and t.isdigit():
            items.append((v, int(t), f))
    items.sort(key=lambda x: (VOWELS.index(x[0]), x[1]))
    return items


# ══════════════════════════════════════════
# wav2vec2 로딩 (백그라운드 스레드 → 동기 대기)
# ══════════════════════════════════════════

def init_wav2vec(load_timeout=180, proto_timeout=60) -> Wav2VecVowelClassifier:
    wv = Wav2VecVowelClassifier()
    done = threading.Event()
    err  = [None]

    def on_ready():
        done.set()

    def on_error(e):
        err[0] = e
        done.set()

    wv.start_loading(on_ready=on_ready, on_error=on_error)

    if not done.wait(load_timeout):
        raise RuntimeError("wav2vec2 모델 로딩 타임아웃")
    if err[0]:
        raise RuntimeError(f"wav2vec2 로딩 실패: {err[0]}")

    # 합성 prototype 빌드 대기 (백그라운드 _build_default_prototypes)
    deadline = time.time() + proto_timeout
    while not wv.has_prototypes and time.time() < deadline:
        time.sleep(0.5)
    return wv


# ══════════════════════════════════════════
# UI 판단 로직 복제 (ui_window._tick lines 808-819)
# ══════════════════════════════════════════

def ui_decide(wv_vowel, wv_conf, f1, f2, raw_f3, agreement):
    if wv_vowel != "?" and wv_conf > WV_CONF_USE:
        return wv_vowel, wv_conf
    vowel, conf = classify_vowel(f1, f2, GENDER, f3=raw_f3, scale=1.0)
    if agreement > AGR_BOOST and vowel != "?":
        conf = min(1.0, conf + agreement * 0.15)
    return vowel, conf


# ══════════════════════════════════════════
# 파일 1개 평가
# ══════════════════════════════════════════

def evaluate_one(path: Path, true_v: str,
                 engine: FormantEngine,
                 wv: Wav2VecVowelClassifier) -> dict:
    audio = load_wav(path)
    chunk = extract_middle_window(audio)
    chunk = chunk - np.mean(chunk)             # DC removal (ui_window:580)

    engine.reset_kalman()                       # 파일별 독립 평가
    res = engine.extract(chunk, GENDER, ceilings=CEILINGS)

    f1, f2, f3       = res["f1"], res["f2"], res["f3"]
    raw_f1           = res.get("raw_f1")
    raw_f2           = res.get("raw_f2")
    raw_f3           = res.get("raw_f3")
    agreement        = res.get("agreement", 0.0)
    f0               = res.get("f0")
    hnr              = res.get("hnr")

    # 1. wav2vec2 분류기
    if wv.is_ready:
        try:
            wv_vowel, wv_conf = wv.classify(
                chunk, sr=SAMPLE_RATE,
                f1=f1, f2=f2, f3=f3, gender=GENDER,
            )
        except Exception:
            wv_vowel, wv_conf = "?", 0.0
    else:
        wv_vowel, wv_conf = "?", 0.0

    # 2. UI 통합 결과 (Full system)
    pred_full, conf_full = ui_decide(
        wv_vowel, wv_conf, f1, f2, raw_f3, agreement,
    )

    # 3. classify_vowel 단독 (포먼트 Mahalanobis)
    pred_cv, conf_cv = classify_vowel(
        f1, f2, GENDER, f3=raw_f3, scale=1.0,
    )

    # 4. _formant_only 단독 (wav2vec 우회)
    try:
        pred_fo, conf_fo = wv._formant_only(
            f1, f2, f3, GENDER, audio=chunk, sr=SAMPLE_RATE,
        )
    except Exception:
        pred_fo, conf_fo = "?", 0.0

    return dict(
        true=true_v, path=path.name,
        f0=f0, hnr=hnr, agreement=agreement,
        f1=f1, f2=f2, f3=f3,
        raw_f1=raw_f1, raw_f2=raw_f2, raw_f3=raw_f3,
        wv_vowel=wv_vowel, wv_conf=wv_conf,
        pred_full=pred_full, conf_full=conf_full,
        pred_cv=pred_cv, conf_cv=conf_cv,
        pred_fo=pred_fo, conf_fo=conf_fo,
    )


# ══════════════════════════════════════════
# 통계
# ══════════════════════════════════════════

def accuracy_breakdown(results, key):
    """{vowel: [correct, total]}"""
    by_v = {v: [0, 0] for v in VOWELS}
    for r in results:
        by_v[r["true"]][1] += 1
        if r[key] == r["true"]:
            by_v[r["true"]][0] += 1
    return by_v


def confusion_matrix(results, key):
    """rows=true, cols=pred (포함 ?)"""
    cols = VOWELS + ["?"]
    col_i = {v: i for i, v in enumerate(cols)}
    M = np.zeros((len(VOWELS), len(cols)), dtype=int)
    for r in results:
        i = VOWELS.index(r["true"])
        pred = r[key]
        j = col_i.get(pred, len(VOWELS))
        M[i, j] += 1
    return M, cols


def raw_stats(results):
    """{vowel: {raw_f1: (n, mean, sd), ...}}"""
    out = {}
    for v in VOWELS:
        rs = [r for r in results if r["true"] == v]
        bucket = {}
        for k in ["raw_f1", "raw_f2", "raw_f3"]:
            vals = [r[k] for r in rs if r[k] is not None]
            mean = float(np.mean(vals)) if vals else None
            sd   = float(np.std(vals))  if len(vals) > 1 else 0.0
            bucket[k] = (len(vals), mean, sd)
        out[v] = bucket
    return out


# ══════════════════════════════════════════
# 산출물
# ══════════════════════════════════════════

def _fmt(v):
    return f"{v:.1f}" if isinstance(v, (int, float)) else ""


def write_csv(results, path: Path):
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "file", "true",
            "f0", "hnr", "agreement",
            "raw_f1", "raw_f2", "raw_f3",
            "kf_f1", "kf_f2", "kf_f3",
            "wv_vowel", "wv_conf",
            "pred_full", "conf_full",
            "pred_cv",   "conf_cv",
            "pred_fo",   "conf_fo",
        ])
        for r in results:
            w.writerow([
                r["path"], r["true"],
                _fmt(r["f0"]), _fmt(r["hnr"]), f"{r['agreement']:.3f}",
                _fmt(r["raw_f1"]), _fmt(r["raw_f2"]), _fmt(r["raw_f3"]),
                _fmt(r["f1"]),     _fmt(r["f2"]),     _fmt(r["f3"]),
                r["wv_vowel"], f"{r['wv_conf']:.3f}",
                r["pred_full"], f"{r['conf_full']:.3f}",
                r["pred_cv"],   f"{r['conf_cv']:.3f}",
                r["pred_fo"],   f"{r['conf_fo']:.3f}",
            ])


def plot_confusion(results, key, title, out_path: Path):
    M, cols = confusion_matrix(results, key)
    fig, ax = plt.subplots(figsize=(7.5, 6.0))
    im = ax.imshow(M, cmap="Blues", aspect="auto")
    ax.set_xticks(range(len(cols)))
    ax.set_yticks(range(len(VOWELS)))
    ax.set_xticklabels(cols, fontsize=12)
    ax.set_yticklabels(VOWELS, fontsize=12)
    ax.set_xlabel("예측 (Predicted)", fontsize=11)
    ax.set_ylabel("정답 (True)", fontsize=11)
    ax.set_title(title, fontsize=13)
    vmax = M.max() if M.max() > 0 else 1
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = int(M[i, j])
            if v == 0:
                continue
            color = "white" if v > vmax * 0.5 else "black"
            ax.text(j, i, str(v), ha="center", va="center",
                    color=color, fontsize=11)
    fig.colorbar(im, ax=ax, label="count")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_vowel_space(results, out_path: Path):
    fig, ax = plt.subplots(figsize=(10, 7.5))

    # _REFS["female"] 타원 (mean ± 1.5 SD ≈ 코드 시각화 기준)
    refs  = VC_REFS["female"]
    theta = np.linspace(0, 2 * np.pi, 80)
    for v, (m1, sd1, m2, sd2, _, _) in refs.items():
        rx = sd2 * 1.5
        ry = sd1 * 1.5
        ex = m2 + rx * np.cos(theta)
        ey = m1 + ry * np.sin(theta)
        ax.plot(ex, ey, color=VOWEL_COLORS[v],
                linewidth=1.4, linestyle="--", alpha=0.55)
        ax.text(m2, m1, v, color=VOWEL_COLORS[v],
                fontsize=15, fontweight="bold",
                ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.15",
                          facecolor="#0d0d1a", edgecolor="none", alpha=0.7))

    # 측정 raw F1/F2 점
    for v in VOWELS:
        rs = [r for r in results
              if r["true"] == v
              and r["raw_f1"] is not None and r["raw_f2"] is not None]
        if not rs:
            continue
        xs = [r["raw_f2"] for r in rs]
        ys = [r["raw_f1"] for r in rs]
        ax.scatter(xs, ys, s=110, c=VOWEL_COLORS[v],
                   alpha=0.85, edgecolors="black", linewidth=0.9,
                   label=f"{v} (n={len(rs)})")

    ax.invert_xaxis()       # 전통 음성학 좌표
    ax.invert_yaxis()
    ax.set_xlabel("F2 (Hz)", fontsize=11)
    ax.set_ylabel("F1 (Hz)", fontsize=11)
    ax.set_title("F1/F2 모음 공간 — 측정값(점) vs _REFS[female] 타원(점선, ±1.5σ)",
                 fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def write_md_report(results, raw, path: Path):
    L = []
    L.append("# VoiceTypo Baseline 평가 보고서\n")
    L.append(f"- 데이터: {len(results)}개 wav (모음 {len(VOWELS)} × ~5)")
    L.append(f"- 성별 가정: `{GENDER}`")
    L.append(f"- 분석 윈도우: 중앙 {ANALYSIS_WIN_SEC*1000:.0f} ms ({WIN_SAMPLES} samples)")
    L.append(f"- Praat ceilings: `{CEILINGS}`")
    L.append("")

    # ── A. 분류 정확도 ──
    L.append("## A. 분류 정확도 (3개 분류기 비교)\n")
    L.append("| 분류기 | 맞음/전체 | 정확도 |")
    L.append("|---|---:|---:|")
    for name, key in [("Full system (wav2vec + 포먼트 통합)", "pred_full"),
                      ("classify_vowel (포먼트 Mahalanobis)",  "pred_cv"),
                      ("_formant_only (wav2vec 우회)",          "pred_fo")]:
        c = sum(1 for r in results if r[key] == r["true"])
        t = len(results)
        L.append(f"| {name} | {c}/{t} | **{c/t*100:.1f}%** |")
    L.append("")

    # 모음별
    L.append("### 모음별 정확도\n")
    L.append("| 모음 | Full system | classify_vowel | _formant_only |")
    L.append("|---|---|---|---|")
    by_full = accuracy_breakdown(results, "pred_full")
    by_cv   = accuracy_breakdown(results, "pred_cv")
    by_fo   = accuracy_breakdown(results, "pred_fo")
    for v in VOWELS:
        c1, t1 = by_full[v]
        c2, t2 = by_cv[v]
        c3, t3 = by_fo[v]
        a1 = f"{c1/t1*100:3.0f}%" if t1 else "—"
        a2 = f"{c2/t2*100:3.0f}%" if t2 else "—"
        a3 = f"{c3/t3*100:3.0f}%" if t3 else "—"
        L.append(f"| {v} | {c1}/{t1} ({a1}) | {c2}/{t2} ({a2}) | {c3}/{t3} ({a3}) |")
    L.append("")

    # ── B. 혼동 행렬 ──
    for name, key in [("Full system",    "pred_full"),
                      ("classify_vowel", "pred_cv"),
                      ("_formant_only",  "pred_fo")]:
        M, cols = confusion_matrix(results, key)
        L.append(f"## B. 혼동 행렬 — {name}\n")
        L.append("rows = 정답, cols = 예측. `?` = 분류 거부\n")
        header = "| 정답＼예측 | " + " | ".join(cols) + " |"
        sep    = "|" + "|".join(["---"] * (len(cols) + 1)) + "|"
        L.append(header)
        L.append(sep)
        for i, v in enumerate(VOWELS):
            row_vals = [str(M[i, j]) if M[i, j] else "·"
                        for j in range(len(cols))]
            L.append(f"| **{v}** | " + " | ".join(row_vals) + " |")
        L.append("")

    # ── C. raw F1/F2/F3 ──
    L.append("## C. raw F1/F2/F3 분포 (vs _REFS[\"female\"])\n")
    L.append("⚠️ = |차이| ≥ 100 Hz\n")
    refs = VC_REFS["female"]
    L.append("| 모음 | n | F1 평균±SD | _REFS F1 | F1 차이 | F2 평균±SD | _REFS F2 | F2 차이 | F3 평균±SD | _REFS F3 | F3 차이 |")
    L.append("|---|---:|---|---:|---:|---|---:|---:|---|---:|---:|")

    def _diff(meas, ref):
        if meas is None:
            return "—"
        d = meas - ref
        warn = " ⚠️" if abs(d) >= 100 else ""
        return f"{d:+.0f}{warn}"

    for v in VOWELS:
        n1, m1, s1 = raw[v]["raw_f1"]
        n2, m2, s2 = raw[v]["raw_f2"]
        n3, m3, s3 = raw[v]["raw_f3"]
        rm1, _, rm2, _, rm3, _ = refs[v]
        f1s = f"{m1:.0f} ± {s1:.0f}" if m1 is not None else "—"
        f2s = f"{m2:.0f} ± {s2:.0f}" if m2 is not None else "—"
        f3s = f"{m3:.0f} ± {s3:.0f}" if m3 is not None else "—"
        L.append(
            f"| {v} | {n1} | {f1s} | {rm1} | {_diff(m1, rm1)} "
            f"| {f2s} | {rm2} | {_diff(m2, rm2)} "
            f"| {f3s} | {rm3} | {_diff(m3, rm3)} |"
        )
    L.append("")

    # ── D. 파일별 상세 ──
    L.append("## D. 파일별 결과\n")
    L.append("✓/✗ = full pred 정오. `cv` = classify_vowel, `fo` = _formant_only\n")
    L.append("| 파일 | 정답 | F0 | raw F1 | raw F2 | raw F3 | wv_vowel(conf) | full | cv | fo |")
    L.append("|---|---|---:|---:|---:|---:|---|---|---|---|")

    def _x(v): return f"{v:.0f}" if v is not None else "—"

    def _mark(p, t):
        if p == t:
            return f"**{p}** ✓"
        return f"{p} ✗"

    for r in results:
        L.append(
            f"| {r['path']} | {r['true']} | {_x(r['f0'])} "
            f"| {_x(r['raw_f1'])} | {_x(r['raw_f2'])} | {_x(r['raw_f3'])} "
            f"| {r['wv_vowel']} ({r['wv_conf']:.2f}) "
            f"| {_mark(r['pred_full'], r['true'])} "
            f"| {_mark(r['pred_cv'],   r['true'])} "
            f"| {_mark(r['pred_fo'],   r['true'])} |"
        )
    L.append("")

    path.write_text("\n".join(L), encoding="utf-8")


# ══════════════════════════════════════════
# 메인
# ══════════════════════════════════════════

def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("VoiceTypo baseline 평가")
    print("=" * 60)

    files = collect_files()
    if not files:
        print(f"⚠ 데이터셋 비어있음: {DATASET_DIR}")
        return
    print(f"  데이터셋: {len(files)}개 wav")
    print(f"  성별 가정: {GENDER}")
    print(f"  분석 윈도우: 중앙 {ANALYSIS_WIN_SEC*1000:.0f} ms")
    print()

    print("FormantEngine 초기화...")
    engine = FormantEngine()

    print("Wav2Vec2 모델 로딩 (~10–60초)...")
    t0 = time.time()
    try:
        wv = init_wav2vec()
        print(f"  → 로딩 완료 ({time.time()-t0:.1f}s). prototype: "
              f"{'OK' if wv.has_prototypes else '미준비'}")
    except Exception as e:
        print(f"⚠ wav2vec2 로딩 실패: {e}")
        print("  → 포먼트 분류만 평가 진행")
        wv = Wav2VecVowelClassifier()
    print()

    # 파일별 평가
    print("─" * 60)
    print("파일별 평가")
    print("─" * 60)
    results = []
    for i, (true_v, take, path) in enumerate(files, 1):
        print(f"  [{i:2d}/{len(files)}] {path.name:<20s}", end=" ")
        try:
            r = evaluate_one(path, true_v, engine, wv)
            results.append(r)
            mark = "✓" if r["pred_full"] == true_v else "✗"
            print(f"{mark} full={r['pred_full']}({r['conf_full']:.2f})  "
                  f"cv={r['pred_cv']}  fo={r['pred_fo']}")
        except Exception as e:
            print(f"⚠ ERROR: {e}")
            traceback.print_exc()

    if not results:
        print("⚠ 결과 없음. 종료.")
        return

    # 요약
    print()
    print("=" * 60)
    print("요약")
    print("=" * 60)
    for name, key in [("Full system        ", "pred_full"),
                      ("classify_vowel     ", "pred_cv"),
                      ("_formant_only      ", "pred_fo")]:
        c = sum(1 for r in results if r[key] == r["true"])
        t = len(results)
        print(f"  {name}: {c:2d}/{t} = {c/t*100:5.1f}%")
    print()
    print(f"  {'모음':<4} | {'Full':<11} | {'cv':<11} | {'fo':<11}")
    print(f"  ----+-------------+-------------+-------------")
    by_full = accuracy_breakdown(results, "pred_full")
    by_cv   = accuracy_breakdown(results, "pred_cv")
    by_fo   = accuracy_breakdown(results, "pred_fo")
    for v in VOWELS:
        cf, tf = by_full[v]
        cv, tv = by_cv[v]
        co, to = by_fo[v]
        af = f"{cf}/{tf} ({cf/tf*100:3.0f}%)" if tf else "—"
        ac = f"{cv}/{tv} ({cv/tv*100:3.0f}%)" if tv else "—"
        ao = f"{co}/{to} ({co/to*100:3.0f}%)" if to else "—"
        print(f"  {v:<4} | {af:<11} | {ac:<11} | {ao:<11}")
    print()

    # 산출물
    raw = raw_stats(results)
    write_md_report(results, raw, RESULTS_DIR / "accuracy_report.md")
    write_csv(results,         RESULTS_DIR / "raw_distribution.csv")
    plot_confusion(results, "pred_full", "Full system 혼동 행렬",
                   RESULTS_DIR / "confusion_matrix.png")
    plot_vowel_space(results,  RESULTS_DIR / "vowel_space.png")

    print("산출물:")
    for f in ["accuracy_report.md", "confusion_matrix.png",
              "vowel_space.png", "raw_distribution.csv"]:
        print(f"  - {RESULTS_DIR / f}")


if __name__ == "__main__":
    main()
