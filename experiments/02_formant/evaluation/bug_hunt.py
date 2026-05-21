"""
evaluation/bug_hunt.py — baseline 22.9% 버그 헌트

수행 항목:
  1. wav2vec2 모델 실제 로딩/사용 검증 (self._ready, _default_proto)
  2. 단일 파일(아_01.wav) 단계별 추적
       wav 메타 → FormantEngine → K-NN 코사인 유사도 → 분기 → 최종
  3. raw F1/F2 분포 vs 학계 평균 비교 (하영우·오재혁 2017)
  4. 결론: 추출 vs 분류 어느 쪽이 깨졌는가

산출: evaluation/bug_hunt.md
"""

import sys
import time
import csv
import threading
from pathlib import Path
from collections import defaultdict

import numpy as np
from scipy.io import wavfile

# Windows 콘솔 UTF-8
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import SAMPLE_RATE, ANALYSIS_WIN_SEC
from formant_engine import FormantEngine
from vowel_classifier import _REFS as VC_REFS
from wav2vec_classifier import (
    Wav2VecVowelClassifier,
    formant_vowel_probs,
    _normalizer,
)


# ══════════════════════════════════════════
# 상수
# ══════════════════════════════════════════
VOWELS    = ["아", "에", "이", "오", "우", "으", "어"]
HERE      = Path(__file__).resolve().parent
DATASET   = HERE / "dataset"
RESULTS   = HERE / "results"
REPORT    = HERE / "bug_hunt.md"
TARGET    = DATASET / "아_01.wav"
GENDER    = "female"
CEILINGS  = [3500, 4800, 5200]
WIN       = int(SAMPLE_RATE * ANALYSIS_WIN_SEC)

# 학계 평균: 하영우·오재혁 (2017), 여성 아나운서 8명
ACADEMIC_REFS = {
    "아": (996, 1503),
    "에": (477, 2514),
    "이": (289, 2716),
    "오": (363,  642),
    "우": (332,  832),
    "으": (344, 1711),
    "어": (629,  950),
}


# ══════════════════════════════════════════
# 출력 헬퍼: stdout + markdown 동시 작성
# ══════════════════════════════════════════
md_buf = ["# VoiceTypo Bug Hunt 보고서", ""]

def section(title: str, level: int = 2):
    bar = "=" * 60
    print(f"\n{bar}")
    print(title)
    print(bar)
    md_buf.append("")
    md_buf.append(f"{'#' * level} {title}")
    md_buf.append("")

def subsection(title: str):
    print(f"\n--- {title} ---")
    md_buf.append("")
    md_buf.append(f"### {title}")
    md_buf.append("")

def code(content_fn):
    """함수 내부에서 say(s)로 라인 출력 → 결과를 ``` 코드블록에 wrap"""
    md_buf.append("```")
    content_fn()
    md_buf.append("```")
    md_buf.append("")

def say(s: str = ""):
    print(s)
    md_buf.append(s)

def md(s: str = ""):
    md_buf.append(s)


# ══════════════════════════════════════════
# 0. 분류기 초기화 상태
# ══════════════════════════════════════════
section("0. 분류기 초기화 상태 검증")

engine = FormantEngine()
wv = Wav2VecVowelClassifier()
done = threading.Event()
err  = [None]

def _on_ready():
    done.set()

def _on_error(e):
    err[0] = e
    done.set()

def _step0():
    say("wav2vec2 로딩 중...")
    t0 = time.time()
    wv.start_loading(on_ready=_on_ready, on_error=_on_error)
    ok = done.wait(180)
    say(f"  done    = {ok}")
    say(f"  elapsed = {time.time()-t0:.1f}s")
    say(f"  err     = {err[0]}")
    say()
    say(f"wv.is_ready  = {wv.is_ready}     # property")
    say(f"wv._ready    = {wv._ready}     # private 속성 직접 확인")
    say(f"wv._error    = {wv._error}")
    say()
    say("기본 prototype 빌드 대기...")
    deadline = time.time() + 60
    while not wv.has_prototypes and time.time() < deadline:
        time.sleep(0.5)
    say(f"wv.has_prototypes      = {wv.has_prototypes}")
    say(f"len(wv._default_proto) = {len(wv._default_proto)}")
    say(f"keys: {sorted(wv._default_proto.keys())}")
    say()
    say("프로토타입 임베딩 검사 (정상이면 norm≈1.0):")
    for v in VOWELS:
        if v in wv._default_proto:
            p = wv._default_proto[v]
            say(f"  {v}: shape={p.shape}  dtype={p.dtype}  norm={np.linalg.norm(p):.4f}")
        else:
            say(f"  {v}: ⚠ 없음")
    say()
    say(f"_normalizer.ready = {_normalizer.ready}")

code(_step0)


# ══════════════════════════════════════════
# 1. 단일 파일 단계별 추적
# ══════════════════════════════════════════
section(f"1. 단일 파일 단계별 추적: {TARGET.name}")

if not TARGET.exists():
    say(f"⚠ {TARGET} 없음")
    REPORT.write_text("\n".join(md_buf), encoding="utf-8")
    sys.exit(1)

# 1.1 wav 메타데이터
subsection("1.1 WAV 메타데이터")
sr_, raw = wavfile.read(str(TARGET))

def _step1_1():
    say(f"파일: {TARGET.name}")
    say(f"샘플레이트: {sr_} Hz   (시스템 SAMPLE_RATE = {SAMPLE_RATE} Hz)")
    say(f"길이: {len(raw)} samples = {len(raw)/sr_:.3f}s")
    say(f"dtype: {raw.dtype}")
    say(f"min/max: {raw.min()} / {raw.max()}")
    say(f"평균: {float(np.mean(raw)):.4f}")
code(_step1_1)

# normalize
audio = raw.astype(np.float32) if raw.dtype == np.float32 else raw.astype(np.float32) / 32768.0
if audio.ndim > 1:
    audio = audio[:, 0]

# 중앙 윈도우
n = len(audio)
start = (n - WIN) // 2
chunk = audio[start:start + WIN].copy()
chunk = chunk - np.mean(chunk)


# 1.2 FormantEngine.extract()
subsection("1.2 FormantEngine.extract()")

def _step1_2():
    global res
    say(f"중앙 윈도우 추출: [{start} : {start+WIN}] ({WIN} samples = {ANALYSIS_WIN_SEC*1000:.0f}ms)")
    say(f"chunk RMS: {np.sqrt(np.mean(chunk**2)):.4f}")
    say()
    engine.reset_kalman()
    res = engine.extract(chunk, GENDER, ceilings=CEILINGS)
    for k in ("f0", "hnr", "is_voiced", "agreement", "confidence",
              "raw_f1", "raw_f2", "raw_f3", "f1", "f2", "f3", "jitter"):
        v = res.get(k)
        if isinstance(v, float):
            say(f"  {k:12s} = {v:.3f}")
        else:
            say(f"  {k:12s} = {v}")
code(_step1_2)


# 1.3 wav2vec2 K-NN
subsection("1.3 wav2vec2 K-NN 단계")

def _step1_3():
    global feat, knn_vowel, knn_conf, sims_sorted
    a16k = wv._prep_audio(chunk, SAMPLE_RATE)
    say(f"_prep_audio: {len(a16k)} samples @ 16kHz (= {len(a16k)/16000:.3f}s)")
    feat = wv._extract_hidden(a16k)
    if feat is None:
        say("⚠ _extract_hidden() returned None")
        knn_vowel, knn_conf = "?", 0.0
        sims_sorted = []
        return
    say(f"feat: shape={feat.shape}  norm={np.linalg.norm(feat):.4f}")
    say()
    say("코사인 유사도 vs default_proto:")
    sims = {v: float(np.dot(feat, wv._default_proto[v])) for v in VOWELS
            if v in wv._default_proto}
    sims_sorted = sorted(sims.items(), key=lambda x: -x[1])
    for v, s in sims_sorted:
        mark = "  ←best" if v == sims_sorted[0][0] else ""
        say(f"  {v}: {s:+.4f}{mark}")
    best_v, best_s = sims_sorted[0]
    second_s = sims_sorted[1][1]
    margin = best_s - second_s
    confidence = min(margin * 5.0, 1.0)
    say()
    say(f"best={best_v}  top sim={best_s:.4f}  margin={margin:.4f}  conf={confidence:.3f}")
    pass_thresh = best_s >= 0.45
    say(f"_knn_classify 임계 (top sim ≥ 0.45): {'PASS' if pass_thresh else 'FAIL → returns (?, conf)'}")
    if pass_thresh:
        knn_vowel, knn_conf = best_v, confidence
    else:
        knn_vowel, knn_conf = "?", confidence
    say()
    say(f"K-NN 반환값: ({knn_vowel}, {knn_conf:.3f})")
code(_step1_3)


# 1.4 formant_vowel_probs
subsection("1.4 formant_vowel_probs (Bark Mahalanobis)")

def _step1_4():
    global fmt_best, fmt_conf
    f1 = res["f1"]; f2 = res["f2"]
    if f1 is None or f2 is None or f1 < 80 or f2 < 250:
        say(f"⚠ f1/f2 부적합 (f1={f1}, f2={f2}) → formant_vowel_probs 건너뜀")
        fmt_best, fmt_conf = "?", 0.0
        return
    fmt_prob = formant_vowel_probs(f1, f2, GENDER)
    sorted_fmt = sorted(fmt_prob.items(), key=lambda x: -x[1])
    say(f"입력: f1={f1:.1f}  f2={f2:.1f}  gender={GENDER}")
    say(f"_normalizer.ready = {_normalizer.ready}  (정규화 적용 여부)")
    say()
    for v, p in sorted_fmt:
        mark = "  ←best" if v == sorted_fmt[0][0] else ""
        say(f"  {v}: {p:.4f}{mark}")
    fmt_best, fmt_conf = sorted_fmt[0]
code(_step1_4)


# 1.5 classify() 분기 결정
subsection("1.5 classify() 분기 결정")

def _step1_5():
    say(f"knn_vowel = {knn_vowel}  knn_conf = {knn_conf:.3f}")
    say(f"fmt_best  = {fmt_best}  fmt_conf  = {fmt_conf:.3f}")
    say()
    cond1 = (knn_vowel == fmt_best) and (knn_conf >= 0.10)
    cond2 = fmt_conf > 0.35
    cond3 = knn_conf < 0.08
    say(f"① knn_vowel == fmt_best AND knn_conf >= 0.10  → {cond1}")
    say(f"② fmt_conf > 0.35                            → {cond2}")
    say(f"③ knn_conf < 0.08                            → {cond3}")
    say()
    if cond1:
        say("→ 경로 ①: K-NN+포먼트 일치 → KNN×1.3 부스트 후 반환")
    elif cond2:
        say("→ 경로 ②: 포먼트 단독 신뢰 → _formant_only() 위임")
    elif cond3:
        say("→ 경로 ③: KNN 신뢰도 매우 낮음 → _formant_only or ?")
    else:
        say("→ 기본: KNN 결과 그대로 반환")
code(_step1_5)


# 1.6 실제 wv.classify() 호출 결과
subsection("1.6 실제 호출 결과 (sanity check)")

def _step1_6():
    actual = wv.classify(
        chunk, sr=SAMPLE_RATE,
        f1=res["f1"], f2=res["f2"], f3=res["f3"],
        gender=GENDER,
    )
    say(f"wv.classify(...) = {actual}")
    say()
    fo = wv._formant_only(
        res["f1"], res["f2"], res["f3"],
        GENDER, audio=chunk, sr=SAMPLE_RATE,
    )
    say(f"wv._formant_only(...) = {fo}")
code(_step1_6)


# ══════════════════════════════════════════
# 2. raw F1/F2 분포 vs 학계 평균
# ══════════════════════════════════════════
section("2. raw F1/F2 분포 vs 학계 평균")

csv_path = RESULTS / "raw_distribution.csv"
summary = []

if not csv_path.exists():
    msg = f"⚠ {csv_path} 없음 — 먼저 evaluate.py 실행 필요"
    print(msg)
    md(msg)
else:
    by_vowel = defaultdict(lambda: {"f1": [], "f2": [], "f3": []})
    with open(csv_path, "r", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            v = row["true"]
            for k in ("f1", "f2", "f3"):
                vstr = row[f"raw_{k}"].strip()
                if vstr:
                    try:
                        by_vowel[v][k].append(float(vstr))
                    except ValueError:
                        pass

    md("학계 평균: 하영우·오재혁 (2017), 여성 아나운서 8명 평균")
    md()
    md("⚠️ = |측정 − 학계| ≥ 100 Hz")
    md()
    md("| 모음 | n | F1 측정±SD | F1 학계 | F1 Δ | F2 측정±SD | F2 학계 | F2 Δ |")
    md("|---|---:|---|---:|---:|---|---:|---:|")

    print()
    for v in VOWELS:
        f1s = by_vowel[v]["f1"]
        f2s = by_vowel[v]["f2"]
        n = max(len(f1s), len(f2s))
        f1m = float(np.mean(f1s)) if f1s else None
        f2m = float(np.mean(f2s)) if f2s else None
        f1sd = float(np.std(f1s)) if len(f1s) > 1 else 0.0
        f2sd = float(np.std(f2s)) if len(f2s) > 1 else 0.0
        af1, af2 = ACADEMIC_REFS[v]
        d1 = (f1m - af1) if f1m is not None else None
        d2 = (f2m - af2) if f2m is not None else None
        m1 = " ⚠️" if d1 is not None and abs(d1) >= 100 else ""
        m2 = " ⚠️" if d2 is not None and abs(d2) >= 100 else ""
        f1ms = f"{f1m:.0f} ± {f1sd:.0f}" if f1m is not None else "—"
        f2ms = f"{f2m:.0f} ± {f2sd:.0f}" if f2m is not None else "—"
        d1s  = f"{d1:+.0f}{m1}" if d1 is not None else "—"
        d2s  = f"{d2:+.0f}{m2}" if d2 is not None else "—"
        md(f"| {v} | {n} | {f1ms} | {af1} | {d1s} | {f2ms} | {af2} | {d2s} |")
        print(f"  {v}  n={n}  F1={f1ms} (학계{af1}, Δ{d1s})  F2={f2ms} (학계{af2}, Δ{d2s})")
        summary.append((v, len(f1s), len(f2s), f1m, f2m, d1, d2))


# ══════════════════════════════════════════
# 3. 결론
# ══════════════════════════════════════════
section("3. 결론")

if not summary:
    say("CSV 데이터 없음 — 결론 도출 불가")
else:
    bad_f1 = sum(1 for _, _, _, _, _, d1, _ in summary if d1 is not None and abs(d1) >= 100)
    bad_f2 = sum(1 for _, _, _, _, _, _, d2 in summary if d2 is not None and abs(d2) >= 100)
    bad_either = sum(1 for _, _, _, _, _, d1, d2 in summary
                     if (d1 is not None and abs(d1) >= 100)
                     or (d2 is not None and abs(d2) >= 100))
    n_extracted_f1 = sum(1 for _, n1, _, _, _, _, _ in summary if n1 > 0)
    n_extracted_f2 = sum(1 for _, _, n2, _, _, _, _ in summary if n2 > 0)
    total_files_per_vowel_expected = 5
    n_expected = len(VOWELS) * total_files_per_vowel_expected

    n_f1_total = sum(n1 for _, n1, _, _, _, _, _ in summary)
    n_f2_total = sum(n2 for _, _, n2, _, _, _, _ in summary)

    say(f"포먼트 추출 성공률:")
    say(f"  raw_f1 추출됨: {n_f1_total}/{n_expected}")
    say(f"  raw_f2 추출됨: {n_f2_total}/{n_expected}")
    say()
    say(f"학계 대비 100Hz 초과 차이:")
    say(f"  F1: {bad_f1}/{len(VOWELS)} 모음")
    say(f"  F2: {bad_f2}/{len(VOWELS)} 모음")
    say(f"  최소 한 축 어긋남: {bad_either}/{len(VOWELS)} 모음")
    say()

    md("**진단**")
    md()
    if n_f1_total < n_expected * 0.7 or n_f2_total < n_expected * 0.7:
        diag = (
            f"포먼트 추출 자체가 자주 실패함 (raw_f1: {n_f1_total}/{n_expected}, "
            f"raw_f2: {n_f2_total}/{n_expected}). 분류 이전 단계에서 정보 손실 발생.\n\n"
            f"→ **포먼트 추출 (FormantEngine) 결함이 1차 원인**."
        )
    elif bad_either >= 5:
        diag = (
            f"추출은 대부분 성공하지만 측정값이 학계와 광범위하게 어긋남 "
            f"({bad_either}/{len(VOWELS)} 모음에서 ≥100Hz 차이).\n\n"
            f"→ **포먼트 추출 정확도 자체가 낮음**. 분류기를 고쳐도 한계 명확."
        )
    elif bad_either <= 2:
        diag = (
            f"추출은 학계와 대체로 일치 (≥100Hz 차이는 {bad_either}/{len(VOWELS)} 모음뿐).\n\n"
            f"→ **분류기 로직 또는 _REFS 값에 문제**. 추출은 정상."
        )
    else:
        diag = (
            f"일부 모음에서 추출이 어긋남 ({bad_either}/{len(VOWELS)}).\n\n"
            f"→ 추출 + 분류 양쪽 모두 부분적 결함."
        )
    say(diag)
    md(diag)


# ══════════════════════════════════════════
# 저장
# ══════════════════════════════════════════
REPORT.write_text("\n".join(md_buf), encoding="utf-8")
print(f"\n보고서 저장: {REPORT}")
