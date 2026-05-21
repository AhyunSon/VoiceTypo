"""
wav2vec_classifier.py — wav2vec2 레이어-8 K-NN 한국어 단모음 분류기

분류 모드 (우선순위 순):
  1. 개인 K-NN  — 보정 후 저장된 프로토타입 (90-95%)
  2. 기본 K-NN  — 시작 시 표준 포먼트 합성음으로 자동 생성 (보정 불필요)
  3. 포먼트 Mahalanobis — K-NN 신뢰도 낮을 때 보조

핵심 원리:
  wav2vec2 CTC(자모 확률)는 고립 모음에서 아 편향이 심해 사용 불가.
  대신 레이어 8 은닉 벡터(768d)는 음소를 직접 인코딩 → 코사인 유사도 분류.
  (InterSpeech 2024: 레이어 6-10이 음소 분류에 최적)
"""

from __future__ import annotations

import threading
from pathlib import Path

import numpy as np

from config import SAMPLE_RATE

MODEL_NAME  = "Kkonjeong/wav2vec2-base-korean"
TARGET_SR   = 16000
PROTO_FILE  = Path(__file__).parent / "user_proto.npz"
PROTO_LAYER = 8   # wav2vec2-base 12레이어 중 음소 표현 최적 레이어

_JAMO_TO_VOWEL = {
    'ㅏ': '아', 'ㅓ': '어', 'ㅗ': '오',
    'ㅜ': '우', 'ㅡ': '으', 'ㅣ': '이',
    'ㅔ': '에', 'ㅐ': '에',
}
VOWELS = ["아", "에", "이", "오", "우", "으", "어"]

# ── Bark 스케일 포먼트 분류기 ────────────────────────────────────
# Bark 스케일 = 심리음향적 주파수 척도 → Hz보다 화자 독립적
# 특히 F2-F1 분산도를 Bark 공간에서 사용하면
# 성도 길이 차이(성별/화자)에 의한 변동이 크게 줄어든다.

# 기준값: config.py VOWEL_REFS/VOWEL_REFS_MALE 중심값과 완전 일치
# (F1_center, F1_sigma_hz, F2_center, F2_sigma_hz)
# σ = VOWEL_REFS 범위 / 3  (mean ± 1.5 SD → range = 3 SD)
_POP_REFS_HZ = {
    "female": {
        "아": (978, 100, 1397, 175),
        "에": (548, 100, 2125, 185),
        "이": (352,  78, 2787, 250),
        "오": (487,  88,  840, 160),   # F2σ=160
        "우": (367,  78,  660, 210),   # F2σ=210 (넓음): LPC 과추정 대비, F1이 주 변별자
        "으": (435,  90, 1404, 217),
        "어": (671, 109, 1212, 178),   # F1σ=109 → 넓게 → 인식 범위 확대
    },
    "male": {
        "아": (831,  88, 1145, 143),
        "에": (466,  88, 1743, 152),
        "이": (299,  69, 2285, 205),
        "오": (414,  78,  689, 130),
        "우": (312,  69,  541, 175),   # F2σ 넓힘: LPC 과추정 대비
        "으": (370,  79, 1151, 178),
        "어": (570,  96,  994, 146),
    },
}

# 표준 포먼트 합성 파라미터 {vowel: [(F0, [(freq, bw), ...]), ...]}
# 다양한 F0로 화자 다양성 커버
_SYNTH_PARAMS = {
    "아": [
        (130, [(670,115),(1070,150),(2800,220)]),
        (180, [(790,120),(1220,140),(2800,220)]),
        (220, [(820,130),(1260,150),(2850,230)]),
    ],
    "에": [
        (130, [(415,100),(1800,130),(2900,210)]),
        (190, [(490,105),(2050,125),(2950,210)]),
        (240, [(510,110),(2100,130),(3000,215)]),
    ],
    "이": [
        (130, [(250, 75),(2350,140),(3200,230)]),
        (200, [(290, 80),(2700,145),(3300,235)]),
        (250, [(310, 85),(2750,150),(3350,240)]),
    ],
    "오": [
        (120, [(390,120),( 740,180),(2600,220)]),
        (160, [(460,130),( 840,185),(2650,225)]),
        (200, [(480,135),( 880,190),(2700,230)]),
    ],
    "우": [
        (110, [(320, 95),( 540,190),(2500,220)]),
        (150, [(380,105),( 620,200),(2550,225)]),
        (190, [(400,110),( 650,205),(2600,230)]),
    ],
    "으": [
        (130, [(350, 95),(1210,155),(2400,220)]),
        (170, [(410,100),(1380,160),(2450,225)]),
        (210, [(430,105),(1420,165),(2500,230)]),
    ],
    "어": [
        (130, [(545,110),(1080,145),(2700,215)]),
        (175, [(640,115),(1230,150),(2750,220)]),
        (215, [(660,120),(1270,155),(2800,225)]),
    ],
}

_calib_refs: dict = {}


# ── 자동 화자 정규화 (보정 없이 이/아 앵커로 성도 차이 보정) ──────────
class _VowelNormalizer:
    """
    이(이)와 아(아)를 앵커로 사용해 화자별 성도 길이 차이를 자동 보정.

    이유:
      • 이: F1 최소, F2 최대 → 어떤 화자든 F1<380, F2>2200이면 거의 확실히 이
      • 아: F1 최대 (개모음) → F1>750, F2<1700이면 거의 확실히 아
      이 두 극단값으로 화자의 포먼트 공간 스케일을 추정해 전체 모음 분류에 반영.

    효과: 성도 짧은 화자(포먼트 높음) / 긴 화자(포먼트 낮음) 모두 집단 평균으로 정규화.
    약 30-60초 사용 후 우/어/으 정확도 향상.
    """
    _MIN_FRAMES = 8    # 정규화 시작 최소 프레임
    _MAX_FRAMES = 50   # 누적 최대 (최근값 위주)

    # 집단 앵커 기준 (VOWEL_REFS female 중심값)
    _POP_I_F2 = 2787.0
    _POP_A_F1 =  978.0

    def __init__(self):
        self._i_f2: list = []
        self._a_f1: list = []
        self.scale_f1 = 1.0
        self.scale_f2 = 1.0
        self.ready    = False

    def update(self, f1: float, f2: float) -> None:
        if not (f1 > 80 and f2 > 300):
            return
        # 이 앵커: F1 매우 낮고 F2 매우 높음 → 거의 확실히 이
        if f1 < 380 and f2 > 2200:
            self._i_f2.append(f2)
            if len(self._i_f2) > self._MAX_FRAMES:
                self._i_f2.pop(0)
        # 아 앵커: F1 높고 F2 중간 (어와 구분: F1>750)
        if f1 > 750 and 1000 < f2 < 1700:
            self._a_f1.append(f1)
            if len(self._a_f1) > self._MAX_FRAMES:
                self._a_f1.pop(0)

        if (len(self._i_f2) >= self._MIN_FRAMES
                and len(self._a_f1) >= self._MIN_FRAMES):
            user_i_f2 = float(np.median(self._i_f2))
            user_a_f1 = float(np.median(self._a_f1))
            # 집단 기준값으로 스케일 계산 (clamp: 0.7~1.4)
            self.scale_f2 = float(np.clip(
                self._POP_I_F2 / max(user_i_f2, 1500.0), 0.70, 1.40))
            self.scale_f1 = float(np.clip(
                self._POP_A_F1 / max(user_a_f1,  500.0), 0.70, 1.40))
            self.ready = True

    def normalize(self, f1: float, f2: float) -> tuple:
        if not self.ready:
            return f1, f2
        return f1 * self.scale_f1, f2 * self.scale_f2

    def reset(self):
        self._i_f2.clear()
        self._a_f1.clear()
        self.ready    = False
        self.scale_f1 = 1.0
        self.scale_f2 = 1.0


_normalizer = _VowelNormalizer()


def get_normalizer_status() -> str:
    """UI 상태 표시용"""
    if not _normalizer.ready:
        n_i = len(_normalizer._i_f2)
        n_a = len(_normalizer._a_f1)
        return f"자동보정 준비중 이={n_i}/8 아={n_a}/8"
    return (f"자동보정 완료 "
            f"F1×{_normalizer.scale_f1:.2f} F2×{_normalizer.scale_f2:.2f}")


def set_calibration(calib_data: dict) -> None:
    global _calib_refs
    refs = {}
    for vowel, d in calib_data.items():
        f1    = float(d.get("f1",    0))
        f2    = float(d.get("f2",    0))
        f1_sd = max(float(d.get("f1_sd", 80)),  45.0)
        f2_sd = max(float(d.get("f2_sd", 120)), 75.0)
        if f1 > 50 and f2 > 200:
            refs[vowel] = (f1, f1_sd, f2, f2_sd)
    _calib_refs = refs


def clear_calibration() -> None:
    global _calib_refs
    _calib_refs = {}


def _norm_log(x, mu, sigma):
    return -0.5 * ((x - mu) / sigma) ** 2


def _hz_to_bark(f: float) -> float:
    """Zwicker Bark 스케일 변환 (심리음향적 선형화)"""
    f = max(f, 1.0)
    return 13.0 * np.arctan(0.76 * f / 1000.0) + 3.5 * np.arctan((f / 7500.0) ** 2)


def formant_vowel_probs(f1: float, f2: float, gender: str = "female") -> dict:
    """
    Bark 스케일 F1 + (F2-F1) 분산도 기반 모음 확률.

    Bark 변환 후 F2-F1 분산도 사용 → Hz 대비 화자 독립성 크게 향상.
    (성도 길이 차이가 Bark 분산도에서는 ±0.3B 이내로 압축됨)
    """
    if _calib_refs and len(_calib_refs) >= 4:
        # 개인 보정: Hz 기준 F1/F2 Mahalanobis
        log_p = {}
        for v, (m1, s1, m2, s2) in _calib_refs.items():
            log_p[v] = _norm_log(f1, m1, s1) + _norm_log(f2, m2, s2)
    else:
        # 기본: 모음별 σ를 Hz→Bark 변환 적용 (VOWEL_REFS 범위 기반)
        # 오 F2σ(좁음) vs 우 F2σ(넓음) → 오/우 경계 선명해짐
        # 자동 화자 정규화 적용 (이/아 앵커 충분히 쌓인 경우만)
        f1, f2 = _normalizer.normalize(f1, f2)
        b1 = _hz_to_bark(f1)
        b2 = _hz_to_bark(f2)

        refs = _POP_REFS_HZ.get(gender, _POP_REFS_HZ["female"])
        log_p = {}
        for v, (rf1, sf1, rf2, sf2) in refs.items():
            rb1   = _hz_to_bark(rf1)
            rb2   = _hz_to_bark(rf2)
            # Hz sigma → Bark sigma (Bark 비선형성 보정)
            sb1   = max(_hz_to_bark(rf1 + sf1) - rb1, 0.05)
            sb2   = max(_hz_to_bark(rf2 + sf2) - rb2, 0.05)
            log_p[v] = _norm_log(b1, rb1, sb1) + _norm_log(b2, rb2, sb2)

    max_lp = max(log_p.values())
    p = {v: float(np.exp(lp - max_lp)) for v, lp in log_p.items()}
    total = sum(p.values())
    return {v: p[v] / total for v in p}


# ── 합성 모음 생성 유틸 ──────────────────────────────────────────

def _synth_vowel(f0: float, formants: list, sr: int = TARGET_SR,
                 dur: float = 0.35) -> np.ndarray:
    """배음 + 포먼트 필터링으로 모음 합성 (Klatt 단순화)"""
    t   = np.linspace(0, dur, int(sr * dur))
    sig = np.zeros_like(t)
    n_harm = max(1, int(sr / 2 / f0))
    for h in range(1, n_harm + 1):
        freq = h * f0
        amp  = sum(np.exp(-((freq - fm) / bw) ** 2) for fm, bw in formants)
        sig += amp * np.sin(2 * np.pi * freq * t)
    peak = np.max(np.abs(sig))
    return (sig / (peak + 1e-8)).astype(np.float32)


# ── 리샘플링 ────────────────────────────────────────────────────

def _resample(audio: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    if src_sr == dst_sr:
        return audio.astype(np.float32)
    n_out = int(len(audio) * dst_sr / src_sr)
    if n_out == 0:
        return np.zeros(1, dtype=np.float32)
    idx  = np.linspace(0, len(audio) - 1, n_out)
    lo   = np.clip(idx.astype(int), 0, len(audio) - 2)
    frac = idx - lo
    return (audio[lo] * (1 - frac) + audio[lo + 1] * frac).astype(np.float32)


# ── 분류기 ──────────────────────────────────────────────────────

class Wav2VecVowelClassifier:

    def __init__(self):
        self._model     = None
        self._processor = None
        self._ready     = False
        self._error     = None
        self._lock      = threading.Lock()

        self._proto: dict[str, np.ndarray]         = {}  # 개인 보정 프로토타입
        self._default_proto: dict[str, np.ndarray] = {}  # 기본 합성 프로토타입
        self._proto_accum: dict[str, list]         = {}  # 보정 중 누적

    # ── 로딩 ───────────────────────────────────────────────────

    def start_loading(self, on_ready=None, on_error=None):
        def _load():
            try:
                from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

                proc  = Wav2Vec2Processor.from_pretrained(MODEL_NAME)
                model = Wav2Vec2ForCTC.from_pretrained(MODEL_NAME)
                model.eval()

                with self._lock:
                    self._processor = proc
                    self._model     = model
                    self._ready     = True

                if on_ready:
                    on_ready()

                # 모델 로딩 완료 후 기본 프로토타입 빌드 (백그라운드)
                self._build_default_prototypes()

            except Exception as e:
                self._error = str(e)
                if on_error:
                    on_error(str(e))

        threading.Thread(target=_load, daemon=True).start()

    @property
    def is_ready(self) -> bool:
        return self._ready

    @property
    def load_error(self):
        return self._error

    @property
    def has_prototypes(self) -> bool:
        return bool(self._proto) or bool(self._default_proto)

    @property
    def has_personal_prototypes(self) -> bool:
        return bool(self._proto)

    # ── 기본 합성 프로토타입 빌드 ──────────────────────────────

    def _build_default_prototypes(self):
        """
        표준 포먼트 합성음 → 레이어-8 임베딩 → 기본 프로토타입.
        모델 로딩 완료 후 백그라운드에서 실행 (~3-5초).
        """
        proto = {}
        for vowel, samples in _SYNTH_PARAMS.items():
            feats = []
            for f0, formants in samples:
                audio_16k = _synth_vowel(f0, formants, sr=TARGET_SR)
                feat = self._extract_hidden(audio_16k)
                if feat is not None:
                    feats.append(feat)
            if feats:
                mean = np.mean(feats, axis=0)
                norm = np.linalg.norm(mean)
                proto[vowel] = mean / (norm + 1e-8)

        self._default_proto = proto

    # ── 레이어-8 임베딩 추출 ────────────────────────────────────

    def _extract_hidden(self, audio_16k: np.ndarray) -> np.ndarray | None:
        try:
            import torch
            with self._lock:
                proc  = self._processor
                model = self._model

            inputs = proc(audio_16k, sampling_rate=TARGET_SR,
                          return_tensors="pt", padding=False)
            with torch.no_grad():
                out = model(**inputs, output_hidden_states=True)

            hs   = out.hidden_states[PROTO_LAYER]        # (1, T, 768)
            feat = hs.squeeze(0).mean(0).cpu().numpy()   # (768,)
            norm = np.linalg.norm(feat)
            return feat / (norm + 1e-8)
        except Exception:
            return None

    def _prep_audio(self, audio: np.ndarray, sr: int) -> np.ndarray:
        a16k = _resample(audio.astype(np.float32), sr, TARGET_SR)
        min_len = int(TARGET_SR * 0.1)
        if len(a16k) < min_len:
            a16k = np.pad(a16k, (0, min_len - len(a16k)))
        return a16k

    # ── K-NN 분류 (개인 or 기본 프로토타입) ────────────────────

    def _knn_classify(self, audio_16k: np.ndarray,
                      proto: dict) -> tuple[str, float]:
        feat = self._extract_hidden(audio_16k)
        if feat is None:
            return "?", 0.0

        sims = {v: float(np.dot(feat, p)) for v, p in proto.items()}
        sorted_sims = sorted(sims.values(), reverse=True)
        best_vowel  = max(sims, key=sims.get)
        margin      = sorted_sims[0] - sorted_sims[1] if len(sorted_sims) > 1 else 0.0
        confidence  = min(margin * 5.0, 1.0)

        # 기본 프로토타입: 합성음↔실제음 차이 고려해 임계값 낮춤
        # 개인 프로토타입: 더 엄격한 기준 적용 불필요 (유사도 자체가 높음)
        if not self._proto and sorted_sims[0] < 0.55:
            return "?", confidence

        return best_vowel, confidence

    # ── 프로토타입 K-NN + 포먼트 보조 ──────────────────────────

    def classify(self, audio: np.ndarray, sr: int = SAMPLE_RATE,
                 f1: float = None, f2: float = None,
                 gender: str = "female") -> tuple[str, float]:
        if not self._ready:
            return "?", 0.0
        try:
            a16k = self._prep_audio(audio, sr)

            # 개인 보정 프로토타입이 있으면 K-NN 사용 (보정 후 고정밀)
            if self._proto:
                knn_vowel, knn_conf = self._knn_classify(a16k, self._proto)

                # 포먼트 융합: K-NN 신뢰도 낮을 때 보완
                if f1 and f2 and f1 > 80 and f2 > 250:
                    _normalizer.update(f1, f2)  # 앵커 학습은 항상 진행
                    fmt_prob = formant_vowel_probs(f1, f2, gender)
                    fmt_best = max(fmt_prob, key=fmt_prob.get)
                    fmt_conf = fmt_prob[fmt_best]

                    if knn_conf < 0.25 and fmt_conf > 0.50:
                        return fmt_best, fmt_conf * 0.75
                    elif knn_vowel == fmt_best:
                        return knn_vowel, min(knn_conf * 1.3, 1.0)

                if knn_conf < 0.08:
                    return "?", knn_conf
                return knn_vowel, knn_conf

            # 자동 화자 정규화 데이터 누적 (이/아 앵커)
            if f1 and f2 and f1 > 80 and f2 > 250:
                _normalizer.update(f1, f2)

            # 보정 없음 → Bark 스케일 포먼트 분류기 (화자 독립적)
            return self._formant_only(f1, f2, gender)

        except Exception:
            return "?", 0.0

    def _formant_only(self, f1, f2, gender) -> tuple[str, float]:
        """보정 없을 때 Bark 스케일 포먼트만으로 분류"""
        if not (f1 and f2 and f1 > 80 and f2 > 250):
            return "?", 0.0
        prob = formant_vowel_probs(f1, f2, gender)
        best = max(prob, key=prob.get)
        conf = prob[best]
        # 임계값 낮춤: 어처럼 넓은 모음도 포착되도록 (EMA가 최종 필터링)
        return (best, conf) if conf > 0.15 else ("?", conf)

    # ── 개인 보정 프로토타입 API ────────────────────────────────

    def add_prototype(self, vowel: str, audio: np.ndarray, sr: int = SAMPLE_RATE):
        if not self._ready:
            return
        a16k = self._prep_audio(audio, sr)
        feat = self._extract_hidden(a16k)
        if feat is not None:
            self._proto_accum.setdefault(vowel, []).append(feat)

    def fit_prototypes(self):
        self._proto = {}
        for vowel, feats in self._proto_accum.items():
            arr  = np.stack(feats)
            mean = arr.mean(axis=0)
            norm = np.linalg.norm(mean)
            self._proto[vowel] = mean / (norm + 1e-8)
        self._proto_accum = {}

    def save_prototypes(self) -> bool:
        if not self._proto:
            return False
        try:
            np.savez(PROTO_FILE,
                     **{v.encode('utf-8').hex(): p for v, p in self._proto.items()})
            return True
        except Exception:
            return False

    def load_prototypes(self) -> bool:
        if not PROTO_FILE.exists():
            return False
        try:
            data = np.load(PROTO_FILE)
            self._proto = {bytes.fromhex(k).decode('utf-8'): data[k]
                           for k in data.files}
            return bool(self._proto)
        except Exception:
            return False

    def clear_prototypes(self):
        self._proto.clear()
        self._proto_accum.clear()
        if PROTO_FILE.exists():
            PROTO_FILE.unlink()
