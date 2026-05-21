"""config.py — 모든 설정값 중앙 관리"""

# ── 오디오 ────────────────────────────────────────────────
SAMPLE_RATE = 44100
CHANNELS    = 1
BLOCK_SIZE  = 441            # 10 ms 콜백 블록

# ── 분석 / UI ────────────────────────────────────────────
ANALYSIS_WIN_SEC = 0.30      # 청크 길이 (초)
UPDATE_MS        = 30        # UI 갱신 주기 (ms)
DISPLAY_SECS     = 10        # 시계열 표시 범위 (초)

# ── 성별 자동 판단 ────────────────────────────────────────
GENDER_THRESH_HZ = 165

# ── VAD ───────────────────────────────────────────────────
# 작품 의도: 희미한 발음도 인식. 임계값 완화 + RMS 위주.
# autocorr/zcr 이 breathy 음성 차단 → 매우 낮춤.
CALIB_SECS      = 2.0
VAD_RMS_MULT    = 2.5      # 3.5→2.5: 작은 음량도 통과
ADAPT_RATE      = 0.008
AUTOCORR_THRESH = 0.05     # 0.15→0.05: breathy 통과
ZCR_THRESH      = 0.60     # 0.35→0.60: whisper 도 통과

# ── HNR 게이팅 ────────────────────────────────────────────
HNR_MIN_DB    = 5.0        # 미만 → 신뢰도 ×0.5 (시각화 alpha 활용 가능)
HNR_VOICE_MIN = 0.0        # 2.0→0.0: HNR 게이팅 비활성 (희미한 발음 허용)

# ── pyworld ──────────────────────────────────────────────
# 작품 의도: 희미한 발음/속삭임도 추출 → 0.05 (실질 비활성)
PYWORLD_VOICED_FRAC_MIN = 0.05
PYWORLD_FRAME_PERIOD    = 10.0

# ── Praat 포먼트 ─────────────────────────────────────────
FORMANT_CEILINGS = [5500]
MAX_BW           = {1: 600, 2: 900, 3: 1300}
SAMPLE_POS       = [0.25, 0.50, 0.75]

# 성별별 Praat 파라미터
# female: max_formants=4 + window=50ms = Hillenbrand 권장. 5는 LPC 가
#   5500Hz 안에 빡빡해 F1 BW ↑ → reject 됨.
# female f1_range 280+: F0 harmonic (200-280Hz) 을 F1 으로 오인 방지.
PARAMS = {
    "male": dict(
        pitch_floor=50,    pitch_ceiling=300,
        max_formants=5,    max_formant_freq=5000,
        window_length=0.030, pre_emphasis=50,
        label="남성", color="#5599FF",
        f1_range=(100, 900),  f2_range=(380, 2800), f3_range=(1500, 4000),
    ),
    "female": dict(
        pitch_floor=150,   pitch_ceiling=400,
        max_formants=4,    max_formant_freq=5000,
        window_length=0.050, pre_emphasis=50,
        label="여성", color="#FF77BB",
        f1_range=(280, 1100), f2_range=(380, 3200), f3_range=(1800, 4800),
    ),
}

# ── 한국어 모음 F1/F2 참조값 (Yoon 2015) ──────────────────
VOWEL_REFS = {
    "아": dict(F1=(828, 1128), F2=(1135, 1660), color="#FF4444"),
    "에": dict(F1=(398,  698), F2=(1848, 2402), color="#FFAA22"),
    "이": dict(F1=(235,  469), F2=(2412, 3162), color="#FFFF44"),
    "오": dict(F1=(355,  619), F2=(618,  1062), color="#44FF88"),
    "우": dict(F1=(250,  484), F2=(479,   841), color="#44DDFF"),
    "으": dict(F1=(300,  570), F2=(1078, 1730), color="#4488FF"),
    "어": dict(F1=(508,  834), F2=(945,  1479), color="#CC55FF"),
}

VOWEL_REFS_MALE = {
    "아": dict(F1=(699,  963), F2=(931,  1359), color="#FF4444"),
    "에": dict(F1=(334,  598), F2=(1515, 1971), color="#FFAA22"),
    "이": dict(F1=(196,  402), F2=(1978, 2592), color="#FFFF44"),
    "오": dict(F1=(297,  531), F2=(508,   870), color="#44FF88"),
    "우": dict(F1=(209,  415), F2=(393,   689), color="#44DDFF"),
    "으": dict(F1=(252,  488), F2=(884,  1418), color="#4488FF"),
    "어": dict(F1=(426,  714), F2=(775,  1213), color="#CC55FF"),
}
