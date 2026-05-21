"""SVM 기반 모음 분류기 + 캘리브레이션.

캘리브레이션으로 모음별 MFCC 벡터를 수집하고
SVM(RBF 커널)으로 학습. 실시간 추론.

사용법:
    clf = VowelClassifier()

    # 캘리브레이션
    clf.calibrate_start("아")
    for chunk in audio_chunks:
        clf.calibrate_feed(chunk, sr)
    clf.calibrate_end()
    # ... 8개 모음 반복
    clf.train()

    # 추론
    clf.feed(chunk, sr)
    vowel, confidence = clf.get_result()
"""

import json
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from typing import Optional, Tuple

from .features import extract_mfcc, CMVN, N_MFCC

VOWELS = ["아", "어", "오", "우", "으", "이", "에", "애"]
DEBOUNCE_FRAMES = 3  # 연속 N프레임 동일해야 전환


class VowelClassifier:
    def __init__(self):
        self._cmvn = CMVN(window_size=50)
        self._svm: Optional[SVC] = None
        self._scaler = StandardScaler()
        self._trained = False

        # 캘리브레이션 데이터
        self._cal_data = {v: [] for v in VOWELS}
        self._cal_current: Optional[str] = None
        self._cal_buffer = []

        # 추론 상태
        self._current_vowel = ""
        self._current_conf = 0.0
        self._candidate = ""
        self._candidate_count = 0

    @property
    def is_trained(self) -> bool:
        return self._trained

    @property
    def vowels(self):
        return VOWELS

    # ── 캘리브레이션 ──

    def calibrate_start(self, vowel: str):
        """특정 모음의 캘리브레이션 시작."""
        assert vowel in VOWELS, f"Unknown vowel: {vowel}"
        self._cal_current = vowel
        self._cal_buffer = []

    def calibrate_feed(self, audio: np.ndarray, sr: int):
        """캘리브레이션 중 오디오 청크 입력."""
        if self._cal_current is None:
            return
        mfcc = extract_mfcc(audio, sr)
        self._cmvn.update(mfcc)
        normalized = self._cmvn.normalize(mfcc)
        self._cal_buffer.append(normalized)

    def calibrate_end(self):
        """현재 모음 캘리브레이션 종료."""
        if self._cal_current and self._cal_buffer:
            self._cal_data[self._cal_current].extend(self._cal_buffer)
        self._cal_current = None
        self._cal_buffer = []

    def get_calibration_counts(self) -> dict:
        """각 모음별 수집된 벡터 수."""
        return {v: len(d) for v, d in self._cal_data.items()}

    # ── 학습 ──

    def train(self) -> bool:
        """수집된 캘리브레이션 데이터로 SVM 학습.
        Returns: 학습 성공 여부.
        """
        X, y = [], []
        for vowel, vectors in self._cal_data.items():
            for v in vectors:
                X.append(v)
                y.append(vowel)

        if len(set(y)) < 2:
            print("[VowelClassifier] Need at least 2 vowels to train")
            return False

        X = np.array(X, dtype=np.float32)
        y = np.array(y)

        self._scaler.fit(X)
        X_scaled = self._scaler.transform(X)

        self._svm = SVC(kernel='rbf', probability=True, C=10.0, gamma='scale')
        self._svm.fit(X_scaled, y)
        self._trained = True
        print(f"[VowelClassifier] Trained on {len(X)} vectors, "
              f"{len(set(y))} vowels")
        return True

    # ── 실시간 추론 ──

    def feed(self, audio: np.ndarray, sr: int):
        """오디오 프레임 입력 → 모음 분류."""
        if not self._trained:
            return

        mfcc = extract_mfcc(audio, sr)
        self._cmvn.update(mfcc)
        normalized = self._cmvn.normalize(mfcc)
        scaled = self._scaler.transform(normalized.reshape(1, -1))

        pred = self._svm.predict(scaled)[0]
        proba = self._svm.predict_proba(scaled)[0]
        conf = float(np.max(proba))

        # 디바운싱
        if pred == self._candidate:
            self._candidate_count += 1
        else:
            self._candidate = pred
            self._candidate_count = 1

        if self._candidate_count >= DEBOUNCE_FRAMES:
            self._current_vowel = self._candidate
            self._current_conf = conf

    def get_result(self) -> Tuple[str, float]:
        """현재 인식 결과.
        Returns: (vowel, confidence)
        """
        return self._current_vowel, self._current_conf

    # ── 저장/로드 ──

    def save(self, path: str):
        """모델 + 캘리브레이션 데이터 저장."""
        import pickle
        data = {
            'cal_data': {v: [vec.tolist() for vec in vecs]
                         for v, vecs in self._cal_data.items()},
            'svm': pickle.dumps(self._svm) if self._svm else None,
            'scaler': pickle.dumps(self._scaler),
            'trained': self._trained,
        }
        with open(path, 'w') as f:
            json.dump({k: v if not isinstance(v, bytes) else v.hex()
                       for k, v in data.items()}, f)
        print(f"[VowelClassifier] Saved to {path}")

    def load(self, path: str):
        """저장된 모델 로드."""
        import pickle
        with open(path, 'r') as f:
            data = json.load(f)

        self._cal_data = {v: [np.array(vec, dtype=np.float32) for vec in vecs]
                          for v, vecs in data['cal_data'].items()}
        if data['svm']:
            self._svm = pickle.loads(bytes.fromhex(data['svm']))
        self._scaler = pickle.loads(bytes.fromhex(data['scaler']))
        self._trained = data['trained']
        print(f"[VowelClassifier] Loaded from {path}")
