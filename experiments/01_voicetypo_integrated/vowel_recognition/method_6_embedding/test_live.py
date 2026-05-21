"""Method 6 실시간 모음 인식 테스트.

모드 A: 정량 평가 — 모음별 N회 녹음 → 정확도/혼동행렬/지연 측정
모드 B: 자유 테스트 — 실시간 인식 결과 표시

사용법:
  # 모드 A: 정량 평가 (모음당 10회)
  python -m vowel_recognition.method_6_embedding.test_live --mode eval --trials 10

  # 모드 B: 자유 테스트
  python -m vowel_recognition.method_6_embedding.test_live --mode free
"""

import sys
import os
import argparse
import time
import pickle
import threading
import wave
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import torch
from transformers import Wav2Vec2Model, Wav2Vec2FeatureExtractor
from audio_capture.capture import AudioCapture
from pitch_detection.yin import YinDetector
from pitch_detection.vad import VoiceActivityDetector

VOWELS = ["아", "어", "오", "우", "으", "이", "에"]
TARGET_VOWELS = ["오", "우"]

# ─────────────────────────────────────
# XLSR-53 임베딩 추출기 (실시간용)
# ─────────────────────────────────────
class LiveEmbeddingExtractor:
    """실시간용 XLSR-53 임베딩 추출. Layer 16 + Layer 5-7 동시 추출."""

    def __init__(self, model_name='facebook/wav2vec2-large-xlsr-53'):
        print(f"[모델 로딩] {model_name}...", flush=True)
        self._fe = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
        self._model = Wav2Vec2Model.from_pretrained(model_name)
        self._model.eval()
        self._target_sr = 16000
        print("[모델 로딩] 완료.", flush=True)

    def extract(self, audio: np.ndarray, sr: int):
        """오디오 → (layer16_emb, layer567_emb, elapsed_ms).

        Returns:
            layer16_emb: np.ndarray (1024,) — Stage 1용
            layer567_emb: np.ndarray (1024,) — Stage 2용
            elapsed_ms: float — 추론 시간(ms)
        """
        # 리샘플링
        if sr != self._target_sr:
            ratio = self._target_sr / sr
            n_out = int(len(audio) * ratio)
            indices = np.arange(n_out) / ratio
            idx = np.clip(indices.astype(int), 0, len(audio) - 1)
            audio = audio[idx]

        inputs = self._fe(
            audio, sampling_rate=self._target_sr,
            return_tensors="pt", padding=False
        )

        t0 = time.perf_counter()
        with torch.no_grad():
            outputs = self._model(**inputs, output_hidden_states=True)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        hidden = outputs.hidden_states

        # Layer 16
        h16 = hidden[16].squeeze(0)  # (T, 1024)
        emb16 = self._pool(h16)

        # Layer 5, 6, 7: 각 레이어 풀링 후 평균 (학습 데이터와 동일한 방식)
        emb5 = self._pool(hidden[5].squeeze(0))
        emb6 = self._pool(hidden[6].squeeze(0))
        emb7 = self._pool(hidden[7].squeeze(0))
        emb567 = (emb5 + emb6 + emb7) / 3.0

        return emb16, emb567, elapsed_ms

    @staticmethod
    def _pool(frames):
        """에너지 상위 50% 프레임 mean pooling."""
        energy = frames.norm(dim=1)
        k = max(1, len(energy) // 2)
        top_idx = torch.topk(energy, k).indices
        selected = frames[top_idx]
        return selected.mean(dim=0).numpy().astype(np.float32)


# ─────────────────────────────────────
# 2단계 분류기
# ─────────────────────────────────────
class TwoStageClassifier:
    """사전 학습된 2단계 분류기 로드 & 예측."""

    def __init__(self, model_path):
        print(f"[분류기 로딩] {model_path}...", flush=True)
        with open(model_path, 'rb') as f:
            data = pickle.load(f)
        self._s1_scaler = data['stage1']['scaler']
        self._s1_clf = data['stage1']['clf']
        self._s2_scaler = data['stage2']['scaler']
        self._s2_clf = data['stage2']['clf']
        self._target = data['stage2']['target_vowels']
        print("[분류기 로딩] 완료.", flush=True)

    def predict(self, emb_s1, emb_s2, debug=False):
        """2단계 예측.

        Args:
            emb_s1: Layer 16 임베딩 (1024,)
            emb_s2: Layer 5-7 임베딩 (1024,)
            debug: True면 상세 로그 출력

        Returns:
            (vowel, confidence, prob_dict)
        """
        # Stage 1
        X1 = self._s1_scaler.transform(emb_s1.reshape(1, -1))
        pred1 = self._s1_clf.predict(X1)[0]
        proba1 = self._s1_clf.predict_proba(X1)[0]
        classes1 = self._s1_clf.classes_

        if debug:
            top3 = sorted(zip(classes1, proba1), key=lambda x: x[1], reverse=True)[:3]
            top3_str = "  ".join(f"{v}:{p:.0%}" for v, p in top3)
            print(f"     [S1] {pred1}  ({top3_str})")

        # Stage 2: 오/우로 판정된 경우만
        if pred1 in self._target:
            X2 = self._s2_scaler.transform(emb_s2.reshape(1, -1))
            pred2 = self._s2_clf.predict(X2)[0]
            proba2 = self._s2_clf.predict_proba(X2)[0]
            classes2 = self._s2_clf.classes_

            if debug:
                p_str = "  ".join(f"{v}:{p:.0%}" for v, p in zip(classes2, proba2))
                print(f"     [S2] {pred1} -> {pred2}  ({p_str})")

            # 최종: Stage 2 결과 사용
            final = pred2
            conf = float(max(proba2))
            prob_dict = {cls: float(p) for cls, p in zip(classes1, proba1)}
            ou_sum = sum(prob_dict.get(v, 0) for v in self._target)
            for cls, p in zip(classes2, proba2):
                prob_dict[cls] = float(p) * ou_sum
        else:
            final = pred1
            conf = float(max(proba1))
            prob_dict = {cls: float(p) for cls, p in zip(classes1, proba1)}

        return final, conf, prob_dict


# ─────────────────────────────────────
# 오디오 버퍼 (VAD 기반 발화 구간 수집)
# ─────────────────────────────────────
class UtteranceCollector:
    """VAD로 발화 구간을 감지하여 하나의 발화를 수집."""

    def __init__(self, sr=44100, min_duration=0.15, max_duration=3.0,
                 pre_padding=0.05):
        self.sr = sr
        self.min_samples = int(min_duration * sr)
        self.max_samples = int(max_duration * sr)
        self.pre_pad_samples = int(pre_padding * sr)

        self._yin = YinDetector(sr)
        self._vad = VoiceActivityDetector()

        self._buffer = []
        self._pre_buffer = []  # VAD 전 약간의 버퍼
        self._collecting = False
        self._ready = False
        self._utterance = None
        self._lock = threading.Lock()
        self.enabled = False  # 외부에서 제어

    def on_audio(self, chunk, sr):
        if not self.enabled:
            return

        freq, rms = self._yin.detect(chunk)
        self._vad.update(rms, freq)

        with self._lock:
            if self._ready:
                return  # 이전 발화가 아직 소비 안 됨

            if self._vad.is_active:
                if not self._collecting:
                    # 발화 시작
                    self._collecting = True
                    self._buffer = list(self._pre_buffer)  # pre-padding 포함
                self._buffer.append(chunk.copy())

                # 최대 길이 초과
                total = sum(len(c) for c in self._buffer)
                if total >= self.max_samples:
                    self._finalize()
            else:
                # pre-buffer 유지
                self._pre_buffer.append(chunk.copy())
                total_pre = sum(len(c) for c in self._pre_buffer)
                while total_pre > self.pre_pad_samples and len(self._pre_buffer) > 1:
                    total_pre -= len(self._pre_buffer[0])
                    self._pre_buffer.pop(0)

                if self._collecting:
                    # 발화 종료
                    total = sum(len(c) for c in self._buffer)
                    if total >= self.min_samples:
                        self._finalize()
                    else:
                        # 너무 짧으면 버림
                        self._buffer = []
                        self._collecting = False

    def _finalize(self):
        self._utterance = np.concatenate(self._buffer)
        self._buffer = []
        self._collecting = False
        self._ready = True

    def get_utterance(self):
        """수집된 발화 반환. 없으면 None."""
        with self._lock:
            if self._ready:
                utt = self._utterance
                self._utterance = None
                self._ready = False
                return utt
            return None

    def reset(self):
        with self._lock:
            self._buffer = []
            self._pre_buffer = []
            self._collecting = False
            self._ready = False
            self._utterance = None
            self._vad.reset()

    @property
    def is_collecting(self):
        return self._collecting


# ─────────────────────────────────────
# 모드 A: 정량 평가
# ─────────────────────────────────────
def save_wav(path, audio, sr):
    """float32 오디오를 16-bit WAV로 저장."""
    audio_16 = np.clip(audio * 32767, -32768, 32767).astype(np.int16)
    with wave.open(path, 'w') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(audio_16.tobytes())


def mode_eval(extractor, classifier, capture, collector, trials_per_vowel):
    # 녹음 저장 폴더
    save_dir = os.path.join(os.path.dirname(__file__), 'live_recordings')
    os.makedirs(save_dir, exist_ok=True)
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    session_dir = os.path.join(save_dir, f'session_{timestamp}')
    os.makedirs(session_dir, exist_ok=True)

    print("\n" + "=" * 60)
    print("  모드 A: 정량 평가")
    print("=" * 60)
    print(f"  모음 {len(VOWELS)}개 x {trials_per_vowel}회 = 총 {len(VOWELS) * trials_per_vowel}회")
    print(f"  녹음 저장: {session_dir}")
    print()
    print("  진행 방식:")
    print("    1) Enter  -> 녹음 시작 (마이크 열림)")
    print("    2) 모음을 약 1초간 발성  (예: '아---')")
    print("    3) Enter  -> 녹음 종료 & 인식 결과 표시")
    print()
    print("  TIP: 약 0.5~1.5초 정도 소리내면 됩니다.")
    print("=" * 60)

    results = []  # [(gt, pred, conf, latency_ms, wav_path), ...]

    for vi, vowel in enumerate(VOWELS):
        print(f"\n{'─' * 60}")
        print(f"  [{vi+1}/{len(VOWELS)}] 모음 \"{vowel}\" ({trials_per_vowel}회)")
        print(f"{'─' * 60}")

        for trial in range(trials_per_vowel):
            # ── Step 1: Enter로 녹음 시작 ──
            input(f"\n  >> {trial+1}/{trials_per_vowel}: "
                  f"Enter를 눌러 녹음 시작 -> \"{vowel}\" 발성 ->")

            # 수집 시작: Enter~Enter 사이 모든 오디오를 raw 버퍼에 쌓기
            collector.reset()
            collector.enabled = True
            raw_chunks = []
            original_callback = collector.on_audio

            def capture_all(chunk, sr, _chunks=raw_chunks):
                _chunks.append(chunk.copy())

            capture.add_listener(capture_all)

            print(f"     ** 녹음 중 ** \"{vowel}\"를 약 1초간 발성하세요...")

            # ── Step 2: Enter로 녹음 종료 ──
            input("     발성 끝나면 Enter -> ")

            # 녹음 종료
            collector.enabled = False
            capture.remove_listener(capture_all)

            # 수집된 발화 가져오기 (VAD 기반)
            utterance = collector.get_utterance()

            if utterance is None:
                # VAD가 아직 finalize 안 했으면, 버퍼에 있는 것을 직접 수거
                with collector._lock:
                    if collector._buffer:
                        utterance = np.concatenate(collector._buffer)
                        collector._buffer = []
                        collector._collecting = False

            # VAD 실패 시 raw 전체 사용 (묵음 제거 후)
            if utterance is None or len(utterance) < collector.min_samples:
                if raw_chunks:
                    raw_all = np.concatenate(raw_chunks)
                    # 앞뒤 묵음 간단 트리밍
                    rms_threshold = 0.01
                    frame_size = 2048
                    for start in range(0, len(raw_all) - frame_size, frame_size):
                        if np.sqrt(np.mean(raw_all[start:start+frame_size]**2)) > rms_threshold:
                            break
                    for end in range(len(raw_all), frame_size, -frame_size):
                        if np.sqrt(np.mean(raw_all[end-frame_size:end]**2)) > rms_threshold:
                            break
                    trimmed = raw_all[max(0, start - frame_size):min(len(raw_all), end + frame_size)]
                    if len(trimmed) >= collector.min_samples:
                        utterance = trimmed

            if utterance is None or len(utterance) < collector.min_samples:
                print("     [소리가 감지되지 않음 - 다시 시도하세요]")
                continue

            # WAV 저장
            wav_name = f"{vowel}_{trial+1:02d}.wav"
            wav_path = os.path.join(session_dir, wav_name)
            save_wav(wav_path, utterance, capture.sample_rate)

            duration = len(utterance) / capture.sample_rate
            sys.stdout.write(f"     녹음: {duration:.2f}초 -> 분석 중... ")
            sys.stdout.flush()

            # 임베딩 추출 + 분류
            t0 = time.perf_counter()
            emb16, emb567, model_ms = extractor.extract(utterance, capture.sample_rate)
            pred, conf, prob_dict = classifier.predict(emb16, emb567, debug=True)
            total_ms = (time.perf_counter() - t0) * 1000

            correct = pred == vowel
            mark = "O" if correct else "X"
            results.append((vowel, pred, conf, total_ms, wav_path))

            # 결과 표시
            if correct:
                print(f"\r     => {pred}({conf*100:.0f}%)  {mark}  "
                      f"{total_ms:.0f}ms  ({duration:.2f}초)  [{wav_name}]")
            else:
                print(f"\r     => {pred}({conf*100:.0f}%)  {mark} "
                      f"(정답:{vowel})  {total_ms:.0f}ms  ({duration:.2f}초)  [{wav_name}]")

    # ── 결과 요약 ──
    print_eval_results(results, session_dir)


def print_eval_results(results, session_dir=None):
    if not results:
        print("\n결과 없음.")
        return

    print("\n\n" + "=" * 60)
    print("  정량 평가 결과")
    print("=" * 60)

    total = len(results)
    correct = sum(1 for r in results if r[0] == r[1])
    latencies = [r[3] for r in results]

    print(f"\n  전체 정확도: {correct}/{total} ({correct/total*100:.1f}%)")
    print(f"  평균 지연: {np.mean(latencies):.0f}ms "
          f"(min:{np.min(latencies):.0f}, max:{np.max(latencies):.0f})")
    if session_dir:
        print(f"  녹음 저장: {session_dir}")

    # 모음별
    print(f"\n  모음별 정확도:")
    print(f"  {'모음':>4s}  {'시도':>4s}  {'정답':>4s}  {'정확도':>6s}  {'평균지연':>8s}")
    print(f"  {'~'*4}  {'~'*4}  {'~'*4}  {'~'*6}  {'~'*8}")

    for v in VOWELS:
        v_results = [r for r in results if r[0] == v]
        if not v_results:
            continue
        v_correct = sum(1 for r in v_results if r[0] == r[1])
        v_total = len(v_results)
        v_acc = v_correct / v_total * 100
        v_lat = np.mean([r[3] for r in v_results])
        print(f"  {v:>4s}  {v_total:>4d}  {v_correct:>4d}  {v_acc:>5.1f}%  {v_lat:>7.0f}ms")

    # 혼동행렬
    active = [v for v in VOWELS if any(r[0] == v for r in results)]
    print(f"\n  혼동행렬:")
    print(f"  {'':>6s}", end='')
    for v in active:
        print(f"  {v:>4s}", end='')
    print()

    for v in active:
        print(f"  {v:>6s}", end='')
        v_results = [(r[0], r[1]) for r in results if r[0] == v]
        for v2 in active:
            cnt = sum(1 for gt, pred in v_results if pred == v2)
            if cnt == 0:
                print(f"  {'.' :>4s}", end='')
            else:
                print(f"  {cnt:>4d}", end='')
        print()

    # 주요 오인
    errors = [(r[0], r[1]) for r in results if r[0] != r[1]]
    if errors:
        print(f"\n  오인 목록:")
        for gt, pred in errors:
            print(f"    {gt} -> {pred}")


# ─────────────────────────────────────
# 모드 B: 자유 테스트
# ─────────────────────────────────────
def mode_free(extractor, classifier, capture, collector):
    print("\n" + "=" * 60)
    print("  모드 B: 자유 테스트")
    print("=" * 60)
    print("  마이크에 모음을 말하면 실시간으로 인식합니다.")
    print("  Ctrl+C로 종료.")
    print("=" * 60 + "\n")

    collector.reset()
    collector.enabled = True

    history = []

    try:
        while True:
            utt = collector.get_utterance()
            if utt is not None:
                duration = len(utt) / capture.sample_rate

                t0 = time.perf_counter()
                emb16, emb567, model_ms = extractor.extract(utt, capture.sample_rate)
                pred, conf, prob_dict = classifier.predict(emb16, emb567, debug=True)
                total_ms = (time.perf_counter() - t0) * 1000

                # 확률 바
                bar_len = 20
                filled = int(conf * bar_len)
                bar = '#' * filled + '.' * (bar_len - filled)

                # 상위 3개 확률
                top3 = sorted(prob_dict.items(), key=lambda x: x[1], reverse=True)[:3]
                top3_str = "  ".join(f"{v}:{p:.0%}" for v, p in top3)

                print(f"  [{pred}] {conf:.0%}  {bar}  {total_ms:.0f}ms  "
                      f"({duration:.2f}s)  {top3_str}")

                history.append((pred, conf, total_ms))
            else:
                if collector.is_collecting:
                    sys.stdout.write("\r  (녹음 중...)        ")
                    sys.stdout.flush()
                time.sleep(0.02)

    except KeyboardInterrupt:
        collector.enabled = False
        print("\n\n  종료.")
        if history:
            avg_lat = np.mean([ms for _, _, ms in history])
            avg_conf = np.mean([c for _, c, _ in history])
            print(f"  총 {len(history)}회 인식, 평균 신뢰도: {avg_conf:.0%}, 평균 지연: {avg_lat:.0f}ms")


# ═══════════════════════════════════════
# 메인
# ═══════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="Method 6 실시간 테스트")
    parser.add_argument('--mode', default='eval', choices=['eval', 'free'],
                        help='eval: 정량 평가, free: 자유 테스트')
    parser.add_argument('--trials', type=int, default=10,
                        help='모음당 시도 횟수 (eval 모드)')
    parser.add_argument('--device', type=int, default=None,
                        help='오디오 입력 디바이스 번호')
    args = parser.parse_args()

    # 모델 경로
    model_dir = os.path.dirname(__file__)
    model_path = os.path.join(model_dir, 'twostage_model.pkl')
    if not os.path.exists(model_path):
        print(f"모델 파일 없음: {model_path}")
        print("먼저 pretrain_models.py를 실행하세요.")
        sys.exit(1)

    # ── 초기화 ──
    print("=" * 60)
    print("  Method 6: XLSR-53 2단계 실시간 모음 인식")
    print("=" * 60)

    extractor = LiveEmbeddingExtractor()
    classifier = TwoStageClassifier(model_path)

    # 오디오 캡처
    capture = AudioCapture(sample_rate=44100, blocksize=2048, device=args.device)
    collector = UtteranceCollector(sr=44100)
    capture.add_listener(collector.on_audio)

    # 마이크 정보
    try:
        dev_info = capture.default_device()
        print(f"\n[마이크] {dev_info['name']}")
    except Exception:
        print(f"\n[마이크] device={args.device or 'default'}")
    print(f"  샘플레이트: {capture.sample_rate}Hz, 블록: {capture.blocksize}")

    # 웜업 (첫 추론은 느리므로)
    print("\n[웜업] 더미 추론 실행 중...", flush=True)
    dummy = np.random.randn(16000).astype(np.float32) * 0.01
    _, _, warmup_ms = extractor.extract(dummy, 16000)
    print(f"[웜업] 완료 ({warmup_ms:.0f}ms)")

    # 캡처 시작
    capture.start()
    print("\n[마이크] 시작됨.\n")

    try:
        if args.mode == 'eval':
            mode_eval(extractor, classifier, capture, collector, args.trials)
        else:
            mode_free(extractor, classifier, capture, collector)
    finally:
        capture.stop()
        print("[마이크] 종료됨.")


if __name__ == '__main__':
    main()
