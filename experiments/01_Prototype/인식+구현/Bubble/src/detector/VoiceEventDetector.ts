import type { MaybeVowel, Vowel, VoiceFrame } from '../types/audio';
import type { VoiceEvent } from '../types/events';
import { hzToSemitone, pitchToLevel } from '../audio/pitchMath';
import { lerp } from '../animation/mathUtils';
import {
  ENERGY_SMOOTHING,
  INTENSITY_BASELINE_SMOOTHING,
  INTENSITY_EVENT_DELTA,
  PITCH_EVENT_SEMITONES,
  PITCH_SMOOTHING,
  RECOGNITION_LOST_FRAMES,
  RECOGNITION_LOST_MIN_ENERGY,
  SILENCE_ENERGY,
  START_ENERGY,
  TRAJECTORY_KEYPOINTS,
  TREMOR_FULL_SCALE_SEMITONES,
  VOWEL_CHANGE_DEBOUNCE_FRAMES,
} from '../config/bubbleConfig';
import { popSettings } from '../config/popSettings';

/**
 * VoiceEventDetector
 * ==================
 * 프롬프트 §3의 핵심 모듈. "이전 프레임과 비교"해 의미 있는 변화가 있을 때만
 * 타입드 이벤트(VoiceEvent[])를 발행한다.
 *
 * 입력:  매 분석 주기(≈25ms)의 VoiceFrame (파형 → 파생 특징을 합친 것)
 * 출력:  그 프레임에서 감지된 이벤트 배열 (없으면 빈 배열)
 *
 * 이 클래스가 "프레임 원시 데이터를 소비하는 마지막 지점"이다.
 * 하류의 BubbleManager 는 이벤트만 본다(프레임 직접 참조 금지).
 *
 * 발행 이벤트와 한 특징→한 자유도 대응:
 *   UTTERANCE_START/END  : 버블 생성/완성·상승
 *   VOWEL_RESOLVED       : null 스폰 버블에 첫 글자 채움(인식 지연 해소)
 *   VOWEL_CHANGED        : 모음 = 새 세그먼트 + 경계선
 *   INTENSITY_CHANGED    : 세기 = 세그먼트 크기
 *   PITCH_SHIFT          : 피치 = 종횡비
 *   RECOGNITION_LOST/REGAINED : 인식 불가 처리
 *   (떨림은 UTTERANCE_END.tremor 로 실어 보낸다 → 상승 경로 곡률)
 */
export class VoiceEventDetector {
  // ---- 발화 상태 ----
  private speaking = false;
  private utteranceStart = 0;
  private lastSoundTime = 0;

  // ---- 스무딩된 특징(프레임 간 비교 기준) ----
  private smoothedEnergy = 0;
  private smoothedPitch: number | null = null;

  // ---- 이벤트 디바운스/기준값 ----
  private lastEmittedEnergy = 0;
  private lastEmittedSemitone = 0;
  /** 현재 세그먼트의 기준 세기(이동평균). 상대 dB 판정의 기준선(세기 판정 안정화). */
  private segBaselineEnergy = 0;
  /** 세기 변화 후보가 임계를 넘긴 시각(ms). -1이면 후보 없음. 유지 시간 확정용. */
  private intCandidateSince = -1;

  // ---- 변화 속도(급변/점진) 상태 ----
  private slopeSamples: { t: number; e: number }[] = []; // 기울기 측정 슬라이딩 윈도우
  private stepAnchorEnergy = 0; // 마지막 로브 생성 시 세기(점진 누적 기준)
  private gradLastStepTime = 0; // 마지막 단계 로브 시각(최소 간격용)

  // ---- 재발성(음절 반복) 딥 상태 ----
  private inDip = false;
  private dipStart = 0;
  private dipBaseline = 0; // 딥 진입 시점의 기준 세기(회복/크기 판정용)
  private dipPitchAbsentSince = -1; // 현재 연속 피치 공백 시작(ms), -1=피치 있음
  private dipMaxPitchGap = 0; // 딥 중 최대 연속 피치 공백(ms)
  private dipArmed = true; // 딥 재진입 허용 여부(지속 하강의 딥 반복 진입 방지)

  // ---- 모음 상태 ----
  private currentVowel: MaybeVowel = null; // 현재 세그먼트의 확정 모음
  private vowelResolved = false; // null 스폰 후 첫 모음이 채워졌는가
  private pendingVowel: Vowel | null = null; // 전환 디바운스 후보
  private pendingCount = 0;

  // ---- 인식 상태 ----
  private recognitionLost = false;
  private lostCount = 0;

  // ---- 떨림 누적(발화 중) ----
  private prevRawPitch: number | null = null;
  private jitterSum = 0;
  private jitterCount = 0;

  // ---- 피치 궤적 기록(발화 중) → 완성 후 비행 y (3차 수정 3) ----
  private pitchSamples: number[] = [];

  /** 시작/정지 시 상태 초기화. */
  reset(): void {
    this.speaking = false;
    this.smoothedEnergy = 0;
    this.smoothedPitch = null;
    this.currentVowel = null;
    this.vowelResolved = false;
    this.pendingVowel = null;
    this.pendingCount = 0;
    this.recognitionLost = false;
    this.lostCount = 0;
    this.prevRawPitch = null;
    this.jitterSum = 0;
    this.jitterCount = 0;
    this.segBaselineEnergy = 0;
    this.intCandidateSince = -1;
    this.slopeSamples = [];
    this.stepAnchorEnergy = 0;
    this.gradLastStepTime = 0;
    this.dipArmed = true;
    this.resetDip();
    this.pitchSamples = [];
  }

  /**
   * 한 프레임을 처리하고 이번에 감지된 이벤트들을 반환한다.
   */
  process(frame: VoiceFrame): VoiceEvent[] {
    const events: VoiceEvent[] = [];
    const t = frame.time;

    // (0) 특징 스무딩 — 스파이크로 크기/종횡비가 튀지 않도록.
    this.smoothedEnergy = lerp(this.smoothedEnergy, frame.energy, ENERGY_SMOOTHING);
    if (frame.pitch != null) {
      this.smoothedPitch =
        this.smoothedPitch == null
          ? frame.pitch
          : lerp(this.smoothedPitch, frame.pitch, PITCH_SMOOTHING);
    }

    // 떨림: 스무딩 전 원시 피치의 프레임 간 반음 지터를 누적(발화 중일 때만 의미).
    if (this.speaking && frame.pitch != null && this.prevRawPitch != null) {
      this.jitterSum += Math.abs(hzToSemitone(frame.pitch) - hzToSemitone(this.prevRawPitch));
      this.jitterCount++;
    }
    if (frame.pitch != null) this.prevRawPitch = frame.pitch;

    // (1) 발화 시작 판정(히스테리시스).
    if (!this.speaking) {
      if (this.smoothedEnergy >= START_ENERGY) {
        this.beginUtterance(frame, events);
      }
      return events; // 아직 침묵이면 더 볼 것 없음
    }

    // ---- 여기부터는 발화 중 ----

    // 피치 궤적 기록(완성 후 비행 y 재생용). 유효 피치일 때만.
    if (this.smoothedPitch != null) this.pitchSamples.push(this.smoothedPitch);

    // (2) 발성 종료 판정(재발성 수정 2): **에너지 침묵 + 피치 소실**이 함께
    //   utteranceBreakMs 이상 지속돼야 종료(새 버블). 피치가 이어지면(재발성 딥) 유지한다.
    const voicePresent = this.smoothedEnergy >= SILENCE_ENERGY || frame.pitch != null;
    if (voicePresent) {
      this.lastSoundTime = t;
    } else if (t - this.lastSoundTime > popSettings.utteranceBreakMs) {
      this.endUtterance(t, events);
      return events;
    }

    // (3) 인식 가능/불가 추적.
    this.trackRecognition(frame, events);

    // (4) 모음 해소/전환 (인식된 프레임에서만).
    if (frame.recognized && frame.vowel != null) {
      this.trackVowel(frame.vowel, t, events);
    }

    // (5a-0) 재발성(음절 반복) 딥 감지 — 세기 수준 변화보다 먼저(딥에서 재발성 우선, 공존 규칙 3).
    this.trackReutterance(frame, t, events);

    // (5a) 세기 큰 변화 → INTENSITY_SEGMENT + 기준선(segBaselineEnergy) 갱신.
    //   딥 중에도 후보는 계속 쌓되 **확정(발행)만 유보**한다(공존 규칙 3). → 딥이 회복하면
    //   재발성이 후보를 취소하고, 회복 없이 수준이 눌러앉으면(딥 타임아웃) 곧바로 수준 변화로 확정.
    this.trackIntensity(frame, t, events);

    // (5b) 크기 미세 추종 → INTENSITY_CHANGED (현재 세그먼트 크기 보간).
    //   버블 크기 동결 수정: 크기 목표는 순간 RMS가 아니라 **세그먼트 기준 세기
    //   (segBaselineEnergy, 이동 평균)** 를 따른다. → 어택·릴리즈·순간 출렁임에 크기가
    //   요동하지 않고, 세기가 실제로 달라졌을 때만 부드럽게 따라간다. 릴리즈 감쇠가 종료 직전
    //   크기를 끌어내리지 않으므로, UTTERANCE_END 동결 시 커진 크기 그대로 고정된다.
    if (this.vowelResolved) {
      const be = this.segBaselineEnergy;
      if (Math.abs(be - this.lastEmittedEnergy) >= INTENSITY_EVENT_DELTA) {
        this.lastEmittedEnergy = be;
        events.push({ type: 'INTENSITY_CHANGED', time: t, energy: be });
      }
    }

    // (6) 피치 변화 → PITCH_SHIFT (활성 세그먼트 종횡비).
    //   ⚠️ 4차 수정 3: 모음 전환이 디바운스 중(pendingVowel)이면 억제한다.
    //   그래야 새 피치가 곧 만들어질 새 세그먼트에 실리고, 곧 동결될 앞 세그먼트를
    //   끌어당기지 않는다("발화 후반 피치 변화가 앞 세그먼트 모양을 바꾸지 않는다").
    if (this.smoothedPitch != null && this.pendingVowel == null) {
      const st = hzToSemitone(this.smoothedPitch);
      if (Math.abs(st - this.lastEmittedSemitone) >= PITCH_EVENT_SEMITONES) {
        this.lastEmittedSemitone = st;
        events.push({ type: 'PITCH_SHIFT', time: t, level: pitchToLevel(this.smoothedPitch) });
      }
    }

    return events;
  }

  /** 발화 시작 처리 + UTTERANCE_START 발행. */
  private beginUtterance(frame: VoiceFrame, events: VoiceEvent[]): void {
    this.speaking = true;
    this.utteranceStart = frame.time;
    this.lastSoundTime = frame.time;

    // per-utterance 누적/기준 초기화
    this.lastEmittedEnergy = this.smoothedEnergy;
    this.segBaselineEnergy = this.smoothedEnergy;
    this.intCandidateSince = -1;
    this.slopeSamples = [];
    this.stepAnchorEnergy = this.smoothedEnergy;
    this.gradLastStepTime = frame.time;
    this.dipArmed = true;
    this.resetDip();
    this.lastEmittedSemitone = this.smoothedPitch != null ? hzToSemitone(this.smoothedPitch) : 0;
    this.recognitionLost = false;
    this.lostCount = 0;
    this.jitterSum = 0;
    this.jitterCount = 0;
    this.pitchSamples = [];
    this.pendingVowel = null;
    this.pendingCount = 0;

    // 대개 시작 시점엔 모음이 아직 null(인식 지연) → 나중에 VOWEL_RESOLVED.
    this.currentVowel = frame.recognized ? frame.vowel : null;
    this.vowelResolved = this.currentVowel != null;

    const level = this.smoothedPitch != null ? pitchToLevel(this.smoothedPitch) : 0;
    events.push({
      type: 'UTTERANCE_START',
      time: frame.time,
      energy: this.smoothedEnergy,
      level,
      vowel: this.currentVowel,
    });
  }

  /** 발화 종료 처리 + UTTERANCE_END(길이·떨림) 발행. */
  private endUtterance(t: number, events: VoiceEvent[]): void {
    const durationMs = this.lastSoundTime - this.utteranceStart;
    // 떨림: 프레임 간 평균 반음 지터 → 0~1 정규화.
    const avgJitter = this.jitterCount > 0 ? this.jitterSum / this.jitterCount : 0;
    const tremor = Math.min(1, avgJitter / TREMOR_FULL_SCALE_SEMITONES);
    // 피치 궤적: 발화 내 최소~최대로 정규화한 0~1 시퀀스(완성 후 비행 y 재생).
    const pitchTrajectory = this.normalizePitchTrajectory();

    events.push({ type: 'UTTERANCE_END', time: t, durationMs, tremor, pitchTrajectory });

    this.speaking = false;
    this.currentVowel = null;
    this.vowelResolved = false;
    this.prevRawPitch = null;
    this.resetDip();
  }

  /**
   * 기록한 (스무딩된) 피치 샘플(Hz) → 발화 내 최소~최대로 정규화한 0~1 궤적.
   * 4차 수정 1: 소수의 키포인트(TRAJECTORY_KEYPOINTS)로 리샘플링해 큰 흐름만 남긴다
   * (계단식/지그재그 제거). 유효 샘플이 2개 미만이거나 변화가 없으면 빈 배열.
   */
  private normalizePitchTrajectory(): number[] {
    const s = this.pitchSamples;
    if (s.length < 2) return [];
    let min = Infinity;
    let max = -Infinity;
    for (const v of s) {
      if (v < min) min = v;
      if (v > max) max = v;
    }
    const range = max - min;
    if (range < 1e-6) return []; // 피치 변화 없음 → y 힌트 없음

    // 키포인트로 리샘플링(구간 평균으로 부드럽게).
    const k = Math.min(TRAJECTORY_KEYPOINTS, s.length);
    const out: number[] = [];
    for (let i = 0; i < k; i++) {
      const lo = Math.floor((i / k) * s.length);
      const hi = Math.max(lo + 1, Math.floor(((i + 1) / k) * s.length));
      let sum = 0;
      let n = 0;
      for (let j = lo; j < hi && j < s.length; j++) {
        sum += s[j];
        n++;
      }
      out.push(((n > 0 ? sum / n : s[lo]) - min) / range);
    }
    return out;
  }

  /**
   * 재발성(음절 반복) 감지 — 피치 연속성 기반. (재발성 수정 1·3)
   * 에너지 딥(기준의 dipRatio 아래로 하강 → 회복)을 재발성으로 판정하되, 딥 구간 피치 공백이
   * pitchGapMaxMs 이하(=피치 연속)일 때만 인정한다. 확정 시 SYLLABLE_REPEAT 를 **즉시**
   * 발행(유지 시간 조건 없음). 스파이크(상승→하강)는 여기서 다루지 않는다 — 하강→회복만 라우팅.
   */
  private trackReutterance(frame: VoiceFrame, t: number, events: VoiceEvent[]): void {
    const P = popSettings;
    // 활성 세그먼트 + 온셋 유예 지난 뒤에만.
    if (!this.vowelResolved || t - this.utteranceStart < P.onsetGraceMs) {
      this.resetDip();
      return;
    }
    const e = frame.energy; // 딥은 원시 에너지로 본다(스무딩이 딥을 가리므로).
    const baseline = this.segBaselineEnergy;
    if (!(baseline > 1e-6)) return;

    if (!this.inDip) {
      // 에너지가 딥 임계 위로 회복하면 다음 딥을 재장전(지속 하강의 반복 진입 방지).
      if (e >= baseline * P.dipRatio) this.dipArmed = true;
      // 무장 상태 + 하강 → 딥 진입. (누적 리셋은 회복 시 onLobeCreated 가, 지속 하강은
      //  trackIntensity 의 수준 감소 경로가 담당하므로 여기서 후보를 건드리지 않는다.)
      if (this.dipArmed && e < baseline * P.dipRatio) {
        this.inDip = true;
        this.dipStart = t;
        this.dipBaseline = baseline;
        this.dipMaxPitchGap = 0;
        this.dipPitchAbsentSince = frame.pitch == null ? t : -1;
      }
      return;
    }

    // ---- 딥 진행 중 ----
    // 피치 공백(최대 연속) 추적.
    if (frame.pitch == null) {
      if (this.dipPitchAbsentSince < 0) this.dipPitchAbsentSince = t;
      this.dipMaxPitchGap = Math.max(this.dipMaxPitchGap, t - this.dipPitchAbsentSince);
    } else {
      this.dipPitchAbsentSince = -1;
    }

    // 피치 공백이 한계 초과 → 재발성 아님(끊김). 딥 취소(→ 종료 로직/침묵이 처리).
    if (this.dipMaxPitchGap > P.pitchGapMaxMs) {
      this.dipArmed = false; // 끊김(피치 소실) — 에너지 회복 전까지 재진입 금지
      this.resetDip();
      return;
    }

    const dipDur = t - this.dipStart;
    // 회복: 에너지가 기준 부근으로 복귀(진입 임계 재상향 통과).
    if (e >= this.dipBaseline * P.dipRatio) {
      if (dipDur >= P.dipMinMs && dipDur <= P.dipMaxMs) {
        // 재발성 확정 → 같은 모음 경계 없는 새 로브(즉시). 크기 = 이 음절 기준 세기(딥 이전 수준).
        events.push({ type: 'SYLLABLE_REPEAT', time: t, energy: this.dipBaseline });
        this.onLobeCreated(this.dipBaseline, t); // 기준선·앵커·윈도우 리셋(재발성 우선)
      }
      this.dipArmed = true; // 에너지 회복됨
      this.resetDip();
      return;
    }
    // 회복 없이 너무 오래 → 재발성 아님(지속 하강 = 수준 감소). trackIntensity 가 처리하도록
    //  넘기고, 에너지 회복 전까지 딥 재진입을 막는다.
    if (dipDur > P.dipMaxMs) {
      this.dipArmed = false;
      this.resetDip();
    }
  }

  private resetDip(): void {
    this.inDip = false;
    this.dipPitchAbsentSince = -1;
    this.dipMaxPitchGap = 0;
  }

  /**
   * 세기 변화 → INTENSITY_SEGMENT. 변화 **속도**를 슬라이딩 윈도우 기울기로 분류해
   * 로브 개수에 매핑한다(확 커짐 vs 점점 커짐).
   *  - 급변: 윈도우(slopeWindowMs) 안 변화 ≥ intensityChangeDb(=SUDDEN_DB) → 유지 시간 후 분할 1회(로브 2개).
   *  - 점진: 윈도우당 변화 < SUDDEN_DB. 느린 드리프트는 기준선 EMA 가 흡수하되 **누적 dB**(마지막 로브 대비)를
   *    추적해, gradualStepDb 도달마다 단계 로브를 추가(로브 여러 개). → 점진 변화가 조용히 사라지지 않는다.
   *  - 감소도 부호만 반대로 동일 적용. 딥(재발성) 중에는 발행 유보(공존 규칙).
   * 보조) 시작 유예(onsetGraceMs) 동안은 판정 없이 기준선·앵커 산출에만 쓴다.
   * (모음 전환 세그먼트 경로는 trackVowel, 재발성은 trackReutterance 가 담당.)
   */
  private trackIntensity(frame: VoiceFrame, t: number, events: VoiceEvent[]): void {
    const P = popSettings;
    const e = this.smoothedEnergy;

    // 실제로 그 모음을 "내고 있는" 프레임에서만(끝자락 페이드/무음 제외) + 모음 확정 후.
    if (!(frame.recognized && this.vowelResolved)) {
      this.intCandidateSince = -1; // 후보 폐기(흡수)
      return;
    }

    // 시작 유예: 이 구간 세기는 기준선·앵커 산출에만.
    if (t - this.utteranceStart < P.onsetGraceMs) {
      this.segBaselineEnergy = e;
      this.stepAnchorEnergy = e;
      this.gradLastStepTime = t;
      this.intCandidateSince = -1;
      this.slopeSamples.length = 0;
      return;
    }
    if (this.segBaselineEnergy <= 1e-6) this.segBaselineEnergy = e;
    if (this.stepAnchorEnergy <= 1e-6) this.stepAnchorEnergy = e;

    // (1) 변화 "속도" 분류 — 슬라이딩 윈도우(slopeWindowMs) 기울기(dB/창).
    this.slopeSamples.push({ t, e });
    while (this.slopeSamples.length > 1 && this.slopeSamples[0].t < t - P.slopeWindowMs) {
      this.slopeSamples.shift();
    }
    const eOld = this.slopeSamples[0].e;
    const windowDb = 20 * Math.log10(Math.max(e, 1e-6) / Math.max(eOld, 1e-6));
    const sudden = Math.abs(windowDb) >= P.intensityChangeDb; // SUDDEN_DB = intensityChangeDb

    if (sudden) {
      // 급변: 유지 시간 후 세그먼트 분할 1회(딥 중 유보). 점진 누적은 하지 않는다. → 로브 2개.
      if (this.intCandidateSince < 0) this.intCandidateSince = t;
      if (t - this.intCandidateSince >= P.intensityHoldMs && !this.inDip) {
        events.push({ type: 'INTENSITY_SEGMENT', time: t, energy: e });
        this.onLobeCreated(e, t);
      }
      return;
    }

    // (2) 점진: 윈도우당 변화 < SUDDEN_DB. 급변 후보 취소(스파이크 억제) + 느린 드리프트는
    //   기준선(크기용)이 EMA 로 따라가되, **마지막 로브(stepAnchor) 대비 누적 dB** 를 추적한다.
    this.intCandidateSince = -1;
    this.segBaselineEnergy = lerp(this.segBaselineEnergy, e, INTENSITY_BASELINE_SMOOTHING);
    const stepDb = 20 * Math.log10(Math.max(e, 1e-6) / Math.max(this.stepAnchorEnergy, 1e-6));
    // (3) 누적이 GRADUAL_STEP_DB 도달 + 최소 간격 경과 → 단계 로브(딥 중 유보).
    if (
      Math.abs(stepDb) >= P.gradualStepDb &&
      t - this.gradLastStepTime >= P.gradualStepMinMs &&
      !this.inDip
    ) {
      events.push({ type: 'INTENSITY_SEGMENT', time: t, energy: e });
      this.onLobeCreated(e, t);
    }
  }

  /** 로브 하나가 생성된 뒤 세기 기준선·점진 앵커·기울기 창을 초기화한다. */
  private onLobeCreated(e: number, t: number): void {
    this.segBaselineEnergy = e; // 새 로브 수준을 기준선으로
    this.stepAnchorEnergy = e; // 점진 누적 기준 리셋
    this.gradLastStepTime = t;
    this.intCandidateSince = -1;
    this.slopeSamples.length = 0; // 재트리거·회복 스파이크 방지
    this.lastEmittedEnergy = e;
  }

  /**
   * 인식 가능/불가 상태 변화를 이벤트로.
   *
   * ⚠️ 핵심: RECOGNITION_LOST 는 "소리를 내고 있는데 인식이 안 되는" 경우여야 한다.
   * 발화 끝의 자연스러운 침묵(에너지가 SILENCE_ENERGY 아래로 잦아듦)은 인식 실패가 아니라
   * 발화 종료로 향하는 정상 과정이므로, 그 구간은 lost 로 세지 않는다.
   * (이 게이트가 없으면 모든 발화가 끝자락 침묵에서 글자 붕괴(DEBRIS)로 오작동한다.)
   */
  private trackRecognition(frame: VoiceFrame, events: VoiceEvent[]): void {
    // 3차 수정 2: "크게 내고 있는데" 인식이 안 될 때만 lost 로 센다.
    // 발화 끝 볼륨 감쇠(SILENCE~MIN_ENERGY 구간)는 인식 실패로 치지 않는다.
    const voicing = this.smoothedEnergy >= RECOGNITION_LOST_MIN_ENERGY;
    if (frame.recognized) {
      if (this.recognitionLost) {
        this.recognitionLost = false;
        if (frame.vowel != null) {
          events.push({ type: 'RECOGNITION_REGAINED', time: frame.time, vowel: frame.vowel });
        }
      }
      this.lostCount = 0;
    } else if (voicing) {
      this.lostCount++;
      // 짧은 무성 구간(파열음 등)에서 곧바로 무너지지 않도록 연속 프레임으로 확정.
      if (!this.recognitionLost && this.lostCount >= RECOGNITION_LOST_FRAMES) {
        this.recognitionLost = true;
        events.push({ type: 'RECOGNITION_LOST', time: frame.time });
      }
    } else {
      // 침묵(발화 종료로 향하는 중) — 인식 실패로 치지 않는다.
      this.lostCount = 0;
    }
  }

  /** 모음 최초 해소(VOWEL_RESOLVED) 또는 전환(VOWEL_CHANGED, 디바운스). */
  private trackVowel(vowel: Vowel, t: number, events: VoiceEvent[]): void {
    if (!this.vowelResolved) {
      // null 스폰 버블에 첫 글자 채움.
      this.currentVowel = vowel;
      this.vowelResolved = true;
      this.pendingVowel = null;
      this.pendingCount = 0;
      // 세기 기준선을 "발성이 자리잡은" 이 시점으로 리셋 + 후보 폐기 →
      // 발화 온셋의 자연스러운 볼륨 상승이 세그먼트를 낳지 않게 한다.
      this.onLobeCreated(this.smoothedEnergy, t); // 새 세그먼트 기준선·점진 앵커·윈도우 리셋
      events.push({ type: 'VOWEL_RESOLVED', time: t, vowel });
      return;
    }

    if (vowel === this.currentVowel) {
      // 같은 모음 지속 → 전환 후보 리셋(세그먼트 유지).
      this.pendingVowel = null;
      this.pendingCount = 0;
      return;
    }

    // 다른 모음 → 디바운스 후 확정.
    if (vowel === this.pendingVowel) {
      this.pendingCount++;
    } else {
      this.pendingVowel = vowel;
      this.pendingCount = 1;
    }
    if (this.pendingCount >= VOWEL_CHANGE_DEBOUNCE_FRAMES) {
      this.currentVowel = vowel;
      this.pendingVowel = null;
      this.pendingCount = 0;
      // 모음이 바뀜 → 새 세그먼트(경계선 있음). 크기 초기값용 에너지 동봉.
      this.onLobeCreated(this.smoothedEnergy, t); // 새 세그먼트 기준선·점진 앵커·윈도우 리셋
      events.push({ type: 'VOWEL_CHANGED', time: t, vowel, energy: this.smoothedEnergy });
    }
  }
}
