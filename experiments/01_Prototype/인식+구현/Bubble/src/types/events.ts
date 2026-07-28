/**
 * VoiceEventDetector 가 발행하는 "타입드 이벤트" 정의.
 *
 * 왜 이벤트인가(프롬프트 §3):
 *  - 세기·길이·피치·떨림은 "프레임 간 변화 관찰"이 필요하다.
 *  - 그래서 VoiceEventDetector 가 이전 프레임과 현재 프레임을 비교해
 *    의미 있는 변화가 있을 때만 이벤트를 발행하고,
 *  - BubbleManager 는 프레임 원시 데이터를 보지 않고 이 이벤트만 구독한다.
 *
 * 한 특징 → 한 시각 자유도 매핑(프롬프트 §2 + 수정안):
 *  - INTENSITY_CHANGED (세기)  → 세그먼트 크기(미세 변화)
 *  - INTENSITY_SEGMENT (세기)  → 경계 없는 새 세그먼트(큰 변화, 수정 4)
 *  - PITCH_SHIFT      (피치)   → 버블 수직 위치(상하 이동, 수정 2)
 *  - UTTERANCE_END.tremor (떨림) → 이동 방향 수직축 진동(수정 1)
 *  - VOWEL_CHANGED / VOWEL_RESOLVED (모음) → 글자 + 세그먼트 경계
 *  - RECOGNITION_LOST/REGAINED → 인식 불가 처리
 */

import type { MaybeVowel, Vowel } from './audio';

export type VoiceEventType =
  | 'UTTERANCE_START'
  | 'UTTERANCE_END'
  | 'VOWEL_RESOLVED'
  | 'VOWEL_CHANGED'
  | 'INTENSITY_CHANGED'
  | 'INTENSITY_SEGMENT'
  | 'SYLLABLE_REPEAT'
  | 'PITCH_SHIFT'
  | 'RECOGNITION_LOST'
  | 'RECOGNITION_REGAINED';

/** 모든 이벤트가 공통으로 갖는 시각(performance.now(), ms). */
interface BaseEvent {
  time: number;
}

/**
 * 발화 시작.
 *  - 버블을 즉시 생성한다(FORMING). 인식 결과가 아직이면 vowel=null 로 시작하고
 *    (spawn-then-detach: 스프링 오버슈트로 톡 튀어나오게) 나중에 VOWEL_RESOLVED 로 채운다.
 */
export interface UtteranceStartEvent extends BaseEvent {
  type: 'UTTERANCE_START';
  /** 시작 시점 에너지 0~1 (첫 세그먼트 초기 크기). */
  energy: number;
  /** 시작 시점 피치 레벨 (-1 최저 ~ +1 최고 → 수직 위치), 검출 실패 시 0. */
  level: number;
  /** 시작 시점 모음(대개 null: 인식 지연). */
  vowel: MaybeVowel;
}

/**
 * 발화 종료.
 *  - 버블을 완성하고 FLOATING(상승)으로 전이한다.
 *  - tremor(떨림 누적치)를 실어 보내 상승 경로 곡률에 쓴다.
 */
export interface UtteranceEndEvent extends BaseEvent {
  type: 'UTTERANCE_END';
  /** 발화 전체 길이(ms). false-start 취소 판정에도 사용. */
  durationMs: number;
  /** 떨림 정도 0~1 (피치 지터 누적 → 정규화). 이동 방향 수직축 진동 진폭. */
  tremor: number;
  /**
   * 발성 중 피치 궤적을 발화 내 최소~최대로 정규화한 0~1 시퀀스. (3차 수정 3)
   * 완성 후 비행 y좌표 재생에 쓴다. 유효 피치가 부족하면 빈 배열(→ y 궤적 없음).
   */
  pitchTrajectory: number[];
}

/**
 * null 로 시작한 버블에 첫 모음이 도착(인식 지연 해소).
 *  - 글자를 채운다. 첫 세그먼트에는 경계선을 만들지 않는다.
 */
export interface VowelResolvedEvent extends BaseEvent {
  type: 'VOWEL_RESOLVED';
  vowel: Vowel;
}

/**
 * 발화 도중 모음이 바뀜.
 *  - 새 세그먼트를 만든다. 모음이 바뀌었으므로 hasBoundary=true (경계선 있음).
 *  - 붙는 방향은 BubbleManager 가 랜덤으로 정한다(프롬프트 §1-3).
 */
export interface VowelChangedEvent extends BaseEvent {
  type: 'VOWEL_CHANGED';
  vowel: Vowel;
  /** 전환 시점 에너지 0~1 (새 세그먼트 초기 크기). */
  energy: number;
}

/**
 * 세기(에너지) 변화가 임계값을 넘음.
 *  - 현재(마지막) 세그먼트의 목표 크기를 갱신한다. 스프링으로 보간.
 */
export interface IntensityChangedEvent extends BaseEvent {
  type: 'INTENSITY_CHANGED';
  energy: number;
}

/**
 * 세기가 "크게" 바뀜(같은 모음 유지 중). (수정 4)
 *  - 경계선 없는(hasBoundary=false) 새 세그먼트를 붙여 땅콩 형태를 만든다.
 *  - 새 세그먼트 크기 = 바뀐 세기.
 */
export interface IntensitySegmentEvent extends BaseEvent {
  type: 'INTENSITY_SEGMENT';
  energy: number;
}

/**
 * 재발성(음절 반복) — 피치 연속성 기반. (재발성 감지)
 *  - "아아아"처럼 한 호흡 안에서 같은 모음을 반복 → 에너지 딥(하강→회복)이 발생하되
 *    피치 트랙은 이어진다. 이때 같은 모음의 경계 없는(hasBoundary=false) 새 로브를
 *    **즉시**(유지 시간 조건 없이) 붙인다.
 *  - 새 로브 크기 = 해당 음절의 기준 세기(energy).
 */
export interface SyllableRepeatEvent extends BaseEvent {
  type: 'SYLLABLE_REPEAT';
  energy: number;
}

/**
 * 피치 변화가 임계값을 넘음. (2차 수정 3: 다시 종횡비로)
 *  - 버블 전체의 목표 종횡비를 갱신한다. 스프링으로 보간.
 */
export interface PitchShiftEvent extends BaseEvent {
  type: 'PITCH_SHIFT';
  /** 피치 레벨 (-1 최저=가로로 넓적 ~ 0 원형 ~ +1 최고=세로로 길쭉). */
  level: number;
}

/**
 * 인식 가능 → 불가로 바뀜(발화 도중).
 *  - 이미 글자가 있으면 글자가 무너지고 잔해로 쌓인다(프롬프트 §2-5).
 */
export interface RecognitionLostEvent extends BaseEvent {
  type: 'RECOGNITION_LOST';
}

/** 인식 불가 → 다시 가능. (확장 지점: 현재 버블에 글자를 다시 채울 수 있음) */
export interface RecognitionRegainedEvent extends BaseEvent {
  type: 'RECOGNITION_REGAINED';
  vowel: Vowel;
}

/** 모든 음성 이벤트의 판별 유니온. */
export type VoiceEvent =
  | UtteranceStartEvent
  | UtteranceEndEvent
  | VowelResolvedEvent
  | VowelChangedEvent
  | IntensityChangedEvent
  | IntensitySegmentEvent
  | SyllableRepeatEvent
  | PitchShiftEvent
  | RecognitionLostEvent
  | RecognitionRegainedEvent;
