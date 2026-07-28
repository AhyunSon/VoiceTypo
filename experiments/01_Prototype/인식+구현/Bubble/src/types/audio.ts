/**
 * 오디오 파생 스트림의 타입.
 *
 * 구현 원칙(프롬프트 §3): "오디오 원천은 파형 하나이며 모든 특징은 파생 스트림이다."
 *  - AudioManager 가 마이크 파형 하나를 제공하고,
 *  - Pitch/Energy/Formant 분석기가 그 파형에서 프레임 단위 특징을 뽑아
 *  - VoiceFrame 하나로 합쳐진다.
 *  - VoiceEventDetector 가 이 VoiceFrame 을 "이전 프레임과 비교"해 타입드 이벤트를 발행한다.
 *
 * ⚠️ 경계 규칙: VoiceFrame(프레임 원시 데이터)을 소비하는 것은 VoiceEventDetector 까지다.
 *   BubbleManager 는 프레임을 직접 참조하지 않고 이벤트만 구독한다.
 */

/** 최종적으로 사용하는 7개의 모음 (중성 분해 기준). */
export type Vowel = '아' | '어' | '오' | '우' | '으' | '이' | '에';

/** 아직 인식 결과가 도착하지 않았거나(비동기 지연) 분류 실패 상태. */
export type MaybeVowel = Vowel | null;

export const DEFAULT_VOWEL: Vowel = '아';

/** 7개 모음 순서(참조/디버그용). */
export const VOWELS: readonly Vowel[] = ['아', '어', '오', '우', '으', '이', '에'];

/**
 * 매 분석 주기(≈25ms)마다 파형에서 뽑아 합친 한 프레임.
 * VoiceEventDetector 의 입력이자, 프레임 간 비교의 단위.
 */
export interface VoiceFrame {
  /** 이 프레임 시각 (performance.now(), ms). */
  time: number;
  /** 기본 주파수 F0 (Hz). 검출 실패 시 null. */
  pitch: number | null;
  /** Pitchy 명료도(clarity) 0~1. 유성음일수록 1 에 가깝다. */
  clarity: number;
  /** RMS 에너지 0~1 로 정규화. */
  energy: number;
  /**
   * 이 프레임의 모음.
   *  - null 이면 "아직 인식 불가"(무성음/분류 실패/인식 지연).
   *  - 값이 있으면 인식 성공.
   */
  vowel: MaybeVowel;
  /**
   * 이 프레임에서 모음(음성) 인식이 가능한 상태인가.
   *  - 포먼트 기반 소스에서는 "분류 성공 && 에너지 충분" 을 의미한다.
   *  - Web Speech 소스로 교체 시 "최근 인식 텍스트 존재" 로 매핑된다(§speech).
   */
  recognized: boolean;
}
