import { PITCH_MAX_HZ, PITCH_MIN_HZ } from '../config/bubbleConfig';
import { clamp } from '../animation/mathUtils';

/** 주파수(Hz) → 반음(semitone). 기준은 목소리 최저음(PITCH_MIN_HZ). */
export function hzToSemitone(hz: number): number {
  return 12 * Math.log2(hz / PITCH_MIN_HZ);
}

/**
 * 피치(Hz) → 피치 레벨 (-1 ~ +1).
 *  - 낮은 음(범위 하단) → -1
 *  - 높은 음(범위 상단) → +1
 *  - 목소리 범위의 기하평균 부근이 0(중앙).
 *
 * 수정 2: 이 레벨은 버블의 "수직 위치"에 매핑된다(높음=위, 낮음=아래).
 * 로그(반음) 스케일로 지각과 맞춘다.
 */
export function pitchToLevel(hz: number): number {
  const lo = hzToSemitone(PITCH_MIN_HZ); // 0
  const hi = hzToSemitone(PITCH_MAX_HZ);
  const t = clamp((hzToSemitone(hz) - lo) / (hi - lo), 0, 1);
  return t * 2 - 1;
}
