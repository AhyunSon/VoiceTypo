import type { Bubble, BubbleSegment } from '../types/bubble';
import { clamp, easeInOut, lerp } from '../animation/mathUtils';
import {
  OPACITY_FULL_MS,
  POP_ACT_PROPORTIONS,
  SEGMENT_RADIUS_MAX,
  SEGMENT_RADIUS_MIN,
} from '../config/bubbleConfig';
import { popSettings } from '../config/popSettings';

/** 세기(에너지 0~1) → 세그먼트 목표 반지름(px). (크기 = 세기) */
export function energyToRadius(energy: number): number {
  return lerp(SEGMENT_RADIUS_MIN, SEGMENT_RADIUS_MAX, clamp(energy, 0, 1));
}

/** 터짐 3막 실제 지속(초). 비율 × popSettings.popDurationS (A6, 패널 가변). */
export function popDurations(): { tension: number; burst: number; residual: number } {
  const total = Math.max(0.15, popSettings.popDurationS);
  return {
    tension: POP_ACT_PROPORTIONS.tension * total,
    burst: POP_ACT_PROPORTIONS.burst * total,
    residual: POP_ACT_PROPORTIONS.residual * total,
  };
}

/** 발음 지속(ms) → 불투명도 0~1. (길이 = 불투명도, A4: 범위는 패널 가변) */
export function heldToOpacity(heldMs: number): number {
  return lerp(popSettings.opacityMin, popSettings.opacityMax, clamp(heldMs / OPACITY_FULL_MS, 0, 1));
}

/**
 * 정규화 궤적(0~1 배열)을 위치 u(0~1)에서 선형보간 샘플. (3차 수정 3)
 * 빈 배열이면 0.
 */
export function sampleTrajectory(traj: number[], u: number): number {
  const n = traj.length;
  if (n === 0) return 0;
  if (n === 1) return traj[0];
  const x = clamp(u, 0, 1) * (n - 1);
  const i = Math.floor(x);
  if (i >= n - 1) return traj[n - 1];
  const f = x - i;
  return traj[i] * (1 - f) + traj[i + 1] * f;
}

/**
 * 궤적 샘플의 부드러운 버전(피치 부유 수정). 키포인트 사이 보간에 easeInOut 을 적용해
 * 각 키포인트에서 속도가 0에 가까워져, 방향 전환(올라가다 내려가기)이 각지지 않고 둥글게 넘어간다.
 */
export function sampleTrajectorySmooth(traj: number[], u: number): number {
  const n = traj.length;
  if (n === 0) return 0;
  if (n === 1) return traj[0];
  const x = clamp(u, 0, 1) * (n - 1);
  const i = Math.floor(x);
  if (i >= n - 1) return traj[n - 1];
  const f = easeInOut(x - i);
  return traj[i] * (1 - f) + traj[i + 1] * f;
}

/** 삼각파(주기 2): 0→1→0→… . pingpong(감쇠 왕복) 재생용. */
export function triangleWave(x: number): number {
  const m = ((x % 2) + 2) % 2;
  return m <= 1 ? m : 2 - m;
}

/**
 * 종횡비를 가진 세그먼트의 "접합 축 방향 실효 반지름"(px). (4차 수정 2·3)
 * 반지름 r 인 원을 (sx,sy)로 스케일한 타원의, 방향 angle 쪽 경계까지의 거리.
 *   타원 반경(θ) = 1 / sqrt((cosθ/a)² + (sinθ/b)²),  a=r·sx, b=r·sy
 */
export function effectiveRadius(radius: number, aspectValue: number, angle: number): number {
  const { sx, sy } = aspectToScale(aspectValue);
  const a = Math.max(1e-3, radius * sx);
  const b = Math.max(1e-3, radius * sy);
  const c = Math.cos(angle);
  const s = Math.sin(angle);
  return 1 / Math.sqrt((c / a) * (c / a) + (s / b) * (s / b));
}

/**
 * 종횡비 값(-1~+1) → (scaleX, scaleY).
 *  - aspect<0 (낮은 음): 가로로 눌림(scaleX>1, scaleY<1) = 납작.
 *  - aspect>0 (높은 음): 세로로 늘어남(scaleX<1, scaleY>1) = 홀쭉.
 * 면적이 대략 보존되도록 대칭으로 스케일.
 */
export function aspectToScale(aspect: number): { sx: number; sy: number } {
  const a = clamp(aspect, -1, 1);
  const d = a * popSettings.aspectStrength; // A5: 강도 패널 가변
  return { sx: 1 - d, sy: 1 + d };
}

/** 세그먼트 중심의 절대 좌표(버블 pos + 세그먼트 offset). */
export function segmentCenter(bubble: Bubble, seg: BubbleSegment): { x: number; y: number } {
  return { x: bubble.pos.x + seg.offset.x, y: bubble.pos.y + seg.offset.y };
}

/**
 * 버블의 로컬(offset) 경계 상자 — 종횡비 변형 전 기준.
 * pop 트리거(최상단)와 렌더 변형 중심 계산에 쓴다.
 */
export function bubbleLocalBounds(bubble: Bubble): {
  minX: number;
  maxX: number;
  minY: number;
  maxY: number;
  cx: number;
  cy: number;
  maxR: number;
} {
  let minX = Infinity;
  let maxX = -Infinity;
  let minY = Infinity;
  let maxY = -Infinity;
  let maxR = 0;
  for (const s of bubble.segments) {
    const r = s.radius.value;
    minX = Math.min(minX, s.offset.x - r);
    maxX = Math.max(maxX, s.offset.x + r);
    minY = Math.min(minY, s.offset.y - r);
    maxY = Math.max(maxY, s.offset.y + r);
    maxR = Math.max(maxR, r);
  }
  if (!bubble.segments.length) {
    minX = maxX = minY = maxY = 0;
  }
  return {
    minX,
    maxX,
    minY,
    maxY,
    cx: (minX + maxX) / 2,
    cy: (minY + maxY) / 2,
    maxR,
  };
}
