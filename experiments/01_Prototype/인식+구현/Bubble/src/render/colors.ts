/**
 * 비눗방울 팔레트.
 * 색상은 고정하고 불투명도(alpha)만 특징(길이·터짐 진행)에 따라 바꾼다.
 * (에너지=크기, 길이=불투명도 원칙 유지 — 색은 시각 자유도로 쓰지 않는다.)
 */

/** 글자 잉크(RGB). */
export const INK = { r: 28, g: 42, b: 58 };

/** 비눗물 막 틴트(rim/fill). 옅은 하늘·청록. */
export const FILM_INNER = '255, 255, 255'; // 하이라이트 코어
export const FILM_MID = '196, 228, 246';
export const FILM_EDGE = '150, 200, 235';
export const FILM_RIM = '255, 255, 255';

/** rgba 문자열 헬퍼. */
export function rgba(rgb: string, alpha: number): string {
  return `rgba(${rgb}, ${alpha.toFixed(3)})`;
}

/** 잉크 rgba. */
export function ink(alpha: number): string {
  return `rgba(${INK.r}, ${INK.g}, ${INK.b}, ${alpha.toFixed(3)})`;
}
