/** 선형 보간. */
export function lerp(a: number, b: number, t: number): number {
  return a + (b - a) * t;
}

/** 값 제한. */
export function clamp(v: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, v));
}

/** ease-in-out (cubic). 0~1 → 0~1. */
export function easeInOut(t: number): number {
  return t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2;
}

/** smoothstep. 0~1 → 0~1 (양끝 완만). */
export function smoothstep(t: number): number {
  const x = clamp(t, 0, 1);
  return x * x * (3 - 2 * x);
}

/** [0,1] 구간 정규화(역보간). a==b 면 0. */
export function inverseLerp(a: number, b: number, v: number): number {
  if (a === b) return 0;
  return clamp((v - a) / (b - a), 0, 1);
}

/**
 * 결정적(deterministic) 의사난수. 시드 → 0~1.
 * Math.random() 대신 시드 기반을 쓰면 같은 버블이 프레임마다 흔들리지 않고
 * 재현 가능한 무작위 요소(하이라이트 위치, 울퉁불퉁 윤곽)를 만든다.
 */
export function hashNoise(seed: number): number {
  const x = Math.sin(seed * 127.1 + 311.7) * 43758.5453;
  return x - Math.floor(x);
}
