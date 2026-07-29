import {
  FORMANT_F1_MAX_HZ,
  FORMANT_F1_MIN_HZ,
  FORMANT_F2_MAX_HZ,
  FORMANT_F2_MIN_HZ,
  FORMANT_MIN_PROMINENCE,
} from '../config/bubbleConfig';

/** 검출된 포먼트 쌍. */
export interface Formants {
  f1: number;
  f2: number;
}

/**
 * FormantAnalyzer
 * ---------------
 * 주파수 스펙트럼에서 처음 두 포먼트(F1, F2)를 추정한다(FFT 포락선 피크피킹).
 * 대분류(전설/후설/개폐)는 가능하나 화자·성별 정밀도는 제한적 → 확장 시 LPC 로 교체 가능.
 *
 * (기존 프로젝트에서 그대로 재사용.)
 *
 * ※ 이 프로젝트의 모음 소스는 speech/VowelSource 로 추상화되어 있다.
 *   현재는 이 포먼트 기반 분류가 구체 구현이고, Web Speech(중성 분해) 소스로 교체 가능하다.
 */
export class FormantAnalyzer {
  analyze(mag: Float32Array, sampleRate: number, pitchHz: number | null): Formants | null {
    const bins = mag.length;
    if (bins < 8) return null;
    const binHz = sampleRate / (2 * bins);

    const f0 = pitchHz && pitchHz > 0 ? pitchHz : 200;
    let win = Math.round((1.3 * f0) / binHz);
    win = Math.max(3, Math.min(41, win | 1)); // 홀수화 + 캡
    const env = this.smooth(mag, win);

    let maxEnv = 0;
    for (let i = 0; i < env.length; i++) if (env[i] > maxEnv) maxEnv = env[i];
    if (maxEnv <= 0) return null;

    const f1 = this.bandPeak(env, binHz, FORMANT_F1_MIN_HZ, FORMANT_F1_MAX_HZ);
    if (!f1) return null;
    const f2 = this.bandPeak(env, binHz, Math.max(FORMANT_F2_MIN_HZ, f1.freq + 250), FORMANT_F2_MAX_HZ);
    if (!f2) return null;

    if (f1.val < FORMANT_MIN_PROMINENCE * maxEnv) return null;
    if (f2.val < FORMANT_MIN_PROMINENCE * 0.5 * maxEnv) return null;

    return { f1: f1.freq, f2: f2.freq };
  }

  private smooth(mag: Float32Array, window: number): Float32Array {
    const n = mag.length;
    const half = Math.floor(window / 2);
    const out = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      const lo = Math.max(0, i - half);
      const hi = Math.min(n - 1, i + half);
      let sum = 0;
      for (let j = lo; j <= hi; j++) sum += mag[j];
      out[i] = sum / (hi - lo + 1);
    }
    return out;
  }

  private bandPeak(
    env: Float32Array,
    binHz: number,
    loHz: number,
    hiHz: number,
  ): { freq: number; val: number } | null {
    const lo = Math.max(1, Math.floor(loHz / binHz));
    const hi = Math.min(env.length - 1, Math.ceil(hiHz / binHz));
    if (hi <= lo) return null;
    let bestVal = -Infinity;
    let bestIdx = -1;
    for (let i = lo; i <= hi; i++) {
      if (env[i] > bestVal) {
        bestVal = env[i];
        bestIdx = i;
      }
    }
    if (bestIdx < 0) return null;
    return { freq: bestIdx * binHz, val: bestVal };
  }
}
