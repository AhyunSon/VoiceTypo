import { ENERGY_REFERENCE } from '../config/bubbleConfig';

/**
 * EnergyAnalyzer
 * --------------
 * 시간영역 샘플의 RMS 에너지를 0~1 로 정규화한다. (세기 → 버블 크기의 원천)
 *
 *   RMS = sqrt(mean(x[i]^2)) / ENERGY_REFERENCE, clamp 0~1
 *
 * (기존 프로젝트에서 그대로 재사용.)
 * TODO(적응형): ENERGY_REFERENCE 고정 대신 배경 RMS 분포로 자동 정규화.
 */
export class EnergyAnalyzer {
  analyze(buffer: Float32Array): number {
    let sumSquares = 0;
    for (let i = 0; i < buffer.length; i++) {
      const s = buffer[i];
      sumSquares += s * s;
    }
    const rms = Math.sqrt(sumSquares / buffer.length);
    const normalized = rms / ENERGY_REFERENCE;
    return Math.min(1, Math.max(0, normalized));
  }
}
