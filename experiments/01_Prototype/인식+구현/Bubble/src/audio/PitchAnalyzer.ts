import { PitchDetector } from 'pitchy';
import { PITCH_CLARITY_THRESHOLD, PITCH_MIN_HZ, PITCH_MAX_HZ } from '../config/bubbleConfig';

/**
 * PitchAnalyzer
 * -------------
 * Pitchy(McLeod Pitch Method) 로 기본 주파수 F0 를 검출한다.
 *  - clarity 가 임계값보다 낮거나 목소리 범위를 벗어나면 pitch=null.
 *
 * (기존 프로젝트에서 그대로 재사용.)
 */
export class PitchAnalyzer {
  private detector: PitchDetector<Float32Array>;

  constructor(inputLength: number) {
    this.detector = PitchDetector.forFloat32Array(inputLength);
  }

  analyze(buffer: Float32Array, sampleRate: number): { pitch: number | null; clarity: number } {
    const [pitch, clarity] = this.detector.findPitch(buffer, sampleRate);
    if (clarity < PITCH_CLARITY_THRESHOLD) return { pitch: null, clarity };
    if (pitch < PITCH_MIN_HZ || pitch > PITCH_MAX_HZ) return { pitch: null, clarity };
    return { pitch, clarity };
  }
}
