import type { Vowel } from '../types/audio';
import { DEFAULT_VOWEL } from '../types/audio';

/**
 * 7개 모음의 대표 포먼트(F1, F2) 중심값(Hz). 성인 한국어 단모음 평균(참고용).
 *  - F1: 개구도(입 벌림) — 클수록 '아'처럼 열린 소리
 *  - F2: 전후설 — 클수록 '이/에'처럼 전설(앞)
 *
 * (기존 프로젝트에서 그대로 재사용.)
 */
const REFERENCE: { vowel: Vowel; f1: number; f2: number }[] = [
  { vowel: '아', f1: 800, f2: 1300 },
  { vowel: '어', f1: 650, f2: 1000 },
  { vowel: '오', f1: 450, f2: 850 },
  { vowel: '우', f1: 350, f2: 800 },
  { vowel: '으', f1: 350, f2: 1400 },
  { vowel: '에', f1: 500, f2: 2000 },
  { vowel: '이', f1: 300, f2: 2450 },
];

/**
 * VowelClassifier
 * ---------------
 * (F1, F2) 를 로그 공간 최근접 중심으로 분류한다.
 */
export class VowelClassifier {
  classify(f1: number, f2: number): Vowel {
    if (!(f1 > 0) || !(f2 > 0)) return DEFAULT_VOWEL;
    let best: Vowel = DEFAULT_VOWEL;
    let bestDist = Infinity;
    for (const ref of REFERENCE) {
      const d1 = Math.log(f1 / ref.f1);
      const d2 = Math.log(f2 / ref.f2);
      const dist = d1 * d1 + d2 * d2;
      if (dist < bestDist) {
        bestDist = dist;
        best = ref.vowel;
      }
    }
    return best;
  }
}
