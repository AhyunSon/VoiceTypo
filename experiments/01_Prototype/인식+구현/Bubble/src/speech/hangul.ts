import type { Vowel } from '../types/audio';

/**
 * 한글 음절 → 중성(모음) 분해 → 7모음(아어오우으이에) 매핑.
 *
 * 프롬프트 컨텍스트의 speech/ 모듈: "한글 인식 → 중성 분해로 모음 7종 추출".
 * Web Speech 가 돌려준 텍스트에서 마지막 음절의 중성을 뽑아 7모음으로 축약한다.
 *
 * 현재(1차)는 포먼트 기반 VowelSource 가 기본이므로 이 함수는 Web Speech 소스
 * (speech/VowelSource.ts 의 WebSpeechVowelSource, 아직 미배선)를 위한 준비물이다.
 */

/** 표준 순서의 21개 중성(medial). 유니코드 조합 순서와 동일. */
const MEDIALS = [
  'ㅏ', 'ㅐ', 'ㅑ', 'ㅒ', 'ㅓ', 'ㅔ', 'ㅕ', 'ㅖ', 'ㅗ', 'ㅘ', 'ㅙ',
  'ㅚ', 'ㅛ', 'ㅜ', 'ㅝ', 'ㅞ', 'ㅟ', 'ㅠ', 'ㅡ', 'ㅢ', 'ㅣ',
] as const;

/**
 * 21개 중성 → 7모음 축약표.
 * 이중모음/반모음은 지각적으로 가까운 단모음으로 매핑한다(대략치, [TUNE] 가능).
 *  - ㅘ(wa)→아, ㅝ(wo)→어, ㅚ/ㅙ/ㅞ→에, ㅟ/ㅢ→이 등.
 */
const MEDIAL_TO_VOWEL: Record<string, Vowel> = {
  ㅏ: '아', ㅐ: '에', ㅑ: '아', ㅒ: '에', ㅓ: '어', ㅔ: '에', ㅕ: '어',
  ㅖ: '에', ㅗ: '오', ㅘ: '아', ㅙ: '에', ㅚ: '에', ㅛ: '오', ㅜ: '우',
  ㅝ: '어', ㅞ: '에', ㅟ: '이', ㅠ: '우', ㅡ: '으', ㅢ: '이', ㅣ: '이',
};

const HANGUL_BASE = 0xac00; // '가'
const HANGUL_LAST = 0xd7a3; // '힣'
const MEDIAL_COUNT = 21;
const FINAL_COUNT = 28;

/** 한 음절의 중성 문자를 반환. 완성형 음절이 아니면 null. */
export function medialOf(syllable: string): string | null {
  const code = syllable.codePointAt(0);
  if (code == null || code < HANGUL_BASE || code > HANGUL_LAST) return null;
  const sIndex = code - HANGUL_BASE;
  const medialIndex = Math.floor(sIndex / FINAL_COUNT) % MEDIAL_COUNT;
  return MEDIALS[medialIndex] ?? null;
}

/**
 * 인식 텍스트에서 "가장 최근 음절"의 모음을 7모음 중 하나로 반환.
 * 완성형 한글이 없으면 null(→ 인식 불가로 취급).
 */
export function lastVowelOf(text: string): Vowel | null {
  for (let i = text.length - 1; i >= 0; i--) {
    const m = medialOf(text[i]);
    if (m) return MEDIAL_TO_VOWEL[m] ?? null;
  }
  return null;
}
