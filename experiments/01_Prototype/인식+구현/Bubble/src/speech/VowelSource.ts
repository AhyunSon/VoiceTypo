import type { MaybeVowel, Vowel } from '../types/audio';
import { FormantAnalyzer } from '../audio/FormantAnalyzer';
import { VowelClassifier } from '../audio/VowelClassifier';
import { VOWEL_MIN_ENERGY, VOWEL_SMOOTH_WINDOW } from '../config/bubbleConfig';
import { lastVowelOf } from './hangul';

/**
 * 한 프레임의 모음 판독 결과.
 *  - vowel: 분류된 모음 또는 null(인식 불가/지연).
 *  - recognized: 이 프레임에서 음성 인식이 "가능"했는가.
 *    RECOGNITION_LOST 판정의 원천이 된다.
 */
export interface VowelReading {
  vowel: MaybeVowel;
  recognized: boolean;
}

/**
 * VowelSource
 * -----------
 * "모음"이라는 특징의 출처를 추상화한다.
 * 프롬프트 §3의 원칙(모음 = 글자/경계선)을 만족하는 어떤 소스든 끼울 수 있다.
 *
 *  - 지금(1차): FormantVowelSource — 파형에서 포먼트로 프레임 단위 분류(동기, 지연 없음).
 *  - 확장: WebSpeechVowelSource — Web Speech(비동기·지연) → 중성 분해(§speech).
 *
 * BubbleManager 는 이 결과를 직접 보지 않는다. VoiceEventDetector 만 소비한다.
 */
export interface VowelSource {
  /**
   * @param mag        선형 크기 스펙트럼(포먼트용) 또는 null
   * @param sampleRate 샘플레이트(Hz)
   * @param pitch      검출된 F0(Hz) 또는 null
   * @param energy     정규화 에너지 0~1
   */
  read(mag: Float32Array | null, sampleRate: number, pitch: number | null, energy: number): VowelReading;
  reset(): void;
}

/**
 * FormantVowelSource (기본, 1차에서 사용)
 * --------------------------------------
 * 포먼트(F1·F2) → 최근접 모음. 프레임 단위 흔들림은 최빈값(mode) 창으로 완화한다.
 *
 * 인식 불가(recognized=false) 조건:
 *  - 에너지가 무성음 수준(VOWEL_MIN_ENERGY 미만)
 *  - 피치 검출 실패(무성음/잡음)
 *  - 포먼트 피크가 두드러지지 않음(prominence 미달)
 * → 이때 vowel=null 로 돌려, "인식 지연/불가"를 상위(Detector)가 이벤트로 판정한다.
 */
export class FormantVowelSource implements VowelSource {
  private formant = new FormantAnalyzer();
  private classifier = new VowelClassifier();
  /** 최빈값 창 버퍼(모음 안정화). */
  private window: Vowel[] = [];

  read(mag: Float32Array | null, sampleRate: number, pitch: number | null, energy: number): VowelReading {
    // 무성음/저에너지/피치 실패 → 이 프레임은 인식 불가.
    if (energy < VOWEL_MIN_ENERGY || pitch == null || mag == null) {
      return { vowel: null, recognized: false };
    }
    const formants = this.formant.analyze(mag, sampleRate, pitch);
    if (!formants) {
      return { vowel: null, recognized: false };
    }
    const raw = this.classifier.classify(formants.f1, formants.f2);
    const smoothed = this.pushMode(raw);
    return { vowel: smoothed, recognized: true };
  }

  reset(): void {
    this.window = [];
  }

  /** 최근 VOWEL_SMOOTH_WINDOW 프레임의 최빈값을 반환. */
  private pushMode(v: Vowel): Vowel {
    this.window.push(v);
    if (this.window.length > VOWEL_SMOOTH_WINDOW) this.window.shift();
    const counts = new Map<Vowel, number>();
    let best = v;
    let bestN = 0;
    for (const w of this.window) {
      const n = (counts.get(w) ?? 0) + 1;
      counts.set(w, n);
      if (n > bestN) {
        bestN = n;
        best = w;
      }
    }
    return best;
  }
}

/**
 * WebSpeechVowelSource (확장 지점 · 1차 미배선)
 * --------------------------------------------
 * Web Speech API 는 비동기·지연이 있으므로 프레임 read() 시점에는
 * "가장 최근에 도착한 인식 텍스트"의 마지막 모음을 돌려준다(§2-5 인식 지연 대응의 기반).
 *
 * ⚠️ 1차에서는 사용하지 않는다(정확도/지연 특성 검증 후 도입). 인터페이스만 맞춰 둔다.
 *   실제 배선 시:
 *     - start()/stop() 을 useBubbleVisualizer 의 start/stop 에 연결
 *     - interimResults=true 로 부분 인식을 받아 지연을 줄임
 *     - false-start(짧은 발화) 취소는 Detector/Manager 의 MIN_UTTERANCE_MS 로 처리
 */
export class WebSpeechVowelSource implements VowelSource {
  private recognition: any = null; // SpeechRecognition (webkit) — lib.dom 미정의라 any
  private latestVowel: MaybeVowel = null;
  private lastResultAt = 0;
  /** 이 시간(ms) 안에 도착한 결과만 유효로 본다(오래된 인식은 만료). [TUNE] */
  private readonly staleMs = 1200;

  start(): void {
    const Ctor =
      (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition;
    if (!Ctor) return; // 미지원 브라우저
    const rec = new Ctor();
    rec.lang = 'ko-KR';
    rec.continuous = true;
    rec.interimResults = true;
    rec.onresult = (e: any) => {
      const res = e.results[e.results.length - 1];
      const text: string = res[0]?.transcript ?? '';
      const v = lastVowelOf(text);
      if (v) {
        this.latestVowel = v;
        this.lastResultAt = performance.now();
      }
    };
    rec.start();
    this.recognition = rec;
  }

  stop(): void {
    this.recognition?.stop?.();
    this.recognition = null;
  }

  read(): VowelReading {
    const fresh = performance.now() - this.lastResultAt < this.staleMs;
    if (fresh && this.latestVowel) return { vowel: this.latestVowel, recognized: true };
    return { vowel: null, recognized: false };
  }

  reset(): void {
    this.latestVowel = null;
    this.lastResultAt = 0;
  }
}
