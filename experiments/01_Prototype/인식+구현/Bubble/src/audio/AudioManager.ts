import { FFT_SIZE } from '../config/bubbleConfig';

/**
 * AudioManager
 * ------------
 * 마이크 권한 요청 → AudioContext / MediaStreamSource / AnalyserNode 구성 →
 * 시간영역(time-domain) 샘플 버퍼를 제공한다.
 *
 * "오디오 원천은 파형 하나"라는 원칙(프롬프트 §3)의 그 파형이 여기서 나온다.
 * PitchAnalyzer / EnergyAnalyzer / FormantAnalyzer 가 이 버퍼를 입력으로 쓴다.
 *
 * (기존 "버블 디지인 확인" 프로젝트에서 그대로 가져와 재사용.)
 */
export class AudioManager {
  private audioContext: AudioContext | null = null;
  private stream: MediaStream | null = null;
  private source: MediaStreamAudioSourceNode | null = null;
  private analyser: AnalyserNode | null = null;

  /** 시간영역 샘플 재사용 버퍼(-1~1). 타입은 초기화식에서 추론시켜 lib.dom 변화에 견고. */
  private timeBuffer = new Float32Array(FFT_SIZE);
  /** 주파수영역 dB 스펙트럼 원본. */
  private freqDbBuffer = new Float32Array(FFT_SIZE / 2);
  /** dB → 선형 크기 변환 결과(포먼트 분석 입력). */
  private magBuffer = new Float32Array(FFT_SIZE / 2);

  async start(): Promise<void> {
    this.stream = await navigator.mediaDevices.getUserMedia({
      audio: {
        echoCancellation: false,
        noiseSuppression: false,
        autoGainControl: false,
      },
      video: false,
    });

    this.audioContext = new AudioContext();
    this.source = this.audioContext.createMediaStreamSource(this.stream);
    this.analyser = this.audioContext.createAnalyser();
    this.analyser.fftSize = FFT_SIZE;
    // 출력(destination)에 연결하지 않아 하울링을 막는다.
    this.source.connect(this.analyser);

    this.timeBuffer = new Float32Array(this.analyser.fftSize);
    this.freqDbBuffer = new Float32Array(this.analyser.frequencyBinCount);
    this.magBuffer = new Float32Array(this.analyser.frequencyBinCount);
  }

  get sampleRate(): number {
    return this.audioContext?.sampleRate ?? 44100;
  }

  get bufferLength(): number {
    return this.analyser?.fftSize ?? this.timeBuffer.length;
  }

  /** 최신 시간영역 샘플. 같은 버퍼 재사용(복사가 필요하면 호출측에서). */
  getTimeDomainData(): Float32Array | null {
    if (!this.analyser) return null;
    this.analyser.getFloatTimeDomainData(this.timeBuffer);
    return this.timeBuffer;
  }

  /** 최신 주파수 "선형 크기" 스펙트럼(포먼트 분석 입력). dB → 10^(dB/20). */
  getFrequencyData(): Float32Array | null {
    if (!this.analyser) return null;
    this.analyser.getFloatFrequencyData(this.freqDbBuffer);
    for (let i = 0; i < this.freqDbBuffer.length; i++) {
      this.magBuffer[i] = Math.pow(10, this.freqDbBuffer[i] / 20);
    }
    return this.magBuffer;
  }

  stop(): void {
    this.source?.disconnect();
    this.analyser?.disconnect();
    this.stream?.getTracks().forEach((t) => t.stop());
    this.audioContext?.close();
    this.source = null;
    this.analyser = null;
    this.stream = null;
    this.audioContext = null;
  }
}
