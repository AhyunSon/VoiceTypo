import type { MaybeVowel, Vowel, VoiceFrame } from '../types/audio';
import { VOWELS } from '../types/audio';
import { clamp } from '../animation/mathUtils';
import { ENERGY_REFERENCE, REMOTE_STALE_MS } from '../config/bubbleConfig';

/**
 * RemoteVoiceSource
 * =================
 * 파이썬 캘리브레이션 인식기(`음성 인식/bubble_serve.py`)가 WebSocket 으로 보내는
 * 프레임별 페이로드({vowel, margin, f0, rms, silence})를 받아, 브라우저 마이크 없이
 * VoiceFrame 을 합성한다. 이 프레임은 기존 파이프라인(VoiceEventDetector →
 * BubbleManager)에 그대로 들어간다.
 *
 * 경계 규칙(프롬프트 §3)은 유지된다: 이 소스는 VoiceFrame 까지만 만들고,
 * BubbleManager 는 여전히 이벤트만 구독한다.
 *
 * 매핑:
 *  - pitch   ← f0 (Hz, 무성음이면 null)
 *  - energy  ← rms / ENERGY_REFERENCE (EnergyAnalyzer 와 동일 정규화, 0~1 clamp)
 *  - vowel   ← 페이로드 vowel ('?'/무음/미지원 값이면 null)
 *  - recognized ← 유효 모음이면 true
 *  - clarity ← 참고값(디텍터는 사용하지 않음). 유성음 1, 아니면 0.
 *
 * 스트림이 끊기면(REMOTE_STALE_MS 초과) 무음 프레임을 돌려, 살아 있던 발성이
 * 정상적으로 완성·비행하도록 한다(멈춘 모음이 고정되지 않게).
 */
export type RemoteStatus = 'disconnected' | 'connecting' | 'connected' | 'error';

interface RemotePayload {
  silence?: boolean;
  vowel?: string;
  raw?: string;
  second?: string;
  margin?: number;
  f0?: number | null;
  rms?: number;
  error?: string;
}

const VOWEL_SET = new Set<string>(VOWELS);

export class RemoteVoiceSource {
  private ws: WebSocket | null = null;
  private url = '';
  private latest: RemotePayload | null = null;
  private lastMsgAt = 0;
  private _status: RemoteStatus = 'disconnected';
  private manualClose = false;
  private reconnectTimer: number | null = null;

  /** 상태 변화 콜백(UI 표시용). */
  onStatus?: (s: RemoteStatus) => void;

  get status(): RemoteStatus {
    return this._status;
  }

  /** WebSocket 접속 시작(자동 재접속 포함). */
  connect(url: string): void {
    this.url = url;
    this.manualClose = false;
    this.open();
  }

  private open(): void {
    this.setStatus('connecting');
    try {
      const ws = new WebSocket(this.url);
      this.ws = ws;
      ws.onopen = () => this.setStatus('connected');
      ws.onmessage = (e) => {
        try {
          const p = JSON.parse(e.data as string) as RemotePayload;
          if (p.error) {
            console.warn('[remote] 인식 오류:', p.error);
            return;
          }
          this.latest = p;
          this.lastMsgAt = performance.now();
        } catch {
          /* 형식 오류 프레임은 무시 */
        }
      };
      ws.onerror = () => this.setStatus('error');
      ws.onclose = () => {
        this.ws = null;
        if (this.manualClose) {
          this.setStatus('disconnected');
          return;
        }
        this.setStatus('connecting');
        this.scheduleReconnect();
      };
    } catch {
      this.setStatus('error');
      this.scheduleReconnect();
    }
  }

  private scheduleReconnect(): void {
    if (this.manualClose || this.reconnectTimer != null) return;
    this.reconnectTimer = window.setTimeout(() => {
      this.reconnectTimer = null;
      if (!this.manualClose) this.open();
    }, 1000);
  }

  /** 접속 종료(자동 재접속 중단). */
  disconnect(): void {
    this.manualClose = true;
    if (this.reconnectTimer != null) {
      clearTimeout(this.reconnectTimer);
      this.reconnectTimer = null;
    }
    this.ws?.close();
    this.ws = null;
    this.latest = null;
    this.setStatus('disconnected');
  }

  /** 최신 페이로드로 VoiceFrame 합성. 스트림이 끊기면(stale) 무음 프레임. */
  readFrame(now: number): VoiceFrame {
    const p = this.latest;
    const stale = now - this.lastMsgAt > REMOTE_STALE_MS;
    if (!p || stale || p.silence) {
      const rms = p && !stale ? p.rms ?? 0 : 0;
      return {
        time: now,
        pitch: null,
        clarity: 0,
        energy: clamp(rms / ENERGY_REFERENCE, 0, 1),
        vowel: null,
        recognized: false,
      };
    }
    const f0 = typeof p.f0 === 'number' && p.f0 > 0 ? p.f0 : null;
    const vw: MaybeVowel = p.vowel && VOWEL_SET.has(p.vowel) ? (p.vowel as Vowel) : null;
    return {
      time: now,
      pitch: f0,
      clarity: f0 != null ? 1 : 0,
      energy: clamp((p.rms ?? 0) / ENERGY_REFERENCE, 0, 1),
      vowel: vw,
      recognized: vw != null,
    };
  }

  reset(): void {
    this.latest = null;
    this.lastMsgAt = 0;
  }

  private setStatus(s: RemoteStatus): void {
    if (this._status === s) return;
    this._status = s;
    this.onStatus?.(s);
  }
}
