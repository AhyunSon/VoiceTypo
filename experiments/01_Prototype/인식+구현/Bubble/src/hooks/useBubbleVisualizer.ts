import { useCallback, useEffect, useRef, useState } from 'react';
import type { RefObject } from 'react';
import type { MaybeVowel, VoiceFrame } from '../types/audio';
import type { Dimensions } from '../types/common';
import type { UnrecognizedMode } from '../types/bubble';
import { AudioManager } from '../audio/AudioManager';
import { PitchAnalyzer } from '../audio/PitchAnalyzer';
import { EnergyAnalyzer } from '../audio/EnergyAnalyzer';
import { FormantVowelSource } from '../speech/VowelSource';
import { RemoteVoiceSource } from '../speech/RemoteVoiceSource';
import type { RemoteStatus } from '../speech/RemoteVoiceSource';
import { VoiceEventDetector } from '../detector/VoiceEventDetector';
import { BubbleManager } from '../bubble/BubbleManager';
import { BubbleRenderer } from '../render/BubbleRenderer';
import { ANALYSIS_INTERVAL_MS, REMOTE_WS_URL } from '../config/bubbleConfig';
import { popSettings } from '../config/popSettings';

/** 엔진 국면(HUD 표시용). */
export type Phase = 'idle' | 'forming' | 'floating';

/** 입력 소스: 브라우저 마이크(로컬 포먼트) vs 파이썬 인식기(WebSocket). */
export type InputSource = 'mic' | 'remote';

export interface Stats {
  pitch: number | null;
  energy: number;
  vowel: MaybeVowel;
  bubbles: number;
  phase: Phase;
}

/** 인식 로그 오버레이용 스냅샷(B3). */
export interface OverlayData {
  vowel: MaybeVowel;
  recognized: boolean;
  pitch: number | null;
  energy: number;
  events: { t: number; label: string }[]; // 최근 이벤트(타임스탬프 ms)
}

/**
 * useBubbleVisualizer
 * ===================
 * 데이터 흐름(프롬프트 §3):
 *   마이크 파형(AudioManager)
 *     → 25ms 마다: Pitch/Energy 분석 + VowelSource(포먼트) → VoiceFrame 합성
 *     → VoiceEventDetector.process(frame) → VoiceEvent[]
 *     → BubbleManager.handle(event)  (프레임 직접 참조 없음)
 *   매 프레임(≈60fps):
 *     → BubbleManager.update(dt) → BubbleRenderer.render()
 */
export function useBubbleVisualizer(canvasRef: RefObject<HTMLCanvasElement | null>) {
  const [isRunning, setIsRunning] = useState(false);
  const [stats, setStats] = useState<Stats>({
    pitch: null,
    energy: 0,
    vowel: null,
    bubbles: 0,
    phase: 'idle',
  });
  const [overlay, setOverlay] = useState<OverlayData | null>(null);
  const [inputSource, setInputSourceState] = useState<InputSource>('remote');
  const [remoteStatus, setRemoteStatus] = useState<RemoteStatus>('disconnected');

  // ---- 엔진 구성요소 ----
  const audioRef = useRef<AudioManager | null>(null);
  const pitchRef = useRef<PitchAnalyzer | null>(null);
  const energyRef = useRef<EnergyAnalyzer | null>(null);
  const vowelSrcRef = useRef<FormantVowelSource>(new FormantVowelSource());
  const remoteRef = useRef<RemoteVoiceSource>(new RemoteVoiceSource());
  const sourceRef = useRef<InputSource>('remote'); // 기본 입력 = 파이썬 인식기(WebSocket)
  const detectorRef = useRef<VoiceEventDetector>(new VoiceEventDetector());
  const managerRef = useRef<BubbleManager>(new BubbleManager());
  const rendererRef = useRef<BubbleRenderer | null>(null);

  // ---- 루프 상태 ----
  const rafRef = useRef<number | null>(null);
  const lastTimeRef = useRef<number>(0);
  const analysisAccRef = useRef<number>(0);
  const hudAccRef = useRef<number>(0);
  const dimsRef = useRef<Dimensions>({ width: 0, height: 0 });
  const lastFrameRef = useRef<VoiceFrame | null>(null);
  /** 임시 신호 패널용: 최신 피치(Hz)·에너지. 분석 틱(25ms)마다 갱신, 패널이 rAF 로 읽음. */
  const signalRef = useRef<{ pitch: number | null; energy: number }>({ pitch: null, energy: 0 });
  /** 성능 계측(지시 5): 평활 FPS·프레임 작업시간(ms)·최근 프레임시간 히스토리. */
  const perfRef = useRef<{ fps: number; frameMs: number; hist: number[] }>({ fps: 0, frameMs: 0, hist: [] });
  const eventsRef = useRef<{ t: number; label: string }[]>([]); // 최근 이벤트(B3 오버레이)

  /** 캔버스 크기를 컨테이너에 맞추고 DPR 반영. */
  const resizeCanvas = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const parent = canvas.parentElement;
    if (!parent) return;

    const dpr = window.devicePixelRatio || 1;
    const width = parent.clientWidth;
    const height = parent.clientHeight;

    canvas.width = Math.round(width * dpr);
    canvas.height = Math.round(height * dpr);
    canvas.style.width = `${width}px`;
    canvas.style.height = `${height}px`;

    const ctx = canvas.getContext('2d');
    if (ctx) {
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      rendererRef.current = new BubbleRenderer(ctx);
    }
    dimsRef.current = { width, height };
    managerRef.current.setDimensions(dimsRef.current);
  }, [canvasRef]);

  useEffect(() => {
    resizeCanvas();
    const drawStatic = () => {
      const m = managerRef.current;
      rendererRef.current?.render(m.getBubbles(), m.getStackedGlyphs(), dimsRef.current);
    };
    const onResize = () => {
      resizeCanvas();
      drawStatic();
    };
    window.addEventListener('resize', onResize);
    drawStatic();
    return () => window.removeEventListener('resize', onResize);
  }, [resizeCanvas]);

  // 원격 인식기 연결 상태를 UI 로 전달.
  useEffect(() => {
    const r = remoteRef.current;
    r.onStatus = (st) => setRemoteStatus(st);
    return () => {
      r.onStatus = undefined;
    };
  }, []);

  /** 메인 루프. 분석은 25ms 주기, 업데이트·렌더는 매 프레임. */
  const loop = useCallback((now: number) => {
    const manager = managerRef.current;
    const renderer = rendererRef.current;
    const dims = dimsRef.current;

    const dtMs = now - lastTimeRef.current;
    lastTimeRef.current = now;
    const dt = Math.min(dtMs / 1000, 0.05);

    // ---- (1) 분석 + 이벤트 발행 (25ms) ----
    analysisAccRef.current += dtMs;
    if (analysisAccRef.current >= ANALYSIS_INTERVAL_MS) {
      analysisAccRef.current = 0;
      // 입력 소스에 따라 VoiceFrame 을 만든다.
      //   - remote: 파이썬 인식기가 보낸 최신 페이로드로 합성(마이크 미사용).
      //   - mic:    브라우저 마이크 파형에서 피치/에너지/포먼트 분석.
      let frame: VoiceFrame | null = null;
      if (sourceRef.current === 'remote') {
        frame = remoteRef.current.readFrame(now);
      } else {
        const audio = audioRef.current;
        const timeBuf = audio?.getTimeDomainData();
        if (audio && timeBuf && pitchRef.current && energyRef.current) {
          const { pitch, clarity } = pitchRef.current.analyze(timeBuf, audio.sampleRate);
          const energy = energyRef.current.analyze(timeBuf);
          const mag = audio.getFrequencyData();
          const { vowel, recognized } = vowelSrcRef.current.read(mag, audio.sampleRate, pitch, energy);
          frame = { time: now, pitch, clarity, energy, vowel, recognized };
        }
      }
      if (frame) {
        lastFrameRef.current = frame;
        signalRef.current = { pitch: frame.pitch, energy: frame.energy }; // 임시 신호 패널

        // 프레임 → 이벤트 → 매니저. (매니저는 프레임을 보지 않는다)
        const events = detectorRef.current.process(frame);
        for (const ev of events) manager.handle(ev);

        // B3 오버레이용: 이벤트 로그 누적(최근 12개).
        if (popSettings.showOverlay && events.length) {
          for (const ev of events) {
            const v = 'vowel' in ev && ev.vowel ? `(${ev.vowel})` : '';
            eventsRef.current.push({ t: Math.round(now), label: ev.type + v });
          }
          if (eventsRef.current.length > 12) {
            eventsRef.current = eventsRef.current.slice(-12);
          }
        }
      }
    }

    // ---- (2) 버블 업데이트 + 렌더 ----
    manager.update(dt, dims);
    renderer?.render(manager.getBubbles(), manager.getStackedGlyphs(), dims);

    // ---- (3) HUD (100ms) ----
    hudAccRef.current += dtMs;
    if (hudAccRef.current >= 100) {
      hudAccRef.current = 0;
      const frame = lastFrameRef.current;
      const bubbles = manager.getBubbles();
      let phase: Phase = 'idle';
      if (bubbles.some((b) => b.state === 'FORMING' || b.state === 'MERGING')) phase = 'forming';
      else if (bubbles.some((b) => b.state === 'FLOATING' || b.state === 'POPPING')) phase = 'floating';
      setStats({
        pitch: frame?.pitch ?? null,
        energy: frame?.energy ?? 0,
        vowel: frame?.vowel ?? null,
        bubbles: bubbles.length,
        phase,
      });
      // B3: 오버레이 스냅샷(켜져 있을 때만). 꺼지면 null(React 가 동일값 dedupe).
      setOverlay(
        popSettings.showOverlay
          ? {
              vowel: frame?.vowel ?? null,
              recognized: frame?.recognized ?? false,
              pitch: frame?.pitch ?? null,
              energy: frame?.energy ?? 0,
              events: [...eventsRef.current].reverse(),
            }
          : null,
      );
    }

    // ---- (4) 성능 계측 (지시 5): 프레임 작업시간·평활 FPS ----
    const workMs = performance.now() - now;
    const p = perfRef.current;
    const instFps = dtMs > 0 ? 1000 / dtMs : 0;
    p.fps = p.fps === 0 ? instFps : p.fps * 0.9 + instFps * 0.1;
    p.frameMs = p.frameMs === 0 ? workMs : p.frameMs * 0.8 + workMs * 0.2;
    p.hist.push(workMs);
    if (p.hist.length > 120) p.hist.shift();

    rafRef.current = requestAnimationFrame(loop);
  }, []);

  const start = useCallback(async () => {
    if (isRunning) return;
    try {
      if (sourceRef.current === 'remote') {
        // 파이썬 인식기(WebSocket)에 접속. 마이크는 열지 않는다.
        remoteRef.current.reset();
        remoteRef.current.connect(REMOTE_WS_URL);
      } else {
        const audio = new AudioManager();
        await audio.start();
        audioRef.current = audio;
        pitchRef.current = new PitchAnalyzer(audio.bufferLength);
        energyRef.current = new EnergyAnalyzer();
        vowelSrcRef.current.reset();
      }
      detectorRef.current.reset();

      resizeCanvas();

      lastTimeRef.current = performance.now();
      analysisAccRef.current = 0;
      hudAccRef.current = 0;
      rafRef.current = requestAnimationFrame(loop);
      setIsRunning(true);
    } catch (err) {
      console.error('[useBubbleVisualizer] start 실패:', err);
      if (sourceRef.current === 'remote') {
        alert('인식기(WebSocket)에 연결할 수 없습니다. bubble_serve.py 실행과 주소를 확인하세요.');
        remoteRef.current.disconnect();
      } else {
        alert('마이크 권한이 필요합니다. 브라우저 권한을 허용해 주세요.');
        audioRef.current?.stop();
        audioRef.current = null;
      }
    }
  }, [isRunning, loop, resizeCanvas]);

  const stop = useCallback(() => {
    if (rafRef.current != null) cancelAnimationFrame(rafRef.current);
    rafRef.current = null;
    audioRef.current?.stop();
    audioRef.current = null;
    remoteRef.current.disconnect();
    detectorRef.current.reset();
    setIsRunning(false);
    setStats((s) => ({ ...s, phase: 'idle' }));
    // A7 주의: Stop 은 살아 있는 버블을 제거하지 않는다(일괄 소멸 방지).
    const m = managerRef.current;
    rendererRef.current?.render(m.getBubbles(), m.getStackedGlyphs(), dimsRef.current);
  }, []);

  const clear = useCallback(() => {
    managerRef.current.clear();
    eventsRef.current = [];
    const m = managerRef.current;
    rendererRef.current?.render(m.getBubbles(), m.getStackedGlyphs(), dimsRef.current);
    setStats((s) => ({ ...s, bubbles: 0 }));
  }, []);

  /** 인식 불가 표현 방식 전환(bubble-only ↔ bumpy). */
  const setUnrecognizedMode = useCallback((mode: UnrecognizedMode) => {
    managerRef.current.setUnrecognizedMode(mode);
  }, []);

  /** 입력 소스 전환(마이크 ↔ 인식기). 실행 중이 아닐 때 바꾸고 Start 를 권장. */
  const setInputSource = useCallback((src: InputSource) => {
    sourceRef.current = src;
    setInputSourceState(src);
  }, []);

  useEffect(() => {
    return () => {
      if (rafRef.current != null) cancelAnimationFrame(rafRef.current);
      audioRef.current?.stop();
      remoteRef.current.disconnect();
    };
  }, []);

  return {
    isRunning,
    stats,
    overlay,
    inputSource,
    remoteStatus,
    signal: signalRef,
    perf: perfRef,
    start,
    stop,
    clear,
    setUnrecognizedMode,
    setInputSource,
  };
}
