import { useRef, useState } from 'react';
import { useBubbleVisualizer } from './hooks/useBubbleVisualizer';
import type { Phase, InputSource } from './hooks/useBubbleVisualizer';
import type { RemoteStatus } from './speech/RemoteVoiceSource';
import type { UnrecognizedMode } from './types/bubble';
import { TuningPanel } from './components/TuningPanel';
import { RecognitionOverlay } from './components/RecognitionOverlay';
import { SignalPanel } from './components/SignalPanel';
import './App.css';

/**
 * 전시(최종) 모드 플래그. false 로 두면 개발용 터짐 튜닝 패널을 숨긴다.
 * (2차 수정 2: "최종 전시 모드에서는 숨길 수 있게 플래그를 둔다.")
 */
const SHOW_TUNING_PANEL = true;

/**
 * App
 * ---
 * 상단 컨트롤 + 상태(HUD) + 캔버스. 로직은 useBubbleVisualizer 훅이 담당한다.
 *
 * 버블 시스템(버블 구현 1차):
 *   발성 시작 → 발화 지점에 버블 생성(FORMING) → 세기·길이·피치·모음으로 실시간 변형
 *   → 발성 종료 → 완성 후 상승(FLOATING, 떨림→좌우 흔들림) → 상단에서 터짐(POPPING 3막).
 */
const PHASE_LABEL: Record<Phase, string> = {
  idle: '대기',
  forming: '● 발성 중',
  floating: '▲ 상승/터짐',
};

const SOURCE_LABEL: Record<InputSource, string> = {
  mic: '마이크',
  remote: '인식기(WS)',
};

const REMOTE_STATUS_LABEL: Record<RemoteStatus, string> = {
  disconnected: '연결 안 됨',
  connecting: '연결 중…',
  connected: '● 연결됨',
  error: '오류',
};

export default function App() {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const {
    isRunning,
    stats,
    overlay,
    inputSource,
    remoteStatus,
    signal,
    perf,
    start,
    stop,
    clear,
    setUnrecognizedMode,
    setInputSource,
  } = useBubbleVisualizer(canvasRef);
  const [mode, setMode] = useState<UnrecognizedMode>('bubble-only');

  const onModeChange = (m: UnrecognizedMode) => {
    setMode(m);
    setUnrecognizedMode(m);
  };

  return (
    <div className="app">
      <header className="toolbar">
        <div className="brand">
          <span className="brand-mark">소리짓 · 비눗방울</span>
          <span className="brand-sub">실시간 음성 → 버블 세그먼트 (구현 1차)</span>
        </div>

        <div className="controls">
          <button className="btn btn-primary" onClick={start} disabled={isRunning}>
            Start
          </button>
          <button className="btn" onClick={stop} disabled={!isRunning}>
            Stop
          </button>
          <button className="btn" onClick={clear}>
            Clear
          </button>

          {/* 입력 소스: 브라우저 마이크 vs 파이썬 인식기(WebSocket) */}
          <label className="mode-toggle">
            입력
            <select
              value={inputSource}
              onChange={(e) => setInputSource(e.target.value as InputSource)}
              disabled={isRunning}
              title={`현재 입력: ${SOURCE_LABEL[inputSource]}`}
            >
              <option value="mic">마이크</option>
              <option value="remote">인식기(WS)</option>
            </select>
          </label>

          {/* 인식 불가(처음부터) 표현 방식 — 두 안을 플래그로 전환(프롬프트 §2-5) */}
          <label className="mode-toggle">
            인식불가
            <select value={mode} onChange={(e) => onModeChange(e.target.value as UnrecognizedMode)}>
              <option value="bubble-only">방울만</option>
              <option value="bumpy">울퉁불퉁</option>
            </select>
          </label>
        </div>

        <div className="hud">
          <Metric label="상태" value={PHASE_LABEL[stats.phase]} />
          <Metric label="Pitch" value={stats.pitch ? `${stats.pitch.toFixed(0)} Hz` : '—'} />
          <Metric label="Energy" value={stats.energy.toFixed(2)} />
          <Metric label="Vowel" value={stats.vowel ?? '—'} />
          <Metric label="Bubbles" value={String(stats.bubbles)} />
          {inputSource === 'remote' && (
            <Metric label="인식기" value={REMOTE_STATUS_LABEL[remoteStatus]} />
          )}
        </div>
      </header>

      <main className="stage">
        {!isRunning && <div className="hint">Start 를 누르고 소리를 내보세요 — 모음(아·어·오·우·으·이·에)이 버블이 됩니다. (기본 입력은 파이썬 인식기(WS)입니다 — 먼저 bubble_serve.py 를 실행하세요. 없으면 입력을 ‘마이크’로 바꿔 브라우저 포먼트 인식을 씁니다.)</div>}
        <canvas ref={canvasRef} className="canvas" />
        {overlay && <RecognitionOverlay data={overlay} />}
        {/* 신호 패널: 항상 마운트하고, 표시는 튜닝 패널의 '신호 패널' 체크박스(popSettings.showSignal)로 토글 */}
        <SignalPanel signal={signal} />
        {SHOW_TUNING_PANEL && <TuningPanel perf={perf} />}
      </main>
    </div>
  );
}

function Metric({ label, value }: { label: string; value: string }) {
  return (
    <div className="metric">
      <span className="metric-label">{label}</span>
      <span className="metric-value">{value}</span>
    </div>
  );
}
