import { useEffect, useRef } from 'react';
import type { RefObject } from 'react';
import { popSettings } from '../config/popSettings';
import './SignalPanel.css';

/**
 * SignalPanel (임시 디버그 패널)
 * =============================
 * 화면 옆에 현재 **피치(주파수, Hz)** 와 **에너지** 를 실시간으로 보여준다.
 * 훅이 노출한 signal ref(분석 틱 25ms마다 갱신)를 패널 자체 requestAnimationFrame 으로
 * 읽어, React 리렌더 없이 부드럽게 곡선/숫자를 갱신한다.
 *
 * 임시용 — App 의 SHOW_SIGNAL_PANEL 플래그로 끄거나 파일째 지우면 된다.
 */
export interface Signal {
  pitch: number | null;
  energy: number;
}

const PITCH_MIN = 70; // 그래프 y 하한(Hz) — Python F0_MIN 과 맞춤
const PITCH_MAX = 400; // 그래프 y 상한(Hz) — F0_MAX
const CAP = 200; // 곡선 표본 개수(대략 3~4초)
const GRID_HZ = [100, 200, 300];
const NOTE = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'];

function noteName(hz: number): string {
  const n = Math.round(12 * Math.log2(hz / 440)) + 69;
  return NOTE[((n % 12) + 12) % 12] + (Math.floor(n / 12) - 1);
}

export function SignalPanel({ signal }: { signal: RefObject<Signal> }) {
  const rootRef = useRef<HTMLDivElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const pitchValRef = useRef<HTMLSpanElement | null>(null);
  const noteRef = useRef<HTMLSpanElement | null>(null);
  const enValRef = useRef<HTMLSpanElement | null>(null);
  const enBarRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    const pitchHist: (number | null)[] = [];
    const enHist: number[] = [];
    let raf = 0;

    const yOf = (hz: number, h: number) => {
      const c = Math.max(PITCH_MIN, Math.min(PITCH_MAX, hz));
      return h - ((c - PITCH_MIN) / (PITCH_MAX - PITCH_MIN)) * h;
    };

    const draw = () => {
      // 튜닝 패널의 '신호 패널' 체크박스로 실시간 토글(엔진이 멈춰 있어도 즉시 반영).
      const on = popSettings.showSignal;
      if (rootRef.current) rootRef.current.style.display = on ? '' : 'none';
      if (!on) {
        raf = requestAnimationFrame(draw);
        return;
      }
      const s = signal.current ?? { pitch: null, energy: 0 };
      pitchHist.push(s.pitch);
      if (pitchHist.length > CAP) pitchHist.shift();
      enHist.push(s.energy);
      if (enHist.length > CAP) enHist.shift();

      const cv = canvasRef.current;
      if (cv) {
        const ctx = cv.getContext('2d');
        if (ctx) {
          const w = cv.width;
          const h = cv.height;
          ctx.clearRect(0, 0, w, h);

          // 배경 격자 + Hz 라벨
          ctx.font = '9px ui-monospace, monospace';
          ctx.textBaseline = 'middle';
          for (const hz of GRID_HZ) {
            const y = yOf(hz, h);
            ctx.strokeStyle = 'rgba(255,255,255,0.10)';
            ctx.lineWidth = 1;
            ctx.beginPath();
            ctx.moveTo(0, y);
            ctx.lineTo(w, y);
            ctx.stroke();
            ctx.fillStyle = 'rgba(255,255,255,0.35)';
            ctx.fillText(`${hz}`, 2, y - 5);
          }

          // 에너지: 하단 반투명 영역
          ctx.beginPath();
          ctx.moveTo(0, h);
          for (let i = 0; i < enHist.length; i++) {
            const x = (i / CAP) * w;
            const y = h - Math.max(0, Math.min(1, enHist[i])) * h * 0.9;
            ctx.lineTo(x, y);
          }
          ctx.lineTo((Math.max(1, enHist.length) / CAP) * w, h);
          ctx.closePath();
          ctx.fillStyle = 'rgba(245, 200, 66, 0.16)';
          ctx.fill();

          // 피치 곡선(청록). null 구간은 끊는다.
          ctx.strokeStyle = '#38e0d0';
          ctx.lineWidth = 1.6;
          ctx.beginPath();
          let pen = false;
          for (let i = 0; i < pitchHist.length; i++) {
            const p = pitchHist[i];
            const x = (i / CAP) * w;
            if (p == null || p <= 0) {
              pen = false;
              continue;
            }
            const y = yOf(p, h);
            if (!pen) {
              ctx.moveTo(x, y);
              pen = true;
            } else {
              ctx.lineTo(x, y);
            }
          }
          ctx.stroke();
        }
      }

      // 숫자 판독(React 리렌더 없이 텍스트만 갱신)
      const p = s.pitch;
      if (pitchValRef.current) pitchValRef.current.textContent = p && p > 0 ? `${p.toFixed(0)}` : '—';
      if (noteRef.current) noteRef.current.textContent = p && p > 0 ? noteName(p) : '—';
      if (enValRef.current) enValRef.current.textContent = s.energy.toFixed(2);
      if (enBarRef.current) enBarRef.current.style.width = `${Math.max(0, Math.min(1, s.energy)) * 100}%`;

      raf = requestAnimationFrame(draw);
    };
    raf = requestAnimationFrame(draw);
    return () => cancelAnimationFrame(raf);
  }, [signal]);

  return (
    <div className="signal-panel" ref={rootRef}>
      <div className="sp-head">신호(임시)</div>
      <div className="sp-readout">
        <div className="sp-metric">
          <span className="sp-label">주파수</span>
          <span className="sp-value">
            <span ref={pitchValRef}>—</span>
            <span className="sp-unit">Hz</span>
          </span>
        </div>
        <div className="sp-metric">
          <span className="sp-label">피치</span>
          <span className="sp-value">
            <span ref={noteRef}>—</span>
          </span>
        </div>
      </div>
      <canvas ref={canvasRef} className="sp-canvas" width={224} height={96} />
      <div className="sp-energy">
        <span className="sp-label">에너지</span>
        <div className="sp-bar">
          <div ref={enBarRef} className="sp-bar-fill" />
        </div>
        <span ref={enValRef} className="sp-en-val">0.00</span>
      </div>
    </div>
  );
}
