import { useEffect, useRef } from 'react';
import type { RefObject } from 'react';

/**
 * PerfMeter (성능 계측, 지시 5)
 * ===========================
 * 튜닝 패널 상단에 FPS·프레임 작업시간(ms)과 최근 프레임시간 스파크라인을 표시한다.
 * 훅이 노출한 perf ref(매 프레임 갱신)를 자체 requestAnimationFrame 으로 읽어 React 리렌더 없이 갱신.
 * 목표선: 16ms(빨간 점선). 그 아래로 유지되면 60fps.
 */
export interface Perf {
  fps: number;
  frameMs: number;
  hist: number[];
}

const MS_SCALE = 33; // 그래프 상단 = 33ms(≈30fps)

export function PerfMeter({ perf }: { perf: RefObject<Perf> }) {
  const fpsRef = useRef<HTMLSpanElement | null>(null);
  const msRef = useRef<HTMLSpanElement | null>(null);
  const cvRef = useRef<HTMLCanvasElement | null>(null);

  useEffect(() => {
    let raf = 0;
    const draw = () => {
      const p = perf.current ?? { fps: 0, frameMs: 0, hist: [] };
      if (fpsRef.current) {
        fpsRef.current.textContent = p.fps.toFixed(0);
        fpsRef.current.style.color = p.fps >= 58 ? '#38e0d0' : p.fps >= 45 ? '#f5c842' : '#ff6b6b';
      }
      if (msRef.current) msRef.current.textContent = p.frameMs.toFixed(1);
      const cv = cvRef.current;
      if (cv) {
        const ctx = cv.getContext('2d');
        if (ctx) {
          const w = cv.width;
          const h = cv.height;
          ctx.clearRect(0, 0, w, h);
          // 16ms 목표선
          const y16 = h - (16 / MS_SCALE) * h;
          ctx.strokeStyle = 'rgba(255,107,107,0.5)';
          ctx.lineWidth = 1;
          ctx.setLineDash([3, 3]);
          ctx.beginPath();
          ctx.moveTo(0, y16);
          ctx.lineTo(w, y16);
          ctx.stroke();
          ctx.setLineDash([]);
          // 프레임시간 선
          const hist = p.hist;
          ctx.strokeStyle = '#38e0d0';
          ctx.lineWidth = 1.4;
          ctx.beginPath();
          for (let i = 0; i < hist.length; i++) {
            const x = (i / 120) * w;
            const ms = Math.min(MS_SCALE, Math.max(0, hist[i]));
            const y = h - (ms / MS_SCALE) * h;
            if (i === 0) ctx.moveTo(x, y);
            else ctx.lineTo(x, y);
          }
          ctx.stroke();
        }
      }
      raf = requestAnimationFrame(draw);
    };
    raf = requestAnimationFrame(draw);
    return () => cancelAnimationFrame(raf);
  }, [perf]);

  return (
    <div className="tp-perf">
      <div className="tp-perf-row">
        <span className="tp-perf-fps">
          <span ref={fpsRef}>0</span> fps
        </span>
        <span className="tp-perf-ms">
          <span ref={msRef}>0</span> ms/frame
        </span>
      </div>
      <canvas ref={cvRef} className="tp-perf-canvas" width={212} height={40} />
    </div>
  );
}
