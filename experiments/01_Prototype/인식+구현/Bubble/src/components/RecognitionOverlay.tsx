import type { OverlayData } from '../hooks/useBubbleVisualizer';
import './RecognitionOverlay.css';

/**
 * RecognitionOverlay (B3)
 * =======================
 * 개발 플래그(popSettings.showOverlay)로 켜는 오버레이.
 * 현재 감지된 모음/인식 여부/피치/에너지와 최근 이벤트를 타임스탬프와 함께 표시한다.
 * 목적: 음성이 안 들리는 시연 영상만으로도 "무엇이 입력되었는지" 판별.
 *
 * (현재 모음 소스는 포먼트라 "인식 텍스트" = 감지 모음. Web Speech 배선 시
 *  interim 텍스트를 여기에 함께 표시하면 된다.)
 */
export function RecognitionOverlay({ data }: { data: OverlayData }) {
  return (
    <div className="rec-overlay">
      <div className="rec-line">
        <span className="rec-k">모음</span>
        <span className="rec-v">{data.vowel ?? '—'}</span>
        <span className={`rec-badge ${data.recognized ? 'on' : 'off'}`}>
          {data.recognized ? '인식' : '미인식'}
        </span>
      </div>
      <div className="rec-line">
        <span className="rec-k">Pitch</span>
        <span className="rec-v">{data.pitch ? `${data.pitch.toFixed(0)}Hz` : '—'}</span>
        <span className="rec-k">Energy</span>
        <span className="rec-v">{data.energy.toFixed(2)}</span>
      </div>
      <div className="rec-events">
        {data.events.length === 0 && <div className="rec-ev muted">이벤트 없음</div>}
        {data.events.map((e, i) => (
          <div className="rec-ev" key={`${e.t}-${i}`}>
            <span className="rec-t">{(e.t % 100000).toString().padStart(5, '0')}</span>
            <span>{e.label}</span>
          </div>
        ))}
      </div>
    </div>
  );
}
