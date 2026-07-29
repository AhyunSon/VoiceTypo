import { useState } from 'react';
import type { RefObject } from 'react';
import type { EdgeMode, PopMode, PopSettings } from '../config/popSettings';
import { exportPopSettings, popSettings, updatePopSettings } from '../config/popSettings';
import { PerfMeter } from './PerfMeter';
import type { Perf } from './PerfMeter';
import './TuningPanel.css';

/**
 * TuningPanel (2차 수정 2, 5차 확장)
 * =================================
 * 런타임 파라미터를 시연 중 조정하는 개발용 패널. popSettings(가변 싱글턴)를
 * 직접 갱신하면 엔진/렌더러가 다음 프레임부터 참조한다.
 * 최종 전시 모드에서는 App 의 SHOW_TUNING_PANEL 플래그로 숨긴다.
 */
const MODE_LABELS: Record<PopMode, string> = {
  'random-lifetime': '랜덤 수명',
  'fixed-lifetime': '수명 고정',
  edge: '가장자리 도달 시',
  never: '터지지 않음(관찰)',
};
const EDGE_LABELS: Record<EdgeMode, string> = {
  reflect: '반사',
  passthrough: '통과 후 제거',
};

export function TuningPanel({ perf }: { perf?: RefObject<Perf> }) {
  const [s, setS] = useState<PopSettings>({ ...popSettings });
  const [collapsed, setCollapsed] = useState(false);
  const [copied, setCopied] = useState(false);

  const set = (patch: Partial<PopSettings>) => {
    updatePopSettings(patch);
    setS((prev) => ({ ...prev, ...patch }));
  };

  const copyJson = async () => {
    try {
      await navigator.clipboard.writeText(exportPopSettings());
      setCopied(true);
      setTimeout(() => setCopied(false), 1200);
    } catch {
      console.log('[popSettings]', exportPopSettings());
    }
  };

  if (collapsed) {
    return (
      <button className="tuning-fab" onClick={() => setCollapsed(false)} title="튜닝 패널 열기">
        ⚙ 튜닝
      </button>
    );
  }

  return (
    <div className="tuning-panel">
      <div className="tp-head">
        <span className="tp-title">튜닝</span>
        <button className="tp-collapse" onClick={() => setCollapsed(true)} title="접기">
          –
        </button>
      </div>

      {perf && <PerfMeter perf={perf} />}

      <div className="tp-scroll">
        <div className="tp-divider">터짐</div>
        <label className="tp-row">
          <span>모드</span>
          <select value={s.mode} onChange={(e) => set({ mode: e.target.value as PopMode })}>
            {(Object.keys(MODE_LABELS) as PopMode[]).map((m) => (
              <option key={m} value={m}>
                {MODE_LABELS[m]}
              </option>
            ))}
          </select>
        </label>
        {s.mode === 'random-lifetime' && (
          <>
            <Slider label={`수명 최소 ${s.lifetimeMin.toFixed(1)}s`} min={0.5} max={20} step={0.5}
              value={s.lifetimeMin} onChange={(v) => set({ lifetimeMin: Math.min(v, s.lifetimeMax) })} />
            <Slider label={`수명 최대 ${s.lifetimeMax.toFixed(1)}s`} min={0.5} max={20} step={0.5}
              value={s.lifetimeMax} onChange={(v) => set({ lifetimeMax: Math.max(v, s.lifetimeMin) })} />
          </>
        )}
        {s.mode === 'fixed-lifetime' && (
          <Slider label={`고정 수명 ${s.fixedLifetime.toFixed(1)}s`} min={0.5} max={20} step={0.5}
            value={s.fixedLifetime} onChange={(v) => set({ fixedLifetime: v })} />
        )}
        <Slider label={`터짐 비율 ${Math.round(s.popRatio * 100)}%`} min={0} max={1} step={0.05}
          value={s.popRatio} onChange={(v) => set({ popRatio: v })} />
        <Slider label={`터짐 길이 ${s.popDurationS.toFixed(2)}s`} min={0.2} max={1.0} step={0.05}
          value={s.popDurationS} onChange={(v) => set({ popDurationS: v })} />

        <div className="tp-divider">비행</div>
        <Slider label={`표류 최소 ${s.driftMin.toFixed(0)}px/s`} min={0} max={200} step={2}
          value={s.driftMin} onChange={(v) => set({ driftMin: Math.min(v, s.driftMax) })} />
        <Slider label={`표류 최대 ${s.driftMax.toFixed(0)}px/s`} min={0} max={200} step={2}
          value={s.driftMax} onChange={(v) => set({ driftMax: Math.max(v, s.driftMin) })} />
        <Slider label={`피치 상하폭 ${s.pitchYRange.toFixed(0)}px`} min={0} max={300} step={5}
          value={s.pitchYRange} onChange={(v) => set({ pitchYRange: v })} />

        <div className="tp-divider">피치 부유</div>
        <Slider label={`재생 배율 ×${s.trajectoryTimeScale.toFixed(2)} (재생=수명×배율)`} min={0.25} max={3} step={0.05}
          value={s.trajectoryTimeScale} onChange={(v) => set({ trajectoryTimeScale: v })} />
        <Slider label={`수직 속도 상한 ${s.maxVerticalSpeed.toFixed(0)}px/s`} min={5} max={150} step={1}
          value={s.maxVerticalSpeed} onChange={(v) => set({ maxVerticalSpeed: v })} />
        <label className="tp-row">
          <span>재생 종료 후</span>
          <select value={s.pitchIdleMode}
            onChange={(e) => set({ pitchIdleMode: e.target.value as PopSettings['pitchIdleMode'] })}>
            <option value="pingpong">감쇠 왕복</option>
            <option value="bob">사인 부유</option>
          </select>
        </label>
        <Slider label={`왕복 감쇠율 ${s.pitchDecayRate.toFixed(2)}`} min={0.1} max={0.95} step={0.05}
          value={s.pitchDecayRate} onChange={(v) => set({ pitchDecayRate: v })} />
        <Slider label={`부유 진폭 ${s.idleBobAmp.toFixed(0)}px`} min={0} max={40} step={1}
          value={s.idleBobAmp} onChange={(v) => set({ idleBobAmp: v })} />

        <div className="tp-divider">가장자리 / 바닥</div>
        <label className="tp-row">
          <span>정책</span>
          <select value={s.edgeMode} onChange={(e) => set({ edgeMode: e.target.value as EdgeMode })}>
            {(Object.keys(EDGE_LABELS) as EdgeMode[]).map((m) => (
              <option key={m} value={m}>
                {EDGE_LABELS[m]}
              </option>
            ))}
          </select>
        </label>
        <Slider label={`반사 유지율 ${Math.round(s.edgeBounce * 100)}%`} min={0.2} max={1} step={0.05}
          value={s.edgeBounce} onChange={(v) => set({ edgeBounce: v })} />

        <div className="tp-divider">모양 / 불투명</div>
        <Slider label={`종횡비 강도 ${s.aspectStrength.toFixed(2)}`} min={0} max={0.9} step={0.05}
          value={s.aspectStrength} onChange={(v) => set({ aspectStrength: v })} />
        <Slider label={`불투명 최소 ${s.opacityMin.toFixed(2)}`} min={0} max={1} step={0.02}
          value={s.opacityMin} onChange={(v) => set({ opacityMin: Math.min(v, s.opacityMax) })} />
        <Slider label={`불투명 최대 ${s.opacityMax.toFixed(2)}`} min={0} max={1} step={0.02}
          value={s.opacityMax} onChange={(v) => set({ opacityMax: Math.max(v, s.opacityMin) })} />

        <div className="tp-divider">세기 판정</div>
        <Slider label={`변화 임계 ${s.intensityChangeDb.toFixed(1)}dB`} min={2} max={18} step={0.5}
          value={s.intensityChangeDb} onChange={(v) => set({ intensityChangeDb: v })} />
        <Slider label={`유지 시간 ${s.intensityHoldMs.toFixed(0)}ms`} min={0} max={500} step={10}
          value={s.intensityHoldMs} onChange={(v) => set({ intensityHoldMs: v })} />
        <Slider label={`시작 유예 ${s.onsetGraceMs.toFixed(0)}ms`} min={0} max={400} step={10}
          value={s.onsetGraceMs} onChange={(v) => set({ onsetGraceMs: v })} />

        <div className="tp-divider">변화 속도(급변/점진)</div>
        <Slider label={`기울기 창 ${s.slopeWindowMs.toFixed(0)}ms`} min={100} max={600} step={20}
          value={s.slopeWindowMs} onChange={(v) => set({ slopeWindowMs: v })} />
        <Slider label={`점진 단계 ${s.gradualStepDb.toFixed(1)}dB`} min={1} max={8} step={0.5}
          value={s.gradualStepDb} onChange={(v) => set({ gradualStepDb: v })} />
        <Slider label={`단계 최소 간격 ${s.gradualStepMinMs.toFixed(0)}ms`} min={50} max={400} step={10}
          value={s.gradualStepMinMs} onChange={(v) => set({ gradualStepMinMs: v })} />
        <Slider label={`글자 간격 ${s.glyphEveryNLobes.toFixed(0)}로브마다`} min={1} max={5} step={1}
          value={s.glyphEveryNLobes} onChange={(v) => set({ glyphEveryNLobes: Math.round(v) })} />

        <div className="tp-divider">재발성(음절 반복)</div>
        <Slider label={`딥 비율 ${Math.round(s.dipRatio * 100)}%`} min={0.1} max={0.9} step={0.05}
          value={s.dipRatio} onChange={(v) => set({ dipRatio: v })} />
        <Slider label={`딥 최소 ${s.dipMinMs.toFixed(0)}ms`} min={0} max={150} step={5}
          value={s.dipMinMs} onChange={(v) => set({ dipMinMs: Math.min(v, s.dipMaxMs) })} />
        <Slider label={`딥 최대 ${s.dipMaxMs.toFixed(0)}ms`} min={80} max={500} step={10}
          value={s.dipMaxMs} onChange={(v) => set({ dipMaxMs: Math.max(v, s.dipMinMs) })} />
        <Slider label={`피치 공백 허용 ${s.pitchGapMaxMs.toFixed(0)}ms`} min={0} max={400} step={10}
          value={s.pitchGapMaxMs} onChange={(v) => set({ pitchGapMaxMs: v })} />
        <Slider label={`발성 끊김 ${s.utteranceBreakMs.toFixed(0)}ms`} min={200} max={800} step={20}
          value={s.utteranceBreakMs} onChange={(v) => set({ utteranceBreakMs: v })} />

        <div className="tp-divider">하단 쌓임</div>
        <label className="tp-row tp-check">
          <input type="checkbox" checked={s.stackEnabled} onChange={(e) => set({ stackEnabled: e.target.checked })} />
          <span>글자 쌓임 켜기</span>
        </label>
        <Slider label={`유지 시간 ${s.stackClearAfterS.toFixed(1)}s`} min={1} max={20} step={0.5}
          value={s.stackClearAfterS} onChange={(v) => set({ stackClearAfterS: v })} />
        <Slider label={`쌓임 비율 ${Math.round(s.glyphStackRatio * 100)}%`} min={0} max={1} step={0.05}
          value={s.glyphStackRatio} onChange={(v) => set({ glyphStackRatio: v })} />

        <div className="tp-divider">개발</div>
        <label className="tp-row tp-check">
          <input type="checkbox" checked={s.devLog} onChange={(e) => set({ devLog: e.target.checked })} />
          <span>콘솔 로그(버블 생성/터짐)</span>
        </label>
        <label className="tp-row tp-check">
          <input type="checkbox" checked={s.showOverlay} onChange={(e) => set({ showOverlay: e.target.checked })} />
          <span>인식 로그 오버레이</span>
        </label>
        <label className="tp-row tp-check">
          <input type="checkbox" checked={s.showSignal} onChange={(e) => set({ showSignal: e.target.checked })} />
          <span>신호 패널(주파수·피치)</span>
        </label>
      </div>

      <button className="tp-export" onClick={copyJson}>
        {copied ? '복사됨 ✓' : 'JSON 복사'}
      </button>
    </div>
  );
}

function Slider({
  label, min, max, step, value, onChange,
}: {
  label: string; min: number; max: number; step: number; value: number; onChange: (v: number) => void;
}) {
  return (
    <label className="tp-row tp-slider">
      <span>{label}</span>
      <input type="range" min={min} max={max} step={step} value={value}
        onChange={(e) => onChange(parseFloat(e.target.value))} />
    </label>
  );
}
