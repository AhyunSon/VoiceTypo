import { useStore } from '../store'

export function MicIndicator() {
  const vad = useStore((s) => s.audio.vad)
  const rms = useStore((s) => s.audio.rms)

  const barHeight = Math.min(24, Math.max(4, rms * 300))

  return (
    <div style={{
      display: 'flex',
      alignItems: 'center',
      gap: 8,
      padding: '6px 12px',
      borderRadius: 20,
      background: 'rgba(255,255,255,0.03)',
      border: '1px solid rgba(255,255,255,0.06)',
    }}>
      {/* Mic icon - simple circle */}
      <div style={{
        width: 8,
        height: 8,
        borderRadius: '50%',
        background: vad ? '#4ecdc4' : '#555',
        boxShadow: vad ? '0 0 8px #4ecdc4' : 'none',
        transition: 'all 0.15s',
      }} />

      {/* Volume bars */}
      <div style={{
        display: 'flex',
        alignItems: 'center',
        gap: 2,
        height: 24,
      }}>
        {[0.3, 0.5, 0.7, 0.9, 1.0].map((threshold, i) => (
          <div
            key={i}
            style={{
              width: 2,
              height: vad && rms > threshold * 0.05 ? barHeight * threshold : 3,
              borderRadius: 1,
              background: vad ? '#4ecdc4' : '#333',
              transition: 'height 0.08s ease-out',
              opacity: vad ? 0.8 : 0.3,
            }}
          />
        ))}
      </div>

      <span style={{
        fontSize: 10,
        fontWeight: 300,
        color: '#666',
        letterSpacing: '0.05em',
      }}>
        MIC
      </span>
    </div>
  )
}
