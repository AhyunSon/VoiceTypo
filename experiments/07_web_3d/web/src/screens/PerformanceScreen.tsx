import { useCallback, useRef } from 'react'
import { useStore } from '../store'
import { GlyphRenderer } from '../components/GlyphRenderer'
import { VowelSpace3D } from '../components/VowelSpace3D'
import { MicIndicator } from '../components/MicIndicator'
import { VOWEL_COLORS } from '../types'

export function PerformanceScreen() {
  const setScreen = useStore((s) => s.setScreen)
  const label = useStore((s) => s.vowelSpace.label)
  const vad = useStore((s) => s.audio.vad)
  const freq = useStore((s) => s.audio.freq)
  const rms = useStore((s) => s.audio.rms)
  const speed = useStore((s) => s.vowelSpace.speed)
  const canvasContainerRef = useRef<HTMLDivElement>(null)

  // Capture current frame
  const handleCapture = useCallback(() => {
    const container = canvasContainerRef.current
    if (!container) return

    const canvas = container.querySelector('canvas')
    if (!canvas) return

    try {
      const dataUrl = canvas.toDataURL('image/png')
      useStore.getState().addCapture(dataUrl)
    } catch {
      // Canvas may be tainted
    }
  }, [])

  return (
    <div style={{
      width: '100%',
      height: '100%',
      position: 'relative',
      overflow: 'hidden',
    }}>
      {/* Background depth gradient */}
      <div style={{
        position: 'absolute',
        inset: 0,
        background: 'radial-gradient(ellipse at 50% 45%, #0f0f0f 0%, #080808 60%, #050505 100%)',
        pointerEvents: 'none',
      }} />

      {/* ═══ Main Layer: Central Glyph ═══ */}
      <div
        ref={canvasContainerRef}
        style={{
          position: 'absolute',
          top: '50%',
          left: '50%',
          transform: 'translate(-50%, -50%)',
          zIndex: 1,
        }}
      >
        <GlyphRenderer size={Math.min(window.innerWidth * 0.6, window.innerHeight * 0.7, 600)} />
      </div>

      {/* Current vowel label — subtle, top center */}
      <div style={{
        position: 'absolute',
        top: 40,
        left: '50%',
        transform: 'translateX(-50%)',
        zIndex: 10,
        textAlign: 'center',
      }}>
        <div style={{
          fontSize: 28,
          fontFamily: "'Noto Sans KR', sans-serif",
          fontWeight: 500,
          color: VOWEL_COLORS[label] || '#555',
          opacity: vad ? 0.6 : 0.15,
          transition: 'opacity 0.3s, color 0.3s',
          textShadow: vad ? `0 0 20px ${VOWEL_COLORS[label]}30` : 'none',
        }}>
          {label || '\u00A0'}
        </div>
      </div>

      {/* ═══ Supporting Layer: Mini 3D Vowel Space ═══ */}
      <div style={{
        position: 'absolute',
        bottom: 20,
        right: 20,
        width: 240,
        height: 200,
        zIndex: 10,
        borderRadius: 12,
        overflow: 'hidden',
        border: '1px solid rgba(255,255,255,0.04)',
        background: 'rgba(0,0,0,0.4)',
        backdropFilter: 'blur(10px)',
      }}>
        <VowelSpace3D mini />
      </div>

      {/* ═══ Bottom-left: Audio stats (very subtle) ═══ */}
      <div style={{
        position: 'absolute',
        bottom: 20,
        left: 20,
        zIndex: 10,
        display: 'flex',
        flexDirection: 'column',
        gap: 8,
      }}>
        <MicIndicator />

        {/* Minimal stats */}
        <div style={{
          display: 'flex',
          gap: 16,
          padding: '4px 12px',
          fontSize: 9,
          fontWeight: 300,
          color: '#444',
          letterSpacing: '0.06em',
          fontFamily: "'Inter', monospace",
        }}>
          <span>
            {freq > 0 ? `${Math.round(freq)}Hz` : '---'}
          </span>
          <span>
            {(rms * 100).toFixed(1)}%
          </span>
          <span>
            v:{speed.toFixed(3)}
          </span>
        </div>
      </div>

      {/* ═══ Top-right: Controls ═══ */}
      <div style={{
        position: 'absolute',
        top: 16,
        left: 16,
        zIndex: 10,
        display: 'flex',
        gap: 8,
      }}>
        <button
          onClick={() => setScreen('calibration')}
          style={{
            background: 'transparent',
            border: '1px solid rgba(255,255,255,0.06)',
            color: '#555',
            padding: '5px 14px',
            borderRadius: 16,
            fontSize: 10,
            fontWeight: 300,
            cursor: 'pointer',
            fontFamily: 'inherit',
            letterSpacing: '0.05em',
            transition: 'all 0.2s',
          }}
          onMouseEnter={(e) => {
            e.currentTarget.style.color = '#888'
            e.currentTarget.style.borderColor = 'rgba(255,255,255,0.12)'
          }}
          onMouseLeave={(e) => {
            e.currentTarget.style.color = '#555'
            e.currentTarget.style.borderColor = 'rgba(255,255,255,0.06)'
          }}
        >
          Recalibrate
        </button>

        <button
          onClick={handleCapture}
          style={{
            background: 'transparent',
            border: '1px solid rgba(255,255,255,0.06)',
            color: '#555',
            padding: '5px 14px',
            borderRadius: 16,
            fontSize: 10,
            fontWeight: 300,
            cursor: 'pointer',
            fontFamily: 'inherit',
            letterSpacing: '0.05em',
            transition: 'all 0.2s',
          }}
          onMouseEnter={(e) => {
            e.currentTarget.style.color = '#888'
            e.currentTarget.style.borderColor = 'rgba(255,255,255,0.12)'
          }}
          onMouseLeave={(e) => {
            e.currentTarget.style.color = '#555'
            e.currentTarget.style.borderColor = 'rgba(255,255,255,0.06)'
          }}
        >
          Capture
        </button>

        <button
          onClick={() => setScreen('review')}
          style={{
            background: 'transparent',
            border: '1px solid rgba(255,255,255,0.06)',
            color: '#555',
            padding: '5px 14px',
            borderRadius: 16,
            fontSize: 10,
            fontWeight: 300,
            cursor: 'pointer',
            fontFamily: 'inherit',
            letterSpacing: '0.05em',
            transition: 'all 0.2s',
          }}
          onMouseEnter={(e) => {
            e.currentTarget.style.color = '#888'
            e.currentTarget.style.borderColor = 'rgba(255,255,255,0.12)'
          }}
          onMouseLeave={(e) => {
            e.currentTarget.style.color = '#555'
            e.currentTarget.style.borderColor = 'rgba(255,255,255,0.06)'
          }}
        >
          Review
        </button>
      </div>

      {/* ═══ Vowel weight bars — right side, very subtle ═══ */}
      <div style={{
        position: 'absolute',
        right: 20,
        top: '50%',
        transform: 'translateY(-50%)',
        zIndex: 5,
        display: 'flex',
        flexDirection: 'column',
        gap: 4,
        opacity: vad ? 0.5 : 0.15,
        transition: 'opacity 0.3s',
      }}>
        <VowelWeightBars />
      </div>
    </div>
  )
}

function VowelWeightBars() {
  const weights = useStore((s) => s.vowelSpace.weights)

  const vowels = ['아', '어', '오', '우', '으', '이', '에']

  return (
    <>
      {vowels.map((v) => {
        const w = weights[v] || 0
        return (
          <div key={v} style={{
            display: 'flex',
            alignItems: 'center',
            gap: 6,
          }}>
            <span style={{
              fontSize: 9,
              fontFamily: "'Noto Sans KR', sans-serif",
              color: VOWEL_COLORS[v],
              opacity: w > 0.1 ? 0.8 : 0.3,
              width: 12,
              textAlign: 'right',
            }}>
              {v}
            </span>
            <div style={{
              width: 40,
              height: 2,
              borderRadius: 1,
              background: 'rgba(255,255,255,0.05)',
              overflow: 'hidden',
            }}>
              <div style={{
                width: `${w * 100}%`,
                height: '100%',
                background: VOWEL_COLORS[v],
                borderRadius: 1,
                transition: 'width 0.1s ease-out',
                opacity: 0.7,
              }} />
            </div>
          </div>
        )
      })}
    </>
  )
}
