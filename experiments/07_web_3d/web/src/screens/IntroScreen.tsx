import { useStore } from '../store'
import { GlyphRenderer } from '../components/GlyphRenderer'
import { MicIndicator } from '../components/MicIndicator'

export function IntroScreen() {
  const setScreen = useStore((s) => s.setScreen)
  const hasCalibration = useStore((s) => s.hasCalibration)
  const connectionStatus = useStore((s) => s.connectionStatus)

  return (
    <div style={{
      width: '100%',
      height: '100%',
      display: 'flex',
      flexDirection: 'column',
      alignItems: 'center',
      justifyContent: 'center',
      position: 'relative',
    }}>
      {/* Background subtle gradient */}
      <div style={{
        position: 'absolute',
        inset: 0,
        background: 'radial-gradient(ellipse at 50% 40%, #111 0%, #0a0a0a 70%)',
        pointerEvents: 'none',
      }} />

      {/* Floating particles effect (CSS-based) */}
      <div style={{
        position: 'absolute',
        inset: 0,
        overflow: 'hidden',
        pointerEvents: 'none',
      }}>
        {Array.from({ length: 12 }).map((_, i) => (
          <div
            key={i}
            style={{
              position: 'absolute',
              width: 2,
              height: 2,
              borderRadius: '50%',
              background: 'rgba(255,255,255,0.08)',
              left: `${10 + (i * 7.3) % 80}%`,
              top: `${15 + (i * 11.7) % 70}%`,
              animation: `float ${8 + i * 1.3}s ease-in-out infinite`,
              animationDelay: `${i * 0.5}s`,
            }}
          />
        ))}
      </div>

      {/* Main glyph object */}
      <div style={{ position: 'relative', zIndex: 1 }}>
        <GlyphRenderer size={400} />
      </div>

      {/* Title */}
      <div style={{
        position: 'relative',
        zIndex: 1,
        marginTop: -20,
        textAlign: 'center',
      }}>
        <h1 style={{
          fontSize: 14,
          fontWeight: 300,
          letterSpacing: '0.4em',
          color: '#888',
          textTransform: 'uppercase',
          margin: 0,
        }}>
          VoiceTypo
        </h1>
        <p style={{
          fontSize: 11,
          fontWeight: 300,
          color: '#555',
          marginTop: 8,
          letterSpacing: '0.1em',
        }}>
          voice-driven typographic instrument
        </p>
      </div>

      {/* Buttons */}
      <div style={{
        position: 'relative',
        zIndex: 1,
        marginTop: 48,
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        gap: 12,
      }}>
        <button
          onClick={() => setScreen('calibration')}
          disabled={connectionStatus !== 'connected'}
          style={{
            background: 'transparent',
            border: '1px solid rgba(255,255,255,0.15)',
            color: '#ccc',
            padding: '10px 32px',
            borderRadius: 24,
            fontSize: 12,
            fontWeight: 400,
            letterSpacing: '0.1em',
            cursor: connectionStatus === 'connected' ? 'pointer' : 'default',
            opacity: connectionStatus === 'connected' ? 1 : 0.3,
            transition: 'all 0.3s',
            fontFamily: 'inherit',
          }}
          onMouseEnter={(e) => {
            e.currentTarget.style.borderColor = 'rgba(78,205,196,0.4)'
            e.currentTarget.style.color = '#4ecdc4'
          }}
          onMouseLeave={(e) => {
            e.currentTarget.style.borderColor = 'rgba(255,255,255,0.15)'
            e.currentTarget.style.color = '#ccc'
          }}
        >
          Start Calibration
        </button>

        {hasCalibration && (
          <button
            onClick={() => setScreen('performance')}
            disabled={connectionStatus !== 'connected'}
            style={{
              background: 'transparent',
              border: '1px solid rgba(255,255,255,0.08)',
              color: '#777',
              padding: '8px 28px',
              borderRadius: 24,
              fontSize: 11,
              fontWeight: 300,
              letterSpacing: '0.1em',
              cursor: 'pointer',
              transition: 'all 0.3s',
              fontFamily: 'inherit',
            }}
            onMouseEnter={(e) => {
              e.currentTarget.style.color = '#aaa'
            }}
            onMouseLeave={(e) => {
              e.currentTarget.style.color = '#777'
            }}
          >
            Enter Performance
          </button>
        )}
      </div>

      {/* Mic indicator */}
      <div style={{
        position: 'absolute',
        bottom: 24,
        left: 24,
        zIndex: 1,
      }}>
        <MicIndicator />
      </div>

      {/* Float animation keyframes */}
      <style>{`
        @keyframes float {
          0%, 100% { transform: translateY(0px) translateX(0px); opacity: 0.08; }
          25% { transform: translateY(-15px) translateX(5px); opacity: 0.12; }
          50% { transform: translateY(-8px) translateX(-3px); opacity: 0.06; }
          75% { transform: translateY(-20px) translateX(8px); opacity: 0.1; }
        }
      `}</style>
    </div>
  )
}
