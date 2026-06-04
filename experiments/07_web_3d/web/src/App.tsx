import { useEffect, Component, ReactNode } from 'react'
import { useStore } from './store'
import { connectWebSocket, disconnectWebSocket } from './websocket'
import { IntroScreen } from './screens/IntroScreen'
import { CalibrationScreen } from './screens/CalibrationScreen'
import { PerformanceScreen } from './screens/PerformanceScreen'
import { ReviewScreen } from './screens/ReviewScreen'

// Error boundary to catch runtime errors
class ErrorBoundary extends Component<
  { children: ReactNode },
  { error: Error | null }
> {
  state = { error: null as Error | null }

  static getDerivedStateFromError(error: Error) {
    return { error }
  }

  render() {
    if (this.state.error) {
      return (
        <div style={{
          padding: 40,
          color: '#ff6b6b',
          fontFamily: 'monospace',
          fontSize: 13,
          background: '#0a0a0a',
          height: '100vh',
          overflow: 'auto',
        }}>
          <h2 style={{ marginBottom: 16 }}>Runtime Error</h2>
          <pre style={{ whiteSpace: 'pre-wrap', color: '#ccc' }}>
            {this.state.error.message}
          </pre>
          <pre style={{ whiteSpace: 'pre-wrap', color: '#666', marginTop: 12 }}>
            {this.state.error.stack}
          </pre>
        </div>
      )
    }
    return this.props.children
  }
}

function DebugOverlay() {
  const connectionStatus = useStore((s) => s.connectionStatus)
  const vad = useStore((s) => s.audio.vad)
  const rms = useStore((s) => s.audio.rms)
  const freq = useStore((s) => s.audio.freq)
  const label = useStore((s) => s.vowelSpace.label)
  const centers = useStore((s) => s.vowelCenters)
  const hasCal = useStore((s) => s.hasCalibration)
  const calibration = useStore((s) => s.calibration)
  const screen = useStore((s) => s.screen)
  const locked = useStore((s) => s.lockedCenters)

  return (
    <div style={{
      position: 'fixed',
      bottom: 4,
      right: 4,
      zIndex: 9999,
      fontSize: 9,
      fontFamily: 'monospace',
      color: '#555',
      background: 'rgba(0,0,0,0.7)',
      padding: '4px 8px',
      borderRadius: 4,
      lineHeight: 1.4,
      pointerEvents: 'none',
    }}>
      ws:{connectionStatus} | {screen} | vad:{vad ? 'ON' : 'off'} | rms:{rms.toFixed(3)} | f:{Math.round(freq)}Hz | v:{label || '-'} | centers:{Object.keys(centers).length}
      <br/>
      calVowel:{calibration?.currentVowel || 'none'} | stable:{calibration?.stabilityProgress?.toFixed(2) || '0'} | confirmed:{calibration?.isConfirmed ? 'Y' : 'N'} | locked:{Object.keys(locked).join(',')||'none'}
    </div>
  )
}

export function App() {
  const screen = useStore((s) => s.screen)
  const connectionStatus = useStore((s) => s.connectionStatus)

  useEffect(() => {
    connectWebSocket()
    return () => disconnectWebSocket()
  }, [])

  return (
    <ErrorBoundary>
      <div style={{
        width: '100vw',
        height: '100vh',
        background: '#0a0a0a',
        position: 'relative',
        overflow: 'hidden',
        fontFamily: "'Inter', 'Noto Sans KR', sans-serif",
      }}>
        {/* Connection indicator */}
        <div style={{
          position: 'fixed',
          top: 16,
          right: 16,
          zIndex: 1000,
          display: 'flex',
          alignItems: 'center',
          gap: 6,
          opacity: 0.5,
          fontSize: 11,
          fontWeight: 300,
          letterSpacing: '0.05em',
          color: '#888',
        }}>
          <div style={{
            width: 6,
            height: 6,
            borderRadius: '50%',
            background: connectionStatus === 'connected' ? '#4ecdc4'
              : connectionStatus === 'connecting' ? '#f7dc6f'
              : '#ff6b6b',
            boxShadow: connectionStatus === 'connected'
              ? '0 0 6px #4ecdc4' : 'none',
          }} />
          {connectionStatus === 'connected' ? 'LIVE' : connectionStatus.toUpperCase()}
        </div>

        {/* Screens */}
        {screen === 'intro' && <IntroScreen />}
        {screen === 'calibration' && <CalibrationScreen />}
        {screen === 'performance' && <PerformanceScreen />}
        {screen === 'review' && <ReviewScreen />}

        {/* Debug overlay */}
        <DebugOverlay />
      </div>
    </ErrorBoundary>
  )
}
