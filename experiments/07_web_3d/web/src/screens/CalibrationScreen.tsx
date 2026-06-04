import { useEffect, useState } from 'react'
import { useStore } from '../store'
import { sendCommand } from '../websocket'
import { VowelSpace3D } from '../components/VowelSpace3D'
import { MicIndicator } from '../components/MicIndicator'
import { VOWEL_COLORS } from '../types'

const VOWEL_ORDER = ['아', '어', '오', '우', '으', '이', '에']

const VOWEL_PROMPTS: Record<string, string> = {
  '아': '입을 크게 벌리고 편안하게',
  '어': '입을 약간 벌리고 혀를 뒤로',
  '오': '입술을 둥글게 모아서',
  '우': '입술을 좁게 모아서',
  '으': '입을 자연스럽게, 혀를 뒤쪽에',
  '이': '입을 옆으로 벌리고',
  '에': '입을 약간 벌리고 혀를 앞으로',
}

function VowelButton({ vowel, isCurrent, isDone, onClick }: {
  vowel: string
  isCurrent: boolean
  isDone: boolean
  onClick: () => void
}) {
  const [hovered, setHovered] = useState(false)
  const color = VOWEL_COLORS[vowel]
  const active = isCurrent || hovered

  return (
    <div
      onClick={onClick}
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
      style={{
        display: 'flex',
        alignItems: 'center',
        gap: 10,
        padding: '8px 10px',
        borderRadius: 8,
        background: isCurrent
          ? 'rgba(255,255,255,0.06)'
          : hovered && !isDone
            ? 'rgba(255,255,255,0.03)'
            : 'transparent',
        border: isCurrent
          ? `1px solid ${color}40`
          : '1px solid transparent',
        cursor: 'pointer',
        transition: 'all 0.2s',
        opacity: isDone ? 0.5 : 1,
        userSelect: 'none' as const,
      }}
    >
      <div style={{
        width: 8,
        height: 8,
        borderRadius: '50%',
        background: isDone ? color : active ? color : '#333',
        boxShadow: isDone ? `0 0 8px ${color}` : isCurrent ? `0 0 4px ${color}60` : 'none',
        transition: 'all 0.3s',
      }} />
      <span style={{
        fontSize: 20,
        fontFamily: "'Noto Sans KR', sans-serif",
        fontWeight: isCurrent ? 600 : 300,
        color: isDone ? color : isCurrent ? '#eee' : active ? '#aaa' : '#666',
        transition: 'all 0.2s',
      }}>
        {vowel}
      </span>
      <span style={{ marginLeft: 'auto', fontSize: 9, fontWeight: 300, letterSpacing: '0.05em' }}>
        {isDone ? (
          <span style={{ color, opacity: 0.7 }}>DONE</span>
        ) : isCurrent ? (
          <span style={{ color: '#4ecdc4', opacity: 0.6 }}>REC</span>
        ) : null}
      </span>
    </div>
  )
}

export function CalibrationScreen() {
  const setScreen = useStore((s) => s.setScreen)
  const calibration = useStore((s) => s.calibration)
  const lockedCenters = useStore((s) => s.lockedCenters)
  const vad = useStore((s) => s.audio.vad)
  const [showComplete, setShowComplete] = useState(false)

  // Start calibration session — retry until connected
  useEffect(() => {
    useStore.getState().clearLockedCenters()
    useStore.getState().clearTrajectories()

    const trySend = () => {
      const status = useStore.getState().connectionStatus
      if (status === 'connected') {
        sendCommand('startCalibration')
        return true
      }
      return false
    }

    if (!trySend()) {
      const interval = setInterval(() => {
        if (trySend()) clearInterval(interval)
      }, 300)
      return () => clearInterval(interval)
    }
  }, [])

  // Handle all 7 confirmed → complete
  useEffect(() => {
    if (calibration?.isComplete && !showComplete) {
      setShowComplete(true)
      setTimeout(() => {
        sendCommand('finishCalibration')
      }, 2500)
    }
  }, [calibration?.isComplete, showComplete])

  const [selectedVowel, setSelectedVowel] = useState('')
  const currentVowel = selectedVowel || calibration?.currentVowel || ''
  const stabilityProgress = calibration?.stabilityProgress || 0
  const isConfirmed = calibration?.isConfirmed || false
  const doneCount = Object.keys(lockedCenters).length

  const handleSelectVowel = (vowel: string) => {
    if (selectedVowel === vowel) {
      // 토글: 같은 모음 다시 클릭하면 선택 해제
      setSelectedVowel('')
      sendCommand('deselectVowel')
    } else {
      setSelectedVowel(vowel)
      sendCommand('selectVowel', { vowel })
    }
  }

  return (
    <div style={{ width: '100%', height: '100%', display: 'flex', position: 'relative' }}>
      {/* ═══ Main: 3D Vowel Space ═══ */}
      <div style={{ flex: 1, position: 'relative' }}>
        <VowelSpace3D showCalibration />

        {/* Current vowel instruction overlay — 모음 선택 시에만 표시 */}
        {!showComplete && currentVowel && (
          <div style={{
            position: 'absolute',
            top: '8%',
            left: '50%',
            transform: 'translateX(-50%)',
            textAlign: 'center',
            zIndex: 10,
          }}>
            <div style={{
              fontSize: 72,
              fontWeight: 700,
              fontFamily: "'Noto Sans KR', sans-serif",
              color: VOWEL_COLORS[currentVowel] || '#fff',
              textShadow: `0 0 30px ${(VOWEL_COLORS[currentVowel] || '#fff')}40`,
              opacity: isConfirmed ? 0.3 : 1,
              transition: 'opacity 0.5s, transform 0.5s',
              transform: isConfirmed ? 'scale(0.8)' : 'scale(1)',
            }}>
              {currentVowel}
            </div>

            <p style={{ fontSize: 13, fontWeight: 300, color: '#888', marginTop: 8 }}>
              {isConfirmed ? '확인됨' : VOWEL_PROMPTS[currentVowel] || '발성해 주세요'}
            </p>

            {!isConfirmed && (
              <div style={{ marginTop: 16, fontSize: 11, fontWeight: 300, letterSpacing: '0.05em' }}>
                {vad ? (
                  <span style={{ color: '#4ecdc4' }}>듣고 있습니다...</span>
                ) : (
                  <span style={{ color: '#666' }}>발성을 시작해 주세요</span>
                )}
              </div>
            )}

            {isConfirmed && (
              <div style={{
                marginTop: 12, fontSize: 11,
                color: VOWEL_COLORS[currentVowel] || '#4ecdc4',
                fontWeight: 400, letterSpacing: '0.05em',
                animation: 'fadeIn 0.3s ease-out',
              }}>
                LOCKED
              </div>
            )}
          </div>
        )}

        {/* Completion */}
        {showComplete && (
          <div style={{
            position: 'absolute', top: '50%', left: '50%',
            transform: 'translate(-50%, -50%)', textAlign: 'center',
            zIndex: 10, animation: 'fadeIn 0.5s ease-out',
          }}>
            <div style={{
              fontSize: 18, fontWeight: 400, color: '#e0e0e0',
              letterSpacing: '0.1em', fontFamily: "'Noto Sans KR', sans-serif",
            }}>
              당신의 모음 공간이 완성되었습니다
            </div>
            <div style={{ fontSize: 12, fontWeight: 300, color: '#666', marginTop: 12 }}>
              퍼포먼스 모드로 진입합니다...
            </div>
          </div>
        )}
      </div>

      {/* ═══ Right sidebar ═══ */}
      <div style={{
        width: 220, padding: '32px 20px',
        display: 'flex', flexDirection: 'column', gap: 12,
        borderLeft: '1px solid rgba(255,255,255,0.04)',
        background: 'rgba(0,0,0,0.3)',
        position: 'relative',
        zIndex: 20,
      }}>
        <div style={{
          fontSize: 11, fontWeight: 400, color: '#777',
          letterSpacing: '0.15em', textTransform: 'uppercase', marginBottom: 4,
        }}>
          Calibration
        </div>

        <div style={{ fontSize: 10, color: '#555', marginBottom: 8 }}>
          발성할 모음을 선택하세요
        </div>

        {/* Vowel list */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
          {VOWEL_ORDER.map((vowel) => (
            <VowelButton
              key={vowel}
              vowel={vowel}
              isCurrent={vowel === currentVowel}
              isDone={!!lockedCenters[vowel]}
              onClick={() => handleSelectVowel(vowel)}
            />
          ))}
        </div>

        {/* Stability bar */}
        {!showComplete && currentVowel && !isConfirmed && (
          <div style={{ marginTop: 12 }}>
            <div style={{ fontSize: 10, color: '#555', marginBottom: 6, letterSpacing: '0.05em' }}>
              STABILITY
            </div>
            <div style={{
              width: '100%', height: 3, borderRadius: 2,
              background: 'rgba(255,255,255,0.06)', overflow: 'hidden',
            }}>
              <div style={{
                width: `${stabilityProgress * 100}%`,
                height: '100%', borderRadius: 2, background: '#4ecdc4',
                transition: 'width 0.15s ease-out',
                boxShadow: stabilityProgress > 0.5 ? '0 0 8px #4ecdc460' : 'none',
              }} />
            </div>
          </div>
        )}

        <div style={{ fontSize: 10, color: '#555', letterSpacing: '0.05em', marginTop: 8 }}>
          {doneCount} / 7
        </div>

        <div style={{ marginTop: 'auto', display: 'flex', flexDirection: 'column', gap: 8 }}>
          <button
            onClick={() => sendCommand('retryCalibration')}
            style={{
              background: 'transparent', border: '1px solid rgba(255,255,255,0.08)',
              color: '#777', padding: '6px 16px', borderRadius: 16,
              fontSize: 10, fontWeight: 300, cursor: 'pointer',
              fontFamily: 'inherit', letterSpacing: '0.05em',
            }}
          >
            Retry Current
          </button>
          <button
            onClick={() => {
              setSelectedVowel('')
              useStore.getState().clearLockedCenters()
              useStore.getState().clearTrajectories()
              sendCommand('resetCalibration')
            }}
            style={{
              background: 'transparent', border: '1px solid rgba(255,100,100,0.15)',
              color: '#aa5555', padding: '6px 16px', borderRadius: 16,
              fontSize: 10, fontWeight: 300, cursor: 'pointer',
              fontFamily: 'inherit', letterSpacing: '0.05em',
            }}
          >
            Reset All
          </button>
          <button
            onClick={() => setScreen('intro')}
            style={{
              background: 'transparent', border: 'none', color: '#444',
              padding: '6px 16px', fontSize: 10, fontWeight: 300,
              cursor: 'pointer', fontFamily: 'inherit', letterSpacing: '0.05em',
            }}
          >
            Back
          </button>
        </div>
      </div>

      <div style={{ position: 'absolute', bottom: 20, left: 20, zIndex: 10 }}>
        <MicIndicator />
      </div>

      <style>{`
        @keyframes fadeIn {
          from { opacity: 0; transform: translateY(8px); }
          to { opacity: 1; transform: translateY(0); }
        }
      `}</style>
    </div>
  )
}
