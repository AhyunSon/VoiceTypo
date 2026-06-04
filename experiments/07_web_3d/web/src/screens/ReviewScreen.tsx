import { useStore } from '../store'

export function ReviewScreen() {
  const setScreen = useStore((s) => s.setScreen)
  const captures = useStore((s) => s.captures)

  const handleDownload = (dataUrl: string, index: number) => {
    const link = document.createElement('a')
    link.download = `voicetypo_capture_${index + 1}.png`
    link.href = dataUrl
    link.click()
  }

  return (
    <div style={{
      width: '100%',
      height: '100%',
      display: 'flex',
      flexDirection: 'column',
      padding: 40,
      overflow: 'auto',
    }}>
      {/* Header */}
      <div style={{
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        marginBottom: 40,
      }}>
        <div>
          <h2 style={{
            fontSize: 14,
            fontWeight: 400,
            color: '#888',
            letterSpacing: '0.15em',
            textTransform: 'uppercase',
            margin: 0,
          }}>
            Captured Moments
          </h2>
          <p style={{
            fontSize: 11,
            fontWeight: 300,
            color: '#555',
            marginTop: 4,
          }}>
            {captures.length} capture{captures.length !== 1 ? 's' : ''}
          </p>
        </div>

        <button
          onClick={() => setScreen('performance')}
          style={{
            background: 'transparent',
            border: '1px solid rgba(255,255,255,0.1)',
            color: '#888',
            padding: '8px 24px',
            borderRadius: 20,
            fontSize: 11,
            fontWeight: 300,
            cursor: 'pointer',
            fontFamily: 'inherit',
            letterSpacing: '0.05em',
          }}
        >
          Back to Performance
        </button>
      </div>

      {/* Captures grid */}
      {captures.length === 0 ? (
        <div style={{
          flex: 1,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          flexDirection: 'column',
          gap: 12,
        }}>
          <div style={{
            fontSize: 13,
            color: '#555',
            fontWeight: 300,
          }}>
            아직 캡처된 순간이 없습니다
          </div>
          <div style={{
            fontSize: 11,
            color: '#444',
            fontWeight: 300,
          }}>
            퍼포먼스 중 Capture 버튼을 눌러 순간을 기록하세요
          </div>
        </div>
      ) : (
        <div style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fill, minmax(240, 1fr))',
          gap: 20,
        }}>
          {captures.map((dataUrl, i) => (
            <div
              key={i}
              style={{
                background: 'rgba(255,255,255,0.02)',
                borderRadius: 12,
                overflow: 'hidden',
                border: '1px solid rgba(255,255,255,0.05)',
                cursor: 'pointer',
                transition: 'border-color 0.2s',
              }}
              onClick={() => handleDownload(dataUrl, i)}
              onMouseEnter={(e) => {
                e.currentTarget.style.borderColor = 'rgba(255,255,255,0.12)'
              }}
              onMouseLeave={(e) => {
                e.currentTarget.style.borderColor = 'rgba(255,255,255,0.05)'
              }}
            >
              <img
                src={dataUrl}
                alt={`Capture ${i + 1}`}
                style={{
                  width: '100%',
                  display: 'block',
                  background: '#0a0a0a',
                }}
              />
              <div style={{
                padding: '8px 12px',
                display: 'flex',
                justifyContent: 'space-between',
                alignItems: 'center',
              }}>
                <span style={{
                  fontSize: 10,
                  color: '#555',
                  fontWeight: 300,
                }}>
                  #{i + 1}
                </span>
                <span style={{
                  fontSize: 9,
                  color: '#444',
                  fontWeight: 300,
                  letterSpacing: '0.05em',
                }}>
                  CLICK TO SAVE
                </span>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
