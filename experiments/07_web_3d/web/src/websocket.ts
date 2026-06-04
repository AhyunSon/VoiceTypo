import { useStore } from './store'
import type { ServerMessage, ServerState, ServerConfig } from './types'

let ws: WebSocket | null = null
let reconnectTimer: number | null = null
const pendingCommands: string[] = []

const WS_URL = `ws://${window.location.hostname || 'localhost'}:8765/ws`

export function connectWebSocket() {
  // 이미 연결 중이거나 연결됨이면 무시
  if (ws && (ws.readyState === WebSocket.OPEN || ws.readyState === WebSocket.CONNECTING)) {
    return
  }

  const store = useStore.getState()
  store.setConnectionStatus('connecting')

  try {
    ws = new WebSocket(WS_URL)
  } catch {
    store.setConnectionStatus('error')
    scheduleReconnect()
    return
  }

  ws.onopen = () => {
    console.log('[WS] connected')
    useStore.getState().setConnectionStatus('connected')
    if (reconnectTimer) {
      clearTimeout(reconnectTimer)
      reconnectTimer = null
    }
    // 대기 중인 명령 전송
    while (pendingCommands.length > 0) {
      const payload = pendingCommands.shift()!
      ws!.send(payload)
      console.log('[WS] sent (queued):', payload)
    }
  }

  ws.onclose = () => {
    console.log('[WS] disconnected')
    useStore.getState().setConnectionStatus('disconnected')
    ws = null
    scheduleReconnect()
  }

  ws.onerror = (e) => {
    console.error('[WS] error:', e)
    useStore.getState().setConnectionStatus('error')
  }

  ws.onmessage = (event) => {
    try {
      const msg: ServerMessage = JSON.parse(event.data)
      handleMessage(msg)
    } catch (e) {
      console.warn('[WS] Parse error:', e)
    }
  }
}

function scheduleReconnect() {
  if (reconnectTimer) return
  reconnectTimer = window.setTimeout(() => {
    reconnectTimer = null
    connectWebSocket()
  }, 2000)
}

function handleMessage(msg: ServerMessage) {
  const store = useStore.getState()

  if (msg.type === 'config') {
    const config = msg as ServerConfig
    store.setConfig(config.vowelCenters, config.vowelOrder, config.hasCalibration)
    return
  }

  if (msg.type === 'state') {
    const state = msg as ServerState
    store.updateState(state.audio, state.formants, state.vowelSpace)

    // Add trail point
    if (state.vowelSpace.xyz && state.audio.vad) {
      store.addTrailPoint(state.vowelSpace.xyz)
    }

    // Calibration state
    if (state.calibration) {
      store.setCalibration(state.calibration)

      // Accumulate trajectory points per vowel
      if (state.calibration.currentVowel && state.calibration.trajectory?.length) {
        store.appendTrajectory(
          state.calibration.currentVowel,
          state.calibration.trajectory
        )
      }

      if (state.calibration.justLocked && state.calibration.lockedCenter) {
        store.lockCenter(
          state.calibration.currentVowel,
          state.calibration.lockedCenter as [number, number, number]
        )
      }
    }
    return
  }

  if (msg.type === 'calibrationComplete') {
    const centers = (msg as Record<string, unknown>).vowelCenters as Record<string, [number, number, number]>
    if (centers) {
      store.setConfig(centers, store.vowelOrder, true)
    }
    store.setCalibration(null)
    store.setScreen('performance')
    return
  }
}

export function sendCommand(command: string, data?: Record<string, unknown>) {
  const payload = JSON.stringify({ command, ...data })
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send(payload)
    console.log('[WS] sent:', payload)
  } else {
    console.warn('[WS] queued (not connected yet):', payload)
    pendingCommands.push(payload)
    // 연결 안 됐으면 재연결 시도
    connectWebSocket()
  }
}

export function disconnectWebSocket() {
  if (reconnectTimer) {
    clearTimeout(reconnectTimer)
    reconnectTimer = null
  }
  if (ws) {
    ws.close()
    ws = null
  }
}

// HMR 시 WebSocket 유지
if (import.meta.hot) {
  import.meta.hot.accept()
  // HMR 후 ws가 null이면 재연결
  if (!ws) {
    connectWebSocket()
  }
}
