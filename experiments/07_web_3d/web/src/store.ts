import { create } from 'zustand'
import type {
  AppScreen, ConnectionStatus,
  AudioState, FormantState, VowelSpaceState, CalibrationState,
} from './types'

interface TrailPoint {
  xyz: [number, number, number]
  time: number
}

interface VoiceTypoStore {
  // App navigation
  screen: AppScreen
  setScreen: (s: AppScreen) => void

  // Connection
  connectionStatus: ConnectionStatus
  setConnectionStatus: (s: ConnectionStatus) => void

  // Server config
  vowelCenters: Record<string, [number, number, number]>
  vowelOrder: string[]
  hasCalibration: boolean
  setConfig: (centers: Record<string, [number, number, number]>, order: string[], hasCal: boolean) => void

  // Audio state (updated ~50fps)
  audio: AudioState
  formants: FormantState
  vowelSpace: VowelSpaceState
  updateState: (audio: AudioState, formants: FormantState, vowelSpace: VowelSpaceState) => void

  // Trail (recent positions for 3D visualization)
  trail: TrailPoint[]
  addTrailPoint: (xyz: [number, number, number]) => void

  // Calibration
  calibration: CalibrationState | null
  setCalibration: (c: CalibrationState | null) => void

  // Locked calibration centers (accumulated during session)
  lockedCenters: Record<string, [number, number, number]>
  lockCenter: (vowel: string, center: [number, number, number]) => void
  clearLockedCenters: () => void

  // Per-vowel calibration trajectories (accumulated across vowel switches)
  vowelTrajectories: Record<string, number[][]>
  appendTrajectory: (vowel: string, points: number[][]) => void
  clearTrajectories: () => void

  // Smoothed values for rendering (EMA applied in animation loop)
  smoothedWeights: Record<string, number>
  smoothedXyz: [number, number, number] | null
  smoothedFreq: number
  smoothedRms: number
  updateSmoothed: (dt: number) => void

  // Performance captures
  captures: string[]
  addCapture: (dataUrl: string) => void
}

const TRAIL_MAX = 80
const SMOOTH_FACTOR = 0.12

export const useStore = create<VoiceTypoStore>((set, get) => ({
  screen: 'intro',
  setScreen: (s) => set({ screen: s }),

  connectionStatus: 'connecting',
  setConnectionStatus: (s) => set({ connectionStatus: s }),

  vowelCenters: {},
  vowelOrder: [],
  hasCalibration: false,
  setConfig: (centers, order, hasCal) => set({
    vowelCenters: centers,
    vowelOrder: order,
    hasCalibration: hasCal,
  }),

  audio: { freq: 0, rms: 0, vad: false, vibratoRate: 0, vibratoExtent: 0 },
  formants: { f1: null, f2: null, f3: null },
  vowelSpace: { xyz: null, weights: {}, velocity: [0, 0, 0], label: '', speed: 0 },
  updateState: (audio, formants, vowelSpace) => set({ audio, formants, vowelSpace }),

  trail: [],
  addTrailPoint: (xyz) => set((state) => {
    const now = performance.now()
    const newTrail = [...state.trail, { xyz, time: now }]
    // Keep last N points and remove old ones (> 2 seconds)
    const cutoff = now - 2000
    return { trail: newTrail.filter(p => p.time > cutoff).slice(-TRAIL_MAX) }
  }),

  calibration: null,
  setCalibration: (c) => set({ calibration: c }),

  lockedCenters: {},
  lockCenter: (vowel, center) => set((state) => ({
    lockedCenters: { ...state.lockedCenters, [vowel]: center },
  })),
  clearLockedCenters: () => set({ lockedCenters: {} }),

  vowelTrajectories: {},
  appendTrajectory: (vowel, points) => set((state) => {
    const existing = state.vowelTrajectories[vowel] || []
    return {
      vowelTrajectories: {
        ...state.vowelTrajectories,
        [vowel]: [...existing, ...points],
      },
    }
  }),
  clearTrajectories: () => set({ vowelTrajectories: {} }),

  smoothedWeights: {},
  smoothedXyz: null,
  smoothedFreq: 0,
  smoothedRms: 0,

  updateSmoothed: (_dt) => {
    const state = get()
    const alpha = SMOOTH_FACTOR

    // Smooth weights
    const newWeights: Record<string, number> = {}
    const allKeys = new Set([
      ...Object.keys(state.smoothedWeights),
      ...Object.keys(state.vowelSpace.weights),
    ])
    for (const key of allKeys) {
      const target = state.vowelSpace.weights[key] ?? 0
      const current = state.smoothedWeights[key] ?? 0
      newWeights[key] = current + alpha * (target - current)
    }

    // Smooth xyz
    let newXyz = state.smoothedXyz
    if (state.vowelSpace.xyz) {
      if (newXyz) {
        newXyz = [
          newXyz[0] + alpha * (state.vowelSpace.xyz[0] - newXyz[0]),
          newXyz[1] + alpha * (state.vowelSpace.xyz[1] - newXyz[1]),
          newXyz[2] + alpha * (state.vowelSpace.xyz[2] - newXyz[2]),
        ]
      } else {
        newXyz = [...state.vowelSpace.xyz] as [number, number, number]
      }
    }

    // Smooth scalar values
    const newFreq = state.smoothedFreq + alpha * (state.audio.freq - state.smoothedFreq)
    const newRms = state.smoothedRms + alpha * (state.audio.rms - state.smoothedRms)

    set({
      smoothedWeights: newWeights,
      smoothedXyz: newXyz,
      smoothedFreq: newFreq,
      smoothedRms: newRms,
    })
  },

  captures: [],
  addCapture: (dataUrl) => set((state) => ({
    captures: [...state.captures, dataUrl],
  })),
}))
