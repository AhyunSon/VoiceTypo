// ─── Server → Client Messages ───

export interface AudioState {
  freq: number
  rms: number
  vad: boolean
  vibratoRate: number
  vibratoExtent: number
}

export interface FormantState {
  f1: number | null
  f2: number | null
  f3: number | null
}

export interface VowelSpaceState {
  xyz: [number, number, number] | null
  weights: Record<string, number>
  velocity: [number, number, number]
  label: string
  speed: number
}

export interface CalibrationState {
  currentVowel: string
  progress: [number, number]
  stabilityProgress: number
  isConfirmed: boolean
  isComplete: boolean
  trajectory: number[][]
  justLocked?: boolean
  lockedCenter?: number[]
}

export interface ServerState {
  type: 'state'
  timestamp: number
  audio: AudioState
  formants: FormantState
  vowelSpace: VowelSpaceState
  calibration?: CalibrationState
}

export interface ServerConfig {
  type: 'config'
  vowelCenters: Record<string, [number, number, number]>
  vowelOrder: string[]
  hasCalibration: boolean
}

export type ServerMessage = ServerState | ServerConfig | {
  type: 'calibrationStarted' | 'calibrationComplete'
  [key: string]: unknown
}

// ─── App State ───

export type AppScreen = 'intro' | 'calibration' | 'performance' | 'review'

export type ConnectionStatus = 'connecting' | 'connected' | 'disconnected' | 'error'

// ─── Vowel Colors ───

export const VOWEL_COLORS: Record<string, string> = {
  '아': '#ff6b6b',
  '이': '#4ecdc4',
  '우': '#45b7d1',
  '오': '#f7dc6f',
  '으': '#bb8fce',
  '어': '#82e0aa',
  '에': '#f0b27a',
}

export const VOWEL_LABELS_EN: Record<string, string> = {
  '아': 'a',
  '이': 'i',
  '우': 'u',
  '오': 'o',
  '으': 'eu',
  '어': 'eo',
  '에': 'e',
}
