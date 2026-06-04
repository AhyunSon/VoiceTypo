import { useRef, useEffect, useCallback } from 'react'
import { useStore } from '../store'
import { VOWEL_COLORS } from '../types'

/**
 * GlyphRenderer: Canvas-based Korean vowel glyph morphing.
 *
 * Renders a continuously morphing Hangul glyph based on vowel weights.
 * Uses Canvas 2D with SDF-like soft rendering for clean edges.
 *
 * Visual mapping:
 * - Vowel weights → glyph shape (stroke interpolation)
 * - Pitch → color hue
 * - Volume → size / stroke thickness
 * - Vibrato → edge micro-oscillation
 */

// ─── Vowel Stroke Definitions ───
// Each vowel's medial (중성) is defined as a set of strokes.
// Strokes are normalized to [0,1] coordinate space.

interface Stroke {
  type: 'line' | 'arc'
  points: [number, number][]  // Control points
  thickness: number
}

interface VowelGlyph {
  strokes: Stroke[]
}

// Simplified structural definitions for 7 vowels
// These represent the medial (중성) component
const VOWEL_GLYPHS: Record<string, VowelGlyph> = {
  '아': {  // ㅏ — vertical line + right horizontal
    strokes: [
      { type: 'line', points: [[0.55, 0.15], [0.55, 0.85]], thickness: 1.0 },
      { type: 'line', points: [[0.55, 0.45], [0.78, 0.45]], thickness: 1.0 },
    ],
  },
  '어': {  // ㅓ — vertical line + left horizontal
    strokes: [
      { type: 'line', points: [[0.45, 0.15], [0.45, 0.85]], thickness: 1.0 },
      { type: 'line', points: [[0.22, 0.45], [0.45, 0.45]], thickness: 1.0 },
    ],
  },
  '오': {  // ㅗ — horizontal line + up vertical
    strokes: [
      { type: 'line', points: [[0.2, 0.55], [0.8, 0.55]], thickness: 1.0 },
      { type: 'line', points: [[0.5, 0.28], [0.5, 0.55]], thickness: 1.0 },
    ],
  },
  '우': {  // ㅜ — horizontal line + down vertical
    strokes: [
      { type: 'line', points: [[0.2, 0.45], [0.8, 0.45]], thickness: 1.0 },
      { type: 'line', points: [[0.5, 0.45], [0.5, 0.72]], thickness: 1.0 },
    ],
  },
  '으': {  // ㅡ — single horizontal line
    strokes: [
      { type: 'line', points: [[0.18, 0.5], [0.82, 0.5]], thickness: 1.0 },
    ],
  },
  '이': {  // ㅣ — single vertical line
    strokes: [
      { type: 'line', points: [[0.5, 0.15], [0.5, 0.85]], thickness: 1.0 },
    ],
  },
  '에': {  // ㅔ — two verticals + left horizontal
    strokes: [
      { type: 'line', points: [[0.4, 0.15], [0.4, 0.85]], thickness: 1.0 },
      { type: 'line', points: [[0.6, 0.15], [0.6, 0.85]], thickness: 1.0 },
      { type: 'line', points: [[0.2, 0.45], [0.4, 0.45]], thickness: 1.0 },
    ],
  },
}

// ─── Onset (초성) ─── Simple ㅎ-like shape as constant backdrop
const ONSET_STROKES: Stroke[] = [
  // Simplified ㅎ — circle + horizontals
  { type: 'line', points: [[0.25, 0.18], [0.75, 0.18]], thickness: 0.8 },
  { type: 'line', points: [[0.3, 0.32], [0.7, 0.32]], thickness: 0.7 },
]

// ─── Interpolation Helpers ───

function lerp(a: number, b: number, t: number): number {
  return a + (b - a) * t
}

function interpolateStrokes(
  weights: Record<string, number>,
  time: number,
  vibratoExtent: number,
): Stroke[] {
  // Get top vowels by weight
  const sorted = Object.entries(weights)
    .filter(([k]) => k in VOWEL_GLYPHS)
    .sort((a, b) => b[1] - a[1])

  if (sorted.length === 0) {
    return VOWEL_GLYPHS['아'].strokes
  }

  // Use multi-blend: weighted average of all vowels' strokes
  // First, determine max stroke count
  const maxStrokes = Math.max(
    ...sorted.map(([k]) => VOWEL_GLYPHS[k].strokes.length)
  )

  const result: Stroke[] = []

  for (let si = 0; si < maxStrokes; si++) {
    let totalWeight = 0
    let blendedPoints: [number, number][] = [[0, 0], [0, 0]]
    let blendedThickness = 0
    let hasStroke = false

    for (const [vowel, weight] of sorted) {
      const glyph = VOWEL_GLYPHS[vowel]
      if (si < glyph.strokes.length) {
        const stroke = glyph.strokes[si]
        hasStroke = true

        // Initialize blended points to match max point count
        if (blendedPoints.length < stroke.points.length) {
          while (blendedPoints.length < stroke.points.length) {
            blendedPoints.push([0, 0])
          }
        }

        for (let pi = 0; pi < stroke.points.length; pi++) {
          blendedPoints[pi] = [
            blendedPoints[pi][0] + stroke.points[pi][0] * weight,
            blendedPoints[pi][1] + stroke.points[pi][1] * weight,
          ]
        }

        blendedThickness += stroke.thickness * weight
        totalWeight += weight
      }
    }

    if (hasStroke && totalWeight > 0.01) {
      // Normalize
      for (let pi = 0; pi < blendedPoints.length; pi++) {
        blendedPoints[pi][0] /= totalWeight
        blendedPoints[pi][1] /= totalWeight

        // Add vibrato micro-oscillation
        if (vibratoExtent > 0.05) {
          const freq = 6 + si * 2
          blendedPoints[pi][0] += Math.sin(time * freq + pi * 1.5) * vibratoExtent * 0.003
          blendedPoints[pi][1] += Math.cos(time * freq * 0.7 + pi * 2.1) * vibratoExtent * 0.003
        }
      }

      result.push({
        type: 'line',
        points: blendedPoints as [number, number][],
        thickness: blendedThickness / totalWeight,
      })
    }
  }

  return result
}

// ─── Pitch → Color mapping ───

function pitchToColor(freq: number): string {
  if (freq <= 0) return '#c8c8c8'

  // Map 80-500Hz to hue range
  const t = Math.max(0, Math.min(1, (freq - 80) / 420))

  // Cool (blue/cyan) for low, warm (coral/amber) for high
  const r = Math.round(lerp(100, 235, t))
  const g = Math.round(lerp(160, 130, t * t))
  const b = Math.round(lerp(210, 100, t))

  return `rgb(${r},${g},${b})`
}

// ─── Main Component ───

export function GlyphRenderer({ size = 500 }: { size?: number }) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const animRef = useRef<number>(0)
  const timeRef = useRef(0)

  const draw = useCallback(() => {
    const canvas = canvasRef.current
    if (!canvas) return

    const ctx = canvas.getContext('2d')
    if (!ctx) return

    const state = useStore.getState()
    const w = canvas.width
    const h = canvas.height
    const now = performance.now() / 1000

    // Update smoothed values
    state.updateSmoothed(1 / 60)
    timeRef.current = now

    // Clear
    ctx.clearRect(0, 0, w, h)

    const weights = state.smoothedWeights
    const freq = state.smoothedFreq
    const rms = state.smoothedRms
    const vad = state.audio.vad
    const vibratoExtent = state.audio.vibratoExtent

    // Volume → scale
    const baseScale = 0.6
    const volumeScale = vad ? Math.min(baseScale + rms * 4, 1.2) : baseScale * 0.85
    const idleBreath = vad ? 0 : Math.sin(now * 0.8) * 0.02

    const scale = (volumeScale + idleBreath) * Math.min(w, h)

    // Pitch → color
    const strokeColor = pitchToColor(freq)

    // Volume → stroke width
    const baseWidth = Math.max(3, scale * 0.04)
    const strokeWidth = vad ? baseWidth * (1 + rms * 3) : baseWidth * 0.7

    // Center transform
    ctx.save()
    ctx.translate(w / 2, h / 2)
    ctx.scale(1, 1)

    // ─── Draw onset (subtle, constant) ───
    ctx.globalAlpha = vad ? 0.12 : 0.06
    ctx.strokeStyle = '#999'
    ctx.lineWidth = strokeWidth * 0.6
    ctx.lineCap = 'round'
    ctx.lineJoin = 'round'

    for (const stroke of ONSET_STROKES) {
      ctx.beginPath()
      for (let i = 0; i < stroke.points.length; i++) {
        const x = (stroke.points[i][0] - 0.5) * scale
        const y = (stroke.points[i][1] - 0.5) * scale
        if (i === 0) ctx.moveTo(x, y)
        else ctx.lineTo(x, y)
      }
      ctx.stroke()
    }

    // ─── Draw vowel strokes (morphing) ───
    const strokes = interpolateStrokes(weights, now, vibratoExtent)

    // Main stroke
    ctx.globalAlpha = vad ? 0.9 : 0.3
    ctx.strokeStyle = strokeColor
    ctx.lineWidth = strokeWidth
    ctx.lineCap = 'round'
    ctx.lineJoin = 'round'

    // Soft glow layer
    ctx.shadowColor = strokeColor
    ctx.shadowBlur = vad ? 15 + rms * 40 : 5

    for (const stroke of strokes) {
      ctx.beginPath()
      for (let i = 0; i < stroke.points.length; i++) {
        const x = (stroke.points[i][0] - 0.5) * scale
        const y = (stroke.points[i][1] - 0.5) * scale
        if (i === 0) ctx.moveTo(x, y)
        else ctx.lineTo(x, y)
      }
      ctx.stroke()
    }

    // Second pass: inner bright core
    ctx.shadowBlur = 0
    ctx.globalAlpha = vad ? 0.5 : 0.15
    ctx.strokeStyle = '#ffffff'
    ctx.lineWidth = strokeWidth * 0.3

    for (const stroke of strokes) {
      ctx.beginPath()
      for (let i = 0; i < stroke.points.length; i++) {
        const x = (stroke.points[i][0] - 0.5) * scale
        const y = (stroke.points[i][1] - 0.5) * scale
        if (i === 0) ctx.moveTo(x, y)
        else ctx.lineTo(x, y)
      }
      ctx.stroke()
    }

    ctx.restore()

    // ─── Idle breathing pulse ───
    if (!vad) {
      const pulse = Math.sin(now * 1.2) * 0.5 + 0.5
      ctx.globalAlpha = pulse * 0.03
      const gradient = ctx.createRadialGradient(w / 2, h / 2, 0, w / 2, h / 2, scale * 0.5)
      gradient.addColorStop(0, '#ffffff')
      gradient.addColorStop(1, 'transparent')
      ctx.fillStyle = gradient
      ctx.fillRect(0, 0, w, h)
    }

    animRef.current = requestAnimationFrame(draw)
  }, [])

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return

    // HiDPI setup: canvas works in physical pixels,
    // draw() already uses w/h from canvas.width/height
    const dpr = window.devicePixelRatio || 1
    canvas.width = size * dpr
    canvas.height = size * dpr
    canvas.style.width = `${size}px`
    canvas.style.height = `${size}px`

    animRef.current = requestAnimationFrame(draw)
    return () => cancelAnimationFrame(animRef.current)
  }, [size, draw])

  return (
    <canvas
      ref={canvasRef}
      style={{
        width: size,
        height: size,
        display: 'block',
      }}
    />
  )
}
