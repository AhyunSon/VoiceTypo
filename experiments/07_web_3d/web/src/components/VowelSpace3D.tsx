import { useRef, useMemo, useCallback, useState } from 'react'
import { Canvas, useFrame, useThree } from '@react-three/fiber'
import { OrbitControls } from '@react-three/drei'
import * as THREE from 'three'
import { useStore } from '../store'
import { VOWEL_COLORS } from '../types'

const DEFAULT_CAMERA_POS: [number, number, number] = [1.3, 0.9, 1.5]

// 시점 프리셋: X=F2, Y=-F1, Z=F3
const VIEW_PRESETS: Record<string, { pos: [number, number, number], label: string }> = {
  '3D':      { pos: [1.3, 0.9, 1.5],  label: '3D' },
  'F1 × F2': { pos: [0, 0, 3],        label: 'F1 × F2' },
  'F2 × F3': { pos: [0, 3, 0],        label: 'F2 × F3' },
  'F1 × F3': { pos: [3, 0, 0],        label: 'F1 × F3' },
}

// ─── Anchor Point (vowel center) ───

function AnchorPoint({ name, position, isNearest }: {
  name: string
  position: [number, number, number]
  isNearest: boolean
}) {
  const meshRef = useRef<THREE.Mesh>(null)
  const glowRef = useRef<THREE.Mesh>(null)
  const color = VOWEL_COLORS[name] || '#ffffff'
  const scaleRef = useRef(1.0)

  useFrame(() => {
    const targetScale = isNearest ? 1.6 : 1.0
    scaleRef.current += 0.08 * (targetScale - scaleRef.current)
    if (meshRef.current) {
      meshRef.current.scale.setScalar(scaleRef.current)
    }
    if (glowRef.current) {
      glowRef.current.scale.setScalar(scaleRef.current * 3)
      ;(glowRef.current.material as THREE.MeshBasicMaterial).opacity = isNearest ? 0.12 : 0.04
    }
  })

  return (
    <group position={position}>
      <mesh ref={meshRef}>
        <sphereGeometry args={[0.03, 16, 16]} />
        <meshBasicMaterial color={color} transparent opacity={0.85} />
      </mesh>
      <mesh ref={glowRef}>
        <sphereGeometry args={[0.03, 16, 16]} />
        <meshBasicMaterial color={color} transparent opacity={0.04} depthWrite={false} />
      </mesh>
    </group>
  )
}

// ─── Vowel Labels (rendered as sprite text via canvas texture) ───

function VowelLabels() {
  const vowelCenters = useStore((s) => s.vowelCenters)

  const textures = useMemo(() => {
    const result: Record<string, THREE.CanvasTexture> = {}
    for (const [name] of Object.entries(vowelCenters)) {
      const canvas = document.createElement('canvas')
      canvas.width = 64
      canvas.height = 32
      const ctx = canvas.getContext('2d')!
      ctx.fillStyle = VOWEL_COLORS[name] || '#ffffff'
      ctx.font = '20px "Noto Sans KR", sans-serif'
      ctx.textAlign = 'center'
      ctx.textBaseline = 'middle'
      ctx.fillText(name, 32, 16)
      result[name] = new THREE.CanvasTexture(canvas)
    }
    return result
  }, [vowelCenters])

  return (
    <>
      {Object.entries(vowelCenters).map(([name, pos]) => (
        <sprite key={name} position={[pos[0], pos[1] + 0.07, pos[2]]} scale={[0.12, 0.06, 1]}>
          <spriteMaterial map={textures[name]} transparent opacity={0.7} depthWrite={false} />
        </sprite>
      ))}
    </>
  )
}

// ─── Live Point ───

function LivePoint() {
  const meshRef = useRef<THREE.Mesh>(null)
  const glowRef = useRef<THREE.Mesh>(null)
  const posRef = useRef(new THREE.Vector3(0, 0, 0))
  const targetRef = useRef(new THREE.Vector3(0, 0, 0))

  useFrame(() => {
    const { vowelSpace, audio } = useStore.getState()
    const vad = audio.vad

    if (vowelSpace.xyz && vad) {
      targetRef.current.set(vowelSpace.xyz[0], vowelSpace.xyz[1], vowelSpace.xyz[2])
      // 발성 중에만 마커 이동
      posRef.current.lerp(targetRef.current, 0.15)
    }
    // VAD 꺼지면 마지막 위치에 정지

    if (meshRef.current) {
      meshRef.current.position.copy(posRef.current)
      ;(meshRef.current.material as THREE.MeshBasicMaterial).opacity = vad ? 0.95 : 0.3
    }
    if (glowRef.current) {
      glowRef.current.position.copy(posRef.current)
      ;(glowRef.current.material as THREE.MeshBasicMaterial).opacity = vad ? 0.25 : 0.05
    }
  })

  return (
    <>
      <mesh ref={meshRef}>
        <sphereGeometry args={[0.025, 16, 16]} />
        <meshBasicMaterial color="#ffffff" transparent opacity={0.1} />
      </mesh>
      <mesh ref={glowRef}>
        <sphereGeometry args={[0.06, 16, 16]} />
        <meshBasicMaterial color="#ffffff" transparent opacity={0.02} depthWrite={false} />
      </mesh>
    </>
  )
}

// ─── Trail ───

function Trail() {
  const maxPoints = 80

  const { geometry, material, lineObj } = useMemo(() => {
    const geo = new THREE.BufferGeometry()
    const positions = new Float32Array(maxPoints * 3)
    geo.setAttribute('position', new THREE.BufferAttribute(positions, 3))
    geo.setDrawRange(0, 0)
    const mat = new THREE.LineBasicMaterial({
      color: 0xffffff,
      transparent: true,
      opacity: 0.3,
      blending: THREE.AdditiveBlending,
    })
    const line = new THREE.Line(geo, mat)
    line.frustumCulled = false
    return { geometry: geo, material: mat, lineObj: line }
  }, [])

  useFrame(() => {
    const trail = useStore.getState().trail
    const positions = geometry.attributes.position as THREE.BufferAttribute
    const count = Math.min(trail.length, maxPoints)

    for (let i = 0; i < count; i++) {
      const p = trail[trail.length - count + i]
      positions.setXYZ(i, p.xyz[0], p.xyz[1], p.xyz[2])
    }

    positions.needsUpdate = true
    geometry.setDrawRange(0, count)
    material.opacity = count > 2 ? 0.3 : 0
  })

  return <primitive object={lineObj} />
}

// ─── Subtle Axes ───

function SubtleAxes() {
  // X=F2(전설/후설), Y=-F1(고모음/저모음), Z=F3
  const axisLabels = [
    { dir: [1, 0, 0] as [number, number, number], label: 'F2', color: '#666' },
    { dir: [0, 1, 0] as [number, number, number], label: 'F1', color: '#666' },
    { dir: [0, 0, 1] as [number, number, number], label: 'F3', color: '#666' },
  ]

  const lines = useMemo(() => {
    const group = new THREE.Group()
    for (const { dir } of axisLabels) {
      const geo = new THREE.BufferGeometry().setFromPoints([
        new THREE.Vector3(-dir[0] * 1.2, -dir[1] * 1.2, -dir[2] * 1.2),
        new THREE.Vector3(dir[0] * 1.2, dir[1] * 1.2, dir[2] * 1.2),
      ])
      const mat = new THREE.LineBasicMaterial({ color: 0x555555, transparent: true, opacity: 0.3 })
      group.add(new THREE.Line(geo, mat))
    }
    return group
  }, [])

  const labelTextures = useMemo(() => {
    return axisLabels.map(({ label, color }) => {
      const canvas = document.createElement('canvas')
      canvas.width = 64
      canvas.height = 24
      const ctx = canvas.getContext('2d')!
      ctx.fillStyle = color
      ctx.font = '16px monospace'
      ctx.textAlign = 'center'
      ctx.textBaseline = 'middle'
      ctx.fillText(label, 32, 12)
      return new THREE.CanvasTexture(canvas)
    })
  }, [])

  return (
    <>
      <primitive object={lines} />
      {axisLabels.map(({ dir }, i) => (
        <sprite
          key={i}
          position={[dir[0] * 1.3, dir[1] * 1.3, dir[2] * 1.3]}
          scale={[0.12, 0.045, 1]}
        >
          <spriteMaterial map={labelTextures[i]} transparent opacity={0.6} depthWrite={false} />
        </sprite>
      ))}
    </>
  )
}

// ─── Camera Controls (orbit, zoom, pan + reset) ───

function CameraControls({ onControlsRef }: { onControlsRef?: (ref: any) => void }) {
  const controlsRef = useRef<any>(null)

  useFrame(() => {
    if (controlsRef.current && onControlsRef) {
      onControlsRef(controlsRef.current)
    }
  })

  return (
    <OrbitControls
      ref={controlsRef}
      target={[0, 0, 0]}
      enableDamping
      dampingFactor={0.12}
      rotateSpeed={0.8}
      zoomSpeed={0.8}
      panSpeed={0.6}
      minDistance={0.3}
      maxDistance={6}
      mouseButtons={{
        LEFT: THREE.MOUSE.ROTATE,
        MIDDLE: THREE.MOUSE.PAN,
        RIGHT: THREE.MOUSE.PAN,
      }}
    />
  )
}

// ─── All Vowel Trajectories (persistent per-vowel point clouds) ───

function VowelTrajectories() {
  const vowelTrajectories = useStore((s) => s.vowelTrajectories)
  const currentVowel = useStore((s) => s.calibration?.currentVowel)

  return (
    <>
      {Object.entries(vowelTrajectories).map(([vowel, points]) => {
        if (!points.length) return null
        return (
          <VowelPointCloud
            key={vowel}
            vowel={vowel}
            points={points}
            isCurrent={vowel === currentVowel}
          />
        )
      })}
    </>
  )
}

// 3x3 대칭행렬 고유값 분해 (Jacobi rotation, 안정적)
function eigenDecomposition3x3(cov: number[][]) {
  const a = cov.map(r => [...r])
  // 고유벡터 = 단위행렬로 시작
  const v = [[1,0,0],[0,1,0],[0,0,1]]

  // Jacobi rotation sweep
  for (let sweep = 0; sweep < 30; sweep++) {
    // 비대각 원소 중 가장 큰 것 찾기
    let maxVal = 0, p = 0, q = 1
    for (let i = 0; i < 3; i++)
      for (let j = i + 1; j < 3; j++)
        if (Math.abs(a[i][j]) > maxVal) { maxVal = Math.abs(a[i][j]); p = i; q = j }

    if (maxVal < 1e-12) break

    // 회전 각도
    const diff = a[q][q] - a[p][p]
    let t: number
    if (Math.abs(diff) < 1e-12) {
      t = 1
    } else {
      const phi = diff / (2 * a[p][q])
      t = 1 / (Math.abs(phi) + Math.sqrt(phi * phi + 1))
      if (phi < 0) t = -t
    }
    const c = 1 / Math.sqrt(t * t + 1)
    const s = t * c

    // 행렬 회전
    const app = a[p][p], aqq = a[q][q], apq = a[p][q]
    a[p][p] = app - t * apq
    a[q][q] = aqq + t * apq
    a[p][q] = 0; a[q][p] = 0

    for (let r = 0; r < 3; r++) {
      if (r === p || r === q) continue
      const arp = a[r][p], arq = a[r][q]
      a[r][p] = a[p][r] = c * arp - s * arq
      a[r][q] = a[q][r] = s * arp + c * arq
    }

    // 고유벡터 갱신
    for (let r = 0; r < 3; r++) {
      const vrp = v[r][p], vrq = v[r][q]
      v[r][p] = c * vrp - s * vrq
      v[r][q] = s * vrp + c * vrq
    }
  }

  return {
    eigenvalues: [a[0][0], a[1][1], a[2][2]],
    eigenvectors: [
      new THREE.Vector3(v[0][0], v[1][0], v[2][0]),
      new THREE.Vector3(v[0][1], v[1][1], v[2][1]),
      new THREE.Vector3(v[0][2], v[1][2], v[2][2]),
    ],
  }
}

function VowelPointCloud({ vowel, points, isCurrent }: {
  vowel: string
  points: number[][]
  isCurrent: boolean
}) {
  const color = VOWEL_COLORS[vowel] || '#ffffff'
  const meshRef = useRef<THREE.Points>(null)
  const labelRef = useRef<THREE.Sprite>(null)
  const ellipsoidRef = useRef<THREE.Mesh>(null)
  const lastPointCountRef = useRef(0)

  const { geometry, material } = useMemo(() => {
    const geo = new THREE.BufferGeometry()
    const mat = new THREE.PointsMaterial({
      color,
      size: 0.012,
      transparent: true,
      opacity: 0.7,
      blending: THREE.AdditiveBlending,
      depthWrite: false,
      sizeAttenuation: true,
    })
    return { geometry: geo, material: mat }
  }, [color])

  const labelTexture = useMemo(() => {
    const canvas = document.createElement('canvas')
    canvas.width = 64
    canvas.height = 32
    const ctx = canvas.getContext('2d')!
    ctx.fillStyle = color
    ctx.font = 'bold 22px "Noto Sans KR", sans-serif'
    ctx.textAlign = 'center'
    ctx.textBaseline = 'middle'
    ctx.fillText(vowel, 32, 16)
    return new THREE.CanvasTexture(canvas)
  }, [vowel, color])

  useFrame(() => {
    if (!points.length) return

    const n = points.length
    // Update point cloud geometry
    const positions = new Float32Array(n * 3)
    let cx = 0, cy = 0, cz = 0
    for (let i = 0; i < n; i++) {
      const p = points[i]
      if (p?.length >= 3) {
        positions[i * 3] = p[0]
        positions[i * 3 + 1] = p[1]
        positions[i * 3 + 2] = p[2]
        cx += p[0]; cy += p[1]; cz += p[2]
      }
    }
    geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3))
    geometry.attributes.position.needsUpdate = true

    material.opacity = isCurrent ? 0.85 : 0.5

    // Centroid
    const mx = cx / n, my = cy / n, mz = cz / n

    // Label
    if (labelRef.current) {
      labelRef.current.position.set(mx, my + 0.06, mz)
    }

    // Ellipsoid — 점이 20개 이상이고, 점 수가 바뀔 때만 재계산
    if (ellipsoidRef.current && n >= 20) {
      if (lastPointCountRef.current !== n) {
        lastPointCountRef.current = n

        // 공분산 행렬 계산
        const cov = [[0,0,0],[0,0,0],[0,0,0]]
        for (let i = 0; i < n; i++) {
          const p = points[i]
          if (!p || p.length < 3) continue
          const dx = p[0] - mx, dy = p[1] - my, dz = p[2] - mz
          cov[0][0] += dx*dx; cov[0][1] += dx*dy; cov[0][2] += dx*dz
          cov[1][0] += dy*dx; cov[1][1] += dy*dy; cov[1][2] += dy*dz
          cov[2][0] += dz*dx; cov[2][1] += dz*dy; cov[2][2] += dz*dz
        }
        for (let i = 0; i < 3; i++)
          for (let j = 0; j < 3; j++)
            cov[i][j] /= n

        const { eigenvalues, eigenvectors } = eigenDecomposition3x3(cov)

        // 2σ 타원체 (95% 범위)
        const scale = eigenvalues.map(v => Math.sqrt(Math.max(v, 0)) * 2)

        // 회전 행렬 구성
        const rotMatrix = new THREE.Matrix4()
        const e = eigenvectors
        rotMatrix.set(
          e[0].x, e[1].x, e[2].x, 0,
          e[0].y, e[1].y, e[2].y, 0,
          e[0].z, e[1].z, e[2].z, 0,
          0, 0, 0, 1,
        )

        ellipsoidRef.current.position.set(mx, my, mz)
        ellipsoidRef.current.scale.set(
          Math.max(scale[0], 0.001),
          Math.max(scale[1], 0.001),
          Math.max(scale[2], 0.001),
        )
        ellipsoidRef.current.setRotationFromMatrix(rotMatrix)
      }
      ellipsoidRef.current.visible = true
    } else if (ellipsoidRef.current) {
      ellipsoidRef.current.visible = false
    }
  })

  return (
    <>
      <points ref={meshRef} geometry={geometry} material={material} frustumCulled={false} />
      <sprite ref={labelRef} scale={[0.1, 0.05, 1]}>
        <spriteMaterial
          map={labelTexture}
          transparent
          opacity={isCurrent ? 0.9 : 0.6}
          depthWrite={false}
        />
      </sprite>
      <mesh ref={ellipsoidRef} visible={false}>
        <sphereGeometry args={[1, 24, 16]} />
        <meshBasicMaterial color={color} transparent opacity={0.08} depthWrite={false} side={THREE.DoubleSide} />
      </mesh>
    </>
  )
}

// ─── Locked Centers ───

function LockedCenters() {
  const lockedCenters = useStore((s) => s.lockedCenters)

  // Build label textures for locked vowels
  const textures = useMemo(() => {
    const result: Record<string, THREE.CanvasTexture> = {}
    for (const name of Object.keys(lockedCenters)) {
      const canvas = document.createElement('canvas')
      canvas.width = 64
      canvas.height = 32
      const ctx = canvas.getContext('2d')!
      ctx.fillStyle = VOWEL_COLORS[name] || '#ffffff'
      ctx.font = '20px "Noto Sans KR", sans-serif'
      ctx.textAlign = 'center'
      ctx.textBaseline = 'middle'
      ctx.fillText(name, 32, 16)
      result[name] = new THREE.CanvasTexture(canvas)
    }
    return result
  }, [Object.keys(lockedCenters).join(',')])

  return (
    <>
      {Object.entries(lockedCenters).map(([vowel, center]) => (
        <group key={vowel} position={center}>
          {/* Core point */}
          <mesh>
            <sphereGeometry args={[0.035, 16, 16]} />
            <meshBasicMaterial color={VOWEL_COLORS[vowel] || '#fff'} transparent opacity={0.9} />
          </mesh>
          {/* Glow */}
          <mesh>
            <sphereGeometry args={[0.07, 16, 16]} />
            <meshBasicMaterial color={VOWEL_COLORS[vowel] || '#fff'} transparent opacity={0.15} depthWrite={false} />
          </mesh>
          {/* Label */}
          {textures[vowel] && (
            <sprite position={[0, 0.08, 0]} scale={[0.12, 0.06, 1]}>
              <spriteMaterial map={textures[vowel]} transparent opacity={0.85} depthWrite={false} />
            </sprite>
          )}
        </group>
      ))}
    </>
  )
}

// ─── Main Component ───

interface VowelSpace3DProps {
  mini?: boolean
  showCalibration?: boolean
  style?: React.CSSProperties
}

export function VowelSpace3D({ mini = false, showCalibration = false, style }: VowelSpace3DProps) {
  const vowelCenters = useStore((s) => s.vowelCenters)
  const label = useStore((s) => s.vowelSpace.label)
  const controlsRef = useRef<any>(null)
  const [activeView, setActiveView] = useState('3D')

  const handleViewChange = useCallback((viewName: string) => {
    const c = controlsRef.current
    if (!c) return
    const preset = VIEW_PRESETS[viewName]
    if (!preset) return
    c.object.position.set(...preset.pos)
    c.target.set(0, 0, 0)
    c.update()
    setActiveView(viewName)
  }, [])

  return (
    <div style={{
      width: '100%',
      height: '100%',
      borderRadius: mini ? 12 : 0,
      overflow: 'hidden',
      position: 'relative',
      ...style,
    }}>
      <Canvas
        camera={{ position: DEFAULT_CAMERA_POS, fov: mini ? 40 : 35, near: 0.1, far: 50 }}
        gl={{ antialias: true, alpha: true, toneMapping: THREE.NoToneMapping }}
        style={{ background: 'transparent' }}
        onCreated={({ gl }) => {
          gl.setClearColor(0x000000, 0)
        }}
      >
        <CameraControls onControlsRef={(ref) => { controlsRef.current = ref }} />
        <SubtleAxes />

        {/* Performance mode: show all pre-calibrated anchor points */}
        {!showCalibration && Object.entries(vowelCenters).map(([name, pos]) => (
          <AnchorPoint key={name} name={name} position={pos} isNearest={name === label} />
        ))}
        {!showCalibration && <VowelLabels />}

        <LivePoint />

        {/* Performance mode: trail */}
        {!showCalibration && <Trail />}

        {/* Calibration mode: per-vowel point clouds + locked centers */}
        {showCalibration && (
          <>
            <VowelTrajectories />
            <LockedCenters />
          </>
        )}
      </Canvas>

      {/* View preset buttons */}
      {!mini && (
        <div style={{
          position: 'absolute',
          top: 12,
          left: 12,
          display: 'flex',
          gap: 4,
          zIndex: 10,
        }}>
          {Object.keys(VIEW_PRESETS).map((name) => (
            <button
              key={name}
              onClick={() => handleViewChange(name)}
              style={{
                background: activeView === name ? 'rgba(255,255,255,0.12)' : 'rgba(255,255,255,0.04)',
                border: activeView === name ? '1px solid rgba(255,255,255,0.25)' : '1px solid rgba(255,255,255,0.08)',
                color: activeView === name ? '#ddd' : '#777',
                padding: '4px 10px',
                borderRadius: 8,
                fontSize: 11,
                fontWeight: activeView === name ? 500 : 300,
                cursor: 'pointer',
                fontFamily: 'monospace',
                letterSpacing: '0.03em',
                transition: 'all 0.2s',
              }}
            >
              {name}
            </button>
          ))}
        </div>
      )}
    </div>
  )
}
