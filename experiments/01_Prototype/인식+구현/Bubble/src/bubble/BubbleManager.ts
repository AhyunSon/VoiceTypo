import type { MaybeVowel } from '../types/audio';
import type { Dimensions } from '../types/common';
import type { VoiceEvent } from '../types/events';
import type { Bubble, BubbleSegment, Particle, UnrecognizedMode } from '../types/bubble';
import { Spring } from '../animation/spring';
import { clamp, hashNoise } from '../animation/mathUtils';
import {
  aspectToScale,
  bubbleLocalBounds,
  effectiveRadius,
  energyToRadius,
  heldToOpacity,
  popDurations,
  sampleTrajectorySmooth,
  segmentCenter,
  triangleWave,
} from './bubbleMath';
import { GlyphStack } from './GlyphStack';
import { popSettings } from '../config/popSettings';
import {
  ATTACH_ANGLE_JITTER,
  DEBRIS_GRAVITY,
  GLYPH_SIZE_RATIO,
  IDLE_BOB_HZ,
  MERGE_MAX_RISE_PX,
  MERGE_WINDOW_MS,
  MIN_UTTERANCE_MS,
  NONADJ_SEPARATION_RATIO,
  OFFSCREEN_MARGIN_PX,
  OVERLAP_RATIO,
  POP_PARTICLE_COUNT,
  RESIDUAL_BUOYANCY,
  RESIDUAL_LIFE_S,
  SEGMENT_SOLVER_ITERS,
  SOFT_COLLISION,
  SOFT_COLLISION_STRENGTH,
  SPAWN_X_JITTER_PX,
  SPAWN_Y_RATIO,
  SPREAD_UPWARD_BIAS,
  SPRING_ASPECT,
  SPRING_PITCH_Y,
  SPRING_SIZE,
  SPRING_SPAWN,
  SWAY_ANGULAR_SPEED,
  SWAY_MAX_AMPLITUDE_PX,
} from '../config/bubbleConfig';

/**
 * BubbleManager
 * =============
 * 버블 상태 전이(FORMING → FLOATING → POPPING, + MERGING/DEBRIS)를 관리한다.
 *
 * 프롬프트 §3의 경계 규칙을 지킨다:
 *  - 오직 VoiceEvent 만 구독한다(handle). 프레임 원시 데이터는 절대 참조하지 않는다.
 *  - 한 특징 → 한 시각 자유도:
 *      INTENSITY_CHANGED → 세그먼트 반지름(크기, 미세)
 *      INTENSITY_SEGMENT → 경계 없는 새 세그먼트(땅콩, 수정 4)
 *      PITCH_SHIFT       → 버블 종횡비(모양, 2차 수정 3)
 *      UTTERANCE_END.pitchTrajectory → 완성 후 비행 y궤적(3차 수정 3)
 *      UTTERANCE_END.tremor → 이동 방향 수직축 진동(수정 1)
 *      VOWEL_*           → 글자 + 세그먼트 경계
 *
 * 터짐 정책은 config/popSettings(런타임 가변)을 참조한다(2차 수정 2).
 * 매 프레임 update(dt) 로 스프링 적분·확산·터짐·입자를 진행한다.
 */
export class BubbleManager {
  private bubbles: Bubble[] = [];
  /** 현재 발성 중(FORMING/MERGING)인 버블. 없으면 null. 동시에 하나만 형성된다. */
  private forming: Bubble | null = null;
  /** false-start 되감기용: 이번 발화가 추가하기 시작한 세그먼트의 인덱스. */
  private formingSegStart = 0;

  private nextBubbleId = 1;
  private nextSegId = 1;
  private dims: Dimensions = { width: 0, height: 0 };
  private unrecognizedMode: UnrecognizedMode = 'bubble-only';
  /** 현재 발화의 최신 피치 레벨(-1~+1). 새 세그먼트의 초기 종횡비에 쓴다(4차 수정 3). */
  private currentLevel = 0;
  /** 하단 글자 쌓임(B1). */
  private stack = new GlyphStack();

  setDimensions(dims: Dimensions): void {
    this.dims = dims;
    this.stack.setDimensions(dims);
  }
  setUnrecognizedMode(mode: UnrecognizedMode): void {
    this.unrecognizedMode = mode;
  }
  getBubbles(): readonly Bubble[] {
    return this.bubbles;
  }
  /** 렌더용: 쌓인 글자들(B1). */
  getStackedGlyphs() {
    return this.stack.getAll();
  }
  clear(): void {
    this.bubbles = [];
    this.forming = null;
    this.stack.clear();
  }

  // ===================================================================
  // 이벤트 구독 — 여기가 유일한 입력 경로.
  // ===================================================================
  handle(event: VoiceEvent): void {
    switch (event.type) {
      case 'UTTERANCE_START':
        this.onStart(event.time, event.energy, event.level, event.vowel);
        break;
      case 'VOWEL_RESOLVED':
        this.onVowelResolved(event.vowel);
        break;
      case 'VOWEL_CHANGED':
        this.onVowelChanged(event.vowel, event.energy);
        break;
      case 'INTENSITY_CHANGED':
        this.onIntensity(event.energy);
        break;
      case 'INTENSITY_SEGMENT':
        this.onIntensitySegment(event.energy);
        break;
      case 'SYLLABLE_REPEAT':
        this.onSyllableRepeat(event.energy);
        break;
      case 'PITCH_SHIFT':
        this.onPitchAspect(event.level);
        break;
      case 'RECOGNITION_LOST':
        this.onRecognitionLost();
        break;
      case 'RECOGNITION_REGAINED':
        this.onRecognitionRegained(event.vowel);
        break;
      case 'UTTERANCE_END':
        this.onEnd(event.time, event.durationMs, event.tremor, event.pitchTrajectory);
        break;
    }
  }

  private onStart(time: number, energy: number, level: number, vowel: MaybeVowel): void {
    this.currentLevel = level;
    // 연달아 발성 → 병합(MERGING): 방금 뜨기 시작한(가까이 있는) 버블에 흡수시킨다.
    const merge = this.findMergeTarget(time);
    if (merge) {
      merge.state = 'MERGING';
      merge.lastActivityAt = time;
      merge.vel = { x: 0, y: 0 }; // 형성 동안 확산 정지
      merge.anchor = { x: merge.pos.x, y: merge.pos.y }; // 현재 자리에 도킹
      this.formingSegStart = merge.segments.length;
      // 새 세그먼트(모음은 대개 null) 부착 — 이전 세그먼트는 동결(4차 수정 3).
      this.appendSegment(merge, this.createSegment(vowel, energy, false));
      merge.spawnScale.snap(0.6);
      merge.spawnScale.setTarget(1); // 살짝 튕기며 붙는 느낌
      this.forming = merge;
      return;
    }

    // 새 버블 생성(FORMING). null-vowel 로 먼저 스폰(spawn-then-detach) + 오버슈트.
    const p = this.spawnPoint();
    const seg = this.createSegment(vowel, energy, false); // 첫 세그먼트 aspect = currentLevel
    const spawnSpring = new Spring(0, SPRING_SPAWN.stiffness, SPRING_SPAWN.damping);
    spawnSpring.setTarget(1); // 0→1 언더댐프 = 톡 튀어나오는 오버슈트
    const vSpring = new Spring(0, SPRING_PITCH_Y.stiffness, SPRING_PITCH_Y.damping);

    const bubble: Bubble = {
      id: this.nextBubbleId++,
      state: 'FORMING',
      anchor: { ...p },
      pos: { ...p },
      segments: [seg],
      spawnScale: spawnSpring,
      createdAt: time,
      lastActivityAt: time,
      durationMs: 0,
      base: { ...p },
      vel: { x: 0, y: 0 },
      tremor: 0,
      swayPhase: 0,
      pitchTrajectory: [],
      trajElapsed: 0,
      trajDuration: 0,
      vSpring,
      lifetime: 0,
      age: 0,
      willPop: true,
      popElapsed: 0,
      popBursted: false,
      particles: [],
      dead: false,
      everRecognized: vowel != null,
      unrecognized: false,
      unrecognizedMode: this.unrecognizedMode,
      bumpySeed: 0,
    };
    this.formingSegStart = 0;
    this.bubbles.push(bubble);
    this.forming = bubble;
  }

  /** null 스폰 버블에 첫 모음이 도착 → 글자 채움 + (병합이면) 경계선 결정. */
  private onVowelResolved(vowel: MaybeVowel): void {
    const b = this.forming;
    if (!b || vowel == null) return;
    const idx = b.segments.length - 1;
    const seg = b.segments[idx];
    if (!seg || seg.vowel != null) return;

    seg.vowel = vowel;
    b.everRecognized = true;
    b.unrecognized = false;
    if (idx > 0) {
      // 앞 세그먼트(이전 발화의 끝)와 비교:
      //  - 같은 모음 → 경계선 없음(매끈한 땅콩)
      //  - 다른 모음 → 경계선 있음
      const prev = b.segments[idx - 1];
      seg.hasBoundary = prev.vowel != null && prev.vowel !== vowel;
    }
  }

  /** 발화 도중 모음 전환 → 새 세그먼트(경계선 있음). 이전 세그먼트는 동결(4차 수정 3). */
  private onVowelChanged(vowel: MaybeVowel, energy: number): void {
    const b = this.forming;
    if (!b) return;
    b.everRecognized = true;
    this.appendSegment(b, this.createSegment(vowel, energy, true));
  }

  /** 세기 미세 변화 → 현재(활성=마지막) 세그먼트 목표 크기만. (4차 수정 3) */
  private onIntensity(energy: number): void {
    const b = this.forming;
    if (!b) return;
    const seg = b.segments[b.segments.length - 1];
    if (seg) seg.radius.setTarget(energyToRadius(energy));
  }

  /**
   * 세기 큰 변화(같은 모음 유지) → 경계 없는 새 세그먼트(땅콩). (3차 수정 4)
   *  - 이전 세그먼트 동결 후, 현재 모음을 이어받은 새 세그먼트를 붙인다.
   */
  private onIntensitySegment(energy: number): void {
    const b = this.forming;
    if (!b) return;
    const prev = b.segments[b.segments.length - 1];
    if (!prev) return;
    this.appendSegment(b, this.createSegment(prev.vowel, energy, false));
  }

  /**
   * 재발성(음절 반복) → 같은 모음의 경계 없는 새 로브(땅콩). (재발성 수정 1)
   *  - "아아아"처럼 한 호흡 내 반복 발성 → 음절마다 로브 하나. onIntensitySegment 와 동일 구조지만
   *    피치 연속성 딥으로 판정된다는 점만 다르다(크기 = 그 음절 기준 세기).
   */
  private onSyllableRepeat(energy: number): void {
    const b = this.forming;
    if (!b) return;
    const prev = b.segments[b.segments.length - 1];
    if (!prev) return;
    this.appendSegment(b, this.createSegment(prev.vowel, energy, false));
  }

  /** 피치 변화 → 현재(활성) 세그먼트의 종횡비만 갱신(4차 수정 3). */
  private onPitchAspect(level: number): void {
    this.currentLevel = level;
    const b = this.forming;
    if (!b) return;
    const seg = b.segments[b.segments.length - 1];
    if (seg) seg.aspect.setTarget(level);
  }

  /**
   * 인식 가능 → 불가.
   *  - 이미 글자가 있으면(everRecognized): 글자가 무너져 잔해로(§2-5). 버블 미완성.
   *  - 처음부터 인식 불가면: unrecognized 플래그 + 모드별 표현.
   */
  private onRecognitionLost(): void {
    const b = this.forming;
    if (!b) return;
    if (b.everRecognized) {
      this.collapseToDebris(b);
      this.forming = null;
    } else {
      b.unrecognized = true;
      if (this.unrecognizedMode === 'bumpy') {
        b.bumpySeed = this.nextBubbleId * 13.37 + 1;
      }
      // 'bubble-only' 는 글자 없이 그대로 형성 유지.
    }
  }

  /** 인식 복귀(확장 지점): 현재 세그먼트가 비어 있으면 글자를 채운다. */
  private onRecognitionRegained(vowel: MaybeVowel): void {
    const b = this.forming;
    if (!b || b.state === 'DEBRIS' || vowel == null) return;
    const seg = b.segments[b.segments.length - 1];
    if (seg && seg.vowel == null) {
      seg.vowel = vowel;
      b.everRecognized = true;
      b.unrecognized = false;
    }
  }

  /**
   * 발화 종료. (수정 3: 정상 인식 발성은 여기서 터지지 않는다 — 완성 후 확산만 시작)
   *  - false-start(너무 짧음): 이번 발화 세그먼트를 되감고, 빈 버블이면 제거.
   *  - 정상 종료: 완성 → FLOATING(랜덤 방향 확산 시작). tremor 를 곡률로 전달.
   *    터짐(POPPING)은 발성 종료가 아니라 랜덤 수명(수정 5)으로만 발생한다.
   */
  private onEnd(time: number, durationMs: number, tremor: number, pitchTrajectory: number[]): void {
    const b = this.forming;
    this.forming = null;
    if (!b) return;

    b.durationMs = durationMs;
    b.lastActivityAt = time;

    // 인식 실패로 이미 잔해가 된 경우는 그대로 둔다.
    if (b.state === 'DEBRIS') return;

    if (durationMs < MIN_UTTERANCE_MS) {
      // false-start: 이번 발화가 추가한 세그먼트를 잘라낸다.
      b.segments.splice(this.formingSegStart);
      if (b.segments.length === 0) {
        b.dead = true; // 새 버블이었다면 소멸
      } else {
        this.beginFloating(b, b.tremor, pitchTrajectory); // 병합 대상이었다면 확산 복귀
      }
      return;
    }

    // 3차 수정 2: 정상 인식 이력이 있으면 종료 시 반드시 완성(FLOATING). 절대 터지지 않는다.
    //   (POPPING 경로는 (a) 인식 실패 확정=DEBRIS, (b) FLOATING 수명 도달, 둘뿐.)
    // 정상 완성 → 확산.
    this.beginFloating(b, tremor, pitchTrajectory);

    // 처음부터 인식 불가 확정(모음이 한 번도 안 잡힘).
    if (!b.everRecognized) {
      b.unrecognized = true;
      if (this.unrecognizedMode === 'bumpy' && b.bumpySeed === 0) {
        b.bumpySeed = b.id * 13.37 + 1;
      }
    }
  }

  /**
   * FLOATING 진입 공통 처리.
   *  - x: 랜덤 확산 방향(위쪽 편향) + 떨림. y: 피치 궤적을 스프링으로 부드럽게 추종(4차 수정 1).
   *  - popSettings 에서 수명·터짐 여부(willPop)를 뽑는다(런타임 가변).
   *  - 시각 속성(세그먼트 종횡비·크기·스폰스케일)을 이 시점 값으로 **스냅샷 고정**.
   */
  private beginFloating(b: Bubble, tremor: number, pitchTrajectory: number[]): void {
    b.state = 'FLOATING';
    b.tremor = tremor;
    b.swayPhase = hashNoise(b.id) * Math.PI * 2;
    b.base = { x: b.pos.x, y: b.pos.y };

    const P = popSettings;

    // A1: 버블마다 방향은 randomSpreadDir(개별), 속도는 [driftMin, driftMax]에서 개별 랜덤.
    //   → 정지·일렬 대형 방지. (y 스프링 추종은 pos 계산에만 쓰고 vel 을 감쇠시키지 않음.)
    const dir = this.randomSpreadDir();
    const speed = P.driftMin + Math.random() * Math.max(0, P.driftMax - P.driftMin);
    b.vel = { x: dir.x * speed, y: dir.y * speed };

    // 터짐 정책(런타임 가변). A7: 수명은 이 버블의 완성 시점에 독립적으로 샘플링.
    //   (피치 부유 수정: 궤적 재생 시간이 수명에 의존하므로 수명을 먼저 정한다.)
    b.lifetime =
      P.mode === 'fixed-lifetime'
        ? P.fixedLifetime
        : P.lifetimeMin + Math.random() * Math.max(0, P.lifetimeMax - P.lifetimeMin);
    b.willPop = Math.random() < P.popRatio;
    b.age = 0;

    // 피치 궤적 재생 준비(피치 부유 수정): 재생 시간 = 수명 × 배율.
    //   수명 내내 궤적이 진행되어 '이동'이 아니라 '부유'로 보인다. 종료 후엔 updateFloating 이
    //   pingpong/bob 로 이어받아 정지하지 않는다.
    b.pitchTrajectory = pitchTrajectory;
    b.trajElapsed = 0;
    b.trajDuration = Math.max(0.5, b.lifetime * P.trajectoryTimeScale);
    b.vSpring.snap(0);

    // 시각 속성 스냅샷 고정(스프링 정지) — 이후 오디오 변화에 영향받지 않도록.
    this.freezeVisuals(b);
    // 동결된 반지름/종횡비로 세그먼트 배치를 **최종 1회** 확정(지시 4). 이후 매 프레임 재계산 생략,
    //   렌더러는 이 배치를 스프라이트로 1회 베이크한다.
    this.layoutSegments(b);

    // A1·A7 진단 로그(개발 플래그).
    if (P.devLog) {
      // eslint-disable-next-line no-console
      console.log(
        `[bubble #${b.id}] float dir=(${b.vel.x.toFixed(0)},${b.vel.y.toFixed(0)}) speed=${speed.toFixed(0)}px/s lifetime=${b.lifetime.toFixed(1)}s willPop=${b.willPop}`,
      );
    }
  }

  /**
   * 완성 시점 시각 속성 고정(3차 수정 4).
   * 스폰스케일 + 모든 세그먼트의 종횡비·크기 스프링을 현재 값에 스냅한다.
   */
  private freezeVisuals(b: Bubble): void {
    b.spawnScale.snap(b.spawnScale.value);
    for (const s of b.segments) this.freezeSegment(s);
  }

  /** 세그먼트 하나 동결(종횡비·크기 스프링 정지). (4차 수정 3) */
  private freezeSegment(s: BubbleSegment): void {
    s.aspect.snap(s.aspect.value);
    s.radius.snap(s.radius.value);
  }

  /**
   * 새 세그먼트를 붙이기 전에 직전(활성) 세그먼트를 동결한다. (4차 수정 3)
   * → 이후 실시간 오디오 변화는 새(활성) 세그먼트에만 적용된다.
   */
  private appendSegment(b: Bubble, seg: BubbleSegment): void {
    const prev = b.segments[b.segments.length - 1];
    if (prev) this.freezeSegment(prev);
    b.segments.push(seg);
  }

  /**
   * 세그먼트(로브) 배치. 매 프레임 재계산 — 모든 생성 경로(모음전환·세기·재발성·단계 로브)가
   * 이 한 함수를 거치므로 제약이 우회되지 않는다.
   *  (1) 인접: 랜덤 절대각(offsetAngle)으로 앞 로브에 붙이되 중심거리 d=(r1+r2)_eff×OVERLAP_RATIO.
   *  (2) **비인접 최소 이격**(다중 로브 정리 지시 3): PBD soft-resolve 로 비인접 쌍이
   *      (r_i+r_j)_eff×NONADJ_SEPARATION_RATIO 아래로 못 들어오게 밀어낸다(seg0 고정) →
   *      큰 로브가 인접하지 않은 로브 위에 포개지지 않는다. 매 프레임 같은 입력 → 결과가 튀지 않음.
   */
  private layoutSegments(b: Bubble): void {
    const segs = b.segments;
    const n = segs.length;
    if (n === 0) return;
    segs[0].offset.x = 0;
    segs[0].offset.y = 0;
    if (n === 1) return;

    const target = new Array<number>(n).fill(0); // 인접 목표 중심거리
    let dir = 0;
    for (let i = 1; i < n; i++) {
      const prev = segs[i - 1];
      const cur = segs[i];
      dir = i === 1 ? cur.offsetAngle : dir + cur.attachTurn; // 직전 방향 ±지터 누적(되접힘 방지)
      const rPrev = effectiveRadius(prev.radius.value, prev.aspect.value, dir);
      const rCur = effectiveRadius(cur.radius.value, cur.aspect.value, dir);
      const d = (rPrev + rCur) * OVERLAP_RATIO;
      target[i] = d;
      cur.offset.x = prev.offset.x + Math.cos(dir) * d;
      cur.offset.y = prev.offset.y + Math.sin(dir) * d;
    }
    if (n === 2) return; // 인접 한 쌍뿐 → 비인접 이격 불필요

    // (2) soft resolve(PBD): 인접 등식 + 비인접 부등식(최소 이격). seg0 고정.
    const px = segs.map((sg) => sg.offset.x);
    const py = segs.map((sg) => sg.offset.y);
    const rAt = (i: number, ang: number) =>
      effectiveRadius(segs[i].radius.value, segs[i].aspect.value, ang);
    for (let it = 0; it < SEGMENT_SOLVER_ITERS; it++) {
      for (let i = 1; i < n; i++) {
        const dx = px[i] - px[i - 1];
        const dy = py[i] - py[i - 1];
        const dist = Math.hypot(dx, dy) || 1e-4;
        const diff = (dist - target[i]) / dist;
        const wPrev = i - 1 === 0 ? 0 : 0.5; // seg0 고정
        const wCur = 1 - wPrev;
        px[i - 1] += dx * diff * wPrev;
        py[i - 1] += dy * diff * wPrev;
        px[i] -= dx * diff * wCur;
        py[i] -= dy * diff * wCur;
      }
      for (let i = 0; i < n; i++) {
        for (let j = i + 2; j < n; j++) {
          const dx = px[j] - px[i];
          const dy = py[j] - py[i];
          const dist = Math.hypot(dx, dy) || 1e-4;
          const ang = Math.atan2(dy, dx);
          const minD = (rAt(i, ang) + rAt(j, ang)) * NONADJ_SEPARATION_RATIO;
          if (dist >= minD) continue;
          const push = (dist - minD) / dist; // 음수 → 밀어냄
          const wi = i === 0 ? 0 : 0.5;
          const wj = i === 0 ? 1 : 0.5;
          px[i] += dx * push * wi;
          py[i] += dy * push * wi;
          px[j] -= dx * push * wj;
          py[j] -= dy * push * wj;
        }
      }
      px[0] = 0;
      py[0] = 0;
    }
    for (let i = 0; i < n; i++) {
      segs[i].offset.x = px[i];
      segs[i].offset.y = py[i];
    }
  }

  /** 확산 방향 단위벡터. 위쪽 편향(SPREAD_UPWARD_BIAS)을 섞는다(수정 1). */
  private randomSpreadDir(): { x: number; y: number } {
    const a = Math.random() * Math.PI * 2;
    const bias = clamp(SPREAD_UPWARD_BIAS, 0, 1);
    // 균등 랜덤을 위쪽(0,-1)과 선형 블렌드.
    let x = Math.cos(a) * (1 - bias) + 0 * bias;
    let y = Math.sin(a) * (1 - bias) + -1 * bias;
    const len = Math.hypot(x, y) || 1;
    return { x: x / len, y: y / len };
  }

  // ===================================================================
  // 매 프레임 적분
  // ===================================================================
  update(dt: number, dims: Dimensions): void {
    this.dims = dims;
    this.stack.setDimensions(dims);
    this.stack.update(dt); // 하단 쌓임 진행/소거(B1)

    for (const b of this.bubbles) {
      b.spawnScale.update(dt);
      // 성능(지시 4): 세그먼트 스프링·배치 재계산은 **생성 중(FORMING/MERGING)** 버블만.
      //   완성(FLOATING/POPPING) 버블은 시각 속성이 동결(freezeVisuals)되어 결과가 불변이므로
      //   매 프레임 재계산은 낭비다(배치는 freeze 시점에 1회 확정 → beginFloating). 스프라이트 캐시와 짝.
      if (b.state === 'FORMING' || b.state === 'MERGING') {
        for (const s of b.segments) {
          s.radius.update(dt);
          s.aspect.update(dt);
        }
        this.layoutSegments(b);
      }

      switch (b.state) {
        case 'FORMING':
        case 'MERGING':
          this.updateForming(b, dt);
          break;
        case 'FLOATING':
          this.updateFloating(b, dt);
          break;
        case 'POPPING':
          this.updatePopping(b, dt);
          break;
        case 'DEBRIS':
          this.updateParticles(b, dt);
          if (b.particles.length === 0) b.dead = true;
          break;
      }
    }

    if (SOFT_COLLISION) this.applySoftCollision();

    // 소멸한 버블 정리.
    if (this.bubbles.some((b) => b.dead)) {
      this.bubbles = this.bubbles.filter((b) => !b.dead);
      if (this.forming?.dead) this.forming = null;
    }
  }

  private updateForming(b: Bubble, _dt: number): void {
    // 제자리 고정(2차 수정 1: 생성 중에는 상하 이동 없음, 모양만 변형).
    b.pos.x = b.anchor.x;
    b.pos.y = b.anchor.y;
    // 길이 → 불투명도: 현재(마지막) 세그먼트에만 누적한다.
    // (새 세그먼트가 생기면 이전 세그먼트의 불투명도는 그 지점에서 고정 = 세그먼트별 길이 반영)
    const seg = b.segments[b.segments.length - 1];
    if (seg) {
      seg.heldMs += _dt * 1000;
      seg.opacity = heldToOpacity(seg.heldMs);
    }
    // MERGING 은 스폰 스프링이 안정되면 FORMING 으로 흡수 완료.
    if (b.state === 'MERGING' && b.spawnScale.isSettled(0.02)) b.state = 'FORMING';
  }

  private updateFloating(b: Bubble, dt: number): void {
    b.age += dt;
    b.trajElapsed += dt;

    // 랜덤 방향 표류(x, y 모두 — 궤적은 그 위에 얹는 부드러운 힌트).
    b.base.x += b.vel.x * dt;
    b.base.y += b.vel.y * dt;

    // y 피치 부유(피치 부유 수정): 궤적을 **수명 전체(trajDuration)** 에 펴서 재생하고,
    //   재생이 끝나도 멈추지 않고 감쇠 왕복(pingpong) 또는 사인 부유(bob)로 이어간다.
    //   각 키포인트 전환은 easeInOut(sampleTrajectorySmooth)으로 둥글게, 스프링으로 간접 추종.
    const P = popSettings;
    const traj = b.pitchTrajectory;
    // 미세 사인 부유(idle bob)는 **항상** 얹는다 → 재생이 끝나도 정지하지 않고(기준 3),
    //   피치 변화 없는(궤적 평탄) 버블도 큰 이동 없이 잔잔히 오르내린다(기준 4).
    let target = Math.sin(b.trajElapsed * IDLE_BOB_HZ * Math.PI * 2) * P.idleBobAmp;
    if (traj.length >= 2) {
      const x = b.trajElapsed / Math.max(1e-3, b.trajDuration); // ≥0, 1에서 재생 완료
      if (x <= 1) {
        // 정방향 재생(수명 전체에 펼침).
        target += -(sampleTrajectorySmooth(traj, x) - traj[0]) * P.pitchYRange;
      } else if (P.pitchIdleMode === 'pingpong') {
        // 감쇠 왕복: 삼각파로 궤적을 왕복하되 사이클마다 진폭 감쇠 + 최소 부유 유지.
        const uu = triangleWave(x);
        const amp = Math.pow(P.pitchDecayRate, x - 1);
        target += -(sampleTrajectorySmooth(traj, uu) - traj[0]) * P.pitchYRange * amp;
      } else {
        // 마지막 y 유지 + 작은 사인 부유.
        target += -(traj[traj.length - 1] - traj[0]) * P.pitchYRange;
      }
    }
    const prevY = b.vSpring.value;
    b.vSpring.setTarget(target);
    b.vSpring.update(dt);
    let dy = b.vSpring.value - prevY;
    const maxStep = P.maxVerticalSpeed * dt; // 수직 속도 상한(표류와 같은 자릿수 → 부유감)
    if (Math.abs(dy) > maxStep) {
      dy = Math.sign(dy) * maxStep;
      b.vSpring.value = prevY + dy;
      b.vSpring.velocity = dy / Math.max(1e-4, dt);
    }
    const pitchY = b.vSpring.value;

    // 떨림 → 이동 방향에 **수직인 축**으로 진동(수정 1: 방향이 랜덤이어도 유지).
    b.swayPhase += SWAY_ANGULAR_SPEED * dt;
    const s = Math.sin(b.swayPhase) * b.tremor * SWAY_MAX_AMPLITUDE_PX;
    const speed = Math.hypot(b.vel.x, b.vel.y) || 1;
    const perpX = -b.vel.y / speed;
    const perpY = b.vel.x / speed;
    b.pos.x = b.base.x + perpX * s;
    b.pos.y = b.base.y + pitchY + perpY * s;

    // 수명 기반 터짐(2차 수정 2: 모드에 따라). random/fixed-lifetime 에서만.
    if (b.willPop && (P.mode === 'random-lifetime' || P.mode === 'fixed-lifetime') && b.age >= b.lifetime) {
      this.startPop(b);
      return;
    }

    this.handleEdges(b);
  }

  /**
   * 화면 가장자리/바닥 처리(B2, popSettings 참조).
   *  - edge 모드 + willPop: 가장자리 도달 시 터짐.
   *  - reflect: 부드럽게 반사(edgeBounce 감쇠) + 최소 속도 유지(정지 방지 A1).
   *    바닥은 쌓임 높이(B1) 위에서 반사한다.
   *  - passthrough(또는 날아가는 버블): 완전히 밖으로 나가면 제거.
   */
  private handleEdges(b: Bubble): void {
    const maxR = bubbleLocalBounds(b).maxR;
    const { width, height } = this.dims;
    const P = popSettings;

    const pitchY = b.vSpring.value;
    const ex = b.base.x;
    const ey = b.base.y + pitchY;
    // 하단 반사선: 쌓인 글자 위(B1·B2). 쌓임 없으면 화면 하단.
    const floorY = height - (P.stackEnabled ? this.stack.heightAt(ex) : 0);

    const overLeft = ex - maxR < 0;
    const overRight = ex + maxR > width;
    const overTop = ey - maxR < 0;
    const overBottom = ey + maxR > floorY;
    const atEdge = overLeft || overRight || overTop || overBottom;

    if (atEdge && b.willPop && P.mode === 'edge') {
      this.startPop(b);
      return;
    }

    // 반사: never 모드 전체, 그 외엔 reflect 모드 + 터지는 버블.
    const reflect = P.mode === 'never' ? true : P.edgeMode === 'reflect' && b.willPop;
    if (reflect) {
      const k = clamp(P.edgeBounce, 0, 1);
      if (overLeft) {
        b.base.x = maxR;
        b.vel.x = Math.abs(b.vel.x) * k;
      } else if (overRight) {
        b.base.x = width - maxR;
        b.vel.x = -Math.abs(b.vel.x) * k;
      }
      if (overTop) {
        b.base.y = maxR - pitchY;
        b.vel.y = Math.abs(b.vel.y) * k;
      } else if (overBottom) {
        b.base.y = floorY - maxR - pitchY;
        b.vel.y = -Math.abs(b.vel.y) * k;
      }
      this.enforceMinSpeed(b); // A1: 반복 감쇠로 멈추지 않게 최소 속도 유지
      return;
    }

    // passthrough(날아감): 완전히 밖으로 나가면 제거.
    const m = OFFSCREEN_MARGIN_PX + maxR;
    if (ex < -m || ex > width + m || ey < -m || ey > floorY + m) {
      b.dead = true;
    }
  }

  /** 속도가 driftMin 아래로 떨어지면 방향 유지한 채 하한으로 끌어올린다(A1). */
  private enforceMinSpeed(b: Bubble): void {
    const min = popSettings.driftMin;
    const sp = Math.hypot(b.vel.x, b.vel.y);
    if (sp < 1e-3) {
      const d = this.randomSpreadDir();
      b.vel.x = d.x * min;
      b.vel.y = d.y * min;
    } else if (sp < min) {
      const s = min / sp;
      b.vel.x *= s;
      b.vel.y *= s;
    }
  }

  private updatePopping(b: Bubble, dt: number): void {
    b.popElapsed += dt;
    const { tension, burst, residual } = popDurations();
    // 2막(burst) 진입 순간 한 번만 입자 + 글자 쌓임(B1).
    if (!b.popBursted && b.popElapsed >= tension) {
      this.spawnBurst(b);
      b.popBursted = true;
    }
    this.updateParticles(b, dt);
    if (b.popElapsed >= tension + burst + residual) b.dead = true;
  }

  private updateParticles(b: Bubble, dt: number): void {
    for (const p of b.particles) {
      if (p.kind === 'glyph-debris') p.vy += DEBRIS_GRAVITY * dt;
      else if (p.kind === 'residual') p.vy += RESIDUAL_BUOYANCY * dt;
      p.x += p.vx * dt;
      p.y += p.vy * dt;
      p.rot += p.vrot * dt;
      p.life -= dt;
    }
    if (b.particles.some((p) => p.life <= 0)) {
      b.particles = b.particles.filter((p) => p.life > 0);
    }
  }

  // ===================================================================
  // 전이 도우미
  // ===================================================================
  private startPop(b: Bubble): void {
    b.state = 'POPPING';
    b.popElapsed = 0;
    b.popBursted = false;
  }

  /** 파열 시 잔여 비눗물 입자 + 세그먼트별 링 흔적을 흩뿌린다(3막의 잔여물). */
  private spawnBurst(b: Bubble): void {
    const scale = b.spawnScale.value;
    const ringLife = popDurations().residual * 0.7;
    // 세그먼트마다 터지는 링. + 글자 일부는 하단에 쌓인다(B1).
    for (const s of b.segments) {
      const wx = b.pos.x + s.offset.x * scale;
      const wy = b.pos.y + s.offset.y * scale;
      b.particles.push({
        x: wx,
        y: wy,
        vx: 0,
        vy: 0,
        life: ringLife,
        maxLife: ringLife,
        size: s.radius.value * scale,
        rot: 0,
        vrot: 0,
        kind: 'burst-ring',
      });
      // 글자 일부는 하단으로 떨어져 쌓인다(B1: 일부는 쌓이고 일부는 사라진다).
      if (s.vowel && popSettings.stackEnabled && Math.random() < popSettings.glyphStackRatio) {
        const sc = aspectToScale(s.aspect.value);
        this.stack.land(wx, wy, s.vowel, s.radius.value * scale * GLYPH_SIZE_RATIO, sc.sx, sc.sy);
      }
    }
    // 사방으로 흩어지는 잔여 물방울.
    const bounds = bubbleLocalBounds(b);
    for (let i = 0; i < POP_PARTICLE_COUNT; i++) {
      const a = (i / POP_PARTICLE_COUNT) * Math.PI * 2 + hashNoise(b.id + i) * 0.6;
      const speed = 60 + hashNoise(b.id * 3 + i) * 140;
      b.particles.push({
        x: b.pos.x + Math.cos(a) * bounds.maxR * 0.4,
        y: b.pos.y + Math.sin(a) * bounds.maxR * 0.4,
        vx: Math.cos(a) * speed,
        vy: Math.sin(a) * speed,
        life: RESIDUAL_LIFE_S * (0.6 + hashNoise(b.id + i * 2) * 0.6),
        maxLife: RESIDUAL_LIFE_S,
        size: 2 + hashNoise(b.id + i * 5) * 4,
        rot: 0,
        vrot: 0,
        kind: 'residual',
      });
    }
  }

  /**
   * 인식 실패로 글자가 무너짐 → 각 글자를 낙하 잔해로. (B1: 일부는 하단에 쌓임)
   */
  private collapseToDebris(b: Bubble): void {
    b.state = 'DEBRIS';
    const particles: Particle[] = [];
    for (const s of b.segments) {
      if (s.vowel == null) continue;
      const c = segmentCenter(b, s);
      // 일부 글자는 하단에 쌓인다(B1).
      if (popSettings.stackEnabled && Math.random() < popSettings.glyphStackRatio) {
        const sc = aspectToScale(s.aspect.value);
        this.stack.land(c.x, c.y, s.vowel, s.radius.value * GLYPH_SIZE_RATIO, sc.sx, sc.sy);
        continue;
      }
      particles.push({
        x: c.x,
        y: c.y,
        vx: (hashNoise(s.seed) * 2 - 1) * 60,
        vy: -20 + hashNoise(s.seed + 1) * 40,
        life: 1.3,
        maxLife: 1.3,
        size: s.radius.value * GLYPH_SIZE_RATIO,
        rot: 0,
        vrot: (hashNoise(s.seed + 2) * 2 - 1) * 4,
        kind: 'glyph-debris',
        text: s.vowel,
      });
    }
    b.particles = particles;
  }

  // ===================================================================
  // 생성/배치 도우미
  // ===================================================================
  private spawnPoint(): { x: number; y: number } {
    const { width, height } = this.dims;
    const cx = width / 2 + (Math.random() - 0.5) * SPAWN_X_JITTER_PX;
    return { x: cx, y: height * SPAWN_Y_RATIO };
  }

  /**
   * 세그먼트 하나 생성.
   *  - 크기(반지름)=세기, 종횡비=현재 피치 레벨(currentLevel, 4차 수정 3).
   *  - offsetAngle 은 랜덤(프롬프트 §1-3: 일렬 아님). 실제 offset 거리는 layoutSegments 가 매 프레임 계산.
   */
  private createSegment(vowel: MaybeVowel, energy: number, hasBoundary: boolean): BubbleSegment {
    const radiusTarget = energyToRadius(energy);
    const radius = new Spring(radiusTarget * 0.5, SPRING_SIZE.stiffness, SPRING_SIZE.damping);
    radius.setTarget(radiusTarget);

    const aspect = new Spring(this.currentLevel, SPRING_ASPECT.stiffness, SPRING_ASPECT.damping);
    aspect.setTarget(this.currentLevel);

    return {
      id: this.nextSegId++,
      vowel,
      hasBoundary,
      radius,
      aspect,
      offset: { x: 0, y: 0 },
      offsetAngle: Math.random() * Math.PI * 2, // 첫 부착 기준 방향(랜덤)
      attachTurn: (Math.random() * 2 - 1) * ATTACH_ANGLE_JITTER, // 이후 부착: 직전 ±지터
      heldMs: 0,
      opacity: heldToOpacity(0),
      seed: Math.random() * 1000,
    };
  }

  /** 병합 대상: 방금 종료돼 아직 멀리 안 퍼진 FLOATING 버블. */
  private findMergeTarget(time: number): Bubble | null {
    for (let i = this.bubbles.length - 1; i >= 0; i--) {
      const b = this.bubbles[i];
      if (b.state !== 'FLOATING' || b.dead) continue;
      if (time - b.lastActivityAt > MERGE_WINDOW_MS) continue;
      const traveled = b.age * Math.hypot(b.vel.x, b.vel.y); // 완성 후 이동 거리(방향 무관)
      if (traveled <= MERGE_MAX_RISE_PX) return b;
    }
    return null;
  }

  /**
   * soft collision (선택 구현, 기본 off — config.SOFT_COLLISION).
   * 겹친 두 버블을 반지름 합 기준으로 부드럽게 밀어낸다("큰 버블이 주변을 밀어냄").
   * FORMING 버블은 제자리 고정이므로 FLOATING 만 이동시킨다.
   */
  private applySoftCollision(): void {
    const active = this.bubbles.filter((b) => b.state === 'FLOATING' || b.state === 'FORMING' || b.state === 'MERGING');
    for (let i = 0; i < active.length; i++) {
      for (let j = i + 1; j < active.length; j++) {
        const a = active[i];
        const b = active[j];
        const ra = bubbleLocalBounds(a).maxR;
        const rb = bubbleLocalBounds(b).maxR;
        const dx = b.pos.x - a.pos.x;
        const dy = b.pos.y - a.pos.y;
        const dist = Math.hypot(dx, dy) || 0.0001;
        const overlap = ra + rb - dist;
        if (overlap <= 0) continue;
        const push = (overlap * SOFT_COLLISION_STRENGTH) / dist;
        // FLOATING 만 실제로 밀린다(FORMING 은 고정).
        if (b.state === 'FLOATING') {
          b.anchor.x += dx * push;
          b.pos.x += dx * push;
          b.pos.y += dy * push;
        }
        if (a.state === 'FLOATING') {
          a.anchor.x -= dx * push;
          a.pos.x -= dx * push;
          a.pos.y -= dy * push;
        }
      }
    }
  }
}
