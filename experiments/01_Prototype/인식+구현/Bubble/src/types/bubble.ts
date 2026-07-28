/**
 * 버블 도메인 모델.
 *
 * 버블은 **BubbleSegment 배열**로 모델링하고, 세그먼트 사이의 관계를
 * `hasBoundary: boolean` 로 표현한다(프롬프트 컨텍스트):
 *  - hasBoundary=false  → 경계선 없이 매끈하게 이어진 땅콩(같은 모음 지속/연속)
 *  - hasBoundary=true   → 경계선이 보이는 두 방울(모음이 바뀜)
 *
 * 생명주기 상태 머신: FORMING → FLOATING → POPPING (+ 세그먼트 병합 시 MERGING).
 * 여기에 인식 실패 시 글자가 무너지는 DEBRIS 를 보조 상태로 둔다.
 */

import type { Spring } from '../animation/spring';
import type { MaybeVowel } from './audio';

export type BubbleState =
  | 'FORMING' // 발성 중: 제자리에서 실시간 변형·세그먼트 추가
  | 'MERGING' // 직전 버블에 새 발화가 흡수되는 짧은 전이(스프링으로 끌어당김)
  | 'FLOATING' // 발성 종료: 완성되어 상승(떨림에 따라 좌우 흔들림)
  | 'POPPING' // 터짐 3막(tension → burst → residual)
  | 'DEBRIS'; // 인식 실패로 글자가 무너져 잔해만 남은 상태(버블 미완성)

/**
 * 인식 불가(처음부터) 케이스의 표현 방식 플래그.
 * 팀 논의에서 우선 선택된 두 안을 모두 구현하고 플래그로 전환한다(프롬프트 §2-5).
 *
 * 기각되지 않은 대안(구현은 보류, 확장 지점으로만 기록):
 *  - 'question'      : '?' 만 제시
 *  - 'nearest-query' : 가장 가까운 모음 + '?'
 *  - 'hourglass'     : 모래시계처럼 상승 전 소멸
 *  - 'scatter'       : 작은 방울 여러 개로 흩어져 소멸
 *  - 'abstract'      : 버블 안 추상 형태
 * 최종 단일안은 7/16 레퍼런스 조사 후 재논의 예정.
 */
export type UnrecognizedMode = 'bubble-only' | 'bumpy';

/**
 * 하나의 세그먼트 = 하나의 방울.
 * 발화 순서를 반영하되, 붙는 방향은 일렬이 아니라 랜덤(offsetAngle).
 */
export interface BubbleSegment {
  id: number;
  /**
   * 이 세그먼트의 모음.
   *  - null: 아직 인식 전(spawn-then-detach). 글자를 그리지 않는다.
   */
  vowel: MaybeVowel;
  /**
   * 앞 세그먼트와의 사이에 경계선이 있는가.
   *  - true  → 모음이 바뀐 지점(눈에 보이는 경계).
   *  - false → 같은 모음이 이어지거나 세기만 변한 매끈한 연결(땅콩).
   * (첫 세그먼트는 앞이 없으므로 false.)
   */
  hasBoundary: boolean;

  /** 반지름(px) 스프링. 목표는 세기(에너지)로 정해진다 → 크기 = 세기. */
  radius: Spring;
  /**
   * 이 세그먼트의 종횡비 스프링. value 는 -1(납작)~0(원형)~+1(홀쭉). (4차 수정 3)
   * 활성 세그먼트만 실시간 피치로 갱신되고, 다음 세그먼트가 시작되면 동결된다.
   * → 한 땅콩 안에 "납작한 아 + 홀쭉한 오"가 공존한다.
   */
  aspect: Spring;

  /**
   * 버블 앵커(중심 기준) 대비 이 세그먼트 중심의 상대 위치(px, 로컬 좌표).
   * 붙는 방향(offsetAngle)은 랜덤이고, 거리는 매 프레임 실효 반지름으로 클램프된다(4차 수정 2).
   */
  offset: { x: number; y: number };
  /**
   * 첫 부착(i=1)의 기준 절대 각도(rad, 랜덤). 이후 로브는 attachTurn 으로 누적한다.
   * 실제 중심거리는 layoutSegments 가 매 프레임 실효 반지름으로 계산한다.
   */
  offsetAngle: number;
  /**
   * 직전 부착 방향 대비 **상대 회전각**(rad, ±ATTACH_ANGLE_JITTER). (다중 로브 정리 지시 3)
   * 체인이 되접혀 고리(=실루엣 틈)를 만들지 않게, 각 로브는 직전 방향에서 이만큼만 꺾여 붙는다.
   */
  attachTurn: number;

  /**
   * 이 세그먼트가 활성(발음)된 누적 시간(ms).
   * 길이 → 불투명도 매핑의 원천: 오래 낸 모음일수록 불투명해진다.
   */
  heldMs: number;
  /** heldMs 로 계산한 현재 불투명도 0~1 (렌더가 사용). */
  opacity: number;

  /** 색/무작위 요소용 시드(하이라이트 위치 등). */
  seed: number;
}

/** 터짐/잔해 입자. */
export interface Particle {
  x: number;
  y: number;
  vx: number;
  vy: number;
  /** 남은 수명(초). */
  life: number;
  /** 최초 수명(초). alpha 계산에 사용. */
  maxLife: number;
  size: number;
  rot: number;
  vrot: number;
  kind: 'burst-ring' | 'residual' | 'glyph-debris';
  /** glyph-debris 일 때 그릴 글자. */
  text?: string;
}

/**
 * 하나의 버블(한 발성의 결과물).
 */
export interface Bubble {
  id: number;
  state: BubbleState;

  /** 발화 지점(고정). FORMING 동안 pos 는 여기에 머문다. */
  anchor: { x: number; y: number };
  /** 현재 중심 위치. FLOATING 에서 상승·좌우 흔들림으로 갱신된다. */
  pos: { x: number; y: number };

  /** 세그먼트 배열(발화 순서). 종횡비·크기는 세그먼트 단위(4차 수정 3). */
  segments: BubbleSegment[];

  /**
   * 스폰 스케일 스프링(0→1, 오버슈트 허용). 버블 전체 균일 스케일.
   * spawn-then-detach: 발화 시작 즉시 톡 튀어나오는 오버슈트 연출.
   */
  spawnScale: Spring;

  createdAt: number;
  /** 마지막으로 발화 활동이 있었던 시각(병합 후보 판정용). */
  lastActivityAt: number;
  /** 완성된 발화 길이(ms). UTTERANCE_END 에서 확정. */
  durationMs: number;

  // ---- 확산(FLOATING) — 수정 1 ----
  /**
   * 흔들림 전 "기준 중심". 완성 후 vel 로 이 점이 이동하고,
   * 떨림 진동은 이 점을 기준으로 vel 수직축에 얹는다(pos = base + 수직진동).
   */
  base: { x: number; y: number };
  /** 확산 속도 벡터(px/s). 랜덤 방향·기본속도(위쪽 편향). x 표류 + 궤적 후 y 표류. */
  vel: { x: number; y: number };
  /** 0~1. 클수록 이동 중 크게 흔들린다(이동 방향 수직축 진동). */
  tremor: number;
  /** 흔들림 위상(rad). 매 프레임 누적. */
  swayPhase: number;

  // ---- 비행 y = 피치 궤적(부드러운 힌트) (4차 수정 1) ----
  /** 발성 중 기록한 정규화 피치 궤적(0~1, 키포인트로 단순화됨). 비어 있으면 y 힌트 없음. */
  pitchTrajectory: number[];
  /** 궤적 재생 경과(초). */
  trajElapsed: number;
  /** 궤적 재생 총 길이(초). 비행 전체에 길게 편다. */
  trajDuration: number;
  /**
   * 피치 y 추종 스프링(px). 궤적 목표를 스프링+속도상한으로 간접 추종 →
   * 순간이동·계단식 없이 부드럽게. value 가 현재 y 오프셋.
   */
  vSpring: Spring;

  // ---- 터짐 — 수정 5 + 2차 수정 2 ----
  /** 완성 시점에 부여되는 수명(초). random/fixed-lifetime 모드에서 age 초과 시 터짐. */
  lifetime: number;
  /** FLOATING 진입 후 경과(초). */
  age: number;
  /** 이 버블이 "터지는" 버블인가(popRatio 로 결정). false 면 날아가 사라진다. */
  willPop: boolean;

  // ---- 터짐 ----
  /** POPPING 진입 후 경과(초). 3막 타이밍 계산에 사용. */
  popElapsed: number;
  /** 파열(burst) 막에서 입자를 한 번만 흩뿌리기 위한 플래그. */
  popBursted: boolean;
  particles: Particle[];
  /** 제거 대상 표시(모든 연출 종료). */
  dead: boolean;

  /**
   * 이 발화에서 모음이 한 번이라도 인식되었는가.
   *  - false 인 채로 UTTERANCE_END 를 맞으면 "처음부터 인식 불가" 버블로 처리.
   */
  everRecognized: boolean;

  // ---- 인식 불가 처리 ----
  /** true 면 "처음부터 인식 불가" 버블. */
  unrecognized: boolean;
  /** 인식 불가 표현 방식. */
  unrecognizedMode: UnrecognizedMode;
  /** bumpy 모드에서 울퉁불퉁 윤곽을 만드는 시드. */
  bumpySeed: number;
}
