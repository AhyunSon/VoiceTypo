/**
 * 런타임 튜닝 설정 (구 "터짐 설정"에서 확장, 5차).
 *
 * 코드 상수가 아니라 **가변 싱글턴 객체**로 두어, 시연 중 튜닝 패널에서 값을 바꾸면
 * BubbleManager/렌더러가 즉시 참조해 반영한다(다시 빌드할 필요 없음).
 *
 * 엔진은 이 객체를 **읽기만** 한다. 값 변경은 UI(TuningPanel)가 담당.
 * (이름은 하위호환을 위해 popSettings/PopSettings 유지.)
 */

/** 터짐 모드. */
export type PopMode =
  | 'random-lifetime' // 완성 시 [min,max] 랜덤 수명 → 그 자리에서 터짐
  | 'fixed-lifetime' // 완성 시 고정 수명 → 그 자리에서 터짐
  | 'edge' // 화면 가장자리 도달 시 터짐
  | 'never'; // 터지지 않음(관찰용)

/** 가장자리/바닥 정책. (B2) */
export type EdgeMode =
  | 'reflect' // 부드럽게 반사(속도 일부 감쇠)
  | 'passthrough'; // 통과 후 화면 밖 제거

export interface PopSettings {
  mode: PopMode;
  /** random-lifetime 수명 범위(초). A7: 간격을 넓게(≥5s) 두면 제각각 터진다. */
  lifetimeMin: number;
  lifetimeMax: number;
  /** fixed-lifetime 고정 수명(초). */
  fixedLifetime: number;
  /** 터짐 비율(0~1). 이 비율만 터지고 나머지는 날아가 사라진다. */
  popRatio: number;
  /** 터짐 3막 애니메이션 전체 길이(초, 0.3~0.6). (A6) */
  popDurationS: number;

  // ---- 확산(비행) (A1) ----
  /** 표류 속도 범위(px/s). 버블마다 이 사이에서 랜덤. driftMin 이 정지 방지 하한. */
  driftMin: number;
  driftMax: number;

  // ---- 가장자리/바닥 (B2) ----
  edgeMode: EdgeMode;
  /** 반사 시 속도 유지율(0~1). 0.8 이면 튕길 때 20% 감쇠. */
  edgeBounce: number;

  // ---- 비행 y = 피치 궤적 (4차 수정 1 + 피치 부유 수정) ----
  pitchYRange: number;
  pitchYPlaybackS: number; // (레거시) 참고용 — 재생 시간은 이제 수명×배율.
  /** 재생 시간 = 버블 수명 × 이 배율. 1.0 이면 수명 내내 궤적 진행. */
  trajectoryTimeScale: number;
  /** 수직 속도 상한(px/s). 표류 속도와 같은 자릿수로 두어 '이동'이 아니라 '부유'로. */
  maxVerticalSpeed: number;
  /** 재생 종료 후: 'pingpong'(감쇠 왕복) / 'bob'(작은 사인 부유). */
  pitchIdleMode: 'pingpong' | 'bob';
  /** pingpong 반복마다 진폭 감쇠율(0~1). 작을수록 빨리 잔잔해짐. */
  pitchDecayRate: number;
  /** idle 부유 진폭(px). 피치 변화 없는 버블도 이만큼 미세하게 오르내림. */
  idleBobAmp: number;

  // ---- 불투명도 (A4) ----
  opacityMin: number;
  opacityMax: number;

  // ---- 피치 → 종횡비 강도 (A5) ----
  /** 0~1. scaleX/scaleY 최대 왜곡. 클수록 납작/홀쭉이 뚜렷. */
  aspectStrength: number;


  // ---- 세기 변화 판정 (어택·릴리즈 오탐 제거) ----
  /** 새 세그먼트 판정 상대 임계(dB). 기준선 대비 |20·log10(현재/기준)| 이 이상이어야 후보. */
  intensityChangeDb: number;
  /** 후보 세기 수준이 이 시간(ms) 유지돼야 세그먼트로 확정. 어택·릴리즈를 걸러낸다. */
  intensityHoldMs: number;
  /** 발성 시작 후 이 시간(ms) 동안은 세기 변화 판정을 하지 않고 기준선 산출에만 쓴다. */
  onsetGraceMs: number;

  // ---- 변화 속도 구분 (확 커짐 vs 점점 커짐) ----
  /** 기울기 측정 슬라이딩 윈도우(ms). 이 창 안 변화가 급변/점진을 가른다. */
  slopeWindowMs: number;
  /** 점진 변화 단계 임계(dB). 마지막 로브 대비 누적 변화가 이만큼마다 새 로브. */
  gradualStepDb: number;
  /** 단계 로브 사이 최소 시간 간격(ms). 경계 근처 연속 생성 방지. */
  gradualStepMinMs: number;
  /** 글자 표시 간격(N로브마다 1개). 1=모두. 로브가 과밀할 때 키운다. */
  glyphEveryNLobes: number;

  // ---- 재발성(음절 반복) 감지 (피치 연속성) ----
  /** 딥 진입 비율(0~1): 에너지가 기준 세기의 이 배 아래로 떨어지면 딥으로 본다. */
  dipRatio: number;
  /** 딥 최소 지속(ms) — 떨림·미세 출렁임 오탐 방지. */
  dipMinMs: number;
  /** 딥 최대 지속(ms) — 이보다 오래 안 회복하면 재발성 아님(페이드/끊김). */
  dipMaxMs: number;
  /** 딥 구간 허용 피치 공백(ms) — 이하이면 '피치 연속'=재발성, 초과면 끊김. */
  pitchGapMaxMs: number;
  /** 발성 종료 판정: 에너지 침묵 + 피치 소실이 이 시간(ms) 이상 지속되면 새 버블. */
  utteranceBreakMs: number;

  // ---- 하단 쌓임 / 테트리스 (B1) ----
  stackEnabled: boolean;
  /** 쌓인 글자가 이 시간(초) 지나면 아래층부터 사라진다. */
  stackClearAfterS: number;
  /** 붕괴/터짐 글자 중 쌓이는 비율(0~1). 나머지는 소멸. */
  glyphStackRatio: number;

  // ---- 개발용 (A1·A2·A7, B3) ----
  devLog: boolean; // 콘솔에 버블 생성/터짐 로그
  showOverlay: boolean; // 인식 로그 오버레이 표시
  showSignal: boolean; // 임시 신호(주파수·피치) 패널 표시
}

/** 기본값. */
export const DEFAULT_POP_SETTINGS: PopSettings = {
  mode: 'random-lifetime',
  lifetimeMin: 4.0,
  lifetimeMax: 12.0, // A7: 8초 간격 → 제각각 터짐
  fixedLifetime: 5.0,
  popRatio: 0.7,
  popDurationS: 0.5,

  driftMin: 34, // A1: 정지 방지 하한
  driftMax: 90,

  edgeMode: 'reflect',
  edgeBounce: 0.7,

  pitchYRange: 110,
  pitchYPlaybackS: 6.0,
  trajectoryTimeScale: 1.0, // 수명 전체에 궤적을 편다
  maxVerticalSpeed: 34, // 수평 표류(driftMin)와 같은 자릿수 → 부유감
  pitchIdleMode: 'pingpong',
  pitchDecayRate: 0.55,
  idleBobAmp: 12,

  opacityMin: 0.4, // A4: 가장 짧은 발성도 명확히 보이게
  opacityMax: 0.95,

  aspectStrength: 0.55, // A5: 정지 화면에서도 납작/홀쭉 구분

  intensityChangeDb: 6, // 어택·릴리즈 오탐 제거: 6dB 이상 상대 변화만 후보
  intensityHoldMs: 180, // 후보가 이만큼 유지돼야 확정(어택 상승/릴리즈 감쇠 걸러냄)
  onsetGraceMs: 120, // 시작 직후 유예(기준선 산출 전용)

  slopeWindowMs: 300, // 기울기 측정 창 — 이 안 6dB↑=급변, 미만=점진
  gradualStepDb: 2.5, // 점진 2.5dB마다 단계 로브
  gradualStepMinMs: 150, // 단계 로브 최소 간격
  glyphEveryNLobes: 1, // 글자 표시 간격(1=모두)

  dipRatio: 0.4, // 기준의 40% 아래로 떨어지면 딥
  dipMinMs: 30, // 떨림 오탐 방지 하한
  dipMaxMs: 250, // 이 안에 회복해야 재발성
  pitchGapMaxMs: 150, // 딥 중 피치 공백 허용치(연속 조건)
  utteranceBreakMs: 400, // 에너지+피치 동시 소실 이 이상 → 새 버블(딥 허용치보다 길게)

  stackEnabled: true,
  stackClearAfterS: 6.0,
  glyphStackRatio: 0.6,

  devLog: false,
  showOverlay: false,
  showSignal: true, // 임시 디버그 패널 — 기본 표시
};

/** 앱 전역에서 공유하는 가변 인스턴스. */
export const popSettings: PopSettings = { ...DEFAULT_POP_SETTINGS };

/** 패널에서 일괄 갱신할 때 사용(부분 갱신). */
export function updatePopSettings(patch: Partial<PopSettings>): void {
  Object.assign(popSettings, patch);
}

/** 현재 설정을 JSON 문자열로(팀 공유용 내보내기). */
export function exportPopSettings(): string {
  return JSON.stringify(popSettings, null, 2);
}
