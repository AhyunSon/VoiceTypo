/**
 * Spring
 * ------
 * 커스텀 스프링 물리(프롬프트: "커스텀 스프링 물리(구현 완료)"에 대응).
 * 세미-임플리싯 오일러 적분의 감쇠 조화진동자.
 *
 *   a = -stiffness·(value - target) - damping·velocity   (질량 1)
 *
 * 쓰임(프롬프트 §2):
 *  - 세그먼트 반지름(세기 → 크기)
 *  - 버블 종횡비(피치 → squash & stretch)
 *  - 스폰 스케일: damping 을 낮춰 **오버슈트**(톡 튀어나오는 spawn-then-detach)
 *
 * damping 이 2·sqrt(stiffness) 보다 작으면 언더댐프(오버슈트), 크면 오버댐프(부드럽게 수렴).
 */
export class Spring {
  value: number;
  velocity = 0;
  target: number;
  stiffness: number;
  damping: number;

  constructor(initial: number, stiffness: number, damping: number) {
    this.value = initial;
    this.target = initial;
    this.stiffness = stiffness;
    this.damping = damping;
  }

  /** 목표만 바꾼다(속도 유지 → 자연스러운 재조준). */
  setTarget(target: number): void {
    this.target = target;
  }

  /** 값·목표를 즉시 일치시키고 속도 0(초기화). */
  snap(value: number): void {
    this.value = value;
    this.target = value;
    this.velocity = 0;
  }

  /** dt(초) 만큼 적분. dt 상한으로 탭 비활성 시 폭주 방지. */
  update(dt: number): number {
    const h = Math.min(dt, 0.05);
    const force = -this.stiffness * (this.value - this.target);
    const damp = -this.damping * this.velocity;
    this.velocity += (force + damp) * h;
    this.value += this.velocity * h;
    return this.value;
  }

  /** 목표에 충분히 수렴했는가(정지 판정). */
  isSettled(epsilon = 0.5): boolean {
    return Math.abs(this.value - this.target) < epsilon && Math.abs(this.velocity) < epsilon;
  }
}
