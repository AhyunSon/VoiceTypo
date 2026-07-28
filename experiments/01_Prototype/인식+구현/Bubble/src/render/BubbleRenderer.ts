import type { Bubble, BubbleSegment } from '../types/bubble';
import type { Dimensions } from '../types/common';
import type { StackedGlyph } from '../bubble/GlyphStack';
import { aspectToScale, effectiveRadius, popDurations } from '../bubble/bubbleMath';
import { clamp, hashNoise, smoothstep } from '../animation/mathUtils';
import { GLYPH_SIZE_RATIO, MEMBRANE_LINE_PX } from '../config/bubbleConfig';
import { popSettings } from '../config/popSettings';
import { FILM_EDGE, FILM_INNER, FILM_MID, FILM_RIM, ink, rgba } from './colors';

const TWO_PI = Math.PI * 2;
const DPR_CAP = 2; // 오프스크린 해상도 상한

interface Sprite {
  canvas: HTMLCanvasElement;
  box: { minX: number; minY: number; w: number; h: number }; // 로컬 CSS px (버블 중심 기준)
  opacity: number; // 베이크 시점의 몸체 불투명도(최대 세그먼트 opacity)
}

/**
 * BubbleRenderer (성능 + 실루엣 경로 분리)
 * =======================================
 * 실루엣은 **로브별 독립 경로 불투명 채움의 union** 으로만 만든다. 한 경로에 ellipse 를 누적하면
 * Canvas 가 연속 호 사이에 직선 연결선을 넣어 "직선 변 부채꼴"과 winding 상쇄 틈이 생기므로,
 * 로브마다 `beginPath()`→`ellipse()`→`fill()` 로 따로 그려 쌓는다(겹침이 곧 union, 틈 불가).
 *
 * 성능: "버블 수 × 합성" → "완성 합성 1회(스프라이트 베이크) + 버블 수 × drawImage".
 *  - FORMING/MERGING(1개): 재사용 오프스크린에 매 프레임 합성 후 메인에 blit.
 *  - FLOATING/POPPING: 완성(동결) 시 스프라이트로 1회 베이크 → 이후 drawImage 로 위치·불투명도만.
 *    POPPING 은 스프라이트를 scale·alpha 변형(재합성 없음). 매 프레임 블러·getImageData 없음.
 *
 * 글자는 실루엣 위에 함께 그려 스프라이트에 베이크(글자-버블 분리 불가).
 */
export class BubbleRenderer {
  /** 버블 → 베이크된 스프라이트. 동결 버블 재사용, FORMING/MERGING 진입 시 무효화. */
  private sprites = new WeakMap<Bubble, Sprite>();
  /** FORMING 라이브 합성용 재사용 오프스크린(프레임당 새로 만들지 않음, 지시 4). */
  private formCanvas: HTMLCanvasElement | null = null;
  private formCtx: CanvasRenderingContext2D | null = null;

  constructor(private ctx: CanvasRenderingContext2D) {}

  render(bubbles: readonly Bubble[], stacked: readonly StackedGlyph[], dims: Dimensions): void {
    const { ctx } = this;
    ctx.clearRect(0, 0, dims.width, dims.height);

    for (const b of bubbles) {
      if (b.state === 'DEBRIS') {
        this.drawParticles(b);
        continue;
      }
      if (b.state === 'FORMING' || b.state === 'MERGING') {
        this.sprites.delete(b); // 생성 중엔 시각 변화 → 스프라이트 무효화(라이브 합성)
        this.drawLive(b);
      } else {
        this.drawSprite(b); // FLOATING/POPPING — 캐시된 스프라이트
      }
      this.drawParticles(b);
    }

    this.drawStacked(stacked);
  }

  // -------------------------------------------------------------------
  // FORMING(라이브) — 재사용 오프스크린에 합성 후 메인에 blit
  // -------------------------------------------------------------------
  private drawLive(b: Bubble): void {
    const segs = b.segments;
    if (segs.length === 0) return;
    const overall = Math.max(0, b.spawnScale.value);
    const ctx = this.ctx;

    if (b.unrecognized && b.unrecognizedMode === 'bumpy') {
      ctx.save();
      ctx.translate(b.pos.x, b.pos.y);
      ctx.scale(overall, overall);
      this.drawBumpy(ctx, b, 1);
      ctx.restore();
      return;
    }

    const box = this.localBox(segs, 5);
    if (box.w <= 0 || box.h <= 0) return;
    const fc = this.ensureFormCtx(box);
    if (!fc) return;
    this.composite(fc, segs, b.unrecognized);

    ctx.save();
    ctx.translate(b.pos.x, b.pos.y);
    ctx.scale(overall, overall);
    ctx.globalAlpha = this.bodyOpacity(segs);
    ctx.drawImage(this.formCanvas!, box.minX, box.minY, box.w, box.h);
    ctx.restore();
  }

  /** FORMING 오프스크린을 버블 박스에 맞춰 리사이즈+클리어하고, 로컬좌표 변환을 걸어 반환. */
  private ensureFormCtx(box: { minX: number; minY: number; w: number; h: number }) {
    if (!this.formCanvas) {
      this.formCanvas = document.createElement('canvas');
      this.formCtx = this.formCanvas.getContext('2d');
    }
    const c = this.formCanvas;
    const cx = this.formCtx;
    if (!cx) return null;
    const w = Math.max(1, Math.ceil(box.w * DPR_CAP));
    const h = Math.max(1, Math.ceil(box.h * DPR_CAP));
    if (c.width !== w || c.height !== h) {
      c.width = w; // 리사이즈는 자동 클리어(같은 캔버스 객체 재사용 — 프레임당 생성 아님)
      c.height = h;
    } else {
      cx.setTransform(1, 0, 0, 1, 0, 0);
      cx.clearRect(0, 0, w, h);
    }
    cx.setTransform(DPR_CAP, 0, 0, DPR_CAP, -box.minX * DPR_CAP, -box.minY * DPR_CAP);
    return cx;
  }

  // -------------------------------------------------------------------
  // FLOATING/POPPING — 베이크된 스프라이트 drawImage
  // -------------------------------------------------------------------
  private drawSprite(b: Bubble): void {
    // 터짐 3막(A6): 긴장(수축·떨림) → 파열(팽창·소멸). 스프라이트를 변형만 한다(재합성 없음).
    let swell = 1;
    let bodyAlpha = 1;
    if (b.state === 'POPPING') {
      const { tension, burst } = popDurations();
      if (b.popElapsed < tension) {
        const t = b.popElapsed / tension;
        swell = 1 - 0.1 * t + 0.03 * Math.sin(t * 40);
      } else if (b.popElapsed < tension + burst) {
        const t = (b.popElapsed - tension) / burst;
        swell = 0.9 + 0.9 * t;
        bodyAlpha = 1 - t;
      } else {
        bodyAlpha = 0;
      }
    }
    if (bodyAlpha <= 0) return;

    let sp = this.sprites.get(b);
    if (!sp) {
      const baked = this.bake(b);
      if (!baked) return;
      sp = baked;
      this.sprites.set(b, sp);
    }
    const overall = Math.max(0, b.spawnScale.value) * swell;
    const ctx = this.ctx;
    ctx.save();
    ctx.translate(b.pos.x, b.pos.y);
    ctx.scale(overall, overall);
    ctx.globalAlpha = clamp(sp.opacity * bodyAlpha, 0, 1);
    ctx.drawImage(sp.canvas, sp.box.minX, sp.box.minY, sp.box.w, sp.box.h);
    ctx.restore();
  }

  /** 완성 버블을 스프라이트로 1회 베이크(실루엣+그라디언트+하이라이트+경계막+글자). */
  private bake(b: Bubble): Sprite | null {
    const segs = b.segments;
    if (segs.length === 0) return null;
    const box = this.localBox(segs, 5);
    if (box.w <= 0 || box.h <= 0) return null;

    const canvas = document.createElement('canvas');
    canvas.width = Math.max(1, Math.ceil(box.w * DPR_CAP));
    canvas.height = Math.max(1, Math.ceil(box.h * DPR_CAP));
    const c = canvas.getContext('2d');
    if (!c) return null;
    c.setTransform(DPR_CAP, 0, 0, DPR_CAP, -box.minX * DPR_CAP, -box.minY * DPR_CAP);

    if (b.unrecognized && b.unrecognizedMode === 'bumpy') {
      this.drawBumpy(c, b, 1);
    } else {
      this.composite(c, segs, b.unrecognized);
    }
    return { canvas, box, opacity: this.bodyOpacity(segs) };
  }

  // -------------------------------------------------------------------
  // 공통 합성(격리 오프스크린, 로컬 좌표) — 로브별 독립 union → source-in 몸체 → 하이라이트 → 경계막 → 글자
  // -------------------------------------------------------------------
  private composite(ctx: CanvasRenderingContext2D, segs: BubbleSegment[], unrecognized: boolean): void {
    const pxScale = ctx.getTransform().a || 1; // 로컬→디바이스 x배율(선폭 보정)
    const box = this.localBox(segs, 2);

    // ---- (1) union: **로브마다 독립 경로**로 불투명 채움 (핵심 수정) ----
    //   한 경로 누적 시 생기는 직선 연결선·winding 상쇄 틈이 구조적으로 불가능하다.
    ctx.globalCompositeOperation = 'source-over';
    ctx.fillStyle = '#ffffff';
    for (const s of segs) {
      const { sx, sy } = aspectToScale(s.aspect.value);
      ctx.beginPath();
      ctx.ellipse(s.offset.x, s.offset.y, s.radius.value * sx, s.radius.value * sy, 0, 0, TWO_PI);
      ctx.fill();
    }

    // ---- (2) 몸체: union 안(source-in)에 FILM_MID + 세로 sheen ----
    ctx.globalCompositeOperation = 'source-in';
    ctx.fillStyle = rgba(FILM_MID, 1);
    ctx.fillRect(box.minX, box.minY, box.w, box.h);
    ctx.globalCompositeOperation = 'source-atop';
    const gy = this.lobeYRange(segs);
    const grad = ctx.createLinearGradient(0, gy.min, 0, gy.max || gy.min + 1);
    grad.addColorStop(0, rgba(FILM_INNER, 0.55));
    grad.addColorStop(0.4, rgba(FILM_MID, 0.28));
    grad.addColorStop(1, rgba(FILM_EDGE, 0.5));
    ctx.fillStyle = grad;
    ctx.fillRect(box.minX, box.minY, box.w, box.h);

    // ---- (3) 로브별 하이라이트(각각 독립 경로, source-atop → union 안) ----
    for (const s of segs) this.highlight(ctx, s);

    // ---- (4) 경계막: hasBoundary 접합부 렌즈 중심선을 가는 곡선 1개(면 아님, source-atop) ----
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    ctx.lineWidth = MEMBRANE_LINE_PX / pxScale;
    ctx.strokeStyle = rgba(FILM_RIM, 0.7);
    for (let i = 1; i < segs.length; i++) {
      if (!segs[i].hasBoundary) continue;
      const m = this.membraneChord(segs[i - 1], segs[i]);
      if (!m) continue;
      ctx.beginPath();
      ctx.moveTo(m.x1, m.y1);
      ctx.quadraticCurveTo(m.cx, m.cy, m.x2, m.y2);
      ctx.stroke();
    }

    // ---- (5) 글자(위에) ----
    ctx.globalCompositeOperation = 'source-over';
    if (!unrecognized) {
      const everyN = Math.max(1, Math.round(popSettings.glyphEveryNLobes));
      for (let i = 0; i < segs.length; i++) {
        const s = segs[i];
        if (s.vowel && i % everyN === 0) this.drawGlyph(ctx, s, s.vowel, s.opacity);
      }
    }
    ctx.globalCompositeOperation = 'source-over';
  }

  private bodyOpacity(segs: BubbleSegment[]): number {
    let o = 0;
    for (const s of segs) o = Math.max(o, s.opacity);
    return clamp(o * 0.92, 0, 1);
  }

  /** 로브의 로컬 y 범위(그라데이션 방향). */
  private lobeYRange(segs: BubbleSegment[]): { min: number; max: number } {
    let min = Infinity;
    let max = -Infinity;
    for (const s of segs) {
      const { sy } = aspectToScale(s.aspect.value);
      const r = s.radius.value * sy;
      min = Math.min(min, s.offset.y - r);
      max = Math.max(max, s.offset.y + r);
    }
    return { min, max };
  }

  /** 세그먼트들의 로컬 바운딩 박스(CSS px, 버블 중심 기준) + 패딩. */
  private localBox(
    segs: BubbleSegment[],
    pad: number,
  ): { minX: number; minY: number; w: number; h: number } {
    let minX = Infinity;
    let minY = Infinity;
    let maxX = -Infinity;
    let maxY = -Infinity;
    for (const s of segs) {
      const { sx, sy } = aspectToScale(s.aspect.value);
      const rx = s.radius.value * sx;
      const ry = s.radius.value * sy;
      minX = Math.min(minX, s.offset.x - rx);
      maxX = Math.max(maxX, s.offset.x + rx);
      minY = Math.min(minY, s.offset.y - ry);
      maxY = Math.max(maxY, s.offset.y + ry);
    }
    if (!isFinite(minX)) return { minX: 0, minY: 0, w: 0, h: 0 };
    return { minX: minX - pad, minY: minY - pad, w: maxX - minX + pad * 2, h: maxY - minY + pad * 2 };
  }

  private highlight(ctx: CanvasRenderingContext2D, s: BubbleSegment): void {
    const r = s.radius.value;
    if (r <= 4) return;
    const { sx, sy } = aspectToScale(s.aspect.value);
    const jitter = hashNoise(s.seed) * 0.4 - 0.2;
    ctx.fillStyle = rgba(FILM_INNER, 0.55);
    ctx.beginPath();
    ctx.ellipse(
      s.offset.x - r * 0.34 * sx,
      s.offset.y - r * 0.4 * sy,
      r * 0.2 * sx,
      r * 0.12 * sy,
      -0.6 + jitter,
      0,
      TWO_PI,
    );
    ctx.fill();
  }

  /** 글자 — 세그먼트 종횡비로 함께 눌리거나 늘어난다(납작한 아 / 홀쭉한 오). */
  private drawGlyph(
    ctx: CanvasRenderingContext2D,
    s: BubbleSegment,
    vowel: string,
    alpha: number,
  ): void {
    const r = s.radius.value;
    if (r <= 6) return;
    const { sx, sy } = aspectToScale(s.aspect.value);
    const size = r * GLYPH_SIZE_RATIO;
    ctx.save();
    ctx.translate(s.offset.x, s.offset.y);
    ctx.scale(sx, sy);
    ctx.font = `700 ${size.toFixed(1)}px 'Pretendard', 'Noto Sans KR', sans-serif`;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillStyle = ink(clamp(0.4 + alpha * 0.6, 0, 0.98));
    ctx.fillText(vowel, 0, 0);
    ctx.restore();
  }

  /** 두 인접 로브의 교차 렌즈 "중심선"(현) 끝점 + 살짝 부푼 제어점(로컬). 안 겹치면 null. */
  private membraneChord(
    a: BubbleSegment,
    b: BubbleSegment,
  ): { x1: number; y1: number; x2: number; y2: number; cx: number; cy: number } | null {
    const dx = b.offset.x - a.offset.x;
    const dy = b.offset.y - a.offset.y;
    const d = Math.hypot(dx, dy);
    if (d < 1e-3) return null;
    const ang = Math.atan2(dy, dx);
    const rA = effectiveRadius(a.radius.value, a.aspect.value, ang);
    const rB = effectiveRadius(b.radius.value, b.aspect.value, ang);
    if (d >= rA + rB) return null;
    if (d <= Math.abs(rA - rB)) return null;
    const aDist = (d * d + rA * rA - rB * rB) / (2 * d);
    const h2 = rA * rA - aDist * aDist;
    if (h2 <= 0) return null;
    const h = Math.sqrt(h2);
    const ux = dx / d;
    const uy = dy / d;
    const px = a.offset.x + ux * aDist;
    const py = a.offset.y + uy * aDist;
    const nx = -uy;
    const ny = ux;
    const bulge = h * 0.18 * (rA >= rB ? 1 : -1);
    return {
      x1: px + nx * h,
      y1: py + ny * h,
      x2: px - nx * h,
      y2: py - ny * h,
      cx: px + ux * bulge,
      cy: py + uy * bulge,
    };
  }

  /** 인식 불가(bumpy): 울퉁불퉁한 윤곽의 방울(로컬 좌표, 로브별 독립). */
  private drawBumpy(ctx: CanvasRenderingContext2D, b: Bubble, bodyAlpha: number): void {
    for (const s of b.segments) {
      const r = s.radius.value;
      if (r <= 0) continue;
      const { sx, sy } = aspectToScale(s.aspect.value);
      ctx.save();
      ctx.translate(s.offset.x, s.offset.y);
      ctx.scale(sx, sy);
      const steps = 40;
      ctx.beginPath();
      for (let k = 0; k <= steps; k++) {
        const a = (k / steps) * TWO_PI;
        const wob =
          1 +
          0.14 * Math.sin(3 * a + b.bumpySeed) +
          0.09 * Math.sin(5 * a + b.bumpySeed * 1.7) +
          0.05 * Math.sin(8 * a + b.bumpySeed * 0.3);
        const rr = r * wob;
        const x = Math.cos(a) * rr;
        const y = Math.sin(a) * rr;
        if (k === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.closePath();
      ctx.fillStyle = rgba(FILM_EDGE, 0.22 * bodyAlpha);
      ctx.fill();
      ctx.lineWidth = 1.8;
      ctx.strokeStyle = rgba(FILM_RIM, 0.6 * bodyAlpha);
      ctx.stroke();
      ctx.restore();
    }
  }

  // -------------------------------------------------------------------
  // 입자 / 쌓인 글자 (메인 좌표계 그대로)
  // -------------------------------------------------------------------
  private drawParticles(b: Bubble): void {
    if (b.particles.length === 0) return;
    const ctx = this.ctx;
    for (const p of b.particles) {
      const lifeT = clamp(p.life / p.maxLife, 0, 1);
      switch (p.kind) {
        case 'burst-ring': {
          const grow = 1 + (1 - lifeT) * 1.8;
          ctx.beginPath();
          ctx.arc(p.x, p.y, Math.max(0.1, p.size * grow), 0, TWO_PI);
          ctx.lineWidth = 1 + 2 * lifeT;
          ctx.strokeStyle = rgba(FILM_RIM, 0.55 * smoothstep(lifeT));
          ctx.stroke();
          break;
        }
        case 'residual': {
          ctx.beginPath();
          ctx.arc(p.x, p.y, p.size, 0, TWO_PI);
          ctx.fillStyle = rgba(FILM_MID, 0.6 * lifeT);
          ctx.fill();
          ctx.beginPath();
          ctx.arc(p.x - p.size * 0.3, p.y - p.size * 0.3, p.size * 0.35, 0, TWO_PI);
          ctx.fillStyle = rgba(FILM_INNER, 0.7 * lifeT);
          ctx.fill();
          break;
        }
        case 'glyph-debris': {
          ctx.save();
          ctx.translate(p.x, p.y);
          ctx.rotate(p.rot);
          ctx.font = `700 ${p.size.toFixed(1)}px 'Pretendard', 'Noto Sans KR', sans-serif`;
          ctx.textAlign = 'center';
          ctx.textBaseline = 'middle';
          ctx.fillStyle = ink(0.75 * lifeT);
          ctx.fillText(p.text ?? '', 0, 0);
          ctx.restore();
          break;
        }
      }
    }
  }

  /** 하단에 쌓인 글자(B1). */
  private drawStacked(glyphs: readonly StackedGlyph[]): void {
    if (glyphs.length === 0) return;
    const ctx = this.ctx;
    for (const g of glyphs) {
      ctx.save();
      ctx.translate(g.x, g.y);
      ctx.scale(g.sx, g.sy);
      ctx.font = `700 ${g.size.toFixed(1)}px 'Pretendard', 'Noto Sans KR', sans-serif`;
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillStyle = ink(clamp(0.8 * g.alpha, 0, 0.95));
      ctx.fillText(g.text, 0, 0);
      ctx.restore();
    }
  }
}
