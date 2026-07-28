import type { Dimensions } from '../types/common';
import { popSettings } from '../config/popSettings';
import { lerp } from '../animation/mathUtils';

/**
 * 하단에 쌓인 글자 하나. (B1)
 */
export interface StackedGlyph {
  x: number;
  y: number; // 현재 y(스프링 없이 targetY로 부드럽게 수렴)
  targetY: number;
  text: string;
  size: number;
  sx: number; // 종횡비(세그먼트에서 물려받음)
  sy: number;
  age: number; // 착지 후 경과(초)
  alpha: number; // 사라질 때 페이드
  fading: boolean;
}

/**
 * GlyphStack (B1)
 * ===============
 * 터진/무너진 버블의 글자 중 일부가 하단에 쌓이고, 오래된(아래층) 것부터
 * 테트리스처럼 사라진다. 열(column) 단위로 높이를 관리한다.
 *
 * "일정 시간(stackClearAfterS) 도달 시 아래층부터 소거"로 구현한다.
 * (일정 높이 기준도 원하면 확장 지점: 열 높이가 임계 넘으면 바닥층 제거.)
 */
export class GlyphStack {
  private cols = new Map<number, StackedGlyph[]>(); // colIndex → 바닥..위 순
  private dims: Dimensions = { width: 0, height: 0 };
  private readonly colW = 52;

  setDimensions(dims: Dimensions): void {
    this.dims = dims;
  }

  clear(): void {
    this.cols.clear();
  }

  /** 현재 쌓인 글자들(렌더용, 평탄화). */
  getAll(): StackedGlyph[] {
    const out: StackedGlyph[] = [];
    for (const arr of this.cols.values()) out.push(...arr);
    return out;
  }

  /** 특정 x 위치의 현재 쌓임 높이(px). 반사(B2)가 이 위에서 튕기도록. */
  heightAt(x: number): number {
    const col = this.colOf(x);
    const arr = this.cols.get(col);
    if (!arr || arr.length === 0) return 0;
    let h = 0;
    for (const g of arr) h += g.size;
    return h;
  }

  /** 전체에서 가장 높은 쌓임(px). B2 하단 반사선 계산용. */
  maxHeight(): number {
    let m = 0;
    for (const arr of this.cols.values()) {
      let h = 0;
      for (const g of arr) h += g.size;
      if (h > m) m = h;
    }
    return m;
  }

  /** 글자 하나를 이 x 열의 맨 위에 착지시킨다(fromY 에서 떨어지듯 내려온다). */
  land(x: number, fromY: number, text: string, size: number, sx: number, sy: number): void {
    const col = this.colOf(x);
    const arr = this.cols.get(col) ?? [];
    this.cols.set(col, arr);
    arr.push({
      x: col * this.colW + this.colW / 2,
      y: fromY,
      targetY: 0,
      text,
      size,
      sx,
      sy,
      age: 0,
      alpha: 1,
      fading: false,
    });
    this.relayout(col);
  }

  update(dt: number): void {
    const clearAfter = popSettings.stackClearAfterS;
    for (const [col, arr] of this.cols) {
      // 바닥층(가장 오래된)이 수명 초과 → 페이드 후 제거.
      const bottom = arr[0];
      if (bottom) {
        bottom.age += dt;
        if (!bottom.fading && bottom.age > clearAfter) bottom.fading = true;
        if (bottom.fading) {
          bottom.alpha -= dt / 0.4; // 0.4초 페이드
          if (bottom.alpha <= 0) {
            arr.shift();
            this.relayout(col);
          }
        }
      }
      // 나머지 글자 age 누적 + y 수렴.
      for (const g of arr) {
        if (g !== bottom) g.age += dt;
        g.y = lerp(g.y, g.targetY, Math.min(1, dt * 10)); // 부드럽게 착지/하강
      }
      if (arr.length === 0) this.cols.delete(col);
    }
  }

  /** 열의 targetY 를 바닥부터 다시 쌓아 계산. */
  private relayout(col: number): void {
    const arr = this.cols.get(col);
    if (!arr) return;
    let stack = 0;
    for (const g of arr) {
      stack += g.size;
      g.targetY = this.dims.height - stack + g.size / 2; // 바닥에서 위로
    }
  }

  private colOf(x: number): number {
    return Math.floor(x / this.colW);
  }
}
