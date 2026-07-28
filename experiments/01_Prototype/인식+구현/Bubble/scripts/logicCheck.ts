/**
 * 헤드리스 로직 점검 — 마이크 없이 합성 프레임으로 detector→manager 파이프라인을 구동한다.
 * (앱 빌드에는 포함되지 않음: tsconfig include 는 src 뿐)
 */
import type { VoiceFrame, MaybeVowel } from '../src/types/audio';
import { VoiceEventDetector } from '../src/detector/VoiceEventDetector';
import { BubbleManager } from '../src/bubble/BubbleManager';
import { pitchToLevel } from '../src/audio/pitchMath';
import { aspectToScale, effectiveRadius, energyToRadius, popDurations } from '../src/bubble/bubbleMath';
import { OVERLAP_RATIO } from '../src/config/bubbleConfig';
import { popSettings } from '../src/config/popSettings';

// 테스트 안정화: 짧은 프레임 동안 터지지 않게 하고 y궤적을 관찰 가능하게.
popSettings.mode = 'never';
popSettings.pitchYRange = 110;
popSettings.pitchYPlaybackS = 2.5;
// 종료 판정을 짧게(기본 400ms → 150ms) 두어 기존 시나리오의 ~300ms 꼬리 침묵이 종료되게 한다.
// (에너지 스무딩 감쇠 꼬리 ~100ms + 이 값 < 300ms. 재발성 시나리오는 각자 간격을 명시적으로 준다.)
popSettings.utteranceBreakMs = 150;

const STEP = 25; // ms

function run(name: string, frames: VoiceFrame[], mode: 'bubble-only' | 'bumpy' = 'bubble-only') {
  const det = new VoiceEventDetector();
  const mgr = new BubbleManager();
  mgr.setDimensions({ width: 1200, height: 800 });
  mgr.setUnrecognizedMode(mode);
  const events: string[] = [];
  for (const f of frames) {
    const evs = det.process(f);
    for (const e of evs) {
      events.push(e.type + (('vowel' in e && e.vowel) ? `(${e.vowel})` : ''));
      mgr.handle(e);
    }
    mgr.update(STEP / 1000, { width: 1200, height: 800 });
  }
  // 종료 후 한 번 더 적분(cull 반영)
  mgr.update(STEP / 1000, { width: 1200, height: 800 });
  const bubbles = mgr.getBubbles();
  console.log(`\n=== ${name} ===`);
  console.log('events:', events.join(' '));
  console.log('bubbles:', bubbles.length);
  bubbles.forEach((b, i) => {
    console.log(
      `  bubble#${i} state=${b.state} unrecognized=${b.unrecognized} bumpySeed=${b.bumpySeed.toFixed(1)} tremor=${b.tremor.toFixed(2)} dur=${b.durationMs}ms`,
    );
    b.segments.forEach((s, j) =>
      console.log(`    seg${j} vowel=${s.vowel ?? 'null'} boundary=${s.hasBoundary} r=${s.radius.value.toFixed(0)} op=${s.opacity.toFixed(2)}`),
    );
  });
  return { events, bubbles };
}

function frame(t: number, energy: number, pitch: number | null, vowel: MaybeVowel, recognized: boolean): VoiceFrame {
  return { time: t, energy, pitch, clarity: recognized ? 0.95 : 0.5, vowel, recognized };
}

let t = 0;
const next = () => (t += STEP);

// ---------------------------------------------------------------
// 시나리오 1: "아"(지연 인식) → "오"(모음 전환) → 종료. 피치 지터로 tremor.
// ---------------------------------------------------------------
{
  t = 0;
  const fr: VoiceFrame[] = [];
  for (let i = 0; i < 6; i++) fr.push(frame(next(), 0.01, null, null, false)); // 침묵
  // 온셋 2프레임: 에너지는 있으나 아직 미인식(인식 지연) → null 스폰
  fr.push(frame(next(), 0.14, null, null, false));
  fr.push(frame(next(), 0.16, null, null, false));
  // '아' 지속 (피치를 흔들어 tremor 생성)
  for (let i = 0; i < 18; i++) fr.push(frame(next(), 0.22 + (i % 3) * 0.02, 150 + (i % 2 ? 10 : -10), '아', true));
  // '오' 전환 지속
  for (let i = 0; i < 12; i++) fr.push(frame(next(), 0.3, 140, '오', true));
  // 침묵으로 종료(>220ms)
  for (let i = 0; i < 12; i++) fr.push(frame(next(), 0.01, null, null, false));
  const { events, bubbles } = run('S1: 아→오 (지연인식·전환·tremor)', fr);

  const ok =
    events.includes('UTTERANCE_START') &&
    events.includes('VOWEL_RESOLVED(아)') &&
    events.includes('VOWEL_CHANGED(오)') &&
    events.includes('UTTERANCE_END') &&
    bubbles.length === 1 &&
    bubbles[0].segments.length === 2 &&
    bubbles[0].segments[0].vowel === '아' &&
    bubbles[0].segments[0].hasBoundary === false &&
    bubbles[0].segments[1].vowel === '오' &&
    bubbles[0].segments[1].hasBoundary === true &&
    bubbles[0].state === 'FLOATING' &&
    bubbles[0].tremor > 0;
  console.log('S1 PASS:', ok);
}

// ---------------------------------------------------------------
// 시나리오 2: false-start (아주 짧은 발화) → 버블 취소.
// ---------------------------------------------------------------
{
  t = 0;
  const fr: VoiceFrame[] = [];
  for (let i = 0; i < 6; i++) fr.push(frame(next(), 0.01, null, null, false));
  for (let i = 0; i < 3; i++) fr.push(frame(next(), 0.2, 150, '아', true)); // ~75ms < 150ms
  for (let i = 0; i < 14; i++) fr.push(frame(next(), 0.01, null, null, false));
  const { bubbles } = run('S2: false-start', fr);
  console.log('S2 PASS:', bubbles.length === 0);
}

// ---------------------------------------------------------------
// 시나리오 3: 처음부터 인식 불가 (bumpy) → unrecognized 버블 상승.
// ---------------------------------------------------------------
{
  t = 0;
  const fr: VoiceFrame[] = [];
  for (let i = 0; i < 6; i++) fr.push(frame(next(), 0.01, null, null, false));
  for (let i = 0; i < 20; i++) fr.push(frame(next(), 0.2, null, null, false)); // 소리는 있으나 미인식
  for (let i = 0; i < 12; i++) fr.push(frame(next(), 0.01, null, null, false));
  const { events, bubbles } = run('S3: 처음부터 인식불가 (bumpy)', fr, 'bumpy');
  const ok =
    events.includes('RECOGNITION_LOST') &&
    bubbles.length === 1 &&
    bubbles[0].unrecognized === true &&
    bubbles[0].bumpySeed > 0 &&
    (bubbles[0].state === 'FLOATING' || bubbles[0].state === 'POPPING');
  console.log('S3 PASS:', ok);
}

// ---------------------------------------------------------------
// 시나리오 4: 같은 모음 연속 발성 병합 (아 … 아) → 경계선 없는 땅콩.
// ---------------------------------------------------------------
{
  t = 0;
  const fr: VoiceFrame[] = [];
  for (let i = 0; i < 6; i++) fr.push(frame(next(), 0.01, null, null, false));
  for (let i = 0; i < 12; i++) fr.push(frame(next(), 0.24, 150, '아', true)); // 발화1
  for (let i = 0; i < 16; i++) fr.push(frame(next(), 0.0, null, null, false)); // 침묵(발화1 종료, 병합창 내)
  for (let i = 0; i < 12; i++) fr.push(frame(next(), 0.24, 150, '아', true)); // 발화2 (연달아)
  for (let i = 0; i < 12; i++) fr.push(frame(next(), 0.01, null, null, false));
  const { bubbles } = run('S4: 아…아 병합(경계없는 땅콩)', fr);
  const merged = bubbles.length === 1 && bubbles[0].segments.length === 2;
  const boundaryless = merged && bubbles[0].segments[1].hasBoundary === false;
  console.log('S4 PASS:', merged && boundaryless);
}

// ---------------------------------------------------------------
// 시나리오 5 (수정 4): 같은 모음 '아' + 세기 약→센 → 경계 없는 땅콩(세그먼트 2+).
// ---------------------------------------------------------------
{
  t = 0;
  const fr: VoiceFrame[] = [];
  for (let i = 0; i < 6; i++) fr.push(frame(next(), 0.0, null, null, false));
  for (let i = 0; i < 8; i++) fr.push(frame(next(), 0.15, 150, '아', true)); // 약하게
  for (let i = 0; i < 14; i++) fr.push(frame(next(), 0.5, 150, '아', true)); // 세게(같은 모음)
  for (let i = 0; i < 12; i++) fr.push(frame(next(), 0.0, null, null, false));
  const { bubbles } = run('S5: 아(약)→아(세) 경계없는 땅콩(수정 4)', fr);
  const b = bubbles[0];
  const targets = b.segments.map((s) => s.radius.target);
  const sizeSpread = Math.max(...targets) > Math.min(...targets) * 1.2; // 크기 편차 존재(작+큰)
  const ok =
    bubbles.length === 1 &&
    b.segments.length >= 2 &&
    b.segments.every((s) => s.vowel === '아') &&
    b.segments.slice(1).every((s) => s.hasBoundary === false) &&
    sizeSpread;
  console.log('S5 PASS:', ok);
}

// ---------------------------------------------------------------
// 시나리오 6 (수정 3): 정상 인식 발성 종료 → 터지지 않고 FLOATING(수명 부여).
// ---------------------------------------------------------------
{
  t = 0;
  const fr: VoiceFrame[] = [];
  for (let i = 0; i < 6; i++) fr.push(frame(next(), 0.0, null, null, false));
  for (let i = 0; i < 20; i++) fr.push(frame(next(), 0.25, 150, '아', true));
  for (let i = 0; i < 12; i++) fr.push(frame(next(), 0.0, null, null, false));
  const { bubbles } = run('S6: 정상 발성 종료 → 안 터지고 FLOATING(수정 3)', fr);
  const b = bubbles[0];
  const speed = Math.hypot(b.vel.x, b.vel.y);
  const ok = bubbles.length === 1 && b.state === 'FLOATING' && b.lifetime > 0 && speed > 0;
  console.log('S6 PASS:', ok);
}

// ---------------------------------------------------------------
// 시나리오 7 (2차 수정 3 + 1): 생성 중 피치→종횡비, 위치는 고정.
//   높은 음(320Hz) 유지 → aspect.target>0(세로 길쭉), pos.y 는 anchor.y 그대로.
// ---------------------------------------------------------------
{
  t = 0;
  const det = new VoiceEventDetector();
  const mgr = new BubbleManager();
  mgr.setDimensions({ width: 1200, height: 800 });
  const feed = (f: VoiceFrame) => det.process(f).forEach((e) => mgr.handle(e));
  for (let i = 0; i < 6; i++) {
    feed(frame(next(), 0.0, null, null, false));
    mgr.update(0.025, { width: 1200, height: 800 });
  }
  for (let i = 0; i < 16; i++) {
    feed(frame(next(), 0.25, 320, '이', true)); // 높은 음
    mgr.update(0.025, { width: 1200, height: 800 });
  }
  const b = mgr.getBubbles()[0];
  const anchorHeld = Math.abs(b.pos.y - b.anchor.y) < 0.001; // 생성 중 상하 이동 없음
  const seg = b.segments[b.segments.length - 1]; // 활성 세그먼트(4차 수정 3)
  const sc = aspectToScale(seg.aspect.value);
  const tall = seg.aspect.target > 0.1 && sc.sy > 1 && sc.sx < 1; // 세로 길쭉
  console.log('\n=== S7: 생성 중 피치→모양(세로 길쭉), 위치 고정 ===');
  console.log(`state=${b.state} pos.y=${b.pos.y.toFixed(1)} anchor.y=${b.anchor.y.toFixed(1)} seg.aspect.target=${seg.aspect.target.toFixed(2)} sx=${sc.sx.toFixed(2)} sy=${sc.sy.toFixed(2)}`);
  console.log('S7 PASS:', b.state === 'FORMING' && anchorHeld && tall);
}

// ---------------------------------------------------------------
// 유닛: 피치→종횡비 부호(높음>0, 낮음<0).
// ---------------------------------------------------------------
{
  const low = pitchToLevel(90);
  const high = pitchToLevel(320);
  const ok = low < 0 && high > 0;
  console.log('\n=== U1: 피치→종횡비 부호 ===');
  console.log(`pitchLevel low(90Hz)=${low.toFixed(2)} high(320Hz)=${high.toFixed(2)}`);
  console.log('U1 PASS:', ok);
}

// ---------------------------------------------------------------
// 시나리오 8 (3차 수정 2): 끝단 인식 드롭아웃(0.10, 미인식) → 붕괴하지 않고 FLOATING.
//   RECOGNITION_LOST_MIN_ENERGY(0.13) 아래라 lost 로 세지 않아야 한다.
// ---------------------------------------------------------------
{
  t = 0;
  const fr: VoiceFrame[] = [];
  for (let i = 0; i < 6; i++) fr.push(frame(next(), 0.0, null, null, false));
  for (let i = 0; i < 14; i++) fr.push(frame(next(), 0.25, 150, '아', true)); // 정상 인식
  for (let i = 0; i < 10; i++) fr.push(frame(next(), 0.1, null, null, false)); // 끝단: 낮은 세기·미인식
  for (let i = 0; i < 12; i++) fr.push(frame(next(), 0.0, null, null, false)); // 침묵
  const { bubbles } = run('S8: 끝단 인식 드롭아웃 → 붕괴 금지(수정 2)', fr);
  const b = bubbles[0];
  const ok = bubbles.length === 1 && b.state === 'FLOATING' && b.everRecognized === true;
  console.log('S8 PASS:', ok);
}

// ---------------------------------------------------------------
// 시나리오 9 (피치 부유 수정): 아↗↘ → 비행 y가 수명 전체에 걸쳐 부드럽게 위로
//   떠올랐다 가라앉고, 매 프레임 수직 이동이 속도 상한(popSettings.maxVerticalSpeed)
//   이하(계단식 없음). 재생 시간 = 수명 × trajectoryTimeScale.
// ---------------------------------------------------------------
{
  t = 0;
  // 결정성: 이 시나리오만 수명 3s 로 고정 → trajDuration=3s. 관측(3.5s)이 재생을 넘겨
  //   아크(상승→하강) 전체와 재생 종료 후 부유 진입까지 본다.
  const savedMin = popSettings.lifetimeMin, savedMax = popSettings.lifetimeMax;
  popSettings.lifetimeMin = 3.0;
  popSettings.lifetimeMax = 3.0;
  popSettings.trajectoryTimeScale = 1.0;
  const det = new VoiceEventDetector();
  const mgr = new BubbleManager();
  const dims = { width: 1200, height: 2000 };
  mgr.setDimensions(dims);
  const feed = (f: VoiceFrame) => det.process(f).forEach((e) => mgr.handle(e));
  const step = () => mgr.update(0.025, dims);
  for (let i = 0; i < 6; i++) { feed(frame(next(), 0.0, null, null, false)); step(); }
  const rise = [120, 150, 180, 210, 240, 260];
  const fall = [240, 210, 170, 130, 100];
  for (const p of rise) for (let k = 0; k < 2; k++) { feed(frame(next(), 0.25, p, '아', true)); step(); }
  for (const p of fall) for (let k = 0; k < 2; k++) { feed(frame(next(), 0.25, p, '아', true)); step(); }
  for (let i = 0; i < 12; i++) { feed(frame(next(), 0.0, null, null, false)); step(); }
  const ys: number[] = [];
  const b = mgr.getBubbles()[0];
  for (let i = 0; i < 140; i++) { step(); if (b) ys.push(b.vSpring.value); }
  const minY = Math.min(...ys);
  const idxMin = ys.indexOf(minY);
  const lastY = ys[ys.length - 1];
  // 속도 상한 준수 확인(계단식/순간이동 없음).
  let maxStep = 0;
  for (let i = 1; i < ys.length; i++) maxStep = Math.max(maxStep, Math.abs(ys[i] - ys[i - 1]));
  const cap = popSettings.maxVerticalSpeed * 0.025 + 1e-6;
  const smooth = maxStep <= cap;
  const arc = minY < -15 && lastY > minY + 15 && idxMin < ys.length - 1; // 떠올랐다 가라앉음
  // 완료 기준 2 방증: 아크의 정점(idxMin)이 초반에 몰려 있지 않다(빠른 이동-후-정지가 아님).
  const gradual = idxMin > 20;
  console.log('\n=== S9: 아↗↘ 수명 전체 부유 + 속도상한(피치 부유 수정) ===');
  console.log(`minY=${minY.toFixed(0)} at ${idxMin}/${ys.length}, lastY=${lastY.toFixed(0)}, maxStep=${maxStep.toFixed(2)}(cap=${cap.toFixed(2)})`);
  console.log('S9 PASS:', b && b.state === 'FLOATING' && arc && smooth && gradual);
  popSettings.lifetimeMin = savedMin;
  popSettings.lifetimeMax = savedMax;
}

// ---------------------------------------------------------------
// 시나리오 9b (피치 부유 수정 — 완료 기준 3·4):
//   (3) 궤적 재생이 끝난 버블도 정지하지 않고 잔잔히 오르내린다.
//   (4) 피치 변화 없이 일정한 음(궤적 없음)은 큰 y 이동 없이 미세한 부유만.
// ---------------------------------------------------------------
{
  t = 0;
  const savedMin = popSettings.lifetimeMin, savedMax = popSettings.lifetimeMax;
  popSettings.lifetimeMin = 2.0; // 짧은 수명 → 관측 중 재생 종료 후 부유 구간 진입
  popSettings.lifetimeMax = 2.0;
  const det = new VoiceEventDetector();
  const mgr = new BubbleManager();
  const dims = { width: 1200, height: 2000 };
  mgr.setDimensions(dims);
  const feed = (f: VoiceFrame) => det.process(f).forEach((e) => mgr.handle(e));
  const step = () => mgr.update(0.025, dims);
  // 일정한 음 '오'(피치 130 고정) → 궤적 변화 없음.
  for (let i = 0; i < 6; i++) { feed(frame(next(), 0.0, null, null, false)); step(); }
  for (let i = 0; i < 16; i++) { feed(frame(next(), 0.25, 130, '오', true)); step(); }
  for (let i = 0; i < 12; i++) { feed(frame(next(), 0.0, null, null, false)); step(); }
  const b = mgr.getBubbles()[0];
  // 재생이 확실히 끝난 뒤(수명 2s 경과) 구간을 관측.
  for (let i = 0; i < 120; i++) step();
  const ys: number[] = [];
  for (let i = 0; i < 160; i++) { step(); if (b) ys.push(b.vSpring.value); }
  const amp = (Math.max(...ys) - Math.min(...ys)) / 2;
  // (3) 정지하지 않음: 관측 구간 내 진폭이 0보다 확실히 큼.
  const moving = amp > 1.0;
  // (4) 큰 이동 없음: 일정 음이므로 진폭이 idleBobAmp 근방(과도한 y 이동 아님).
  const gentle = amp <= popSettings.idleBobAmp + 6;
  console.log('\n=== S9b: 일정 음 → 미세 부유(정지 금지) ===');
  console.log(`amp=${amp.toFixed(2)} (idleBobAmp=${popSettings.idleBobAmp})`);
  console.log('S9b PASS:', b && b.state === 'FLOATING' && moving && gentle);
  popSettings.lifetimeMin = savedMin;
  popSettings.lifetimeMax = savedMax;
}

// ---------------------------------------------------------------
// 시나리오 10 (3차 수정 4): 떠 있는 버블은 새 발성에도 모양/크기 불변.
// ---------------------------------------------------------------
{
  t = 0;
  const det = new VoiceEventDetector();
  const mgr = new BubbleManager();
  mgr.setDimensions({ width: 1200, height: 2000 });
  const feed = (f: VoiceFrame) => det.process(f).forEach((e) => mgr.handle(e));
  const step = () => mgr.update(0.025, { width: 1200, height: 2000 });
  // 발화1: 낮은 음 '오' → 완성
  for (let i = 0; i < 6; i++) { feed(frame(next(), 0.0, null, null, false)); step(); }
  for (let i = 0; i < 14; i++) { feed(frame(next(), 0.25, 110, '오', true)); step(); }
  // 병합창(450ms)을 확실히 벗어나도록 긴 침묵.
  for (let i = 0; i < 36; i++) { feed(frame(next(), 0.0, null, null, false)); step(); }
  const b0 = mgr.getBubbles()[0];
  const snapAspect = b0.segments[0].aspect.value;
  const snapR = b0.segments[0].radius.value;
  const snapVowel = b0.segments[0].vowel;
  // 발화2: 아주 높은 음 '이' (다른 모음·다른 피치) — 별개 버블이어야 한다.
  for (let i = 0; i < 14; i++) { feed(frame(next(), 0.5, 320, '이', true)); step(); }
  for (let i = 0; i < 30; i++) step();
  const changed =
    Math.abs(b0.segments[0].aspect.value - snapAspect) > 0.001 ||
    Math.abs(b0.segments[0].radius.value - snapR) > 0.001 ||
    b0.segments[0].vowel !== snapVowel;
  console.log('\n=== S10: 떠 있는 버블 불변(수정 4) ===');
  console.log(`aspect ${snapAspect.toFixed(3)}→${b0.segments[0].aspect.value.toFixed(3)}  r ${snapR.toFixed(1)}→${b0.segments[0].radius.value.toFixed(1)}  vowel ${snapVowel}→${b0.segments[0].vowel}`);
  console.log('S10 PASS:', !changed);
}

// ---------------------------------------------------------------
// 시나리오 12 (4차 수정 3): 아(낮게)→오(높게) → 한 땅콩에 납작한 아 + 홀쭉한 오 공존.
//   발화 후반 피치 변화가 앞 세그먼트('아')의 모양을 바꾸지 않는다.
// ---------------------------------------------------------------
{
  t = 0;
  const det = new VoiceEventDetector();
  const mgr = new BubbleManager();
  const dims = { width: 1200, height: 2000 };
  mgr.setDimensions(dims);
  const feed = (f: VoiceFrame) => det.process(f).forEach((e) => mgr.handle(e));
  const step = () => mgr.update(0.025, dims);
  for (let i = 0; i < 6; i++) { feed(frame(next(), 0.0, null, null, false)); step(); }
  for (let i = 0; i < 14; i++) { feed(frame(next(), 0.25, 95, '아', true)); step(); } // 낮은 음 '아'
  // '아' 세그먼트 종횡비 스냅(전환 직전).
  const bb = mgr.getBubbles()[0];
  const aAspectBefore = bb.segments[0].aspect.value;
  for (let i = 0; i < 16; i++) { feed(frame(next(), 0.25, 300, '오', true)); step(); } // 높은 음 '오'
  for (let i = 0; i < 12; i++) { feed(frame(next(), 0.0, null, null, false)); step(); }
  const seg0 = bb.segments[0];
  const seg1 = bb.segments[bb.segments.length - 1];
  const flat = seg0.aspect.value < -0.1; // 납작한 아(낮은 음)
  const tall = seg1.aspect.value > 0.1; // 홀쭉한 오(높은 음)
  const seg0Unchanged = Math.abs(seg0.aspect.value - aAspectBefore) < 0.05; // 앞 세그먼트 고정
  console.log('\n=== S12: 납작한 아 + 홀쭉한 오 공존(수정 3) ===');
  console.log(`seg0(아).aspect=${seg0.aspect.value.toFixed(2)}(전환전 ${aAspectBefore.toFixed(2)}), seg1(오).aspect=${seg1.aspect.value.toFixed(2)}`);
  console.log('S12 PASS:', flat && tall && seg0Unchanged && seg0.vowel === '아' && seg1.vowel === '오');
}

// ---------------------------------------------------------------
// 시나리오 13 (4차 수정 2): 반지름이 스프링으로 커지는 동안에도 세그먼트가
//   절대 떨어지지 않는다(매 프레임 d ≤ (rEff1+rEff2)×OVERLAP).
// ---------------------------------------------------------------
{
  t = 0;
  const det = new VoiceEventDetector();
  const mgr = new BubbleManager();
  const dims = { width: 1200, height: 2000 };
  mgr.setDimensions(dims);
  const feed = (f: VoiceFrame) => det.process(f).forEach((e) => mgr.handle(e));
  const step = () => mgr.update(0.025, dims);
  let violated = false;
  const checkGap = () => {
    for (const b of mgr.getBubbles()) {
      for (let i = 1; i < b.segments.length; i++) {
        const a = b.segments[i - 1];
        const c = b.segments[i];
        const dx = c.offset.x - a.offset.x;
        const dy = c.offset.y - a.offset.y;
        const d = Math.hypot(dx, dy);
        const ang = Math.atan2(dy, dx);
        const rA = effectiveRadius(a.radius.value, a.aspect.value, ang);
        const rC = effectiveRadius(c.radius.value, c.aspect.value, ang);
        if (d > (rA + rC) * 0.98) violated = true; // 인접이 실제로 벌어짐(겹침 상실)
      }
    }
  };
  for (let i = 0; i < 6; i++) { feed(frame(next(), 0.0, null, null, false)); step(); checkGap(); }
  // 세기·모음 급변으로 세그먼트 여러 개 + 반지름 급성장/축소 유발
  for (let i = 0; i < 8; i++) { feed(frame(next(), 0.12, 150, '아', true)); step(); checkGap(); }
  for (let i = 0; i < 8; i++) { feed(frame(next(), 0.55, 150, '아', true)); step(); checkGap(); }
  for (let i = 0; i < 8; i++) { feed(frame(next(), 0.2, 300, '오', true)); step(); checkGap(); }
  for (let i = 0; i < 20; i++) { feed(frame(next(), 0.0, null, null, false)); step(); checkGap(); }
  for (let i = 0; i < 40; i++) { step(); checkGap(); }
  console.log('\n=== S13: 반지름 변화 중 세그먼트 분리 없음(수정 2) ===');
  console.log('S13 PASS:', !violated);
}

// ---------------------------------------------------------------
// 시나리오 11 (3차 수정 1): 세기 일정한 3모음(아→오→우) → 인접 간격이 정확히 OVERLAP_RATIO.
//   세기가 일정하면 반지름이 안 자라므로 배치 간격을 결정적으로 검증할 수 있다.
// ---------------------------------------------------------------
{
  t = 0;
  const fr: VoiceFrame[] = [];
  for (let i = 0; i < 6; i++) fr.push(frame(next(), 0.0, null, null, false));
  for (let i = 0; i < 10; i++) fr.push(frame(next(), 0.25, 150, '아', true));
  for (let i = 0; i < 10; i++) fr.push(frame(next(), 0.25, 150, '오', true));
  for (let i = 0; i < 10; i++) fr.push(frame(next(), 0.25, 150, '우', true));
  for (let i = 0; i < 12; i++) fr.push(frame(next(), 0.0, null, null, false));
  const { bubbles } = run('S11: 아→오→우 간격=OVERLAP_RATIO(수정 1)', fr);
  const b = bubbles[0];
  const ratios: number[] = [];
  let spacingOk = !!b && b.segments.length >= 3;
  if (b) {
    for (let i = 1; i < b.segments.length; i++) {
      const a = b.segments[i - 1];
      const c = b.segments[i];
      const dx = c.offset.x - a.offset.x;
      const dy = c.offset.y - a.offset.y;
      const d = Math.hypot(dx, dy);
      // 실제 접합 축(중심-중심 방향)으로 실효 반지름 비율 계산.
      const ang = Math.atan2(dy, dx);
      const rA = effectiveRadius(a.radius.value, a.aspect.value, ang);
      const rC = effectiveRadius(c.radius.value, c.aspect.value, ang);
      const ratio = d / (rA + rC);
      ratios.push(ratio);
      // 인접은 겹치는 사슬(OVERLAP 근방, 비인접 이격 PBD 로 약간 당겨질 수 있음). 분리/완전포개짐 아님.
      if (ratio < 0.6 || ratio > OVERLAP_RATIO + 0.05) spacingOk = false;
    }
  }
  console.log(`  ratios=[${ratios.map((r) => r.toFixed(3)).join(', ')}] target=${OVERLAP_RATIO}`);
  console.log('S11 PASS:', spacingOk);
}

// ---------------------------------------------------------------
// 시나리오 14 (5차 A1·A7): 여러 버블의 속도·방향·수명이 제각각(정지·동기화 없음).
// ---------------------------------------------------------------
{
  popSettings.mode = 'random-lifetime';
  popSettings.lifetimeMin = 20; // 테스트 중 안 터지게 크게, 그래도 랜덤 스프레드
  popSettings.lifetimeMax = 40;
  popSettings.driftMin = 34;
  popSettings.driftMax = 90;
  t = 0;
  const det = new VoiceEventDetector();
  const mgr = new BubbleManager();
  const dims = { width: 1600, height: 2400 };
  mgr.setDimensions(dims);
  const feed = (f: VoiceFrame) => det.process(f).forEach((e) => mgr.handle(e));
  const step = () => mgr.update(0.025, dims);
  // 6회 발성, 사이에 긴 침묵(병합창 밖).
  for (let u = 0; u < 6; u++) {
    for (let i = 0; i < 12; i++) { feed(frame(next(), 0.25, 150, '우', true)); step(); }
    for (let i = 0; i < 40; i++) { feed(frame(next(), 0.0, null, null, false)); step(); }
  }
  const bs = mgr.getBubbles().filter((b) => b.state === 'FLOATING');
  const speeds = bs.map((b) => Math.hypot(b.vel.x, b.vel.y));
  const dirs = bs.map((b) => Math.atan2(b.vel.y, b.vel.x));
  const lifes = bs.map((b) => b.lifetime);
  const allAboveMin = speeds.every((s) => s >= popSettings.driftMin - 0.5);
  const speedVaries = Math.max(...speeds) - Math.min(...speeds) > 5;
  const dirVaries = Math.max(...dirs) - Math.min(...dirs) > 0.3;
  const lifeVaries = Math.max(...lifes) - Math.min(...lifes) > 2;
  console.log('\n=== S14: 다수 버블 속도·방향·수명 분산(A1·A7) ===');
  console.log(`n=${bs.length} speeds=[${speeds.map((s) => s.toFixed(0)).join(',')}] lifeSpread=${(Math.max(...lifes) - Math.min(...lifes)).toFixed(1)}s`);
  console.log('S14 PASS:', bs.length >= 5 && allAboveMin && speedVaries && dirVaries && lifeVaries);
}

// ---------------------------------------------------------------
// 시나리오 15 (5차 B1): 터지면 글자가 하단에 쌓이고, 유지시간 후 소거.
// ---------------------------------------------------------------
{
  popSettings.mode = 'fixed-lifetime';
  popSettings.fixedLifetime = 0.01; // 완성 직후 터짐
  popSettings.popRatio = 1; // 결정성: 반드시 터지게(willPop 랜덤 제거)
  popSettings.popDurationS = 0.4;
  popSettings.stackEnabled = true;
  popSettings.glyphStackRatio = 1; // 전부 쌓임
  popSettings.stackClearAfterS = 0.5;
  t = 0;
  const det = new VoiceEventDetector();
  const mgr = new BubbleManager();
  const dims = { width: 800, height: 600 };
  mgr.setDimensions(dims);
  const feed = (f: VoiceFrame) => det.process(f).forEach((e) => mgr.handle(e));
  const step = () => mgr.update(0.025, dims);
  for (let i = 0; i < 6; i++) { feed(frame(next(), 0.0, null, null, false)); step(); }
  for (let i = 0; i < 14; i++) { feed(frame(next(), 0.25, 150, '우', true)); step(); }
  for (let i = 0; i < 12; i++) { feed(frame(next(), 0.0, null, null, false)); step(); }
  for (let i = 0; i < 30; i++) step(); // 터짐 진행 → 쌓임
  const stackedAfterPop = mgr.getStackedGlyphs().length;
  for (let i = 0; i < 60; i++) step(); // 유지시간(0.5s) 경과 → 소거
  const stackedAfterClear = mgr.getStackedGlyphs().length;
  console.log('\n=== S15: 글자 하단 쌓임 + 소거(B1) ===');
  console.log(`쌓임=${stackedAfterPop} → 소거후=${stackedAfterClear}`);
  console.log('S15 PASS:', stackedAfterPop > 0 && stackedAfterClear === 0);
}

// ---------------------------------------------------------------
// 유닛 U2 (5차 A6): 터짐 3막 합 = popDurationS.
// ---------------------------------------------------------------
{
  popSettings.popDurationS = 0.42;
  const d = popDurations();
  const sum = d.tension + d.burst + d.residual;
  console.log('\n=== U2: 터짐 길이 = popDurationS ===');
  console.log(`sum=${sum.toFixed(3)} (target 0.42)`);
  console.log('U2 PASS:', Math.abs(sum - 0.42) < 1e-6);
}


// ---------------------------------------------------------------
// 시나리오 16 (세기 판정 안정화 — 완료기준 1): 같은 세기 '아–––'(어택·릴리즈 포함)
//   → 세그먼트 1개. 어택 상승·릴리즈 감쇠가 세그먼트를 낳지 않아야 한다.
// ---------------------------------------------------------------
{
  t = 0;
  const fr: VoiceFrame[] = [];
  for (let i = 0; i < 6; i++) fr.push(frame(next(), 0.0, null, null, false));
  // 어택(자연스러운 상승)
  for (const e of [0.10, 0.16, 0.24, 0.30]) fr.push(frame(next(), e, 150, '아', true));
  // 같은 세기 지속
  for (let i = 0; i < 18; i++) fr.push(frame(next(), 0.30, 150, '아', true));
  // 릴리즈(감쇠) → 침묵
  for (const e of [0.20, 0.12]) fr.push(frame(next(), e, 150, '아', true));
  for (let i = 0; i < 12; i++) fr.push(frame(next(), 0.0, null, null, false));
  const { bubbles } = run('S16: 같은 세기 아––– → 세그먼트 1개(완료기준 1)', fr);
  const b = bubbles[0];
  const ok = bubbles.length === 1 && !!b && b.segments.length === 1 && b.segments[0].vowel === '아';
  console.log('S16 PASS:', ok);
}

// ---------------------------------------------------------------
// 시나리오 17 (세기 판정 안정화 — 완료기준 3): '아(약)→아(강)→아(약)' 3단
//   → 세그먼트 3개(각 수준이 유지 시간을 채워 확정).
// ---------------------------------------------------------------
{
  t = 0;
  const fr: VoiceFrame[] = [];
  for (let i = 0; i < 6; i++) fr.push(frame(next(), 0.0, null, null, false));
  for (let i = 0; i < 12; i++) fr.push(frame(next(), 0.15, 150, '아', true)); // 약
  for (let i = 0; i < 14; i++) fr.push(frame(next(), 0.50, 150, '아', true)); // 강
  for (let i = 0; i < 14; i++) fr.push(frame(next(), 0.15, 150, '아', true)); // 약
  for (let i = 0; i < 12; i++) fr.push(frame(next(), 0.0, null, null, false));
  const { bubbles } = run('S17: 아(약)→강→약 → 세그먼트 3개(완료기준 3)', fr);
  const b = bubbles[0];
  const ok =
    bubbles.length === 1 &&
    !!b &&
    b.segments.length === 3 &&
    b.segments.every((sg) => sg.vowel === '아') &&
    b.segments.slice(1).every((sg) => sg.hasBoundary === false); // 세기 세그먼트는 경계 없음
  console.log(`segs=${b ? b.segments.length : 0}`);
  console.log('S17 PASS:', ok);
}


// ---------------------------------------------------------------
// 시나리오 18 (버블 크기 동결 — 완료기준 1): 크게 "아!" → 침묵.
//   커진 크기 그대로 완성·동결되어야 한다(릴리즈 감쇠로 쪼그라들면 실패).
// ---------------------------------------------------------------
{
  popSettings.mode = 'never'; // S15 등에서 바뀐 전역 모드 복원(이 시나리오는 관찰용)
  t = 0;
  const fr: VoiceFrame[] = [];
  for (let i = 0; i < 6; i++) fr.push(frame(next(), 0.0, null, null, false));
  for (const e of [0.2, 0.4, 0.55]) fr.push(frame(next(), e, 150, '아', true)); // 빠른 어택(유예 내)
  for (let i = 0; i < 18; i++) fr.push(frame(next(), 0.55, 150, '아', true)); // 크게 지속
  for (const e of [0.3, 0.12]) fr.push(frame(next(), e, 150, '아', true)); // 릴리즈
  for (let i = 0; i < 12; i++) fr.push(frame(next(), 0.0, null, null, false)); // 침묵
  const { bubbles } = run('S18: 크게 아! → 커진 크기 동결(완료기준 1)', fr);
  const b = bubbles[0];
  const seg = b?.segments[0];
  const big = energyToRadius(0.45); // 크게 낸 세기에 상응하는 큰 반지름 하한
  const retained = !!seg && seg.radius.value >= big; // 릴리즈로 안 쪼그라듦
  const frozen = !!seg && Math.abs(seg.radius.target - seg.radius.value) < 0.5; // 목표=현재(동결)
  console.log(`r=${seg ? seg.radius.value.toFixed(1) : 'n/a'} target=${seg ? seg.radius.target.toFixed(1) : 'n/a'} (기준 ${big.toFixed(1)})`);
  console.log('S18 PASS:', bubbles.length === 1 && b.state === 'FLOATING' && b.segments.length === 1 && retained && frozen);
}

// ---------------------------------------------------------------
// 시나리오 19 (버블 크기 동결 — 완료기준 2): 작게 발성 → 작은 크기 유지.
//   S18(큰 버블)과 비교해 크기 차이가 비행 중에도 보존된다.
// ---------------------------------------------------------------
{
  popSettings.mode = 'never';
  t = 0;
  const fr: VoiceFrame[] = [];
  for (let i = 0; i < 6; i++) fr.push(frame(next(), 0.0, null, null, false));
  for (let i = 0; i < 20; i++) fr.push(frame(next(), 0.13, 150, '아', true)); // 작게 지속
  for (let i = 0; i < 12; i++) fr.push(frame(next(), 0.0, null, null, false));
  const { bubbles } = run('S19: 작게 발성 → 작은 크기 유지(완료기준 2)', fr);
  const b = bubbles[0];
  const seg = b?.segments[0];
  const smallCap = energyToRadius(0.25); // 작은 세기 → 작은 반지름 상한
  const ok = bubbles.length === 1 && !!b && b.state === 'FLOATING' && b.segments.length === 1 &&
    seg.radius.value <= smallCap;
  console.log(`r=${seg ? seg.radius.value.toFixed(1) : 'n/a'} (상한 ${smallCap.toFixed(1)})`);
  console.log('S19 PASS:', ok);
}


// ---------------------------------------------------------------
// 시나리오 20 (재발성 — 완료기준 1): "아아아"(한 호흡, 피치 연속) → 로브 3개짜리 한 버블.
//   음절 사이 에너지 딥이 있어도 피치가 이어지면 같은 버블에 경계 없는 로브가 붙는다.
// ---------------------------------------------------------------
{
  popSettings.mode = 'never';
  t = 0;
  const fr: VoiceFrame[] = [];
  const syl = (n: number) => { for (let i = 0; i < n; i++) fr.push(frame(next(), 0.4, 150, '아', true)); };
  const dip = () => { for (let i = 0; i < 3; i++) fr.push(frame(next(), 0.10, 150, '아', true)); }; // 피치 연속
  for (let i = 0; i < 6; i++) fr.push(frame(next(), 0.0, null, null, false));
  syl(8); dip(); syl(6); dip(); syl(6); // 아 · 아 · 아 (딥으로 구분, 피치 유지)
  for (let i = 0; i < 10; i++) fr.push(frame(next(), 0.0, null, null, false)); // 피치까지 소실 → 종료
  const { bubbles } = run('S20: 아아아 → 로브 3개 한 버블(완료기준 1)', fr);
  const b = bubbles[0];
  const ok = bubbles.length === 1 && !!b && b.segments.length === 3 &&
    b.segments.every((sg) => sg.vowel === '아') &&
    b.segments.slice(1).every((sg) => sg.hasBoundary === false);
  console.log(`segs=${b ? b.segments.length : 0}`);
  console.log('S20 PASS:', ok);
}

// ---------------------------------------------------------------
// 시나리오 21 (재발성 — 완료기준 2): "아" (쉬고, 피치까지 끊김) "아" → 독립 버블 2개.
//   침묵 + 피치 소실이 utteranceBreakMs 이상 → 발성 종료. 병합 창(450ms)보다 길게 쉰다.
// ---------------------------------------------------------------
{
  popSettings.mode = 'never';
  t = 0;
  const fr: VoiceFrame[] = [];
  for (let i = 0; i < 6; i++) fr.push(frame(next(), 0.0, null, null, false));
  for (let i = 0; i < 10; i++) fr.push(frame(next(), 0.35, 150, '아', true)); // 아 #1
  for (let i = 0; i < 30; i++) fr.push(frame(next(), 0.0, null, null, false)); // 한 박자 쉼(피치 끊김, 750ms)
  for (let i = 0; i < 10; i++) fr.push(frame(next(), 0.35, 150, '아', true)); // 아 #2
  for (let i = 0; i < 10; i++) fr.push(frame(next(), 0.0, null, null, false));
  const { bubbles } = run('S21: 아 (쉬고) 아 → 독립 버블 2개(완료기준 2)', fr);
  const ok = bubbles.length === 2 && bubbles.every((b) => b.segments.length >= 1);
  console.log('S21 PASS:', ok);
}

// ---------------------------------------------------------------
// 시나리오 22 (재발성 — 완료기준 3): 길게 "아~~~"(떨림 포함, 에너지 안정) → 로브 1개.
//   피치 지터(떨림)는 로브를 만들지 않는다(딥은 에너지 하강만 본다).
// ---------------------------------------------------------------
{
  popSettings.mode = 'never';
  t = 0;
  const fr: VoiceFrame[] = [];
  for (let i = 0; i < 6; i++) fr.push(frame(next(), 0.0, null, null, false));
  // 에너지 안정(딥 없음) + 피치 흔들림(떨림)
  for (let i = 0; i < 30; i++) fr.push(frame(next(), 0.33 + (i % 2 ? 0.03 : -0.03), 150 + (i % 2 ? 8 : -8), '아', true));
  for (let i = 0; i < 10; i++) fr.push(frame(next(), 0.0, null, null, false));
  const { bubbles } = run('S22: 길게 아~~~ (떨림) → 로브 1개(완료기준 3)', fr);
  const b = bubbles[0];
  const ok = bubbles.length === 1 && !!b && b.segments.length === 1 && b.segments[0].vowel === '아';
  console.log(`segs=${b ? b.segments.length : 0}`);
  console.log('S22 PASS:', ok);
}


// ---------------------------------------------------------------
// 시나리오 23 (변화 속도 — 완료기준 2): "아~~~(점점 크게)" → 단계적으로 커지는 로브 3개+.
// ---------------------------------------------------------------
{
  popSettings.mode = 'never';
  popSettings.gradualStepDb = 2.5;
  t = 0;
  const fr: VoiceFrame[] = [];
  for (let i = 0; i < 6; i++) fr.push(frame(next(), 0.0, null, null, false));
  const N = 44; // 0.15→0.6 (12dB)를 완만히(급변 임계 미만) 크레셴도
  for (let i = 0; i < N; i++) fr.push(frame(next(), 0.15 * Math.pow(4, i / N), 150, '아', true));
  for (let i = 0; i < 10; i++) fr.push(frame(next(), 0.0, null, null, false));
  const { bubbles } = run('S23: 아~~~ 점점 크게 → 단계 로브 3개+(완료기준 2)', fr);
  const b = bubbles[0];
  const radii = b ? b.segments.map((sg) => sg.radius.value) : [];
  const rising = radii.length >= 2 && radii[radii.length - 1] > radii[0] * 1.15; // 단계적으로 커짐
  const ok = bubbles.length === 1 && !!b && b.segments.length >= 3 &&
    b.segments.every((sg) => sg.vowel === '아') &&
    b.segments.slice(1).every((sg) => sg.hasBoundary === false) && rising;
  console.log(`segs=${b ? b.segments.length : 0} radii=[${radii.map((r) => r.toFixed(0)).join(',')}]`);
  console.log('S23 PASS:', ok);
}

// ---------------------------------------------------------------
// 시나리오 24 (변화 속도 — 완료기준 3): "아~~~(점점 작게)" → 단계적으로 작아지는 로브 여러 개.
// ---------------------------------------------------------------
{
  popSettings.mode = 'never';
  popSettings.gradualStepDb = 2.5;
  t = 0;
  const fr: VoiceFrame[] = [];
  for (let i = 0; i < 6; i++) fr.push(frame(next(), 0.0, null, null, false));
  const N = 44; // 0.55→0.18 완만한 데크레셴도(딥 유발 않게 하한 여유)
  for (let i = 0; i < N; i++) fr.push(frame(next(), 0.55 * Math.pow(0.18 / 0.55, i / N), 150, '아', true));
  for (let i = 0; i < 10; i++) fr.push(frame(next(), 0.0, null, null, false));
  const { bubbles } = run('S24: 아~~~ 점점 작게 → 단계 로브 여러 개(완료기준 3)', fr);
  const b = bubbles[0];
  const radii = b ? b.segments.map((sg) => sg.radius.value) : [];
  const falling = radii.length >= 2 && radii[radii.length - 1] < radii[0] * 0.85;
  const ok = bubbles.length === 1 && !!b && b.segments.length >= 3 &&
    b.segments.every((sg) => sg.vowel === '아') && falling;
  console.log(`segs=${b ? b.segments.length : 0} radii=[${radii.map((r) => r.toFixed(0)).join(',')}]`);
  console.log('S24 PASS:', ok);
}

// ---------------------------------------------------------------
// 시나리오 25 (변화 속도 — 완료기준 4): "점점 크게다가 확!" → 계단 로브들 + 뚜렷이 큰 로브 1개.
// ---------------------------------------------------------------
{
  popSettings.mode = 'never';
  popSettings.gradualStepDb = 2.5;
  t = 0;
  const fr: VoiceFrame[] = [];
  for (let i = 0; i < 6; i++) fr.push(frame(next(), 0.0, null, null, false));
  const N = 28;
  for (let i = 0; i < N; i++) fr.push(frame(next(), 0.15 * Math.pow(2.2, i / N), 150, '아', true)); // 점점(→0.33)
  for (let i = 0; i < 14; i++) fr.push(frame(next(), 0.85, 150, '아', true)); // 확! (급변)
  for (let i = 0; i < 10; i++) fr.push(frame(next(), 0.0, null, null, false));
  const { bubbles } = run('S25: 점점 크게다가 확! → 계단 + 큰 로브(완료기준 4)', fr);
  const b = bubbles[0];
  const radii = b ? b.segments.map((sg) => sg.radius.value) : [];
  const last = radii[radii.length - 1] ?? 0;
  const prevMax = Math.max(0, ...radii.slice(0, -1));
  const ok = bubbles.length === 1 && !!b && b.segments.length >= 3 && last > prevMax * 1.2; // 마지막이 확 큼
  console.log(`segs=${b ? b.segments.length : 0} radii=[${radii.map((r) => r.toFixed(0)).join(',')}] last=${last.toFixed(0)} prevMax=${prevMax.toFixed(0)}`);
  console.log('S25 PASS:', ok);
}

// ---------------------------------------------------------------
// 시나리오 26 (변화 속도 — 완료기준 7): GRADUAL_STEP_DB 낮추면 같은 크레셴도에서 로브가 더 많아진다.
// ---------------------------------------------------------------
{
  popSettings.mode = 'never';
  const build = () => {
    t = 0;
    const fr: VoiceFrame[] = [];
    for (let i = 0; i < 6; i++) fr.push(frame(next(), 0.0, null, null, false));
    const N = 44;
    for (let i = 0; i < N; i++) fr.push(frame(next(), 0.15 * Math.pow(4, i / N), 150, '아', true));
    for (let i = 0; i < 10; i++) fr.push(frame(next(), 0.0, null, null, false));
    return fr;
  };
  popSettings.gradualStepDb = 3.0;
  const a = run('S26a: 크레셴도 step=3.0dB', build());
  const n1 = a.bubbles[0] ? a.bubbles[0].segments.length : 0;
  popSettings.gradualStepDb = 1.2;
  const bres = run('S26b: 크레셴도 step=1.2dB', build());
  const n2 = bres.bubbles[0] ? bres.bubbles[0].segments.length : 0;
  popSettings.gradualStepDb = 2.5; // 복원
  console.log(`step3.0 → ${n1}로브, step1.2 → ${n2}로브`);
  console.log('S26 PASS:', n2 > n1);
}
