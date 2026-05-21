# VoiceTypo Textbook — Design System

A design system for a Korean-language textbook on **voice signal processing** (음성 신호처리), written by 허재원 as self-study material for the VoiceTypo project (한국어 음성 → 실시간 타이포그래피 미디어아트). Readership: the author, with potential lab-internal sharing.

The system is built on **Stripe's visual language** — sohne-var typography, deep navy headings, blue-tinted shadows, conservative 4–8px radii, Stripe Purple (#533afd) as the brand anchor — but **strips out the financial/commercial surfaces** (pricing cards, success-payment badges, dashboard previews) and adds **textbook-native components** (수식 박스, 확인 질문 박스, 내 코드에서는 박스, 목차 사이드바, 다음 챕터 카드).

---

## Sources

- **Spec**: Provided inline by the user in chat. No codebase, Figma, screenshots, or existing brand assets were attached.
- **Foundation**: Stripe design language as described in the brief.
- **Target project**: VoiceTypo (외부 프로젝트). This design system is for the companion **textbook**, not the runtime typography system.

---

## Index

| File / Folder | What's in it |
| --- | --- |
| `README.md` | This file. |
| `colors_and_type.css` | CSS variables for colors, shadows, radii, spacing, type. Semantic element styles. Import this first. |
| `SKILL.md` | Agent-skill frontmatter so this folder is portable to Claude Code. |
| `fonts/` | Font substitutions and notes. (sohne-var is licensed — Inter is used as substitute; see flag below.) |
| `assets/` | Logo mark, iconography notes, placeholder illustrations. |
| `preview/` | HTML cards that populate the Design System tab (type specimens, palettes, shadow/radius tokens, components). |
| `ui_kits/textbook/` | Interactive chapter-page recreation showing the full component set in context. |

---

## ⚠ Font substitution flag

**sohne-var is a licensed Stripe font** and cannot be redistributed. The system substitutes **Inter** (Google Fonts, variable weight) for Latin text and **Pretendard Variable** for Korean. Inter is the closest widely-available geometric grotesque and supports the same weight range (300→700) the design calls for. The OpenType `"ss01"` feature is applied globally, though Inter's `ss01` differs from sohne-var's — glyphs for `a`, `g`, `l` will look slightly different.

**Action for you**: If the lab has a sohne-var license, drop the `.woff2` files into `fonts/` and update the `@font-face` declaration in `colors_and_type.css`. Otherwise the Inter substitute is intentional and production-ready for internal use.

**Source Code Pro** is available directly from Google Fonts — used as specified.

---

## Content Fundamentals

### Voice & tone

- **Language**: Korean body (`"~입니다"` 체), with English technical terms in parentheses on first use — e.g. **포먼트(formant)**, **선형 예측 부호화(LPC)**, **단시간 푸리에 변환(STFT)**.
- **Teacher mode, not marketing mode**. The Stripe foundation is "confident, premium, financial" — we strip that out. This textbook is patient, a little warm, careful. Closer to a good graduate TA than a product page.
- **Priorities** (in order): 정확성 > 친절함 > 직관 > 수식. Never invert that order. Never drop a formula onto the page without having built the intuition first.
- **Always answer "왜"**. Don't list formulas. Explain why this formula and not another; what it's doing physically; where it breaks down.
- **Casing**: Korean headings are sentence-case mixed script (한글 + Latin acronyms uppercase). English chapter titles use sentence case. No ALL-CAPS marketing headlines.
- **"I" vs. "you"**: Prefer impersonal Korean constructions — `~해봅시다` (let's), `~할 수 있습니다` (we can). Author voice appears only in the **"내 코드에서는"** boxes, which quote author code with minimal narration.
- **Emoji**: None. Unicode symbols (→, ≈, ·, ∑, ∫) are welcome in running text; emoji are not. The textbook vibe is closer to Tufte than Notion.
- **Numerals**: `"tnum"` OpenType feature for tabular data (tables, chart axes, hyperparameter lists). Proportional figures (`"ss01"` default) for running prose.

### Example phrasings

- Good: *"포먼트(formant)는 성도(vocal tract)의 공명 주파수입니다. 직관적으로는 입 모양에 따라 달라지는 '울림통의 고유 주파수'라고 생각할 수 있습니다."*
- Good (author callout): *"내 코드에서는 선형 예측 계수를 12차로 두었는데, 이는 일반적인 한국어 음성 샘플링에서 경험적으로 잘 맞는 값이기 때문입니다."*
- Avoid: *"이 놀라운 기법을 사용하면 여러분도 손쉽게 음성을 분석할 수 있습니다! 🎙️"* (marketing tone, emoji, 여러분 호격)
- Avoid: *"공식 (1)을 보자."* (명령형, 설명 부재) — instead: *"이제 식 (1)을 살펴봅시다. 좌변의 s[n]은 …"*

### Content structure (per chapter)

Each chapter page is assembled in this fixed order:

1. Chapter number + title (Display Large, 48px, weight 300)
2. **이 챕터에서 배울 것** — 3-bullet overview
3. **직관적 이해** — figures and analogies, no math
4. **수식과 원리** — formula boxes + symbol glossary
5. **Python 구현** — generic reference code
6. **내 코드에서는** — author-code callout (purple box)
7. **인터랙티브 실험** — sliders, where possible
8. **이해 확인** — 3 numbered questions with collapsible answers
9. **다음 챕터 →** — navigation card

---

## Visual Foundations

### Palette

- **Brand anchor**: Stripe Purple `#533afd` for CTAs, links, active-state borders, and the single-accent "내 코드에서는" box stripe.
- **Headings**: Deep navy `#061b31`, never pure black. The warmth matters.
- **Body**: Slate `#64748d`. Labels: `#273951`.
- **Surfaces**: White `#ffffff` body, `#F5F3FF` purple-tint for author-code boxes, `#F6F9FC` cool-gray for formula & quiz boxes, `#0D253D` dark navy for raw code blocks (white text).
- **Decorative-only**: Ruby `#ea2261` and Magenta `#f96bee` exist in the Stripe palette but are **avoided** in the textbook except for occasional signal-waveform overlays. Never used for buttons, links, or chrome.
- **Removed from Stripe**: success/warning payment badges, ruby-magenta marketing gradients, financial-data tables with `tnum` pricing — all stripped.

### Typography

- **Display weight is 300.** That's the signature. No bold headlines.
- **Negative letter-spacing** progressive: -1.4px at 56px, -0.96px at 48px, -0.64px at 32px, relaxing to `normal` at 16px.
- **OpenType**: `"tnum"` for tabular numerals in tables/axes/data rows. The `"ss01"` declaration is kept from the Stripe spec but is a no-op under Pretendard (which does not expose that stylistic set) — it activates automatically if the sans family is ever swapped for one that supports it.
- **Korean body** gets `line-height: 1.75` and `word-break: keep-all` — Latin defaults of 1.4 are too tight for 한글.
- **Monospace**: JetBrains Mono, weight 500 at 12px, line-height 2.0 (deliberately generous for dense signal-processing code). D2Coding serves as the Korean fallback so comments in 한글 keep a monospace advance width.

### Spacing & layout

- **8px base unit** with a dense small end (1, 2, 4, 6, 8, 10, 11, 12, 14, 16, 18, 20). The denser lower range matches Stripe's precision-UI instincts and is genuinely useful for formula margins.
- **Max content width**: ~720px for prose-heavy chapter pages (narrower than Stripe's 1080px marketing pages — this is a textbook, optimize for reading line length). Sidebar adds ~240px.
- **Section rhythm**: sections separated by `var(--vt-s-48)` to `--vt-s-64`. No alternating dark/light bands (those are marketing-site moves; doesn't fit a textbook).
- **Two-column grids** only for before/after waveform comparisons or code-vs-equivalent-math side-by-sides.

### Backgrounds & imagery

- Page is overwhelmingly **white**. No full-bleed photography, no gradients, no textures.
- Imagery is **technical diagrams**: waveforms, spectrograms, formant tracks. Monochrome preferred (navy on white), with purple as single accent when marking a region of interest. Ruby/magenta only when overlaying two signals that need to be visually distinguished.
- **No decorative illustrations**. If a figure doesn't convey data, it doesn't belong.

### Shadows & elevation

Five-level system, blue-tinted per Stripe:

| Level | Shadow | Use |
| --- | --- | --- |
| 0 | none | Page bg, inline text |
| 1 | `0 3px 6px rgba(23,23,23,0.06)` | Hover hints |
| 2 | `0 15px 35px rgba(23,23,23,0.08)` | Standard cards |
| 3 | `rgba(50,50,93,.25) 0 30px 45px -30px, rgba(0,0,0,.1) 0 18px 36px -18px` | Featured cards, next-chapter nav |
| 4 | `rgba(3,3,39,.25) 0 14px 21px -14px, rgba(0,0,0,.1) 0 8px 17px -8px` | Modals |

The blue-tint (`rgba(50,50,93,...)`) is **non-negotiable**. Neutral gray shadows look wrong against this palette.

### Borders & radii

- **Radii**: 4px (buttons/inputs/badges), 5–6px (cards), 8px (featured cards). **Never** pill-shaped, never 12+.
- **Borders**: `1px solid #e5edf5` default. Purple `#b9b9f9` for active/selected. Dashed `1px dashed #362baa` for drop zones (rare in textbook).
- **"내 코드에서는" box**: 4px-wide left border in `#533afd`, `#F5F3FF` background, `4px` radius.

### Motion & interaction

- **Easing**: `ease` or `cubic-bezier(0.4, 0, 0.2, 1)` at 120–180ms for hovers. Nothing bounces — this is a textbook, not a consumer app.
- **Hover**: purple elements darken to `#4434d4`; card shadows intensify from elev-2 → elev-3; border on next-chapter card transitions to `#533afd`.
- **Press**: no shrink/scale animations. Slight background darken only.
- **No animation** on equations, figures, or code blocks — they must read statically first.

### Transparency & blur

- Sticky nav header uses `backdrop-filter: blur(12px)` with white-alpha `rgba(255,255,255,0.85)`. That's the only place blur is used.
- Avoid semi-transparent text (`rgba(0,0,0,0.5)` etc.) — always pick a solid color from the scale.

### Layout rules (fixed elements)

- **Left sidebar** (목차): fixed on desktop ≥1024px, collapses to a drawer below. Width 240px. Active chapter marked with 4px-wide purple left border.
- **Top header**: sticky. 64px tall. Holds brand wordmark + chapter breadcrumb.
- **Content column**: centered, max-width 720px, horizontal padding 32px (desktop) / 20px (mobile).

---

## Iconography

No custom icon set is shipped with the brand. The textbook's iconographic needs are narrow and functional, so we link **[Lucide Icons](https://lucide.dev/)** from CDN for the handful of places icons appear:

- Chapter nav arrows → `chevron-right`
- Collapsible "이해 확인" answers → `chevron-down` / `chevron-up`
- Sidebar toggle on mobile → `menu` / `x`
- External link to author code repo → `external-link`
- Audio playback on waveform figures → `play` / `pause` / `volume-2`
- Copy-code button → `copy` / `check`

**Stroke style**: Lucide's default 2px stroke, `currentColor`, 20×20 viewport. Icons inherit text color (typically `--vt-body` slate or `--vt-purple` for interactive). No filled icons, no brand-colored icon backgrounds.

**Emoji**: None. Ever.

**Unicode symbols as icons**: Welcome in running text for mathematical / phonetic notation (→, ≈, ·, ∑, ∫, ±, ≠, ≤, ≥, ∞, ∂, ∇, ∫, ♪, IPA glyphs like ɐ, ʃ, θ).

**Logo**: A simple wordmark — "VoiceTypo Textbook" set in Inter weight 300 with a small purple ◼︎ glyph prefix. See `assets/logo.svg`. No figural mark; the project is small and a wordmark suits the textbook-first context.

---

## Using the system

```html
<link rel="stylesheet" href="colors_and_type.css" />
```

Then use the CSS variables:

```css
.my-button {
  background: var(--vt-purple);
  color: var(--vt-white);
  padding: var(--vt-s-8) var(--vt-s-16);
  border-radius: var(--vt-radius-4);
  box-shadow: var(--vt-elev-2);
  font: 400 var(--vt-size-body)/1 var(--vt-font-sans);
  font-feature-settings: "ss01";
}
```

Or use the semantic classes on elements: `.vt-h1`, `.vt-body`, `.vt-korean`, `.vt-code-label`, etc.

---

## UI kits

- **`ui_kits/textbook/`** — Full chapter page recreation (1장 포먼트). Sidebar + sticky header + the 9-section chapter flow, with working interactive slider, collapsible quiz answers, and next-chapter navigation between 8 placeholder chapters. Source of truth for how the textbook-specific boxes should look in context.

