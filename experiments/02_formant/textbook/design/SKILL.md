---
name: voicetypo-textbook-design
description: Use this skill to generate well-branded interfaces and assets for the VoiceTypo Textbook — a Korean-language textbook on voice signal processing built on Stripe's design language. Contains essential design guidelines, colors, type, fonts, assets, and a UI kit (chapter page, formula box, "내 코드에서는" callout, quiz box, sidebar, next-chapter card) for prototyping chapter pages, slides, or any author-facing artifact.
user-invocable: true
---

Read the README.md file within this skill, and explore the other available files. Key files:

- `README.md` — brand context, content fundamentals (Korean voice & tone), visual foundations, iconography notes
- `colors_and_type.css` — CSS variables + semantic classes. Import first.
- `fonts/README.md` — font substitution notes (Inter for sohne-var; flag to user if fidelity matters)
- `ui_kits/textbook/` — full chapter page recreation; components are the source of truth for how the textbook-specific boxes (수식, 내 코드에서는, 이해 확인, 다음 챕터) should look
- `preview/` — single-concept cards (palettes, type specimens, radii, shadows, components)
- `assets/` — logo + favicon

If creating visual artifacts (slides, mocks, throwaway prototypes, etc), copy assets out and create static HTML files for the user to view. If working on production code, you can copy assets and read the rules here to become an expert in designing with this brand.

If the user invokes this skill without any other guidance, ask them what they want to build or design — a new chapter page, a slide for a lab talk, a printable cheat-sheet — ask some questions, and act as an expert designer who outputs HTML artifacts or production code, depending on the need.

## Non-negotiables

- Weight 300 for all display type; 400 only for buttons/links/nav. Never bold headlines.
- `font-feature-settings: "ss01"` on every sans element.
- Korean body text: `line-height: 1.75`, `word-break: keep-all`, Pretendard Variable.
- Blue-tinted shadows (`rgba(50,50,93,0.25)`) — never neutral gray.
- Deep navy `#061b31` for headings — never pure black.
- Radii stay 4–8px. No pill shapes.
- Stripe Purple `#533afd` is the ONLY interactive/brand color. Ruby/Magenta are for signal-overlay figures only.
- No emoji. No marketing gradients. No payment/financial UI.
- Chapter pages follow the fixed 9-section structure (title → overview → intuition → math → code → 내 코드에서는 → experiment → 이해 확인 → next-chapter).
