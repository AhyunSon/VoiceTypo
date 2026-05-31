# 레퍼런스 · 크레딧 (저작권 안전 노트)

이 폴더의 스케치들은 아래 작가들의 **개념·방법론에서 영감**을 받았습니다.
**코드는 전부 우리가 직접 새로 작성**했으며, 원작의 소스코드·이미지·에셋을 복제하지 않았습니다.

> 원칙: 아이디어·기법은 저작권 대상이 아님(보호받는 건 "구체적 표현"). 우리는 개념만 흡수하고
> 구현은 자작 → 저작권 문제 없음. 출처를 밝히는 것은 학술적 정직성 + 표절 시비 차단.

## 영감받은 작가 (개념만 참고, 코드 복제 없음)

| 작가 | 작업 | 우리가 가져온 *개념* | 우리 스케치 |
| --- | --- | --- | --- |
| **Golan Levin** + **Zachary Lieberman** | *Messa di Voce* (2003) | "말이 실시간으로 보이는 형태가 된다" | sketch_05 (헌정 — 재현 아님) |
| **Golan Levin** | 음성 파라미터→시각 매핑 어휘집 | 파라미터별 시각 매핑 표 | sketch_01 |
| **Zachary Lieberman** | 데일리 스케치 / Audio→Visual→Render | 작게·많이 스케치하는 작업 방식, 글자 변형 | sketch_02, 04 |
| **Casey Reas** | *Process* / *{Software} Structures* | "규칙이 형태를 만든다" — 모음별 규칙집 | sketch_02 |
| **John Maeda** | *Reactive Books* | 글자가 입력에 살아 반응 | sketch_04 |
| **Daniel Rozin** | *Mirrors* 연작 | 입력→형태 1:1 반사 | sketch_03 |
| **Memo Akten** | *Learning to See* / 생성형 손글씨(RMDN) | "목소리가 획을 생성한다"(학습형) | ai_generative (개념 스탠드인) |

## 주의 (지킬 것)
- **제목 "Messa di Voce"** 는 Levin/Lieberman(Tmema) 작품명 — sketch_05 는 **헌정 표기**이며
  공개/전시 시에는 **우리만의 제목**을 따로 붙인다.
- 원작 소스코드·영상·이미지를 **그대로 가져오지 않는다**. 개념·기술 설명만 참고.
- 공개·전시·상업화 단계에서 한 번 더 "충분히 우리 원본 표현인가" 점검.

## 사용 오픈소스 라이브러리 (라이선스 OK)
- **pygame** (LGPL) · **praat-parselmouth** (GPLv3, 도구로 사용) · **numpy** (BSD)
- **sounddevice** (MIT) · **websockets** (BSD) · **p5.js** (LGPL, CDN 로드)

## 더 읽을 것 (클론 X — 읽기 참고용)
oF/구버전이라 집 맥에서 빌드 비현실적. 개념·구조만 참고:
- `github.com/ofZach/avsysNantes`, `github.com/ofZach/MIT_DrawingPlusPlus` (openFrameworks, C++)
- `github.com/memoakten/ofxMSATensorFlow` (TF r1.1, 구버전 — 생성형 손글씨 구조 참고)
- 개념·영상: Golan Levin `flong.com/archive/projects/`, Casey Reas `reas.com`
