# 🎤 VoiceTypo 데모 가이드 (주간 회의용)

이 폴더(`voicetypo_demo`) **하나만** 노트북에 옮기면 7가지 실험을 실행할 수 있습니다.
20GB 학습 데이터는 빠져 있어요 — **라이브 데모엔 모델 파일만 필요**하고, 그건 다 들어 있습니다.

> ⚠️ **딱 하나, 폴더로 못 옮기는 것:** 파이썬 패키지(torch 등). venv는 컴퓨터마다 달라 복사가 안 돼요.
> 그래서 노트북에서 **한 번만 `pip install`** 하면 됩니다 (아래 0번). 그 뒤론 명령 한 줄로 실행돼요.

---

## ⚡ 30초 요약

| # | 실험 | 실행 (폴더 안에서) | 라이브 난이도 | 인터넷 |
| --- | --- | --- | --- | --- |
| 01 | method7 실시간 (구버전) | `python main.py` | △ 무거움 | 첫 실행 시 ⭕ |
| **02** | **포먼트 (메인)** | `python main.py` | ✅ 쉽고 안정 | ❌ |
| 03 | Whisper SSL | `python scripts/03_run_realtime.py` | ⚠️ 가능 | 첫 실행 시 ⭕(290MB) |
| 04 | 경량 ML | `python scripts/04_live_test_v3.py` | ⚠️ 불안정 | v4만 ⭕ |
| **05** | **시각화 (볼거리)** | `python sketch_02_vowel_letterform.py` | ✅ 쉽고 화려 | ❌ |
| 06 | 통합본 (아현) | `python main_integrated.py` | △ 무거움 | ⭕ |
| **07** | **웹 3D 모음공간 (신규)** | 터미널2개: `python server.py` + `npm run dev` | ✅ 화려·안정 | ❌(설치 후) |

> **회의에서 라이브로 "와" 하는 건 02 + 05 + 07.** 03·04는 마이크가 흔들리면 **녹음된 wav로 대체**(각 폴더 `livetest/`).
> 07은 브라우저에서 F1/F2/F3 3D 공간에 모음 포인트 클라우드를 실시간으로 그려 시각적으로 가장 인상적.

---

## 0. 준비 — 노트북에서 한 번만

1. **Python 3.10+ 설치** — `brew install python` 또는 python.org. 터미널에서 `python3 --version` 확인.
2. **인터넷 연결된 곳에서 미리** 아래를 실행 — 회의장 와이파이 믿지 말 것.
3. 실험마다 가상환경을 따로 만드는 걸 권장 (02·05는 같이 써도 됨).

```bash
# 예시: 02_formant 환경 (맥 / 터미널)
cd voicetypo_demo/02_formant
python3 -m venv .venv
source .venv/bin/activate
pip install PySide6 praat-parselmouth pyworld numpy scipy scikit-learn sounddevice
```

> 🍎 **맥 주의:** 마이크(sounddevice)가 PortAudio 오류를 내면 → `brew install portaudio` 후 `pip install --force-reinstall sounddevice`.
> Apple Silicon(M1~M4)도 위 패키지 전부 arm64 휠이 있어 그대로 설치됩니다.
> venv를 켠 상태(`source .venv/bin/activate`)에선 `python`/`pip` 그대로 쓰면 됩니다 (venv 만들 때만 `python3`).

**실험별 설치 패키지** (requirements.txt 있으면 `pip install -r requirements.txt` 우선):

| 실험 | 설치 |
| --- | --- |
| 01 | `torch transformers praat-parselmouth pyworld scikit-learn sounddevice numpy scipy PySide6` |
| **02** | `PySide6 praat-parselmouth pyworld numpy scipy scikit-learn sounddevice` |
| 03 | `pip install -r requirements.txt` (있음) |
| 04 | `torch transformers numpy scipy sounddevice soundfile librosa` |
| **05** | `pip install -r requirements.txt` (있음: pygame·parselmouth·sounddevice·websockets) |
| 06 | `pip install -r requirements.txt` (있음, 아현 작성) |
| **07** | 서버: `pip install -r requirements_web.txt` (fastapi·uvicorn·sounddevice 등) · 웹: `cd web && npm install` (Node 18+) |

> 실행 중 `ModuleNotFoundError: xxx` 가 뜨면 → `pip install xxx` 한 줄이면 해결.

---

## 데모별 상세

### 01 · method7 실시간 (포먼트 + wav2vec2, 구버전)
- **무엇:** 02의 옛 버전. 앙상블·Kalman 얹은 복잡한 형태(오프라인 22.9%). 02로 단순화하며 54.3%로 올림.
- **실행:** `cd 01_method7_realtime` → `python main.py`
- **첫 실행:** wav2vec2 모델 자동 다운로드(인터넷). 무거움.
- **회의 팁:** 02가 있으니 굳이 안 보여줘도 됨. "이걸 정리해서 02가 됐다"는 *발전 서사*용으로만 짧게.

### 02 · 포먼트 (메인 라인) ⭐ 추천
- **무엇:** F1/F2/F3로 한국어 7모음 실시간 인식 + 시각화 UI. 우리 프로젝트의 본 방향.
- **실행:** `cd 02_formant` → `python main.py` (PySide6 창이 뜸)
- **첫 실행:** 캘리브레이션 다이얼로그가 뜰 수 있음 — 7모음을 한 번씩 발음하면 정확도 ↑.
- **인터넷 불필요.** 가장 안정적. **회의 1순위.**

### 03 · Whisper SSL + MLP
- **무엇:** Whisper 인코더 + MLP로 화자 독립 인식(미관찰 화자 0.66). 모델 `probe.pt` 포함.
- **실행(라이브):** `cd 03_whisper_ssl` → `python scripts/03_run_realtime.py`
- **대체(마이크 불안 시):** `python scripts/06_evaluate_wav_folder.py` 로 `livetest/` 의 녹음 wav 평가
- **첫 실행:** Whisper-base(약 290MB) 자동 다운로드 → **반드시 목요일 전 미리 1회 실행**해서 받아둘 것.

### 04 · 경량 ML (MFCC+CNN / Whisper-tiny)
- **무엇:** 1~5MB 경량 모델 비교. 체크포인트(v1~v4) + 정규화(norm_stats) 포함.
- **실행(라이브):** `cd 04_light_ml` → `python scripts/04_live_test_v3.py`
- **대체:** `python scripts/06_evaluate_wav_folder_v3.py` (livetest wav)
- **주의:** 중단됐던 프로젝트라 라이브가 불안정할 수 있음 → **대체 wav 데모를 기본으로** 잡는 걸 권장. v4 실행 시 Whisper-tiny 자동 다운로드(작음).

### 05 · 시각화 스케치 ⭐ 추천
- **무엇:** 목소리 → 글자/형태 시각화 (멘토 작가들 개념 구현). 인식이 아니라 "보여주기".
- **실행:** `cd 05_visual_sketches` → 먼저 `python voice_input.py`(마이크 값 콘솔 확인) → `python sketch_02_vowel_letterform.py`
- **추천 데모:** `sketch_02`(글자 변형), `sketch_05_messa_di_voce.py`(통합 작품). 창 종료 **ESC**, 화면 비움 **C**.
- **인터넷 불필요.** 시각적으로 화려해서 **회의에서 반응 좋음.**

### 06 · 통합본 (아현 브랜치) — 참고
- **무엇:** 원래 main에 있던 통합 실행본(`main_integrated.py`). 아현이 자기 브랜치로 보존.
- **실행:** `cd 06_integrated` → `python main_integrated.py`
- **주의:** **아현의 작업물**입니다. 회의에선 보통 본인이 데모해요 — 보여줄지 미리 상의 권장. 무겁고(임베딩 데이터 포함) 모델 다운로드가 필요할 수 있음.

### 07 · 웹 3D 모음공간 (React + Three.js) ⭐ 신규 추천
- **무엇:** 목소리 → 포먼트(F1/F2/F3)·피치·비브라토 추출 후, **브라우저 3D 공간에 모음별 포인트 클라우드 + 타원체**를 실시간 시각화. Python(FastAPI+WebSocket) 백엔드 ↔ 웹 프론트엔드(R3F).
- **구성:** `server.py`(마이크→분석→WS 스트리밍, 포트 8765) + `web/`(Vite 개발서버, 포트 3000). 핵심 화면: `web/src/components/VowelSpace3D.tsx`.
- **실행 (터미널 2개 필요):**
  - 터미널 A: `cd 07_web_3d` → `source .venv/bin/activate` → `python server.py` (마이크 없이 미리보기는 `python server.py --sim`)
  - 터미널 B: `cd 07_web_3d/web` → `npm run dev` → 브라우저에서 **http://localhost:3000** 열기
- **인터넷 불필요**(설치 후). 마이크 권한 필요. **회의에서 시각적으로 가장 임팩트 큼.**
- **팁:** 발표 직전 `--sim`으로 한 번 띄워 화면 흐름을 확인해두면 안전. 실제 발표는 `--sim` 빼고 마이크로.

---

## 🔧 트러블슈팅

| 증상 | 해결 |
| --- | --- |
| `ModuleNotFoundError` | `pip install <모듈명>` |
| 마이크 입력 없음 | 시스템 설정 → 개인정보 보호 및 보안 → 마이크 → **터미널(또는 IDE) 허용**. 처음엔 권한 팝업이 뜸 |
| PortAudio 오류 | `brew install portaudio` → `pip install --force-reinstall sounddevice` |
| 모델 다운로드가 느림/실패 | 인터넷 좋은 곳에서 **미리** 1회 실행 (캐시됨). 회의장 와이파이 의존 X |
| Qt/pygame 창이 안 뜸 | 디스플레이/원격접속 환경 확인. 로컬 노트북 화면에서 실행 |
| 추론이 느림(GPU 없음) | 정상 — 작은 모델이라 CPU로도 됨. 03은 첫 추론만 느림 |
| 07 웹이 안 뜸/회색 화면 | 서버(`python server.py`)와 웹(`npm run dev`)을 **둘 다** 켰는지 확인. 브라우저는 http://localhost:3000 |
| 07 `npm` 오류/`tsc` 버전 에러 | `cd 07_web_3d/web && rm -rf node_modules && npm install` 로 재설치 (Node 18+) |

## ✅ 목요일 전 체크리스트

- [ ] 폴더를 노트북에 복사 (USB/클라우드)
- [ ] 인터넷 되는 곳에서 **7개 다 미리 한 번씩 실행** (특히 03 모델 다운로드, 07 `npm install`)
- [ ] 마이크 권한 확인 + 실제로 말해서 인식되는지 확인
- [ ] 02·05·07을 메인 데모로, 03·04는 대체 wav도 준비
- [ ] 06(아현 것) 보여줄지 아현과 상의
- [ ] 07은 터미널 2개(서버+웹) 띄우는 흐름 미리 연습 — 브라우저 http://localhost:3000

---

*데이터 20GB 제외 · 모델·정규화 파일 포함 · 07(웹) 추가로 폴더 용량 증가(node_modules 포함 ~250MB). 학습 재현이 필요하면 원본 데이터(이 컴퓨터의 `voicetypo_new/data`, `realtime_formant/evaluation`)가 따로 있음.*
