# 01 VoiceTypo 통합 프로젝트

작성자: ahyun
브랜치: ahyun
이관일: 2026-05-21

## 무엇

VoiceTypo 의 통합 프로젝트 — 한국어 모음 인식 + 오디오 분석 + 텍스트 효과를
하나로 묶은 실행본(`main_integrated.py`).

원래 `main` 브랜치에 있던 것을, **main 재구축을 위해** ahyun 개인 브랜치로 이관.

## 구성

| 항목 | 내용 |
| --- | --- |
| `main_integrated.py` | 통합 실행 진입점 |
| `vowel_recognition/` | 모음 인식 방법 1~7 (method_1 ~ method_7) |
| `audio_capture/` | 오디오 캡처 |
| `pitch_detection/` | 피치·VAD·비브라토 검출 |
| `text_effects/` | 텍스트 효과 |
| `text_morphing/` | 글자 모핑 |
| `ui/` | UI |
| `PROJECT_OVERVIEW.txt` | 프로젝트 개요 |
| `voicetypo_optimization_v2/v3.md` | 최적화 기록 |

## 상태

`main` 에서 이관된 스냅샷. 이후 통합 작업 방향은 ahyun 이 결정.
이 코드를 기준으로 회의에서 채택된 기능을 `main` 에 재구현하게 됨.
