# web_p5js — 웹(p5.js) 파이프라인 + 전송 레이어

## 무엇 / 왜
스터디로그 4번(Reas/Processing 계보)의 웹 경로.
`02_formant/voice_data.py` 가 "데이터 채널 미정"으로 비워둔 **전송 레이어를 WebSocket 으로 채움**.
→ 분석은 Python, 시각화는 브라우저(p5.js)에서 빠르게 반복.

## 구성
- `bridge.py` — VoiceListener 의 최신 신호를 JSON 으로 ~30fps 브로드캐스트 (ws://localhost:8765)
- `index.html` — p5.js. sketch_02 와 같은 매핑(F1→높이, F2→폭, Pitch→회전, Volume→크기, 모음→형태)

## 실행
```bash
# 02_formant venv 에 websockets 필요 (이미 설치됨)
../02_formant/.venv/bin/pip install websockets   # 처음 한 번
python web_p5js/bridge.py                          # (상위 폴더에서 실행)
# → 브라우저로 web_p5js/index.html 열기 (그냥 파일 더블클릭 / 또는 로컬서버)
```

브라우저 콘솔에 연결 오류가 나면 bridge.py 가 먼저 떠 있는지 확인. 끊기면 1초마다 자동 재연결.

## 왜 웹인가
- 시각 실험 반복이 빠름(새로고침), 공유·전시 배포 쉬움.
- TouchDesigner/Processing 으로도 같은 JSON 을 받게 확장 가능(전송 레이어 재사용).

## 상태
전송 레이어 + p5 매핑 작성·검증 완료. 다음: 브라우저에서 매핑 튜닝, 입자/모핑 효과 추가.
