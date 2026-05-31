"""
bridge.py — Python 음성분석 → WebSocket → 브라우저(p5.js)

voice_data.py 가 "데이터 채널 미정"으로 비워둔 전송 레이어를 채운다.
VoiceListener 의 최신 신호를 JSON 으로 ~30fps 브로드캐스트.

필요: pip install websockets   (02_formant/.venv 에 추가)
실행: python web_p5js/bridge.py   → 그다음 web_p5js/index.html 을 브라우저로 열기
"""

import sys, json, asyncio
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from voice_input import VoiceListener

try:
    import websockets
except ImportError:
    print("websockets 가 없습니다.  ../02_formant/.venv/bin/pip install websockets  후 다시 실행")
    sys.exit(1)

HOST, PORT = "localhost", 8765


async def handler(ws):
    lis = ws.listener
    print("브라우저 연결됨")
    try:
        while True:
            s = lis.latest()
            if s:
                await ws.send(json.dumps(dict(
                    voiced=s.voiced, f0=s.f0, f1=s.f1, f2=s.f2, f3=s.f3,
                    rms=s.rms, jitter=s.jitter, vowel=s.vowel,
                    weights=s.vowel_weights,
                )))
            await asyncio.sleep(0.033)
    except websockets.ConnectionClosed:
        print("브라우저 연결 종료")


async def main():
    listener = VoiceListener().start()

    async def _handler(ws):
        ws.listener = listener
        await handler(ws)

    print(f"ws://{HOST}:{PORT} 대기 중 — index.html 을 브라우저로 여세요 (Ctrl+C 종료)")
    async with websockets.serve(_handler, HOST, PORT):
        await asyncio.Future()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n종료")
