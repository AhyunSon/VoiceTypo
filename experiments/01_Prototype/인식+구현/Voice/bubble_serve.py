"""
bubble_serve.py - 실시간 인식 → 버블 시각화 WebSocket 브리지
============================================================
recognizer.stream_features() 의 프레임별 결과({vowel, margin, f0, rms})를
WebSocket 으로 브로드캐스트한다. 버블 구현(웹앱)이 이 서버에 접속해
그대로 VoiceFrame 으로 만들어 파이프라인에 흘려보낸다.

(README §9 의 "OSC 전송층" 자리 — 시각화팀 환경이 웹앱으로 확정되어 WebSocket 으로 구현.)

실행:
    pip install websockets            # 최초 1회
    python bubble_serve.py            # 화자 이름 입력 → ws://0.0.0.0:8765 로 송출
    python bubble_serve.py 9000       # 포트 지정

접속(브라우저): 버블 구현 앱에서 "인식기" 입력 소스 선택 → ws://localhost:8765
"""
import sys
import json
import asyncio
import threading

try:
    import websockets
except ImportError:
    sys.exit("websockets 패키지가 필요합니다:  pip install websockets")

from recognizer import stream_features, load_model

HOST = "0.0.0.0"
DEFAULT_PORT = 8765


async def serve(name, host, port):
    clients = set()
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue()

    async def handler(ws, *_):
        # 클라이언트(브라우저)는 수신만 한다. 들어오는 메시지는 무시.
        clients.add(ws)
        peer = getattr(ws, "remote_address", None)
        print(f"\n[+] 접속: {peer}  (현재 {len(clients)}개)")
        try:
            async for _msg in ws:
                pass
        except Exception:
            pass
        finally:
            clients.discard(ws)
            print(f"\n[-] 종료: {peer}  (현재 {len(clients)}개)")

    def producer():
        # 블로킹 인식 제너레이터를 별도 스레드에서 돌리고,
        # 프레임을 asyncio 큐로 넘긴다.
        try:
            for frame in stream_features(name):
                loop.call_soon_threadsafe(queue.put_nowait, frame)
        except Exception as exc:  # noqa: BLE001
            loop.call_soon_threadsafe(queue.put_nowait, {"error": str(exc)})

    threading.Thread(target=producer, daemon=True).start()

    async with websockets.serve(handler, host, port):
        print(f"WebSocket 송출 시작: ws://{host}:{port}")
        print("버블 앱에서 '인식기' 소스로 접속하세요. Ctrl+C 로 종료.\n")
        while True:
            frame = await queue.get()
            if "error" in frame:
                print(f"\n[!] 인식 스레드 오류: {frame['error']}")
                continue
            if not clients:
                continue
            msg = json.dumps(frame, ensure_ascii=False)
            await asyncio.gather(*(c.send(msg) for c in list(clients)),
                                 return_exceptions=True)


def main():
    port = DEFAULT_PORT
    if len(sys.argv) > 1:
        try:
            port = int(sys.argv[1])
        except ValueError:
            sys.exit(f"포트 번호가 올바르지 않습니다: {sys.argv[1]}")

    name = input("화자 이름: ").strip()
    # 모델을 먼저 검증해 빠르게 실패(친절한 메시지).
    try:
        load_model(name)
    except FileNotFoundError:
        sys.exit(f"모델 파일이 없습니다: calib_model_{name}.pkl "
                 f"(먼저 calibrate.py → calib_train.py 를 실행하세요)")

    try:
        asyncio.run(serve(name, HOST, port))
    except KeyboardInterrupt:
        print("\n종료.")


if __name__ == "__main__":
    main()
