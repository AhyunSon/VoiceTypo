"""VoiceTypo WebSocket Server.

Python audio analysis backend.
Captures mic → extracts formants/pitch/vibrato/VAD → streams state to web frontend.

Usage:
    python server.py [--port 8765] [--sim]
"""

import asyncio
import json
import argparse
import numpy as np
import sys
import os
import time
import threading
import queue as thread_queue

# Add project root to path
sys.path.insert(0, os.path.dirname(__file__))

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from voicetypo.audio.capture import AudioCapture
from voicetypo.audio.formant import FormantTracker, FormantSimulator, VOWEL_FORMANTS
from voicetypo.vowel.space import formant_to_xyz, DEFAULT_CENTROIDS
from voicetypo.vowel.tracker import VowelTracker
from voicetypo.vowel.calibrator import Calibrator, VOWELS
from pitch_detection.yin import YinDetector
from pitch_detection.vibrato import VibratoAnalyzer
from pitch_detection.vad import VoiceActivityDetector

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Global State ───

class AnalysisEngine:
    """Wraps all audio analysis into a single streaming engine."""

    def __init__(self, sim_mode=False):
        self.sim_mode = sim_mode
        self.sample_rate = 44100

        # Audio capture
        self.capture = None if sim_mode else AudioCapture()

        # Analysis modules
        self.formant_tracker = FormantTracker(sr=self.sample_rate)
        self.yin = YinDetector(sample_rate=self.sample_rate)
        self.vibrato = VibratoAnalyzer(frames_per_sec=self.sample_rate / 882)
        self.vad = VoiceActivityDetector()

        # Vowel tracking
        self.vowel_tracker = VowelTracker()

        # Calibration
        self.calibrator = None
        self._calibration_data = None

        # Simulation
        self.simulator = FormantSimulator() if sim_mode else None

        # Thread-safe queue (audio callback runs in sounddevice thread)
        self._thread_queue = thread_queue.Queue(maxsize=100)
        self._async_queue = asyncio.Queue(maxsize=100)
        self._running = False
        self._clients = set()
        self._loop = None

        # Latest state for new connections
        self._latest_state = None

    def load_calibration(self, path="calibration.npz"):
        """Load saved calibration if available."""
        try:
            data = Calibrator.load(path)
            self._calibration_data = data
            self.vowel_tracker = VowelTracker(centroids=data)
            return True
        except Exception:
            return False

    def start_calibration(self):
        """Begin calibration session."""
        self.calibrator = Calibrator(vowels=['아', '어', '오', '우', '으', '이', '에'])

    def _on_audio(self, audio, sr):
        """Audio callback - runs in sounddevice thread."""
        try:
            # Pitch + RMS
            freq, rms = self.yin.detect(audio)

            # freq fallback — pitch 실패 시 이전 값 유지
            if freq == 0 and hasattr(self, '_prev_freq') and self._prev_freq > 0:
                freq = self._prev_freq
            self._prev_freq = freq

            # VAD
            self.vad.update(rms, freq)

            # Vibrato
            self.vibrato.push(freq, rms)
            vib_rate, vib_extent = self.vibrato.get()

            # Formants
            f1, f2, f3 = self.formant_tracker.update(audio)

            # XYZ position & vowel tracking — only when voice is active
            if self.vad.is_active:
                xyz = formant_to_xyz(f1, f2, f3)
                vpos = self.vowel_tracker.update(xyz)
            else:
                vpos = self.vowel_tracker.last_position()

            # Calibration feed — 유저가 모음을 선택한 경우에만 feed
            cal_state = None
            if self.calibrator and not self.calibrator.is_complete:
                current = self.calibrator.current_vowel
                session = self.calibrator.current_session
                # 발성 시작/끝 노이즈 필터
                if not hasattr(self, '_cal_rms_sum'):
                    self._cal_rms_sum = 0.0
                    self._cal_rms_count = 0
                    self._cal_vad_frames = 0
                if current and self.vad.is_active:
                    self._cal_vad_frames += 1
                    self._cal_rms_sum += rms
                    self._cal_rms_count += 1
                    rms_avg = self._cal_rms_sum / self._cal_rms_count
                    # 처음 3프레임 버림 + 끝부분 RMS 평균의 30% 이하 버림
                    rms_ok = (self._cal_vad_frames > 3) and (
                        self._cal_rms_count < 10 or rms > rms_avg * 0.3
                    )
                else:
                    self._cal_vad_frames = 0
                    rms_ok = False
                if current and self.vad.is_active and rms_ok:
                    result = self.calibrator.feed(f1, f2, f3)
                    if session and len(session.trajectory) > 0 and len(session.trajectory) % 50 == 1:
                        print(f"[CAL] vowel={current}, traj_len={len(session.trajectory)}, f1={f1:.0f}, f2={f2:.0f}")
                    cal_state = {
                        'currentVowel': current,
                        'progress': list(self.calibrator.progress),
                        'stabilityProgress': session.stability_progress if session else 0,
                        'isConfirmed': session.is_confirmed if session else False,
                        'isComplete': self.calibrator.is_complete,
                        'trajectory': [session.trajectory[-1].tolist()] if session and len(session.trajectory) > 0 else [],
                    }
                    if result is not None:
                        cal_state['justLocked'] = True
                        cal_state['lockedCenter'] = result.tolist()
                        # 확정 후 선택 해제 — 유저가 다음 모음을 직접 선택해야 함
                        self.calibrator.deselect()
                else:
                    cal_state = {
                        'currentVowel': current or '',
                        'progress': list(self.calibrator.progress),
                        'stabilityProgress': session.stability_progress if session else 0,
                        'isConfirmed': session.is_confirmed if session else False,
                        'isComplete': self.calibrator.is_complete,
                        'trajectory': [],
                    }

            # Build state
            state = {
                'type': 'state',
                'timestamp': time.time(),
                'audio': {
                    'freq': round(freq, 1),
                    'rms': round(rms, 5),
                    'vad': self.vad.is_active,
                    'vibratoRate': round(vib_rate, 2),
                    'vibratoExtent': round(vib_extent, 3),
                },
                'formants': {
                    'f1': round(f1, 1) if not np.isnan(f1) else None,
                    'f2': round(f2, 1) if not np.isnan(f2) else None,
                    'f3': round(f3, 1) if not np.isnan(f3) else None,
                },
                'vowelSpace': {
                    'xyz': [round(v, 4) for v in vpos.xyz.tolist()] if not np.any(np.isnan(vpos.xyz)) else None,
                    'weights': {k: round(v, 4) for k, v in vpos.weights.items()} if vpos.weights else {},
                    'velocity': [round(v, 5) for v in vpos.velocity.tolist()],
                    'label': vpos.label,
                    'speed': round(vpos.speed, 5),
                },
            }

            if cal_state:
                state['calibration'] = cal_state

            self._latest_state = state

            # Thread-safe put
            try:
                self._thread_queue.put_nowait(state)
            except thread_queue.Full:
                try:
                    self._thread_queue.get_nowait()
                    self._thread_queue.put_nowait(state)
                except Exception:
                    pass

        except Exception as e:
            print(f"[AnalysisEngine] Error: {e}")

    def _sim_tick(self):
        """Simulation mode tick."""
        f1, f2, f3 = self.simulator.next_frame()
        freq = 220.0
        rms = 0.15

        self.vad.is_active = True
        self.vibrato.push(freq, rms)
        vib_rate, vib_extent = self.vibrato.get()

        xyz = formant_to_xyz(f1, f2, f3)
        vpos = self.vowel_tracker.update(xyz)

        # Calibration feed (sim mode)
        cal_state = None
        if self.calibrator and not self.calibrator.is_complete:
            result = self.calibrator.feed(f1, f2, f3)
            session = self.calibrator.current_session
            cal_state = {
                'currentVowel': self.calibrator.current_vowel,
                'progress': list(self.calibrator.progress),
                'stabilityProgress': session.stability_progress if session else 0,
                'isConfirmed': session.is_confirmed if session else False,
                'isComplete': self.calibrator.is_complete,
                'trajectory': session.trajectory[-50:].tolist() if session and len(session.trajectory) > 0 else [],
            }
            if result is not None:
                cal_state['justLocked'] = True
                cal_state['lockedCenter'] = result.tolist()

        state = {
            'type': 'state',
            'timestamp': time.time(),
            'audio': {
                'freq': round(freq, 1),
                'rms': round(rms, 5),
                'vad': True,
                'vibratoRate': round(vib_rate, 2),
                'vibratoExtent': round(vib_extent, 3),
            },
            'formants': {
                'f1': round(f1, 1),
                'f2': round(f2, 1),
                'f3': round(f3, 1),
            },
            'vowelSpace': {
                'xyz': [round(v, 4) for v in vpos.xyz.tolist()],
                'weights': {k: round(v, 4) for k, v in vpos.weights.items()},
                'velocity': [round(v, 5) for v in vpos.velocity.tolist()],
                'label': vpos.label,
                'speed': round(vpos.speed, 5),
            },
        }

        if cal_state:
            state['calibration'] = cal_state

        self._latest_state = state
        return state

    async def start(self):
        """Start audio processing."""
        self._running = True
        self._loop = asyncio.get_event_loop()
        if self.sim_mode:
            asyncio.create_task(self._sim_loop())
        else:
            self.capture.add_listener(self._on_audio)
            self.capture.start()
            # Bridge thread queue → async queue
            asyncio.create_task(self._bridge_queues())

    async def _bridge_queues(self):
        """Transfer items from thread-safe queue to async queue."""
        while self._running:
            try:
                state = self._thread_queue.get_nowait()
                try:
                    self._async_queue.put_nowait(state)
                except asyncio.QueueFull:
                    try:
                        self._async_queue.get_nowait()
                        self._async_queue.put_nowait(state)
                    except Exception:
                        pass
            except thread_queue.Empty:
                pass
            await asyncio.sleep(0.005)  # ~200Hz polling

    async def _sim_loop(self):
        """Simulation loop running at ~50fps."""
        while self._running:
            state = self._sim_tick()
            try:
                self._async_queue.put_nowait(state)
            except asyncio.QueueFull:
                try:
                    self._async_queue.get_nowait()
                    self._async_queue.put_nowait(state)
                except Exception:
                    pass
            await asyncio.sleep(0.02)  # 50fps

    async def stop(self):
        self._running = False
        if not self.sim_mode and self.capture:
            self.capture.stop()

    def get_vowel_centers(self):
        """Return current vowel centers for frontend."""
        centroids = self._calibration_data or DEFAULT_CENTROIDS
        return {
            name: [round(v, 4) for v in c.tolist()]
            for name, c in centroids.items()
        }


# ─── Global engine instance ───
engine = None


@app.on_event("startup")
async def startup():
    global engine
    engine = AnalysisEngine(sim_mode="--sim" in sys.argv)
    engine.load_calibration(
        os.path.join(os.path.dirname(__file__), "calibration.npz")
    )
    await engine.start()
    # Start broadcast loop
    asyncio.create_task(_broadcast_loop())
    print(f"[VoiceTypo] Server started (sim={'--sim' in sys.argv})")


@app.on_event("shutdown")
async def shutdown():
    if engine:
        await engine.stop()


async def _broadcast_loop():
    """Read from async queue and broadcast to all connected clients."""
    while True:
        state = await engine._async_queue.get()
        clients = list(engine._clients)  # snapshot to avoid set mutation during iteration
        if not clients:
            continue
        msg = json.dumps(state, ensure_ascii=False)
        dead = []
        for ws in clients:
            try:
                await ws.send_text(msg)
            except Exception:
                dead.append(ws)
        for ws in dead:
            engine._clients.discard(ws)


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    engine._clients.add(ws)

    # Send initial config
    await ws.send_json({
        'type': 'config',
        'vowelCenters': engine.get_vowel_centers(),
        'vowelOrder': ['아', '어', '오', '우', '으', '이', '에'],
        'hasCalibration': engine._calibration_data is not None,
    })

    try:
        # Only need to receive commands from this client
        await _recv_commands(ws)
    except WebSocketDisconnect:
        pass
    finally:
        engine._clients.discard(ws)


async def _recv_commands(ws: WebSocket):
    """Receive commands from client."""
    print("[WS] _recv_commands started, waiting for messages...")
    while True:
        try:
            raw = await ws.receive_text()
            print(f"[WS] Raw received: {raw[:200]}")
            data = json.loads(raw)
            cmd = data.get('command')
            print(f"[WS] Command: {cmd}")

            if cmd == 'startCalibration':
                engine.start_calibration()
                await ws.send_json({
                    'type': 'calibrationStarted',
                    'vowelOrder': engine.calibrator.vowels,
                })

            elif cmd == 'advanceCalibration':
                if engine.calibrator:
                    engine.calibrator.advance()

            elif cmd == 'retryCalibration':
                if engine.calibrator:
                    engine.calibrator.retry()

            elif cmd == 'skipCalibration':
                if engine.calibrator:
                    engine.calibrator.skip()

            elif cmd == 'selectVowel':
                vowel = data.get('vowel')
                print(f"[WS] selectVowel: {vowel}, calibrator={engine.calibrator is not None}")
                if engine.calibrator and vowel:
                    engine.calibrator.select_vowel(vowel)
                    engine._cal_rms_sum = 0.0
                    engine._cal_rms_count = 0
                    engine._cal_vad_frames = 0
                    print(f"[WS] → now current_vowel={engine.calibrator.current_vowel}")

            elif cmd == 'deselectVowel':
                if engine.calibrator:
                    engine.calibrator.deselect()

            elif cmd == 'resetCalibration':
                engine.start_calibration()
                print("[WS] Calibration reset")

            elif cmd == 'finishCalibration':
                if engine.calibrator and engine.calibrator.is_complete:
                    cal_path = os.path.join(
                        os.path.dirname(__file__), "calibration.npz"
                    )
                    engine.calibrator.save(cal_path)
                    engine._calibration_data = engine.calibrator.centroids
                    engine.vowel_tracker = VowelTracker(
                        centroids=engine.calibrator.centroids
                    )
                    await ws.send_json({
                        'type': 'calibrationComplete',
                        'vowelCenters': engine.get_vowel_centers(),
                    })

        except WebSocketDisconnect:
            break
        except Exception as e:
            print(f"[WS] Command error: {e}")


@app.get("/health")
async def health():
    return {"status": "ok", "sim": engine.sim_mode if engine else False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--sim", action="store_true")
    args = parser.parse_args()

    # Pass --sim through sys.argv for the startup event
    if args.sim and "--sim" not in sys.argv:
        sys.argv.append("--sim")

    uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="info")
