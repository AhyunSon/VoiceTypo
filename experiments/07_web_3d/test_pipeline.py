"""파이프라인 통합 테스트 (시뮬레이션 모드).

마이크 없이 전체 흐름을 검증:
  FormantSimulator → formant_to_xyz → VowelTracker → weights_to_morph
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from voicetypo.audio.formant import FormantSimulator, VOWEL_FORMANTS
from voicetypo.vowel.space import formant_to_xyz, xyz_to_formant, DEFAULT_CENTROIDS
from voicetypo.vowel.tracker import VowelTracker
from voicetypo.sdf.mapper import weights_to_morph
import numpy as np


def test_formant_extraction():
    """포먼트 추출 기본 테스트 (합성 신호)."""
    from voicetypo.audio.formant import extract_formants

    sr = 44100
    dur = 0.02  # 20ms
    t = np.arange(int(sr * dur)) / sr

    # 단순 사인 조합 (F1=500, F2=1500)
    signal = (0.5 * np.sin(2 * np.pi * 500 * t)
              + 0.3 * np.sin(2 * np.pi * 1500 * t)).astype(np.float32)

    f1, f2, f3 = extract_formants(signal, sr)
    print(f"[포먼트 추출] F1={f1:.0f}, F2={f2:.0f}, F3={f3}")
    # LPC는 사인파에서 정확하지 않지만, NaN이 아니면 OK


def test_space_roundtrip():
    """xyz ↔ formant 왕복 변환 정확도."""
    print("\n[공간 변환 왕복 테스트]")
    for name, (f1, f2, f3) in VOWEL_FORMANTS.items():
        xyz = formant_to_xyz(f1, f2, f3)
        f1r, f2r, f3r = xyz_to_formant(xyz)
        err = abs(f1 - f1r) + abs(f2 - f2r) + abs(f3 - f3r)
        status = "OK" if err < 0.1 else "FAIL"
        print(f"  {name}: xyz={xyz.round(3)}  roundtrip_err={err:.6f}  [{status}]")


def test_centroids():
    """기본 중심점 간 거리 확인."""
    print("\n[중심점 거리 행렬]")
    names = list(DEFAULT_CENTROIDS.keys())
    # 우·오 거리가 가장 가까운지 확인
    min_dist = float('inf')
    min_pair = ('', '')
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            d = float(np.linalg.norm(DEFAULT_CENTROIDS[a] - DEFAULT_CENTROIDS[b]))
            if d < min_dist:
                min_dist = d
                min_pair = (a, b)
    print(f"  가장 가까운 쌍: {min_pair[0]}·{min_pair[1]} = {min_dist:.4f}")


def test_simulator_pipeline():
    """시뮬레이터 → 트래커 파이프라인."""
    print("\n[시뮬레이션 파이프라인]")

    sim = FormantSimulator(
        vowel_sequence=['아', '이', '우', '에', '오'],
        hold_frames=15,
        transition_frames=8,
    )
    tracker = VowelTracker(softness=8.0, hyst_frames=6, hyst_ratio=0.7)

    # 가짜 glyph_data (실제 SDF 데이터 대신 더미)
    dummy_glyph = object()
    glyph_data = {v: dummy_glyph for v in VOWEL_FORMANTS}

    prev_label = ''
    transitions = []

    for frame in range(150):
        f1, f2, f3 = sim.next_frame()
        xyz = formant_to_xyz(f1, f2, f3)
        pos = tracker.update(xyz)

        if pos.label and pos.label != prev_label:
            transitions.append((frame, pos.label))
            prev_label = pos.label

        # weights_to_morph 테스트
        morph = weights_to_morph(pos.weights, glyph_data)

        if frame % 20 == 0:
            top_w = sorted(pos.weights.items(), key=lambda x: -x[1])[:3]
            w_str = ' '.join(f'{n}={w:.2f}' for n, w in top_w)
            print(f"  frame {frame:3d}: label={pos.label or '?':2s}  "
                  f"speed={pos.speed:.4f}  {w_str}")

    print(f"\n  전환 이력: ", end='')
    for f, lbl in transitions:
        print(f"[{f}]{lbl}", end=' ')
    print()


def test_nan_handling():
    """NaN 입력 처리."""
    print("\n[NaN 처리 테스트]")
    tracker = VowelTracker()

    # 정상 입력
    xyz_good = formant_to_xyz(798, 1369, 2748)  # '아'
    pos = tracker.update(xyz_good)
    print(f"  정상: label={pos.label}  weights_sum={sum(pos.weights.values()):.4f}")

    # NaN 입력
    xyz_nan = np.array([float('nan'), float('nan'), float('nan')])
    pos = tracker.update(xyz_nan)
    print(f"  NaN:  label={pos.label}  weights={len(pos.weights)}")


if __name__ == '__main__':
    test_formant_extraction()
    test_space_roundtrip()
    test_centroids()
    test_simulator_pipeline()
    test_nan_handling()
    print("\n=== 전체 테스트 완료 ===")
