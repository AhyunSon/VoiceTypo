# 어 진단 보고서 — H1 vs H2

## 가설

- **H1**: 본인 어의 진짜 F1 이 학계 평균(629Hz)보다 낮음 (~300Hz)
  → 알고리즘 수정으로 해결 불가
- **H2**: 진짜 F1 은 ~600Hz, Praat LPC 가 F0 harmonic 에 빠져 못 찾음
  → Cepstral smoothing 으로 복원 가능

## 방법

어 wav 5개 (어_01~05) 의 중앙 1초 구간에 대해:
1. Hamming 윈도우 + FFT → raw spectrum
2. Cepstral liftering (lifter quefrency 4.5 ms) → smoothed envelope
3. envelope 의 peak 검출 (500-800Hz 영역 = 학계 어 F1 영역)
4. Praat 의 F1/F2 와 비교

![어 spectrum](diagnose_eo_spectrum.png)

## 결과

| 파일 | F0 | Praat F1 | Praat F2 | Cepstral peaks ≤3000Hz | 500-800Hz peak |
|---|---:|---:|---:|---|---|
| 어_01.wav | 245 | 319 | 1072 | 242, 989 | ✗ 없음 |
| 어_02.wav | 238 | 296 | 1006 | 970 | ✗ 없음 |
| 어_03.wav | 268 | 355 | 1090 | 280, 561, 1075 | ✓ 561Hz |
| 어_04.wav | 231 | 258 | 1048 | 232, 973 | ✗ 없음 |
| 어_05.wav | 254 | 729 | 1080 | 293, 553, 821 | ✓ 553Hz |

**500-800Hz peak 검출**: 2/5 파일

## 판정

**불명확** — 2/5 만 peak. 추가 진단 또는 cepstral 시도 후 판단.

## 다음 단계

**불명확**:
- 추가 진단 또는 cepstral 시도 후 판단
