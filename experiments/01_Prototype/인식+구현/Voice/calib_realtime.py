"""
calib_realtime.py - 캘리브레이션 모델 실시간 특징 스트림 (콘솔)
프레임마다 {모음(smooth), margin, F0, RMS(raw)} 를 한 줄로 갱신 출력한다.

인식 로직 자체는 recognizer.stream_features() 로 분리되어,
버블 시각화 전송(bubble_serve.py)과 완전히 동일한 코어를 공유한다.
이 파일은 그 결과를 콘솔에 보여주는 얇은 소비자일 뿐이다.
"""
from recognizer import stream_features


def main():
    name = input("화자 이름: ").strip()
    print("발성해보세요. Ctrl+C로 종료.\n")

    for f in stream_features(name):
        if f["silence"]:
            print("\r(무음)                                                  ", end="")
            continue
        f0_str = f"{f['f0']:.0f}Hz" if f["f0"] else "---"
        print(f"\rraw:{f['raw']}(2등:{f['second']})  vowel:{f['vowel']}  "
              f"margin:{f['margin']:.2f}  f0:{f0_str}  "
              f"rms:{f['rms']:.4f}     ", end="")


if __name__ == "__main__":
    main()
