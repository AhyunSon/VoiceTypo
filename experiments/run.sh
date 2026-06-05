#!/bin/bash
# VoiceTypo 데모 런처 — 사용법:  ./run.sh 1   (1~7)
# 07 은 백엔드+프론트 자동 실행.  마이크 없이 미리보기:  ./run.sh 7sim
ROOT="$(cd "$(dirname "$0")" && pwd)"

act() { source "$ROOT/$1/.venv/bin/activate" || { echo "❌ $1 세팅 안됨 — 먼저 ./setup_all.sh 실행"; exit 1; }; }

case "$1" in
  1) cd "$ROOT/01_method7_realtime" && act 01_method7_realtime && python main.py ;;
  2) cd "$ROOT/02_formant"          && act 02_formant          && python main.py ;;
  3) cd "$ROOT/03_whisper_ssl"      && act 03_whisper_ssl      && python scripts/03_run_realtime.py "${@:2}" ;;
  4) cd "$ROOT/04_light_ml"         && act 04_light_ml         && python scripts/04_live_test_v3.py --stage live "${@:2}" ;;
  5) cd "$ROOT/05_visual_sketches"  && act 05_visual_sketches  && python sketch_hub.py ;;
  6) cd "$ROOT/06_integrated"       && act 06_integrated       && python main_integrated.py ;;
  7|7sim)
    D="$ROOT/07_web_3d"
    SIMFLAG=""; [ "$1" = "7sim" ] && SIMFLAG="--sim"
    echo "▶ 백엔드(server.py) 시작…"
    ( cd "$D" && source .venv/bin/activate && python server.py $SIMFLAG ) &
    SRV=$!
    trap "echo; echo '⏹ 백엔드 종료'; kill $SRV 2>/dev/null" EXIT
    sleep 2
    echo "▶ 프론트(npm run dev) 시작 — 브라우저: http://localhost:3000"
    ( sleep 4 && open http://localhost:3000 ) &
    cd "$D/web" && npm run dev
    ;;
  3wav) cd "$ROOT/03_whisper_ssl" && act 03_whisper_ssl && python scripts/06_evaluate_wav_folder.py ;;
  4wav) cd "$ROOT/04_light_ml"    && act 04_light_ml    && python scripts/06_evaluate_wav_folder_v3.py ;;
  *)
    echo "사용법: ./run.sh <번호>"
    echo "  1   method7 실시간      2   포먼트(메인)⭐    3   Whisper SSL"
    echo "  4   경량 ML            5   시각화⭐           6   통합본(아현)"
    echo "  7   웹 3D⭐(서버+웹)    7sim  웹 3D 마이크없이 미리보기"
    echo "  3wav / 4wav   마이크 대신 녹음 wav 평가"
    ;;
esac
