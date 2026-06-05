#!/bin/bash
# VoiceTypo 데모 전체 세팅 — venv(python3.12) 생성 + 패키지 설치 + 07 npm install
# 한 번만 실행하면 됨.  실패한 실험만 다시 돌려도 됨.
ROOT="$(cd "$(dirname "$0")" && pwd)"
PY=python3.12
LOG="$ROOT/setup_log.txt"
: > "$LOG"
ok=(); fail=()

pyenv_setup() {  # $1=폴더  $2...=pip install 인자
  dir="$1"; shift
  echo "===== [$dir] 세팅 시작 =====" | tee -a "$LOG"
  cd "$ROOT/$dir" || { fail+=("$dir(cd)"); return; }
  if [ ! -d .venv ]; then
    $PY -m venv .venv >>"$LOG" 2>&1 || { echo "❌ venv 생성 실패 $dir" | tee -a "$LOG"; fail+=("$dir(venv)"); return; }
  fi
  source .venv/bin/activate
  python -m pip install --upgrade pip wheel "setuptools<81" >>"$LOG" 2>&1
  if python -m pip install "$@" >>"$LOG" 2>&1; then
    echo "✅ [$dir] 완료" | tee -a "$LOG"; ok+=("$dir")
  else
    echo "❌ [$dir] pip 실패 — setup_log.txt 확인" | tee -a "$LOG"; fail+=("$dir(pip)")
  fi
  deactivate
}

pyenv_setup 01_method7_realtime torch transformers praat-parselmouth pyworld scikit-learn sounddevice numpy scipy PySide6 pyqtgraph
pyenv_setup 02_formant         -r requirements.txt
pyenv_setup 03_whisper_ssl     -r requirements.txt
pyenv_setup 04_light_ml        torch torchaudio transformers numpy scipy sounddevice soundfile librosa
pyenv_setup 05_visual_sketches -r requirements.txt
pyenv_setup 06_integrated      -r requirements.txt
pyenv_setup 07_web_3d          -r requirements_web.txt

# 07 프론트엔드
echo "===== [07_web_3d/web] npm install =====" | tee -a "$LOG"
if ( cd "$ROOT/07_web_3d/web" && npm install >>"$LOG" 2>&1 ); then
  echo "✅ [07 web] npm 완료" | tee -a "$LOG"; ok+=("07_web")
else
  echo "❌ [07 web] npm 실패" | tee -a "$LOG"; fail+=("07_web(npm)")
fi

echo "" | tee -a "$LOG"
echo "========== 세팅 결과 ==========" | tee -a "$LOG"
echo "✅ 성공: ${ok[*]}" | tee -a "$LOG"
echo "❌ 실패: ${fail[*]:-없음}" | tee -a "$LOG"
echo "이제 ./run.sh 1 ~ ./run.sh 7 로 실행하세요." | tee -a "$LOG"
