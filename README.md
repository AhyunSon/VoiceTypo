# ahyun — 개인 실험 브랜치

VoiceTypo 프로젝트의 **ahyun 개인 브랜치**입니다.
모든 실험은 `experiments/` 아래 폴더로 분리해서 관리합니다. (협업 가이드 §3 기준)

## 폴더 구조

```
experiments/
└── 01_voicetypo_integrated/   VoiceTypo 통합 프로젝트 (main 에서 이관)
```

각 폴더 안의 `README.md` 에 작성자 / 무엇 / 구성 / 상태가 정리돼 있습니다.

## 규칙

- 이 브랜치는 main 에 직접 merge 하지 않습니다 (실험·작업 기록용).
- 채택된 기능은 담당자가 별도 `feature/` 브랜치에서 main 에 재구현합니다.
- 새 실험은 `experiments/번호_주제/` 폴더를 추가하는 방식으로 진행합니다.
- 음성·모델·대용량 파일은 커밋하지 않습니다 (`.gitignore` 참고).

자세한 협업 규칙은 팀 GitHub 가이드를 따릅니다.
