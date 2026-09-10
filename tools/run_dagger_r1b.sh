#!/usr/bin/env bash
# DAgger round 1b — collect, gate, merge, train, record.
# DAgger 1b — 수집·게이트·병합·학습·기록.
#
# Why this is a file and not a paste:
# 왜 붙여넣기가 아니라 파일인가:
#   Pasting a heredoc that contains a Python file into a terminal truncates.
#   It happened on 2026-09-09 and the corruption interleaved the script with
#   itself, so the failure did not look like a paste failure. Files go through
#   git; the terminal only ever receives one line.
#   파이썬 파일을 담은 heredoc 을 터미널에 붙여넣으면 잘린다. 2026-09-09 에
#   실제로 났고, 깨진 내용이 스크립트끼리 뒤섞여서 **붙여넣기 실패처럼 보이지도
#   않았다.** 파일은 git 으로 넘기고 터미널은 한 줄만 받는다.
#
# Each stage writes an artefact before the next begins, so a failure at stage 3
# does not destroy stages 1 and 2. Re-running skips completed stages.
# 각 단계는 다음 단계 전에 산출물을 남긴다. 3단계에서 죽어도 1·2단계가 살아 있고,
# 다시 돌리면 끝난 단계를 건너뛴다.
#
#   # [서버]
#   bash tools/run_dagger_r1b.sh
#   bash tools/run_dagger_r1b.sh out/dagger_r1b_20260910_1200   # 이어서 돌리기

set -euo pipefail

cd "$(dirname "$0")/.."          # AI/ 로 이동

export AI_THREADS="${AI_THREADS:-2}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1

RUN="${1:-out/dagger_r1b_$(date +%Y%m%d_%H%M%S)}"
LABELS="$RUN/labels"
MERGED="$RUN/merged"
LOGS="$RUN/logs"
mkdir -p "$RUN" "$LOGS"

DEMO=datasets/sim_pick_v5
CKPTS=(
  checkpoints/bc/sim_pick_v5_seed0.pt
  checkpoints/bc/sim_pick_v5_seed1.pt
  checkpoints/bc/sim_pick_v5_seed2.pt
)

say() { printf '\n\033[1m== %s ==\033[0m\n' "$*"; }

say "실행 디렉터리 $RUN"
git rev-parse --short HEAD
git status --short || true

# ---------------------------------------------------------------------------
# 0. 인자 대조 — 규칙: 일괄 실행에 넣는 명령은 전부 --help 로 확인한 뒤 쓴다
# ---------------------------------------------------------------------------
say "0. --help 대조"
python tools/collect_dagger_segments.py --help > "$LOGS/help_collect.txt"
python tools/repeat_runs.py --help          > "$LOGS/help_repeat.txt"
echo "OK"

# ---------------------------------------------------------------------------
# 1. 수집
# ---------------------------------------------------------------------------
if [ -f "$LABELS/dagger_segments_result.json" ]; then
  say "1. 수집 — 이미 있다. 건너뛴다"
else
  say "1. 수집"
  CK_ARGS=()
  for c in "${CKPTS[@]}"; do
    [ -f "$c" ] || { echo "체크포인트 없다: $c" >&2; exit 1; }
    CK_ARGS+=(--policy-ckpt "$c")
  done
  set +e
  python tools/collect_dagger_segments.py \
    "${CK_ARGS[@]}" \
    --episodes 100 \
    --seed-base 4000 \
    --label-start-tick 30 \
    --out "$LABELS" \
    --log 2>&1 | tee "$LOGS/collect.log"
  rc=${PIPESTATUS[0]}
  set -e
  echo "collect exit=$rc" | tee "$RUN/collect_rc.txt"
fi

RESULT="$LABELS/dagger_segments_result.json"
[ -f "$RESULT" ] || { echo "수집 결과 파일이 없다: $RESULT" >&2; exit 1; }

# ---------------------------------------------------------------------------
# 2. MEASURE 초안 — 게이트 통과 여부와 무관하게 남긴다
#    실패한 수집도 기록이다. 실패를 안 남기면 같은 실패를 반복한다.
# ---------------------------------------------------------------------------
say "2. MEASURE 기록"
python tools/write_dagger_measure.py \
  --result "$RESULT" \
  --log "$LOGS/collect.log" \
  --out docs/MEASURE_dagger_segments_0909.md
echo "→ docs/MEASURE_dagger_segments_0909.md"

GO=$(python -c "import json,sys; print(int(json.load(open(sys.argv[1]))['gates']['train_go']))" "$RESULT")
if [ "$GO" != "1" ]; then
  say "수집 게이트 실패 — 학습으로 넘어가지 않는다"
  cat docs/MEASURE_dagger_segments_0909.md
  echo
  echo "게이트를 옮기지 마라. 원인을 먼저 재고, 바꿀 거면 사전등록 문서를 새로 쓴다."
  exit 2
fi

# ---------------------------------------------------------------------------
# 3. 병합
# ---------------------------------------------------------------------------
if [ -f "$MERGED/dataset.json" ]; then
  say "3. 병합 — 이미 있다. 건너뛴다"
else
  say "3. 병합"
  python tools/merge_datasets.py "$DEMO" "$LABELS" "$MERGED" 2>&1 | tee "$LOGS/merge.log"
fi

# ---------------------------------------------------------------------------
# 4. 학습 3회 + 롤아웃
#    ⚠️ 3회는 **고장검사**다. 조건 비교가 아니다 — D-AI-30 은 조건 비교에 5회를
#       요구한다. 이 결과로 "DAgger 가 낫다" 를 주장하면 안 된다.
# ---------------------------------------------------------------------------
if [ -f "$LOGS/repeat_runs.log" ]; then
  say "4. 학습 — 로그가 이미 있다. 건너뛴다 (다시 하려면 로그를 지워라)"
else
  say "4. 학습 3회 (고장검사. 조건 비교 아님 — D-AI-30 은 5회)"
  python tools/repeat_runs.py \
    --data "$MERGED" \
    --runs 3 \
    --episodes 100 \
    --epochs 30 \
    --seed-base 0 \
    --eval-seed-base 3000 \
    --tag dagger_segments_0909 \
    --device cuda \
    --log 2>&1 | tee "$LOGS/repeat_runs.log"
fi

# ---------------------------------------------------------------------------
# 5. 결과 이어붙이기 + FINDINGS 재생성
# ---------------------------------------------------------------------------
say "5. 기록"
{
  printf '\n\n## 3회 학습·롤아웃 (고장검사. 조건 비교 아님 — D-AI-30 은 5회)\n\n'
  printf '```text\n'
  grep -E '시드|평균|범위|합산|게이트|bc[[:space:]]+[0-9]+/100' \
    "$LOGS/repeat_runs.log" || echo "(패턴 일치 없음 — 원본 로그를 봐라)"
  printf '```\n'
  printf '\n원본 로그: `%s`\n' "$LOGS/repeat_runs.log"
} >> docs/MEASURE_dagger_segments_0909.md

python tracking/findings.py

say "최종"
cat docs/MEASURE_dagger_segments_0909.md
say "변경 파일"
git status --short
echo
echo "실행 디렉터리: $RUN"
