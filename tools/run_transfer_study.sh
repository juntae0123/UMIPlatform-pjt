#!/usr/bin/env bash
# End-to-end run for D-AI-18: does simulation training transfer?
# D-AI-18 전체 실행: 시뮬 학습은 전이되는가?
#
# Every stage gates the next one. A failed gate stops the run rather than
# producing a downstream number that looks fine and means nothing.
# 각 단계가 다음 단계의 게이트다. 게이트 실패는 실행을 멈춘다. 그럴듯해 보이지만
# 아무 의미 없는 하류 수치를 만드는 대신에.
#
# 사용법: bash tools/run_transfer_study.sh [GPU_ID] [N_EPISODES]

set -euo pipefail

GPU="${1:-0}"
N_COLLECT="${2:-100}"
N_EVAL=20
DATASET="datasets/sim_pick_v1"
CKPT="checkpoints/bc/bc_sim_pick_v1.pt"

export MUJOCO_GL=egl
export CUDA_VISIBLE_DEVICES="${GPU}"

echo "=============================================="
echo " D-AI-18 전이 계측  ·  GPU ${GPU}  ·  수집 ${N_COLLECT}편"
echo "=============================================="

echo
echo "[0/5] 계측기 검증 — 여기서 실패하면 뒤의 수치는 전부 무효다"
python tools/check_domain.py

echo
echo "[1/5] 조건 A 수집 (도메인 랜덤화 OFF)"
if [ -d "${DATASET}" ] && [ -n "$(ls -A "${DATASET}"/*.npz 2>/dev/null)" ]; then
  echo "  이미 있음 — 건너뜀 (${DATASET})"
else
  python tools/collect_sim.py --episodes "${N_COLLECT}" --jitter 0.05 --out "${DATASET}" --log
fi

echo
echo "[2/5] 계약 검증 — 위반이 하나라도 있으면 학습하지 않는다"
python tools/verify_dataset.py --dataset "${DATASET}" --log

echo
echo "[3/5] BC 실데이터 학습 (첫 실학습. 이전 ckpt 는 random_tensors 였다)"
python tools/train_bc.py --data "${DATASET}" --device cuda --out "${CKPT}" --log

echo
echo "[4/5] 조건 A 롤아웃 — baseline 게이트. 못 넘으면 붕괴율은 재지 않는다"
python tools/eval_rollout.py --episodes "${N_EVAL}" --render \
  --replay-from "${DATASET}" --policy-ckpt "${CKPT}" --log

echo
echo "[5/5] A/B 전이 붕괴율 — D-AI-18 판정"
python tools/measure_transfer.py --episodes "${N_EVAL}" --render \
  --policy-ckpt "${CKPT}" --log

echo
echo "완료. EXP_LOG.jsonl 을 커밋하고 MEASURE 문서로 옮길 것."
