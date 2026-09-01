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
# Counting before deciding, because a half-finished collection is the dangerous
# case: silently reusing it trains on fewer episodes than the run claims, and
# nothing downstream would say so.
# 세기 전에 판단하지 않는다. 중단된 수집분이 위험한 경우다 — 조용히 재사용하면
# 실행이 주장하는 것보다 적은 편수로 학습하게 되고, 하류 어디서도 그 사실이
# 드러나지 않는다.
shopt -s nullglob
EXISTING=("${DATASET}"/*.npz)
shopt -u nullglob
N_HAVE=${#EXISTING[@]}

if [ "${N_HAVE}" -ge "${N_COLLECT}" ]; then
  echo "  기존 ${N_HAVE}편 재사용 (요청 ${N_COLLECT}편)"
elif [ "${N_HAVE}" -gt 0 ]; then
  echo "  ✗ ${DATASET} 에 ${N_HAVE}편만 있다. 요청은 ${N_COLLECT}편이다."
  echo "    중단된 수집분일 수 있고, 이어붙이면 어떤 조건으로 몇 편을 모았는지"
  echo "    알 수 없게 된다. 지우고 다시 받거나 다른 --out 이름을 쓸 것:"
  echo "      rm -rf ${DATASET}"
  exit 1
else
  python tools/collect_sim.py --episodes "${N_COLLECT}" --jitter 0.05 --out "${DATASET}" --log
fi

echo
echo "[2/5] 계약 검증 — 위반이 하나라도 있으면 학습하지 않는다"
python tools/verify_dataset.py "${DATASET}" --write-index --log

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
