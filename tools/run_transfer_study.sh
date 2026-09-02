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
DATASET="datasets/sim_pick_v2"   # 계약 0.2.0 (skill_id 필수). v1 은 0.1.0 이라 무효
CKPT="checkpoints/bc/bc_sim_pick_v2.pt"

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
# collect writes dataset.json only after the whole loop finishes, so its presence
# is the completion marker — not the episode count. A run of 100 legitimately
# yields fewer files because unreachable placements are dropped; judging by count
# would reject a perfectly good dataset. Loose .npz files with no index, on the
# other hand, mean the collection died partway, and reusing those trains on an
# unknown number of episodes under unrecorded conditions.
# collect 는 루프가 전부 끝난 뒤에만 dataset.json 을 쓴다. 따라서 완료 판정 기준은
# 편수가 아니라 인덱스 파일의 존재다. 100편 요청이 그보다 적은 파일을 남기는 것은
# 정상이다 — 도달 불가 배치는 버려진다. 편수로 판정하면 멀쩡한 데이터셋을 거부한다.
# 반대로 인덱스 없이 .npz 만 굴러다니면 수집이 도중에 죽은 것이고, 그걸 재사용하면
# 기록되지 않은 조건에서 몇 편인지 모르는 채로 학습하게 된다.
shopt -s nullglob
EXISTING=("${DATASET}"/*.npz)
shopt -u nullglob
N_HAVE=${#EXISTING[@]}

if [ -f "${DATASET}/dataset.json" ]; then
  N_INDEX=$(python -c "import json,sys; print(len(json.load(open(sys.argv[1],encoding='utf-8'))['episodes']))" "${DATASET}/dataset.json")
  if [ "${N_HAVE}" -ne "${N_INDEX}" ]; then
    echo "  ✗ 인덱스는 ${N_INDEX}편인데 실제 파일은 ${N_HAVE}편이다."
    echo "    수집 이후 파일이 추가·삭제됐다. 무엇으로 학습했는지 귀속이 불가능하다."
    exit 1
  fi
  echo "  완료본 재사용 — ${N_HAVE}편 (인덱스 일치)"
elif [ "${N_HAVE}" -gt 0 ]; then
  echo "  ✗ ${DATASET} 에 .npz ${N_HAVE}편이 있는데 dataset.json 이 없다."
  echo "    수집이 도중에 중단된 것이다. 지우고 다시 받을 것:"
  echo "      rm -rf ${DATASET}"
  exit 1
else
  python tools/collect_sim.py --episodes "${N_COLLECT}" --jitter 0.05 \
    --skill-id pick_place --out "${DATASET}" --log
fi

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
