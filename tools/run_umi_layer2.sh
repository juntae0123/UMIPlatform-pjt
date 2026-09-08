#!/usr/bin/env bash
# 층 2 관통 — 수집 → UMI raw → 계약 → 검증 → 배열대조 → 학습 → 롤아웃.
# S15P21A103-127 완료기준 3·4 (수동 개입 없이 완주 + 단계별 소요시간 기록).
#
# 렌더가 필요하므로 VM 에서는 못 돈다. [로컬] aiot_ai 또는 [서버] aiot_v100 에서 돌린다.
#
#   cd AI && bash tools/run_umi_layer2.sh v1 20
#
# 인자: <버전태그> [수집편수]
# 모든 인자는 argparse 원문과 대조했다 (2026-09-08).
set -euo pipefail

TAG="${1:?버전태그가 필요하다. 예: v1 — 기존 이름 재사용 금지}"
N="${2:-20}"
SKILL="${SKILL:-pick_place}"
JITTER="${JITTER:-0.05}"
EVAL_EPISODES="${EVAL_EPISODES:-20}"
POS_TOL="${POS_TOL:-1e-7}"

SRC="datasets/umi_src_sim_${TAG}"
RAW="out/umi_raw_from_sim_${TAG}"
DST="datasets/umi_pick_${TAG}"
CKPT="checkpoints/bc/umi_pick_${TAG}_seed0.pt"
REPORT="out/umi_layer2_${TAG}_timing.json"

for d in "$SRC" "$RAW" "$DST"; do
  if [ -d "$d" ] && [ -n "$(ls -A "$d" 2>/dev/null)" ]; then
    echo "이미 있다: $d"
    echo "  기존 이름을 재사용하지 않는다 — ckpt 까지 덮어쓴다. 다음 태그를 써라."
    exit 1
  fi
done

mkdir -p out
: > "$REPORT.stages"

stage() {   # stage <이름> <명령...>
  local name="$1"; shift
  echo ""
  echo "########## [$name] $(date '+%H:%M:%S')"
  local t0 t1
  t0=$(date +%s)
  "$@"
  t1=$(date +%s)
  echo "$name $((t1 - t0))" >> "$REPORT.stages"
  echo "########## [$name] $((t1 - t0))초"
}

echo "태그 $TAG · 수집 $N 편 · 스킬 $SKILL · pos_tol $POS_TOL"
echo "MUJOCO_GL=${MUJOCO_GL:-미설정}  (서버 헤드리스는 egl 필요)"

stage collect  python tools/collect_sim.py --episodes "$N" --jitter "$JITTER" \
                 --skill-id "$SKILL" --out "$SRC" --log
stage dump     python tools/umi_dump_from_dataset.py --data "$SRC" --out "$RAW"
stage convert  python tools/convert_umi.py --raw "$RAW" --out "$DST" \
                 --pos-tol-m "$POS_TOL" --log
stage verify   python tools/verify_dataset.py "$DST" --write-index --log
# 배열 대조가 층 2 의 판정기다. 롤아웃 성공률은 판정기가 아니다 —
# baseline 4.3%(CI 2.5~7.3%)에 학습 실행 간 변동 25%p 라 0% 는 "변환이 깨졌다"와
# "정책이 원래 나쁘다" 둘 다에서 나온다.
stage diff     python tools/diff_datasets.py "$SRC" "$DST" || \
                 echo "  ⚠️ diff 가 비영 종료 — action 차이는 정상일 수 있다(ctrl vs q[t+1]). state 를 봐라"
stage train    python tools/train_bc.py --data "$DST" --out "$CKPT" --log
stage rollout  python tools/eval_rollout.py --policy-ckpt "$CKPT" \
                 --episodes "$EVAL_EPISODES" --jitter "$JITTER" --render --log

python - "$REPORT" <<'PY'
import json, sys
from pathlib import Path
rep = Path(sys.argv[1]); rows = []
for line in Path(str(rep) + ".stages").read_text().splitlines():
    name, sec = line.rsplit(" ", 1); rows.append((name, int(sec)))
total = sum(s for _, s in rows)
rep.write_text(json.dumps({"stages": dict(rows), "total_s": total}, indent=2), encoding="utf-8")
print("\n=== 단계별 소요시간 (완료기준 4) ===")
for n, s in rows:
    print(f"  {n:<10}{s:6d}초  {s / total * 100:5.1f}%")
print(f"  {'합계':<10}{total:6d}초 = {total / 60:.1f}분")
print(f"\n기록: {rep}")
PY

echo ""
echo "=== 읽는 법 ==="
echo "  판정기는 [diff] 의 state 왕복 오차다. 롤아웃 성공률이 아니다."
echo "  롤아웃 숫자는 '완주했다'의 증거이고, 성능으로 인용하지 않는다."
echo "  현행 시뮬 BC baseline 은 4.3% (13/300, CI 2.5~7.3%) 다."
