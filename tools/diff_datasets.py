"""두 계약 데이터셋을 배열 수준에서 대조한다. 학습 없이 변환 정확성을 판정한다.
Compare two contract datasets array-by-array — conversion accuracy without training.

왜 롤아웃으로 판정하지 않는가: baseline 이 4.3%(CI 2.5~7.3%)이고 학습 실행 간
변동이 25%p 다 🟢. 0% 는 "변환이 깨졌다"와 "정책이 원래 나쁘다" 둘 다에서 나온다.
**배열이 같으면 학습 결과가 같을 수밖에 없다** — 그게 더 강하고 더 싸다.

    # [로컬]
    cd AI && python tools/diff_datasets.py datasets/umi_src_sim_v1 datasets/umi_pick_v1
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from contract.episode import read_episode  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("left", type=Path)
    ap.add_argument("right", type=Path)
    ap.add_argument("--state-tol", type=float, default=1e-3,
                    help="정규화 단위. IK+정규화 왕복 오차의 허용치")
    args = ap.parse_args()

    lmap = {p.stem: p for p in sorted(args.left.glob("*.npz"))}
    rmap = {p.stem: p for p in sorted(args.right.glob("*.npz"))}
    common = sorted(set(lmap) & set(rmap))
    if not common:
        print(f"공통 에피소드가 없다. left {len(lmap)}편 · right {len(rmap)}편")
        return 1
    only_l, only_r = sorted(set(lmap) - set(rmap)), sorted(set(rmap) - set(lmap))

    print(f"{args.left}  vs  {args.right}")
    print(f"공통 {len(common)}편" + (f" · 왼쪽만 {len(only_l)}" if only_l else "")
          + (f" · 오른쪽만 {len(only_r)}" if only_r else "") + "\n")

    acc = {k: [] for k in ("state", "action", "state_ts", "action_ts")}
    per_joint = {"state": [], "action": []}   # 관절별 최대차
    motion = []                               # 왼쪽의 관절별 mean|action-state| = 학습 신호 크기
    img_same = img_total = 0
    step_mismatch = []

    for name in common:
        a, b = read_episode(lmap[name]), read_episode(rmap[name])
        n = min(a.meta.n_steps, b.meta.n_steps)
        if a.meta.n_steps != b.meta.n_steps:
            step_mismatch.append((name, a.meta.n_steps, b.meta.n_steps))
        acc["state"].append(float(np.abs(a.state[:n] - b.state[:n]).max()))
        acc["action"].append(float(np.abs(a.action[:n] - b.action[:n]).max()))
        acc["state_ts"].append(float(np.abs(a.state_timestamp[:n] - b.state_timestamp[:n]).max()))
        acc["action_ts"].append(float(np.abs(a.action_timestamp[:n] - b.action_timestamp[:n]).max()))
        per_joint["state"].append(np.abs(a.state[:n] - b.state[:n]).max(axis=0))
        per_joint["action"].append(np.abs(a.action[:n] - b.action[:n]).max(axis=0))
        motion.append(np.abs(a.action[:n] - a.state[:n]).mean(axis=0))
        for cam in sorted(set(a.images) & set(b.images)):
            img_total += 1
            if np.array_equal(a.images[cam][:n], b.images[cam][:n]):
                img_same += 1

    print(f"{'필드':<12}{'최대차 중앙':>14}{'최대차 최대':>14}")
    print("-" * 40)
    for k in ("state", "action"):
        v = np.array(acc[k])
        print(f"{k:<12}{np.median(v):14.3e}{v.max():14.3e}")
    for k in ("state_ts", "action_ts"):
        v = np.array(acc[k])
        print(f"{k:<12}{np.median(v):14.3e}{v.max():14.3e}  [초]")

    print(f"\n이미지 바이트 완전 일치 {img_same}/{img_total}")
    if step_mismatch:
        print(f"스텝 수 불일치 {len(step_mismatch)}편: {step_mismatch[:5]}")

    names = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
    js = np.stack(per_joint["state"]).max(axis=0)
    ja = np.stack(per_joint["action"]).max(axis=0)
    mo = np.stack(motion).mean(axis=0)
    print(f"\n{'관절':<15}{'state 최대차':>14}{'action 최대차':>15}{'움직임 평균':>14}{'오차/움직임':>13}")
    print("-" * 71)
    for i, nm in enumerate(names):
        ratio = js[i] / mo[i] if mo[i] > 0 else float("nan")
        print(f"{nm:<15}{js[i]:14.3e}{ja[i]:15.3e}{mo[i]:14.3e}{ratio * 100:12.2f}%")

    st = np.array(acc["state"]).max()
    at = np.array(acc["action"]).max()
    print("\n=== 판정 ===")
    print(f"state 왕복 절대 최대차 {st:.3e} (정규화 단위)")
    print(f"action 왕복 절대 최대차 {at:.3e}")
    print("\n판정 기준은 절대값이 아니라 **학습 신호 대비 비율**이다.")
    print("정책이 배워야 하는 것은 잔차 action-state 이고, 왕복 오차가 그 신호의")
    print("몇 %인지가 의미 있는 수치다. 임의의 절대 허용치는 아무것도 판정하지 못한다.")
    worst = float(np.nanmax(js / np.where(mo > 0, mo, np.nan)))
    print(f"관절별 최악 비율 {worst * 100:.2f}%")
    print("  ⚠️ action 은 0 이 아닌 것이 정상일 수 있다 — 시뮬 수집의 action 은 ctrl(명령)이고")
    print("     UMI 경로는 q[t+1] 이다. 두 출처의 구조적 차이다 (umi_dump_from_dataset 참조).")
    print("     원본이 UMI 경로로 만들어진 데이터셋이면 0 에 가까워야 한다.")
    return 0 if st <= args.state_tol else 1


if __name__ == "__main__":
    raise SystemExit(main())
