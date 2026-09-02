"""Does the policy's action actually depend on what it sees?
정책의 행동이 실제로 보는 것에 따라 달라지는가?

Why this is the question to settle first.
왜 이걸 먼저 가려야 하는가.

A vision policy takes (image, state) and returns an action. If the image
contributes nothing, the policy is a function of joint angles alone — it replays
an average trajectory and cannot find an object that moved. That failure looks
identical to "not accurate enough" from the outside: both score 0%. But the
fixes are opposite. Better action representation and loss weighting help a
policy that responds imprecisely; they do nothing for a policy that is not
responding at all.
시각 정책은 (이미지, 상태)를 받아 행동을 낸다. 이미지가 아무 기여도 하지 않으면
그 정책은 관절각만의 함수이고, 평균 궤적을 재생할 뿐 움직인 물체를 찾지 못한다.
이 실패는 밖에서 보면 "정밀도가 부족하다"와 똑같이 생겼다 — 둘 다 0% 다. 그런데
대책이 정반대다. 행동 표현과 손실 가중은 부정확하게 반응하는 정책을 돕지,
아예 반응하지 않는 정책에는 아무것도 하지 못한다.

How it is measured — swap one input, hold the other.
어떻게 재는가 — 한 입력만 바꾸고 나머지는 고정한다.

At the same timestep of two different episodes the object sits in a different
place, so the images differ and the recorded actions differ. Feeding episode i's
state with episode j's image isolates the image's contribution: whatever the
output moves by came from pixels alone.
서로 다른 두 에피소드의 같은 시점에서는 물체가 다른 자리에 있으므로 이미지가 다르고
기록된 행동도 다르다. 에피소드 i 의 상태에 에피소드 j 의 이미지를 넣으면 이미지의
기여만 분리된다. 출력이 움직인 만큼이 전부 픽셀에서 온 것이다.

⚠️ 교체된 (이미지, 상태) 쌍은 실제로 함께 나타난 적 없는 조합이다. 절대값을
   물리량으로 읽지 마라. 읽어야 할 것은 **비율**이다 — 정답 행동이 물체 위치에
   따라 달라지는 폭 대비 얼마나 달라지는가.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from contract.episode import read_episode
from policy.bc import BCPolicy
from sim.base import Observation
from sim.mujoco.build_scene import DEFAULT_CONFIG, joint_specs, load_config
from tracking.exp_log import file_digest, log_run

# Fixed before the numbers were computed.
# 수치를 계산하기 전에 확정했다.
GATE_IMAGE_RATIO = 0.30
GATES: dict[str, str] = {
    "responds_to_image": (
        f"이미지 교체가 만드는 행동 변화 >= 정답 행동의 에피소드 간 변화 x {GATE_IMAGE_RATIO}. "
        "못 넘으면 정책이 이미지를 사실상 무시하는 것이고, 행동 표현·손실 가중으로는 "
        "고쳐지지 않는다."
    ),
}


@dataclass
class Sensitivity:
    """How much the output moves when one input is swapped.
    한 입력을 교체했을 때 출력이 얼마나 움직이는가."""

    per_joint_image: list[float]
    per_joint_state: list[float]
    per_joint_gt: list[float]
    n_pairs: int
    n_timesteps: int

    def ratio(self) -> list[float]:
        return [
            (img / gt) if gt > 1e-12 else float("nan")
            for img, gt in zip(self.per_joint_image, self.per_joint_gt)
        ]

    def overall_ratio(self) -> float:
        img = float(np.mean(self.per_joint_image))
        gt = float(np.mean(self.per_joint_gt))
        return img / gt if gt > 1e-12 else float("nan")

    def passed(self) -> bool:
        return self.overall_ratio() >= GATE_IMAGE_RATIO


def _obs(images: dict[str, np.ndarray], state: np.ndarray, t: float) -> Observation:
    return Observation(images=images, state=state, timestamp=t)


def measure(
    dataset: Path,
    ckpt: Path,
    *,
    n_episodes: int,
    stride: int,
    device: str,
) -> Sensitivity:
    """Swap images between episodes at matching timesteps and watch the output.
    같은 시점에서 에피소드 간 이미지를 교체하고 출력을 본다."""
    files = sorted(dataset.glob("*.npz"))[:n_episodes]
    if len(files) < 2:
        raise SystemExit(f"에피소드가 2편 이상 필요하다: {dataset}")

    policy = BCPolicy(ckpt, device=device)
    print(f"{policy.describe()}\n")

    eps = [read_episode(p) for p in files]
    cams = eps[0].meta.cameras
    T = min(e.meta.n_steps for e in eps)
    steps = list(range(0, T, max(1, stride)))
    print(f"에피소드 {len(eps)}편 · 시점 {len(steps)}개 (stride {stride}) · 장치 {device}")

    d_image: list[np.ndarray] = []
    d_state: list[np.ndarray] = []
    d_gt: list[np.ndarray] = []

    for t in steps:
        for i in range(len(eps)):
            ei = eps[i]
            base = policy.act(
                _obs({c: ei.images[c][t] for c in cams}, ei.state[t], float(ei.state_timestamp[t]))
            )
            for j in range(len(eps)):
                if i == j:
                    continue
                ej = eps[j]
                # 이미지만 교체: 상태는 i, 픽셀은 j
                swap_img = policy.act(
                    _obs({c: ej.images[c][t] for c in cams}, ei.state[t],
                         float(ei.state_timestamp[t]))
                )
                # 상태만 교체: 픽셀은 i, 상태는 j
                swap_state = policy.act(
                    _obs({c: ei.images[c][t] for c in cams}, ej.state[t],
                         float(ei.state_timestamp[t]))
                )
                d_image.append(np.abs(swap_img - base))
                d_state.append(np.abs(swap_state - base))
                d_gt.append(np.abs(ej.action[t] - ei.action[t]))

    return Sensitivity(
        per_joint_image=[float(v) for v in np.mean(d_image, axis=0)],
        per_joint_state=[float(v) for v in np.mean(d_state, axis=0)],
        per_joint_gt=[float(v) for v in np.mean(d_gt, axis=0)],
        n_pairs=len(d_image),
        n_timesteps=len(steps),
    )


def format_report(s: Sensitivity, cfg: dict[str, Any]) -> str:
    names = [j.name for j in joint_specs(cfg)]
    lines = [
        f"{'관절':15s} {'이미지 교체':>12s} {'상태 교체':>12s} {'정답 차이':>12s} {'이미지/정답':>12s}",
        "-" * 68,
    ]
    for i, jn in enumerate(names):
        r = s.ratio()[i]
        lines.append(
            f"{jn:15s} {s.per_joint_image[i]:12.6f} {s.per_joint_state[i]:12.6f} "
            f"{s.per_joint_gt[i]:12.6f} {r:12.3f}"
        )
    lines.append("-" * 68)
    lines.append(
        f"{'전체':15s} {np.mean(s.per_joint_image):12.6f} "
        f"{np.mean(s.per_joint_state):12.6f} {np.mean(s.per_joint_gt):12.6f} "
        f"{s.overall_ratio():12.3f}"
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=Path("datasets/sim_pick_v1"))
    parser.add_argument("--ckpt", type=Path, default=Path("checkpoints/bc/bc_sim_pick_v1.pt"))
    parser.add_argument("--episodes", type=int, default=6, help="교체에 쓸 에피소드 수")
    parser.add_argument("--stride", type=int, default=10, help="몇 틱마다 잴 것인가")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--author", type=str, default="김준태(트랙B)")
    parser.add_argument("--log", action="store_true")
    args = parser.parse_args()

    cfg = load_config()

    print("게이트 기준 (결과 확인 전 확정):")
    for k, v in GATES.items():
        print(f"  [{k}] {v}")
    print()

    s = measure(
        args.dataset, args.ckpt,
        n_episodes=args.episodes, stride=args.stride, device=args.device,
    )
    print()
    print(format_report(s, cfg))
    print(f"\n교체 쌍 {s.n_pairs}개")

    ratio = s.overall_ratio()
    print("\n게이트 판정:")
    print(
        f"  [responds_to_image] 이미지/정답 {ratio:.3f} >= {GATE_IMAGE_RATIO:.2f} → "
        f"{'통과' if s.passed() else '**실패**'}"
    )
    print()
    if s.passed():
        print(
            "정책이 이미지에 반응한다. 0% 의 원인은 '안 본다'가 아니라 '부정확하다'이므로 "
            "행동 표현(델타)·손실 가중이 유효한 대책이다."
        )
    else:
        print(
            "정책이 이미지를 사실상 무시한다. 관절각만의 함수에 가깝고, 그래서 물체가 "
            "움직이면 찾지 못한다. **델타 행동으로는 고쳐지지 않는다** — 인코더가 학습에 "
            "기여하는지(그래디언트·특징 붕괴), 상태 입력이 답을 지름길로 주고 있는지를 "
            "먼저 봐야 한다."
        )
    print(
        "\n⚠️ 교체된 (이미지, 상태) 쌍은 함께 나타난 적 없는 조합이다. 절대값이 아니라 "
        "비율만 읽어라."
    )

    if args.log:
        rec = log_run(
            experiment="image_sensitivity",
            author=args.author,
            issue="S15P21A103-34",
            conditions={
                "dataset": str(args.dataset),
                "ckpt": str(args.ckpt),
                "episodes": args.episodes,
                "stride": args.stride,
                "device": args.device,
                "config_sha": file_digest(DEFAULT_CONFIG),
                "gates": GATES,
                "method": "같은 시점에서 에피소드 간 이미지/상태를 교체하고 출력 변화를 측정",
            },
            result={
                "per_joint_image": s.per_joint_image,
                "per_joint_state": s.per_joint_state,
                "per_joint_gt": s.per_joint_gt,
                "ratio_per_joint": s.ratio(),
                "overall_ratio": ratio,
                "passed": s.passed(),
                "n_pairs": s.n_pairs,
                "n_timesteps": s.n_timesteps,
            },
        )
        print(f"\nEXP_LOG.jsonl 기록 (git {rec['git_rev']}, dirty={rec['git_dirty']})")

    return 0 if s.passed() else 1


if __name__ == "__main__":
    raise SystemExit(main())
