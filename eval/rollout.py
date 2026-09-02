"""Score policies by rollout success rate — the project's only real metric.
정책을 롤아웃 성공률로 채점한다. 이 프로젝트의 유일한 실제 지표다.

Validation loss is not this. In imitation learning the correlation between loss
and success is weak: loss measures similarity to the demonstrated trajectory,
but the robot succeeds by other paths too, and a low loss still accumulates
error into failure. Loss is only good for "did training break". This is the
number that decides anything.
validation loss 는 이게 아니다. 모방학습에서 손실과 성공률의 상관은 약하다.
손실은 시연 궤적과의 유사도를 재는데, 로봇은 다른 경로로도 성공하고 손실이
낮아도 오차가 누적돼 실패한다. 손실은 "학습이 망가졌나" 확인용이다.
판단을 내리는 수치는 이것이다.

Gate thresholds are set in GATES below, before any result was looked at.
게이트 기준은 아래 GATES 에 있고, 결과를 보기 전에 정했다.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


from sim.mujoco.env import MujocoPickEnv
from policy.base import Policy
from policy.baselines import (
    HoldPolicy,
    ReplayPolicy,
    ScriptedPickPolicy,
    ZeroPolicy,
)

from sim.mujoco.build_scene import DEFAULT_CONFIG, load_config
from tracking.exp_log import file_digest, log_run


# Fixed before any number was produced. Changing these after seeing results is
# not a judgement, it is a rationalisation.
# 어떤 수치도 나오기 전에 확정했다. 결과를 보고 이걸 고치면 판정이 아니라
# 사후 합리화다.
GATES: dict[str, str] = {
    "floor": "학습 정책 성공률 > hold 성공률 + 20%p. 못 넘으면 정책이 무의미하다.",
    "chance": "학습 정책 성공률 > zero 성공률 + 20%p. 못 넘으면 우연과 구분되지 않는다.",
    "task_validity": (
        "replay 성공률 < 30%. 이보다 높으면 고정 궤적으로 풀리는 태스크이므로 "
        "시각 정책이 학습할 것이 없다. 태스크를 다시 설계해야 한다."
    ),
    "ceiling": "scripted 성공률 >= 80%. 못 넘으면 태스크나 씬이 문제이지 정책 문제가 아니다.",
}


@dataclass
class RolloutResult:
    """Outcome of one episode under one policy.
    정책 하나로 돌린 에피소드 하나의 결과."""

    seed: int
    success: bool
    ticks: int
    lift_height_m: float
    object_xy: tuple[float, float]


def rollout(
    env: MujocoPickEnv,
    policy: Policy,
    seed: int,
    object_xy: tuple[float, float] | None = None,
) -> RolloutResult:
    """Run one episode to completion or to the tick limit.
    에피소드 하나를 완료 또는 틱 제한까지 돌린다.

    `object_xy` pins the object instead of drawing it from the seed. It exists for
    one job: checking whether a policy can reproduce the exact episode it was
    trained on. Without it, "overfit one episode and replay it" silently becomes
    "overfit one episode and test generalisation", because jitter=0 puts the object
    at the config default rather than where that episode was recorded.
    `object_xy` 는 시드에서 뽑는 대신 물체를 고정한다. 용도는 하나다 — 정책이 학습한
    바로 그 에피소드를 재현할 수 있는지 확인하는 것. 이게 없으면 "1편 과적합 후 재생"이
    조용히 "1편 과적합 후 일반화 검사"가 된다. jitter=0 은 물체를 그 에피소드가 기록된
    자리가 아니라 config 기본값에 놓기 때문이다.
    """
    obs = env.reset(seed=seed, object_xy=object_xy)
    policy.reset(seed=seed)
    obj = env.object_position()
    success = False
    ticks = 0
    for _ in range(env.max_ticks):
        obs = env.step(policy.act(obs))
        ticks += 1
        if env.is_success():
            success = True
            break
    return RolloutResult(
        seed=seed,
        success=success,
        ticks=ticks,
        lift_height_m=round(env.lift_height(), 5),
        object_xy=(round(float(obj[0]), 5), round(float(obj[1]), 5)),
    )


def evaluate(
    env: MujocoPickEnv,
    policy: Policy,
    seeds: list[int],
    object_xy: tuple[float, float] | None = None,
) -> tuple[float, list[RolloutResult]]:
    """Run one policy over a fixed seed list and return its success rate.
    고정된 시드 목록으로 정책 하나를 돌리고 성공률을 반환한다.

    Every policy sees the SAME seeds, so every policy sees the same object
    placements. Comparing policies across different seeds is not a comparison.
    모든 정책이 **같은** 시드를 본다. 즉 같은 물체 배치를 본다.
    다른 시드로 얻은 성공률끼리 비교하는 것은 비교가 아니다.
    """
    results = [rollout(env, policy, s, object_xy) for s in seeds]
    return sum(r.success for r in results) / len(results), results


def replay_tolerance(
    env: MujocoPickEnv, episode: Path, offsets_m: tuple[float, ...]
) -> dict[str, Any]:
    """How far the object can move before a replayed trajectory stops working.
    재생 궤적이 통하지 않게 되기까지 물체가 얼마나 움직일 수 있는가.

    This is the number that says what a policy actually has to do. If replay
    survives a 5 cm displacement, perception is not required and the task is
    mis-designed. If it dies at 1 cm, the policy must localise the object to
    better than 1 cm — which is a concrete requirement to design against.
    정책이 실제로 무엇을 해야 하는지 말해주는 수치다. 재생이 5cm 이동을 견디면
    인식이 필요 없다는 뜻이고 태스크 설계가 잘못된 것이다. 1cm 에서 죽는다면
    정책은 물체를 1cm 이내로 국소화해야 한다 — 설계에 쓸 수 있는 구체적 요구다.

    A replay that fails even at zero offset is a broken instrument, not a
    finding, so that case is measured first and reported separately.
    0 오프셋에서도 실패하는 재생은 발견이 아니라 고장난 계측기다.
    그래서 그 경우를 먼저 재고 따로 보고한다.
    """
    import json

    meta = json.loads(episode.with_suffix(".json").read_text(encoding="utf-8"))
    recorded_xy = tuple(meta["notes"]["object_init_xy"])
    policy = ReplayPolicy.from_episode(episode)

    def attempt(xy: tuple[float, float]) -> bool:
        obs = env.reset(object_xy=xy)
        policy.reset()
        for _ in range(env.max_ticks):
            obs = env.step(policy.act(obs))
            if env.is_success():
                return True
        return False

    baseline_ok = attempt(recorded_xy)
    rows: list[dict[str, Any]] = []
    for d in offsets_m:
        hits = 0
        for sign in (1.0, -1.0):
            for axis in (0, 1):
                xy = list(recorded_xy)
                xy[axis] += sign * d
                hits += int(attempt((xy[0], xy[1])))
        rows.append({"offset_m": d, "success": hits, "of": 4})
    return {
        "episode": episode.name,
        "recorded_xy": list(recorded_xy),
        "at_recorded_condition": baseline_ok,
        "rows": rows,
    }


def build_policies(
    env: MujocoPickEnv, replay_from: Path | None, policy_ckpt: Path | None = None
) -> list[Policy]:
    """Assemble the baseline set, skipping replay if no episode is available.
    baseline 묶음을 만든다. 재생할 에피소드가 없으면 replay 는 건너뛴다.

    A learned checkpoint joins the same list, so it is scored under exactly the
    same seeds and conditions as the baselines. A learned policy's number
    reported on its own, without the baselines beside it, means nothing.
    학습 체크포인트는 같은 목록에 들어간다. 그래야 baseline 과 **정확히 같은**
    시드·조건으로 채점된다. 학습 정책 수치를 baseline 없이 단독으로 보고하는 것은
    아무 의미가 없다.
    """
    policies: list[Policy] = [HoldPolicy(), ZeroPolicy(), ScriptedPickPolicy(env)]
    if policy_ckpt is not None:
        from policy.bc import BCPolicy

        bc = BCPolicy(policy_ckpt)
        print(f"학습 정책 로드: {bc.describe()}")
        if bc.meta.get("trained_on") == "random_tensors":
            print("⚠️ 이 체크포인트는 **랜덤 텐서로 학습**된 것이다. 평가 결과에 의미가 없다.")
        policies.append(bc)
    if replay_from is not None:
        episodes = sorted(replay_from.glob("*.npz"))
        if not episodes:
            print(f"⚠️ {replay_from} 에 에피소드가 없다. replay baseline 을 건너뛴다.")
        else:
            policies.insert(2, ReplayPolicy.from_episode(episodes[0]))
    return policies


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed-base", type=int, default=1000)
    parser.add_argument("--jitter", type=float, default=0.05,
                        help="object xy randomisation, m / 물체 xy 무작위 범위 (m)")
    parser.add_argument("--replay-from", type=Path, default=None,
                        help="dataset dir to take the replay trajectory from")
    parser.add_argument("--replay-tolerance", action="store_true",
                        help="also measure how far the object may move before replay fails")
    parser.add_argument(
        "--object-xy", type=float, nargs=2, default=None, metavar=("X", "Y"),
        help="물체를 이 좌표에 고정한다. 1편 과적합 재현 검사용 "
             "(에피소드 json 의 notes.object_init_xy 값을 넣는다)",
    )
    parser.add_argument(
        "--from-episode", type=Path, default=None,
        help="이 에피소드가 기록된 물체 위치에서 평가한다. --object-xy 를 자동으로 채운다",
    )
    parser.add_argument("--policy-ckpt", type=Path, default=None,
                        help="학습 정책 체크포인트. baseline 과 같은 조건으로 함께 채점한다")
    parser.add_argument("--render", action="store_true",
                        help="render observations; needed only for vision policies")
    parser.add_argument("--author", type=str, default="김준태(트랙B)")
    parser.add_argument("--log", action="store_true")
    args = parser.parse_args()

    cfg: dict[str, Any] = load_config()
    seeds = [args.seed_base + i for i in range(args.episodes)]

    object_xy: tuple[float, float] | None = None
    if args.from_episode is not None:
        import json as _json

        meta = _json.loads(args.from_episode.with_suffix(".json").read_text(encoding="utf-8"))
        xy = meta["notes"]["object_init_xy"]
        object_xy = (float(xy[0]), float(xy[1]))
        print(f"물체를 {args.from_episode.name} 이 기록된 위치 "
              f"({object_xy[0]:+.4f}, {object_xy[1]:+.4f}) 에 고정한다")
    elif args.object_xy is not None:
        object_xy = (float(args.object_xy[0]), float(args.object_xy[1]))
        print(f"물체를 ({object_xy[0]:+.4f}, {object_xy[1]:+.4f}) 에 고정한다")

    if object_xy is not None and args.episodes > 1:
        print(f"⚠️ 물체가 고정돼 있어 {args.episodes}회가 전부 같은 조건이다. "
              "시뮬은 결정적이므로 성공/실패도 같다 — 표본이 아니다")

    print("게이트 기준 (결과 확인 전 확정):")
    for key, text in GATES.items():
        print(f"  [{key}] {text}")
    print()

    rates: dict[str, float] = {}
    detail: dict[str, list[dict[str, Any]]] = {}
    privileged: dict[str, bool] = {}
    action_space: str | None = None

    with MujocoPickEnv(cfg, render=args.render, object_jitter_m=args.jitter) as env:
        for policy in build_policies(env, args.replay_from, args.policy_ckpt):
            rate, results = evaluate(env, policy, seeds, object_xy)
            if policy.name == "bc":
                action_space = getattr(policy, "action_space", "joint_absolute")
            rates[policy.name] = rate
            privileged[policy.name] = policy.uses_privileged_state
            detail[policy.name] = [r.__dict__ for r in results]
            mark = " (특권정보 사용 — 실물 배포 불가)" if policy.uses_privileged_state else ""
            print(
                f"{policy.name:10s} {sum(r.success for r in results):3d}/{len(results)} "
                f"= {rate * 100:5.1f}%   평균 상승 {np.mean([r.lift_height_m for r in results]) * 100:5.2f}cm{mark}"
            )

    where = (f"물체 고정 ({object_xy[0]:+.4f}, {object_xy[1]:+.4f})"
             if object_xy is not None else f"물체 xy ±{args.jitter * 1000:.0f}mm")
    print(f"\n조건: {where}, 시드 {seeds[0]}~{seeds[-1]}, "
          f"모든 정책이 동일 시드, 틱 제한 {MujocoPickEnv(cfg, render=False).max_ticks}")

    tolerance: dict[str, Any] | None = None
    if args.replay_tolerance and args.replay_from is not None:
        episodes = sorted(args.replay_from.glob("*.npz"))
        if episodes:
            with MujocoPickEnv(cfg, render=False, object_jitter_m=args.jitter) as env:
                tolerance = replay_tolerance(
                    env, episodes[0], (0.005, 0.010, 0.015, 0.020, 0.030, 0.050)
                )
            print(f"\nreplay 위치 허용오차 — 원본 {tolerance['episode']}, "
                  f"기록 위치 {tolerance['recorded_xy']}")
            state = "성공" if tolerance["at_recorded_condition"] else "실패 — 계측기가 고장난 것이므로 아래 수치는 무효"
            print(f"  기록된 바로 그 위치에서: {state}")
            for row in tolerance["rows"]:
                print(f"  물체를 ±{row['offset_m'] * 1000:4.0f}mm 이동 (4방향)  {row['success']}/4")

    print("\n게이트 판정:")
    if "replay" in rates:
        ok = rates["replay"] < 0.30
        print(f"  [task_validity] replay {rates['replay'] * 100:.1f}% < 30% → "
              f"{'통과 — 이 태스크는 관측을 봐야 풀린다' if ok else '실패 — 고정 궤적으로 풀린다. 태스크 재설계 필요'}")
    if "scripted" in rates:
        ok = rates["scripted"] >= 0.80
        print(f"  [ceiling]       scripted {rates['scripted'] * 100:.1f}% >= 80% → "
              f"{'통과' if ok else '실패 — 태스크/씬 문제이지 정책 문제가 아니다'}")
    floor = max(rates.get("hold", 0.0), rates.get("zero", 0.0))
    threshold = floor + 0.20
    if "bc" in rates:
        ok = rates["bc"] > threshold
        print(f"  [floor/chance]  bc {rates['bc'] * 100:.1f}% > {threshold * 100:.1f}% → "
              f"{'통과' if ok else '**실패 — baseline 대비 의미 있는 차이가 아니다**'}")
        if "scripted" in rates and rates["scripted"] > 0:
            gap = rates["scripted"] - rates["bc"]
            print(f"                  상한(scripted) 대비 {gap * 100:.1f}%p 아래 — "
                  f"이미지로 위치를 추정하는 데 드는 비용이다")
    else:
        print(f"  [floor/chance]  학습 정책은 {threshold * 100:.1f}% 를 넘어야 의미가 있다 "
              f"(hold {rates.get('hold', 0) * 100:.1f}%, zero {rates.get('zero', 0) * 100:.1f}%)")

    if args.log:
        rec = log_run(
            experiment="rollout_baselines",
            author=args.author,
            issue="S15P21A103-60",
            conditions={
                "episodes": args.episodes,
                "seeds": [seeds[0], seeds[-1]],
                "jitter_m": args.jitter,
                "object_xy_fixed": list(object_xy) if object_xy else None,
                "render": args.render,
                "config_sha": file_digest(DEFAULT_CONFIG),
                # Without these a rollout number cannot be attributed to a
                # checkpoint. "bc scored 25%" is not a finding if nobody can say
                # which bc, trained in which action space.
                # 이게 없으면 롤아웃 수치를 체크포인트에 귀속시킬 수 없다.
                # 어느 bc 인지, 어느 행동 공간으로 학습된 것인지 말할 수 없으면
                # "bc 가 25% 였다"는 발견이 아니다.
                "policy_ckpt": str(args.policy_ckpt) if args.policy_ckpt else None,
                "policy_action_space": action_space,
                "gates": GATES,
                "metric": "롤아웃 성공률 (validation loss 아님)",
            },
            result={
                "success_rates": rates,
                "uses_privileged_state": privileged,
                "replay_tolerance": tolerance,
                "detail": detail,
            },
        )
        print(f"\nEXP_LOG.jsonl 기록 (git {rec['git_rev']}, dirty={rec['git_dirty']})")


if __name__ == "__main__":
    main()
