"""DAgger 1b — keep the valid runs of ticks, not whole episodes.
DAgger 1b — 에피소드 통째가 아니라 유효한 틱 구간을 남긴다.

Round 1a dropped an entire episode whenever any tick was invalid, and the
80% episode-retention gate became unreachable at episode 68. This is not a
reinterpretation of that result: the data representation changed, so it is a
separate experiment with its own pre-registration.
1a 는 한 틱이라도 무효면 에피소드를 통째로 버렸고, 68편 시점에 유지율 80% 게이트가
도달 불가가 됐다. 이건 그 결과의 재해석이 아니다 — **데이터 표현이 바뀌었으므로
자기 사전등록을 가진 별도 실험이다.**

The policy drives. The expert only answers and its answer is never executed.
정책이 운전한다. 전문가는 답만 하고 그 답은 실행되지 않는다.

⚠️ `state` is MEASURED joint position. Under policy control it can leave the
   contract range: `BCPolicy.act` clips to [-1,1], so it can command a joint
   limit exactly, and a position controller pushed into a limit overshoots it
   (MuJoCo joint limits are soft constraints). Measured 🟢 2026-09-08: v5 seed0
   puts 0.007% of action components exactly on the bound.
   **Neither `state` nor `RANGE_TOLERANCE` is touched here.** Clipping the
   measurement would erase the fact that the policy hits its limits, and that is
   the most expensive failure mode this project has: a broken instrument that
   does not announce itself. Invalid ticks break the segment instead.
⚠️ `state` 는 **측정된** 관절 위치다. 정책이 운전하면 계약 범위를 벗어날 수 있다 —
   `BCPolicy.act` 가 [-1,1] 로 클립하므로 관절 한계를 정확히 명령할 수 있고, 위치
   제어기가 한계로 밀면 넘어간다 (MuJoCo 관절 한계는 soft constraint). 실측 🟢
   2026-09-08: v5 seed0 은 행동 성분의 0.007% 가 경계에 정확히 앉는다.
   **`state` 도 `RANGE_TOLERANCE` 도 여기서 건드리지 않는다.** 측정값을 clip 하면
   정책이 한계를 때린다는 사실이 데이터에서 사라진다. 무효 틱은 **구간을 끊는다.**

⚠️ IK-failure detection does not assume `expert.ik_failures` exists. The class
   does not expose it (checked 2026-09-10) and it may or may not appear as an
   instance attribute, so the script probes once at start-up and prints which
   detection path it is using. A number whose provenance is unknown is worse
   than no number.
⚠️ IK 실패 감지가 `expert.ik_failures` 의 존재를 가정하지 않는다. 클래스에는
   없고(2026-09-10 확인) 인스턴스 속성으로 붙을 수도 있으므로, 시작 시 한 번
   탐지하고 **어느 경로를 쓰는지 출력한다.** 출처를 모르는 숫자는 숫자가 없는
   것보다 나쁘다.

    # [서버]
    python tools/collect_dagger_segments.py \
        --policy-ckpt checkpoints/bc/sim_pick_v5_seed0.pt \
        --policy-ckpt checkpoints/bc/sim_pick_v5_seed1.pt \
        --policy-ckpt checkpoints/bc/sim_pick_v5_seed2.pt \
        --episodes 100 --seed-base 4000 --label-start-tick 30 \
        --out out/dagger_segments --log
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import runtime_limits  # noqa: E402  — numpy/torch 앞에 와야 한다

runtime_limits.claim("collect_dagger_segments")

import numpy as np  # noqa: E402

from contract.episode import (  # noqa: E402
    CONTRACT_VERSION, RANGE_TOLERANCE, Episode, EpisodeMeta,
    read_episode, validate, write_dataset_index, write_episode,
)
from policy.baselines import ScriptedFeedbackPolicy  # noqa: E402
from policy.bc import BCPolicy  # noqa: E402
from sim.mujoco.build_scene import DEFAULT_CONFIG, load_config  # noqa: E402
from sim.mujoco.env import MujocoPickEnv  # noqa: E402
from tracking.exp_log import code_digest, file_digest, log_run  # noqa: E402

CODE_SHA_AT_LAUNCH = code_digest()

# 사전등록 docs/PREREG_dagger_segments_0909.md — 결과 보기 전에 정한 값
GATE_VALID_RATE = 0.95
GATE_STORED_RATE = 0.95
MIN_SEGMENT_TICKS = 2


# --------------------------------------------------------------------------
# 계측기 검증 — 전문가 질의가 시뮬을 건드리지 않는가
# --------------------------------------------------------------------------

def snapshot(env: MujocoPickEnv) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Everything an expert query must leave untouched.
    전문가 질의가 건드리면 안 되는 전부."""
    return (env.data.qpos.copy(), env.data.qvel.copy(),
            env.data.ctrl.copy(), float(env.data.time))


def assert_unchanged(before: tuple, env: MujocoPickEnv, tick: int) -> None:
    """Fail loudly if the expert query moved the simulation.
    전문가 질의가 시뮬을 움직였으면 크게 실패한다.

    Checked every tick, not sampled. If it ever passes silently while the expert
    mutates state, every label comes from a different world than the one the
    policy saw, and nothing downstream can detect it.
    표본이 아니라 매 틱 검사한다. 이게 조용히 통과하는데 전문가가 상태를 바꾸면
    모든 라벨이 정책이 본 것과 다른 세계에서 나오고, 하류에서 감지할 수 없다.
    """
    after = snapshot(env)
    for name, a, b in zip(("qpos", "qvel", "ctrl"), before[:3], after[:3]):
        if not np.array_equal(a, b):
            raise RuntimeError(f"틱 {tick}: 전문가 질의가 MuJoCo {name} 를 바꿨다. 중지")
    if before[3] != after[3]:
        raise RuntimeError(f"틱 {tick}: 전문가 질의가 sim time 을 진행시켰다. 중지")


# --------------------------------------------------------------------------
# IK 실패 감지 — 속성 존재를 가정하지 않는다
# --------------------------------------------------------------------------

IK_COUNTER_NAMES = ("ik_failures", "n_ik_failures", "ik_fail")


def find_ik_counter(expert: Any) -> str | None:
    """Which attribute, if any, counts this expert's IK failures.
    이 전문가의 IK 실패를 세는 속성이 있다면 무엇인가.

    `hasattr` on the class misses attributes assigned in `__init__`, which is how
    `ScriptedPickPolicy` carries its counters. So the probe runs on a live
    instance and the caller prints the result.
    클래스에 대한 `hasattr` 은 `__init__` 에서 붙는 속성을 놓친다 —
    `ScriptedPickPolicy` 의 카운터가 그렇게 붙어 있다. 그래서 **살아 있는
    인스턴스**에서 탐지하고 호출자가 결과를 출력한다.
    """
    for name in IK_COUNTER_NAMES:
        v = getattr(expert, name, None)
        if isinstance(v, (int, np.integer)):
            return name
    return None


def label_is_usable(label: Any) -> bool:
    """A label the environment could actually be commanded with.
    환경에 실제로 명령할 수 있는 라벨인가."""
    if label is None:
        return False
    arr = np.asarray(label, dtype=np.float64)
    return arr.shape == (6,) and bool(np.all(np.isfinite(arr)))


def range_excess(arr: np.ndarray) -> float:
    """How far outside [-1, 1] the array goes. 0.0 if inside.
    배열이 [-1,1] 밖으로 얼마나 나갔나. 안이면 0.0."""
    a = np.asarray(arr, dtype=np.float64)
    if not np.isfinite(a).all():
        return float("inf")
    return float(max(np.max(a) - 1.0, np.max(-a) - 1.0, 0.0))


# --------------------------------------------------------------------------
# 구간 버퍼
# --------------------------------------------------------------------------

def fresh(cameras: list[str]) -> dict[str, Any]:
    return {"images": {c: [] for c in cameras}, "state": [], "action": [], "ts": []}


def close_segment(cur: dict[str, Any], out: list[dict[str, Any]],
                  stats: Counter, cameras: list[str]) -> dict[str, Any]:
    """End the current run of valid ticks; keep it only if long enough.
    현재 유효 구간을 닫는다. 충분히 길 때만 남긴다."""
    n = len(cur["state"])
    if n >= MIN_SEGMENT_TICKS:
        out.append(cur)
    else:
        stats["short_valid_ticks_dropped"] += n
    return fresh(cameras)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--policy-ckpt", type=Path, action="append", required=True,
                   help="운전할 BC 체크포인트. 에피소드마다 순환한다")
    p.add_argument("--out", type=Path, required=True,
                   help="⚠️ 기존 데이터셋 이름 재사용 금지")
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--seed-base", type=int, default=4000)
    p.add_argument("--label-start-tick", type=int, default=30,
                   help="이 틱부터 전문가 라벨을 단다. 앞부분은 라벨이 없어 버린다 — "
                        "BC 자기 행동을 라벨로 쓰면 자기모방이 된다")
    p.add_argument("--max-ticks", type=int, default=200)
    p.add_argument("--jitter", type=float, default=0.05)
    p.add_argument("--device", type=str, default="cpu",
                   help="롤아웃 평가와 같은 장치여야 한다 (L58)")
    p.add_argument("--skill-id", type=str, default="pick_place")
    p.add_argument("--author", type=str, default="김준태(트랙B)")
    p.add_argument("--log", action="store_true")
    args = p.parse_args()

    if args.out.exists() and any(args.out.glob("*.npz")):
        raise SystemExit(
            f"{args.out} 에 이미 에피소드가 있다. 이름을 재사용하지 않는다 — "
            "2026-09-07 에 완주한 v4 98편의 앞 69편을 그렇게 날렸다"
        )
    for c in args.policy_ckpt:
        if not c.exists():
            raise SystemExit(f"체크포인트가 없다: {c}")

    runtime_limits.torch_threads()

    cfg: dict[str, Any] = load_config()
    joints = sorted(cfg["joints"], key=lambda j: j["index"])
    joint_names = [j["name"] for j in joints]
    half = np.array([(float(j["range_rad"][1]) - float(j["range_rad"][0])) / 2.0
                     for j in joints])

    policies = [BCPolicy(c, device=args.device) for c in args.policy_ckpt]

    stats: Counter = Counter()
    phase_invalid: Counter = Counter()
    joint_over: Counter = Counter()
    max_state_excess = 0.0
    max_action_excess = 0.0
    n_written = 0

    # `render=True` 는 선택이 아니다 — BC 가 이미지를 본다.
    with MujocoPickEnv(cfg, render=True, object_jitter_m=args.jitter,
                       max_ticks=args.max_ticks) as env:
        cameras = list(env.camera_names)
        expert = ScriptedFeedbackPolicy(env)

        # 속성 탐지 — 살아 있는 인스턴스에서. 어느 경로를 쓰는지 출력한다.
        ik_attr = find_ik_counter(expert)
        phase_attr = "phase" if hasattr(expert, "phase") else None
        print(f"카메라 {cameras} · 제어 {env.control_rate_hz}Hz · "
              f"max_ticks {env.max_ticks}")
        print(f"IK 실패 감지: "
              + (f"`expert.{ik_attr}` 증가" if ik_attr
                 else "**카운터 없음 → 라벨 유효성(None/비유한/형상)으로만 감지한다.** "
                      "IK 가 실패해도 그럴듯한 라벨을 내면 못 잡는다 🟡"))
        print(f"위상 분포: " + (f"`expert.{phase_attr}`" if phase_attr
                                else "**사용 불가 — 위상별 내역을 못 낸다**"))
        print(f"정책 {len(policies)}개 순환 · 장치 {args.device} · "
              f"라벨 시작 틱 {args.label_start_tick}\n")

        for ep_i in range(args.episodes):
            pol_i = ep_i % len(policies)
            policy = policies[pol_i]
            seed = args.seed_base + ep_i

            obs = env.reset(seed=seed)
            policy.reset(seed)
            if hasattr(expert, "reset"):
                expert.reset(seed)

            cur = fresh(cameras)
            segments: list[dict[str, Any]] = []
            behavior_success = False

            for tick in range(env.max_ticks):
                behavior_action = policy.act(obs)

                if tick >= args.label_start_tick:
                    stats["queries"] += 1
                    before = snapshot(env)
                    ik_before = getattr(expert, ik_attr, 0) if ik_attr else 0
                    label = expert.act(obs)
                    assert_unchanged(before, env, tick)
                    stats["nonmutation_checks"] += 1

                    usable = label_is_usable(label)
                    ik_bad = (not usable) or (
                        ik_attr is not None
                        and getattr(expert, ik_attr, 0) > ik_before
                    )
                    s_ex = range_excess(obs.state)
                    a_ex = range_excess(label) if usable else float("inf")
                    range_bad = (s_ex > RANGE_TOLERANCE or a_ex > RANGE_TOLERANCE)

                    if np.isfinite(s_ex):
                        max_state_excess = max(max_state_excess, s_ex)
                    if usable and np.isfinite(a_ex):
                        max_action_excess = max(max_action_excess, a_ex)

                    if ik_bad or range_bad:
                        if ik_bad:
                            stats["invalid_ik_ticks"] += 1
                            if phase_attr:
                                phase_invalid[str(getattr(expert, phase_attr))] += 1
                        if range_bad:
                            stats["invalid_range_ticks"] += 1
                            over = np.maximum(np.abs(np.asarray(obs.state)) - 1.0, 0.0)
                            if over.max() > 0:
                                joint_over[joint_names[int(np.argmax(over))]] += 1
                        if ik_bad and range_bad:
                            stats["invalid_both_ticks"] += 1
                        cur = close_segment(cur, segments, stats, cameras)
                    else:
                        stats["valid_ticks"] += 1
                        for cam in cameras:
                            if cam not in obs.images:
                                raise RuntimeError(
                                    f"관측에 카메라 {cam} 가 없다. render=True 확인"
                                )
                            cur["images"][cam].append(
                                np.asarray(obs.images[cam], dtype=np.uint8).copy())
                        cur["state"].append(np.asarray(obs.state, dtype=np.float32).copy())
                        cur["action"].append(np.asarray(label, dtype=np.float32).copy())
                        cur["ts"].append(float(obs.timestamp))

                obs = env.step(behavior_action)   # 실행하는 것은 언제나 정책의 행동
                behavior_success = behavior_success or env.is_success()

            cur = close_segment(cur, segments, stats, cameras)
            stats["behavior_successes"] += int(behavior_success)

            for seg_i, seg in enumerate(segments):
                n = len(seg["state"])
                ts = np.asarray(seg["ts"], dtype=np.float64)
                ep = Episode(
                    meta=EpisodeMeta(
                        episode_id=f"dagger_seg_{ep_i:05d}_{seg_i:03d}",
                        skill_id=args.skill_id,
                        task="pick_cube_2cm_dagger_segment",
                        source="sim",
                        success=behavior_success,
                        n_steps=n,
                        control_rate_hz=env.control_rate_hz,
                        cameras=list(cameras),
                        contract_version=CONTRACT_VERSION,
                        collected_by=args.author,
                        config_sha=file_digest(DEFAULT_CONFIG),
                        git_rev=CODE_SHA_AT_LAUNCH,
                        notes={
                            "dagger_round": "1b",
                            "behavior_checkpoint": str(args.policy_ckpt[pol_i]),
                            "behavior_seed": seed,
                            "source_episode_index": ep_i,
                            "segment_index": seg_i,
                            "label_policy": "ScriptedFeedbackPolicy",
                            "label_start_tick": args.label_start_tick,
                            "expert_action_executed": False,
                            "ik_detection": ik_attr or "label_validity_only",
                        },
                    ),
                    images={c: np.stack(seg["images"][c]).astype(np.uint8)
                            for c in cameras},
                    state=np.stack(seg["state"]).astype(np.float32),
                    state_timestamp=ts,
                    action=np.stack(seg["action"]).astype(np.float32),
                    action_timestamp=ts.copy(),
                )
                problems = validate(ep)
                if problems:
                    raise RuntimeError(
                        f"{ep.meta.episode_id} 계약 위반: {problems}\n"
                        "구간 필터가 통과시킨 것을 검증기가 잡았다. 필터가 고장났다"
                    )
                args.out.mkdir(parents=True, exist_ok=True)
                write_episode(ep, args.out)
                stats["stored_ticks"] += n
                stats["segments"] += 1
                n_written += 1

            if (ep_i + 1) % 10 == 0:
                print(f"[{ep_i + 1:3d}/{args.episodes}] "
                      f"질의 {stats['queries']} · 유효 {stats['valid_ticks']} · "
                      f"저장 {stats['stored_ticks']} · 구간 {stats['segments']}",
                      flush=True)

    # ---- 계측기 검증 -------------------------------------------------------
    expected = args.episodes * (args.max_ticks - args.label_start_tick)
    if stats["queries"] != expected:
        raise RuntimeError(f"질의 수 {stats['queries']} != 기대 {expected}. "
                           "에피소드가 조기 종료됐거나 라벨 구간이 어긋났다")
    if stats["nonmutation_checks"] != stats["queries"]:
        raise RuntimeError("불변 검사 횟수가 질의 수와 다르다")

    violations = sum(len(validate(read_episode(f)))
                     for f in sorted(args.out.glob("*.npz")))

    q = stats["queries"] or 1
    valid_rate = stats["valid_ticks"] / q
    stored_rate = stats["stored_ticks"] / q
    gates = {
        "valid_label_rate_ge_95pct": valid_rate >= GATE_VALID_RATE,
        "stored_label_rate_ge_95pct": stored_rate >= GATE_STORED_RATE,
        "contract_violations_zero": violations == 0,
    }
    gates["train_go"] = all(gates.values())

    print(f"\n{'=' * 70}")
    print(f"{'게이트':<30}{'실측':>12}{'기준':>10}  판정")
    print(f"{'유효 라벨 비율':<30}{valid_rate:>12.4f}{GATE_VALID_RATE:>10.2f}  "
          f"{'통과' if gates['valid_label_rate_ge_95pct'] else '실패'}")
    print(f"{'저장 라벨 비율':<30}{stored_rate:>12.4f}{GATE_STORED_RATE:>10.2f}  "
          f"{'통과' if gates['stored_label_rate_ge_95pct'] else '실패'}")
    print(f"{'계약 위반':<30}{violations:>12d}{0:>10d}  "
          f"{'통과' if gates['contract_violations_zero'] else '실패'}")
    print(f"{'→ 학습 진행':<30}{'':>22}  "
          f"**{'GO' if gates['train_go'] else 'NO-GO'}**")
    print(f"{'=' * 70}")
    print(f"무효 틱: IK {stats['invalid_ik_ticks']} · "
          f"범위 {stats['invalid_range_ticks']} · 둘 다 {stats['invalid_both_ticks']}")
    print(f"짧아서 버린 유효 틱 {stats['short_valid_ticks_dropped']} · "
          f"구간 {stats['segments']}개 · 저장 {n_written}편")
    print(f"범위 초과 최대: state {max_state_excess:.3e} · "
          f"action {max_action_excess:.3e} (허용 {RANGE_TOLERANCE:.0e})")
    if joint_over:
        print(f"범위 초과 관절 분포: {dict(joint_over.most_common())}")
    if phase_invalid:
        print(f"IK 실패 위상 분포: {dict(phase_invalid.most_common())}")
    elif stats["invalid_ik_ticks"]:
        print("IK 실패 위상 분포: **못 낸다** — expert 에 phase 속성이 없다")

    print("\n⚠️ state 는 측정값이다. clip 하지 않았고 RANGE_TOLERANCE 도 "
          "바꾸지 않았다. 그 값은 scripted 전문가만 데이터를 만들던 시절 것이고, "
          "contract/episode.py 는 공용이라 혼자 못 바꾼다 — 숫자를 먼저 보고한다")
    print("⚠️ 이 데이터셋은 라벨이 틱 "
          f"{args.label_start_tick} 부터다. 앞부분은 라벨이 없어 버렸다")

    result = {
        "episodes": args.episodes, "queries": stats["queries"],
        "valid_ticks": stats["valid_ticks"], "stored_ticks": stats["stored_ticks"],
        "valid_label_rate": valid_rate, "stored_label_rate": stored_rate,
        "invalid_ik_ticks": stats["invalid_ik_ticks"],
        "invalid_range_ticks": stats["invalid_range_ticks"],
        "invalid_both_ticks": stats["invalid_both_ticks"],
        "short_valid_ticks_dropped": stats["short_valid_ticks_dropped"],
        "segments": stats["segments"],
        "behavior_successes": stats["behavior_successes"],
        "phase_invalid": dict(phase_invalid),
        "range_over_joints": dict(joint_over),
        "max_state_excess": max_state_excess,
        "max_action_excess": max_action_excess,
        "range_tolerance": RANGE_TOLERANCE,
        "contract_violations": violations,
        "ik_detection": ik_attr or "label_validity_only",
        "phase_breakdown_available": phase_attr is not None,
        "label_start_tick": args.label_start_tick,
        "gates": gates,
    }

    if n_written:
        write_dataset_index(args.out, {
            "experimental_only": True, "dagger_round": "1b",
            "behavior_checkpoints": [str(c) for c in args.policy_ckpt],
            "seed_base": args.seed_base, "episodes": args.episodes,
            "label_start_tick": args.label_start_tick,
        })
    else:
        print("\n⚠️ 저장된 구간이 0편이다. 인덱스를 쓰지 않는다")

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "dagger_segments_result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n결과: {args.out}/dagger_segments_result.json")

    if args.log:
        rec = log_run(
            experiment="dagger_r1b_segments",
            author=args.author,
            issue="S15P21A103-34",
            conditions={
                "behavior_checkpoints": [str(c) for c in args.policy_ckpt],
                "behavior_sha": [file_digest(c) for c in args.policy_ckpt],
                "episodes": args.episodes, "seed_base": args.seed_base,
                "label_start_tick": args.label_start_tick,
                "max_ticks": args.max_ticks, "jitter_m": args.jitter,
                "render": True, "policy_device": args.device,
                "config_sha": file_digest(DEFAULT_CONFIG),
                "code_sha_at_launch": CODE_SHA_AT_LAUNCH,
                "prereg": "docs/PREREG_dagger_segments_0909.md",
            },
            result=result,
        )
        print(f"EXP_LOG.jsonl 기록 (git {rec['git_rev']}, dirty={rec['git_dirty']})")

    return 0 if gates["train_go"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
