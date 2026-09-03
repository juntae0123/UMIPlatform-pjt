"""Baselines a learned policy has to be measured against.
학습된 정책이 반드시 대조되어야 할 기준선들.

Project rule: no training code before baselines are measured. The reason is
diagnostic, not ceremonial. Without the numbers below, a BC success rate of 40%
is uninterpretable — it could be excellent or it could be worse than standing
still, and you cannot tell which.
프로젝트 규칙: baseline 측정 전에 학습 코드를 쓰지 않는다. 의례가 아니라 진단
때문이다. 아래 수치가 없으면 BC 성공률 40% 는 해석이 불가능하다. 훌륭한
결과일 수도 있고 가만히 서 있는 것보다 못한 것일 수도 있는데, 구분할 수가 없다.

Four baselines, each answering a different question:
  Hold      가만히 있으면 몇 %인가 — 하한선. 이보다 낮으면 정책이 해롭다.
  Zero      의미 없이 움직이면 몇 %인가 — 우연 성공률.
  Replay    고정 궤적을 재생하면 몇 %인가 — 이 태스크가 관측을 봐야 하는
            태스크인지 판정한다. 높게 나오면 시각 정책이 필요 없다는 뜻이고,
            그건 태스크 설계를 다시 해야 한다는 신호다.
  Scripted  인식이 완벽하면 몇 %인가 — 상한선. 학습 정책이 넘을 수 없다.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from sim.mujoco.build_scene import normalize
from sim.base import Observation
from sim.mujoco.env import MujocoPickEnv
from sim.mujoco.kinematics import pick_waypoints, solve_pose_ik
from policy.base import check_action


class HoldPolicy:
    """Command the joint positions observed at reset, forever.
    리셋 시점에 관측한 관절 위치를 계속 명령한다.

    The true do-nothing floor. Any policy scoring below this is actively worse
    than not moving.
    진짜 하한선. 이보다 낮은 정책은 안 움직이느니만 못한 것이다.
    """

    name = "hold"
    uses_privileged_state = False

    def __init__(self) -> None:
        self._target: np.ndarray | None = None

    def reset(self, seed: int | None = None) -> None:
        """Forget the held pose.
        유지 중인 자세를 잊는다."""
        self._target = None

    def act(self, obs: Observation) -> np.ndarray:
        """Return the first observed state, unchanged.
        처음 관측한 상태를 그대로 반환한다."""
        if self._target is None:
            self._target = obs.state.copy()
        return check_action(self._target, self.name)


class ZeroPolicy:
    """Command the centre of every joint range.
    모든 관절 범위의 중앙을 명령한다.

    Moves, but toward nothing in particular. Separates "the policy learned to
    move usefully" from "moving at all sometimes works".
    움직이기는 하지만 아무 데도 향하지 않는다. "유용하게 움직이는 법을 배웠다"와
    "움직이기만 해도 가끔 된다"를 분리한다.
    """

    name = "zero"
    uses_privileged_state = False

    def reset(self, seed: int | None = None) -> None:
        """Nothing to reset.
        되돌릴 상태 없음."""

    def act(self, obs: Observation) -> np.ndarray:
        """Return an all-zero action.
        전부 0인 행동을 반환한다."""
        return check_action(np.zeros(6, dtype=np.float32), self.name)


class ReplayPolicy:
    """Replay a recorded action sequence, ignoring the observation entirely.
    기록된 행동열을 그대로 재생한다. 관측은 전혀 보지 않는다.

    This is the most informative baseline in the set. If replaying one fixed
    trajectory succeeds even when the object moves, the task does not require
    perception and a vision policy trained on it will learn nothing useful —
    it will learn the trajectory. That is a task-design failure, and it is
    much cheaper to find here than after collecting a hundred demonstrations.
    이 묶음에서 가장 정보가 많은 baseline 이다. 물체가 움직여도 고정 궤적 하나를
    재생해서 성공한다면, 그 태스크는 인식을 필요로 하지 않는다. 그런 데이터로
    학습한 시각 정책은 쓸모 있는 것을 배우지 않는다 — 궤적을 외울 뿐이다.
    그건 태스크 설계의 실패이고, 시연 100개를 찍은 뒤보다 여기서 발견하는 편이
    훨씬 싸다.
    """

    name = "replay"
    uses_privileged_state = False

    def __init__(self, actions: np.ndarray, source: str = "") -> None:
        arr = np.asarray(actions, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[1] != 6:
            raise ValueError(f"actions must be (T, 6), got {arr.shape}")
        self._actions = arr
        self.source = source
        self._i = 0

    @classmethod
    def from_episode(cls, npz_path: Path) -> "ReplayPolicy":
        """Load the action track of a recorded episode.
        기록된 에피소드의 행동 트랙을 읽는다."""
        with np.load(npz_path) as z:
            return cls(z["action"], source=npz_path.name)

    def reset(self, seed: int | None = None) -> None:
        """Rewind to the first recorded action.
        첫 기록 행동으로 되감는다."""
        self._i = 0

    def act(self, obs: Observation) -> np.ndarray:
        """Return the next recorded action, holding the last one at the end.
        다음 기록 행동을 반환한다. 끝에 도달하면 마지막 행동을 유지한다."""
        idx = min(self._i, len(self._actions) - 1)
        self._i += 1
        return check_action(self._actions[idx], self.name)


class ScriptedPickPolicy:
    """Plan the whole pick from the ground-truth object position, then execute.
    물체의 정답 위치로 파지 전체를 계획한 뒤 실행한다.

    Not deployable — it reads simulator state. Its job is to be the ceiling:
    what the task allows when perception is perfect and the only error left is
    control. A learned policy cannot exceed this, and how far below it lands is
    the honest measure of how much perception is costing.
    배포 불가 — 시뮬 상태를 읽는다. 역할은 천장이다. 인식이 완벽하고 남은 오차가
    제어뿐일 때 이 태스크가 허용하는 상한선. 학습 정책은 이걸 넘을 수 없고,
    여기서 얼마나 아래에 떨어지는지가 인식이 치르는 비용의 정직한 척도다.
    """

    name = "scripted"
    uses_privileged_state = True

    def __init__(self, env: MujocoPickEnv) -> None:
        self._env = env
        self._cfg = env.cfg
        self._plan: list[np.ndarray] = []
        self._i = 0

    def reset(self, seed: int | None = None) -> None:
        """Re-plan against the object's current position.
        물체의 현재 위치로 계획을 다시 세운다."""
        self._plan = self._build_plan()
        self._i = 0

    def act(self, obs: Observation) -> np.ndarray:
        """Return the next planned action, holding the last one at the end.
        계획된 다음 행동을 반환한다. 끝에 도달하면 마지막을 유지한다."""
        if not self._plan:
            self.reset()
        idx = min(self._i, len(self._plan) - 1)
        self._i += 1
        return check_action(self._plan[idx], self.name)

    def _build_plan(self) -> list[np.ndarray]:
        """Expand the shared pick waypoints into one action per control tick.
        공유 파지 웨이포인트를 제어 틱당 행동 하나로 펼친다.

        The waypoints come from `pick_waypoints`, the same function the collector
        uses. Before 2026-09-03 this method had its own copy and the two had
        drifted (141 vs 165 ticks) -- the ceiling was not the demonstrator.
        웨이포인트는 수집기와 **같은 함수** `pick_waypoints` 에서 온다. 2026-09-03
        이전에는 이 메서드가 자기 사본을 갖고 있었고 둘이 갈라져 있었다(141 대 165틱)
        — 상한선이 시연자와 다른 것이었다.
        """
        cfg = self._cfg
        g = cfg["grasp"]
        env = self._env
        model = env.model
        rate = env.control_rate_hz
        offset = np.asarray(g["pinch_offset_local"], dtype=float)
        axis = np.asarray(g["approach_axis"], dtype=float)

        q_now = env.joint_positions()
        segments: list[tuple[np.ndarray, float, float]] = []
        seed_q = q_now
        for seg in pick_waypoints(cfg, env.object_position()):
            if seg.target is None:
                segments.append((seed_q, seg.grip, seg.seconds))
                continue
            res = solve_pose_ik(model, seg.target, offset, axis, q_init=seed_q, wrist_roll=0.0)
            if not res.ok:
                if not seg.required:
                    # A failed lift skips its segment, as before the refactor.
                    # 들어올리기 실패는 그 구간만 건너뛴다. 리팩터링 이전과 같다.
                    continue
                # Unreachable: hold still rather than flail. The rollout scores
                # this a failure, which is the correct outcome.
                # 도달 불가면 휘젓지 말고 정지한다. 롤아웃은 실패로 채점하고 그게 맞다.
                return [normalize(q_now, cfg)]
            segments.append((res.qpos, seg.grip, seg.seconds))
            seed_q = res.qpos

        plan: list[np.ndarray] = []
        q_prev = q_now.copy()
        for q_target, grip, seconds in segments:
            n = max(1, int(seconds * rate))
            for k in range(1, n + 1):
                alpha = k / n
                q = (1 - alpha) * q_prev[:5] + alpha * np.asarray(q_target[:5], dtype=float)
                plan.append(normalize(np.concatenate([q, [grip]]), cfg))
            q_prev = np.concatenate([np.asarray(q_target[:5], dtype=float), [grip]])
        return plan
