"""The one seam between a policy and whatever is on the other side of it.
정책과 그 반대편에 있는 것 사이의 유일한 이음매.

S15P21A103-60 asks for the same policy code to run against simulation and
against the real arm. That is only true if the policy never learns which one it
is talking to — so everything simulator-specific lives behind `RobotEnv`, and a
policy only ever sees an `Observation` and returns an action in contract units.
S15P21A103-60 의 요구는 같은 정책 코드가 시뮬과 실물 양쪽에서 도는 것이다.
정책이 자기가 어느 쪽과 얘기하는지 **모를 때만** 그게 성립한다. 그래서 시뮬에
고유한 것은 전부 `RobotEnv` 뒤에 두고, 정책은 `Observation` 을 받아 계약 단위의
행동만 돌려준다.

The real-robot implementation is NOT in this file and does not exist yet. It
belongs to S15P21A103-46 ([ROS] 정책 ckpt 로드→추론 노드). What this file
guarantees is that it has one shape to fill in.
실물 구현은 이 파일에 없고 아직 존재하지도 않는다. S15P21A103-46 의 몫이다.
이 파일이 보장하는 것은, 채워 넣을 형태가 하나로 정해져 있다는 것뿐이다.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import mujoco
import numpy as np

from so101_ai.sim.build_scene import build_model, denormalize, load_config, normalize


@dataclass(frozen=True)
class Observation:
    """What a policy sees at one control tick — data-contract units.
    정책이 제어 틱 한 번에 보는 것. 데이터 계약 단위.

    Field shapes and dtypes match `schema/contract.py` exactly. If they drift
    apart, a policy trained on recorded data will silently receive something
    different at inference time.
    필드의 shape 과 dtype 은 `schema/contract.py` 와 정확히 일치한다. 둘이 갈라지면,
    기록 데이터로 학습한 정책이 추론 시점에 다른 것을 조용히 받게 된다.
    """

    images: dict[str, np.ndarray]  # camera name -> (3, 224, 224) uint8, CHW
    state: np.ndarray  # (6,) float32, normalized to [-1, 1]
    timestamp: float  # seconds since episode start


@runtime_checkable
class RobotEnv(Protocol):
    """What a policy is allowed to know about the world.
    정책이 세상에 대해 알아도 되는 전부."""

    @property
    def control_rate_hz(self) -> float:
        """Ticks per second the policy is called at.
        정책이 호출되는 주기."""
        ...

    @property
    def camera_names(self) -> list[str]:
        """Cameras present in the observation, in a stable order.
        관측에 들어오는 카메라. 순서는 고정."""
        ...

    def reset(self, seed: int | None = None) -> Observation:
        """Put the world back to a start state and return the first observation.
        세상을 시작 상태로 되돌리고 첫 관측을 반환한다."""
        ...

    def step(self, action: np.ndarray) -> Observation:
        """Apply one normalized action and advance one control tick.
        정규화된 행동 하나를 적용하고 제어 틱 하나만큼 진행한다."""
        ...

    def is_success(self) -> bool:
        """Whether the task is currently satisfied.
        지금 태스크가 달성된 상태인가."""
        ...


class MujocoPickEnv:
    """`RobotEnv` backed by the MuJoCo pick scene.
    MuJoCo 픽 씬으로 구현한 `RobotEnv`.

    Rendering is the expensive part — about 60 ms per tick for two 224x224
    cameras on CPU. `render=False` skips it for policies that ignore images,
    which is what makes a 200-rollout baseline sweep finish in minutes instead
    of hours. The observation then carries zero-filled image arrays of the
    correct shape, so a policy that *does* look at images cannot silently
    receive a differently-shaped input.
    렌더링이 비싸다 — CPU에서 224x224 카메라 2대 기준 틱당 약 60ms.
    이미지를 안 보는 정책에는 `render=False` 로 건너뛴다. 200회 롤아웃 스윕이
    시간 단위가 아니라 분 단위로 끝나는 이유다. 이때 관측에는 올바른 shape 의
    0으로 채운 배열이 들어가므로, 이미지를 **보는** 정책이 다른 shape 를 조용히
    받는 일은 생기지 않는다.
    """

    def __init__(
        self,
        cfg: dict[str, Any] | None = None,
        render: bool = True,
        object_jitter_m: float = 0.05,
        max_ticks: int = 200,
    ) -> None:
        self.cfg = cfg if cfg is not None else load_config()
        self.model = build_model(self.cfg)
        self.data = mujoco.MjData(self.model)
        self._render = render
        self._jitter = float(object_jitter_m)
        self.max_ticks = int(max_ticks)

        self._width, self._height = self.cfg["cameras"]["resolution"]
        self._rate = float(self.cfg["control"]["rate_hz"])
        self._substeps = max(1, int((1.0 / self._rate) / self.model.opt.timestep))
        self._cams = [
            mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_CAMERA, i)
            for i in range(self.model.ncam)
        ]
        self._renderer = (
            mujoco.Renderer(self.model, height=self._height, width=self._width)
            if render
            else None
        )

        self._obj_body = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "target_object")
        self._obj_geom = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_GEOM, "target_object_geom"
        )
        self._obj_joint = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, "target_object_free"
        )
        self._obj_qadr = self.model.jnt_qposadr[self._obj_joint]
        self._jaw_geoms = self._collect_jaw_geoms()
        self._base_xy = np.asarray(self.cfg["task"]["object"]["init_pos"][:2], dtype=float)
        self._success_lift = float(self.cfg["grasp"]["success_lift_m"])

        self._t0_z = 0.0
        self._ticks = 0

    # ---- RobotEnv -------------------------------------------------------

    @property
    def control_rate_hz(self) -> float:
        """Ticks per second the policy is called at.
        정책이 호출되는 주기."""
        return self._rate

    @property
    def camera_names(self) -> list[str]:
        """Cameras present in the observation.
        관측에 들어오는 카메라."""
        return list(self._cams)

    def reset(
        self, seed: int | None = None, object_xy: tuple[float, float] | None = None
    ) -> Observation:
        """Randomise the object position and settle the scene.
        물체 위치를 무작위로 놓고 씬을 안정시킨다.

        `object_xy` overrides the randomisation. It exists so a recorded episode
        can be replayed against the exact condition it was recorded under —
        without that, a replay baseline scoring 0% tells you nothing about the
        task, only that the object moved.
        `object_xy` 를 주면 무작위화를 덮어쓴다. 기록된 에피소드를 그것이 기록된
        **바로 그 조건**에서 재생하기 위한 것이다. 이게 없으면 replay baseline 의
        0% 는 태스크에 대해 아무것도 말해주지 않는다. 물체가 움직였다는 것만 말해준다.
        """
        mujoco.mj_resetData(self.model, self.data)
        rng = np.random.default_rng(seed)
        xy = (
            np.asarray(object_xy, dtype=float)
            if object_xy is not None
            else self._base_xy + rng.uniform(-self._jitter, self._jitter, size=2)
        )
        self.data.qpos[self._obj_qadr] = xy[0]
        self.data.qpos[self._obj_qadr + 1] = xy[1]
        mujoco.mj_forward(self.model, self.data)
        self.data.ctrl[:] = self.data.qpos[:6]
        for _ in range(int(0.4 / self.model.opt.timestep)):
            mujoco.mj_step(self.model, self.data)
        self._t0_z = float(self.data.xpos[self._obj_body][2])
        self._ticks = 0
        return self._observe()

    def step(self, action: np.ndarray) -> Observation:
        """Apply one normalized action and advance one control tick.
        정규화된 행동 하나를 적용하고 제어 틱 하나만큼 진행한다."""
        action = np.asarray(action, dtype=np.float32).reshape(6)
        if not np.isfinite(action).all():
            raise ValueError(f"action must be finite, got {action}")
        # Clipping here is a safety limit, not a data-contract decision. The
        # contract question (what to do with out-of-range values in recorded
        # data) is open — see S15P21A103-27.
        # 여기서의 클립은 안전 제한이지 데이터 계약 결정이 아니다. 기록 데이터의
        # 범위 초과 처리 문제는 미결이다 — S15P21A103-27 참조.
        self.data.ctrl[:6] = denormalize(np.clip(action, -1.0, 1.0), self.cfg)
        for _ in range(self._substeps):
            mujoco.mj_step(self.model, self.data)
        self._ticks += 1
        return self._observe()

    def is_success(self) -> bool:
        """Object lifted past the threshold and still held.
        물체가 기준 높이 이상 올라갔고 아직 잡혀 있는가."""
        lifted = float(self.data.xpos[self._obj_body][2]) - self._t0_z
        return lifted >= self._success_lift and self._n_jaw_contacts() > 0

    # ---- simulator-only accessors --------------------------------------
    # A policy must not call these. They exist for scripted baselines and for
    # scoring, where reading privileged state is the point.
    # 정책이 호출하면 안 된다. 스크립트 baseline 과 채점용이다.
    # 특권 정보를 읽는 것 자체가 목적인 자리다.

    def object_position(self) -> np.ndarray:
        """Ground-truth object position — privileged, simulator only.
        물체의 정답 위치. 특권 정보이며 시뮬에만 존재한다."""
        return self.data.xpos[self._obj_body].copy()

    def lift_height(self) -> float:
        """How far the object has risen from its settled height.
        물체가 안착 높이에서 얼마나 올라갔는가."""
        return float(self.data.xpos[self._obj_body][2]) - self._t0_z

    def joint_positions(self) -> np.ndarray:
        """Raw joint angles in radians.
        관절각 원값 (rad)."""
        return self.data.qpos[:6].copy()

    def close(self) -> None:
        """Release the renderer.
        렌더러를 해제한다."""
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

    def __enter__(self) -> "MujocoPickEnv":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # ---- internals ------------------------------------------------------

    def _collect_jaw_geoms(self) -> set[int]:
        ids: set[int] = set()
        for name in ("gripper", "moving_jaw_so101_v1"):
            bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
            if bid < 0:
                continue
            ids |= {
                g
                for g in range(self.model.ngeom)
                if self.model.geom_bodyid[g] == bid and self.model.geom_contype[g] != 0
            }
        return ids

    def _n_jaw_contacts(self) -> int:
        n = 0
        for i in range(self.data.ncon):
            con = self.data.contact[i]
            pair = {con.geom1, con.geom2}
            if self._obj_geom in pair and pair & self._jaw_geoms:
                n += 1
        return n

    def _observe(self) -> Observation:
        images: dict[str, np.ndarray] = {}
        for cam in self._cams:
            if self._renderer is None:
                images[cam] = np.zeros((3, self._height, self._width), dtype=np.uint8)
                continue
            self._renderer.update_scene(self.data, camera=cam)
            images[cam] = np.transpose(self._renderer.render(), (2, 0, 1)).copy()
        return Observation(
            images=images,
            state=normalize(self.data.qpos[:6].copy(), self.cfg),
            timestamp=float(self.data.time),
        )
