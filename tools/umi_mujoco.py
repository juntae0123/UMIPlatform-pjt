"""The wiring layer: MuJoCo kinematics behind the `umi.ik.IKSolver` interface.
배선 층. MuJoCo 기구학을 `umi.ik.IKSolver` 인터페이스 뒤에 둔다.

**This is the only file allowed to import both `umi/` and `sim/`.** `umi/` stays
neutral so the real-data path (S15P21A103-31) can inject a MoveIt/URDF solver
instead; the simulator dependency lives here, at the call site, not in the seam.
**`umi/` 와 `sim/` 을 함께 임포트할 수 있는 유일한 파일이다.** `umi/` 를 중립으로
두어야 실데이터 경로(이슈 31)가 MoveIt/URDF 솔버를 대신 주입할 수 있다. 시뮬
의존성은 이음매가 아니라 호출 지점인 여기 있다.

⚠️ `sim/mujoco/kinematics.py` 와 `sim/mujoco/build_scene.py` 는 시뮬·정책 대화
   단독 소유다. **임포트만 하고 고치지 않는다.** 그쪽이 `solve_pose_ik` 의
   인자·반환을 바꾸면 이 파일이 깨진다 — 소유권 문서 "변경 예고" 에 올려뒀다.

    # [로컬]
    cd AI && python tools/umi_mujoco.py --episodes 5 --steps 90

`main()` 은 관통 검증의 **계측기 검증**이다. 유효한 관절 배치에서 FK 로 pose 를
만들고 IK 로 되돌린다. 그 pose 는 5자유도 다양체 위에 정확히 놓여 있으므로
**오차는 솔버 수렴 한계뿐이어야 한다.** 크게 나오면 좌표계·오프셋·부호 버그다.
이 수치를 UMI 실행가능성으로 보고하지 않는다 — 도달 가능성이 구조적으로 보장된
입력이다.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from paths import DEFAULT_CONFIG, DEFAULT_SCENE  # noqa: E402
from sim.mujoco.build_scene import (  # noqa: E402
    build_model,
    load_config,
    normalize as sim_normalize,
    verify_against_config,
)
from sim.mujoco.kinematics import (  # noqa: E402
    IKResult,
    approach_axis,
    grasp_point,
    solve_pose_ik,
)
from umi.convert import ConversionError, normalize_joints  # noqa: E402
from umi.ik import IKSolution, approach_axis_from_quat, matrix_to_quat, roll_residual_deg  # noqa: E402
from umi.ik import summarize  # noqa: E402

N_JOINTS = 6
GRIPPER_BODY = "gripper"
NORM_CROSSCHECK_ATOL = 1e-6
"""`sim.build_scene.normalize` computes in float32, `umi` in float64. Anything
tighter than this compares the two dtypes, not the two formulas.
`sim.build_scene.normalize` 는 float32, `umi` 는 float64 로 계산한다. 이보다
빡빡하게 잡으면 공식이 아니라 dtype 을 비교하는 것이 된다."""


def joint_ranges(cfg: dict[str, Any]) -> np.ndarray:
    """(6, 2) of [lo, hi] in radians, from the config, in joint-index order.
    설정에서 읽은 (6,2) [lo, hi] 라디안. 관절 인덱스 순서.

    Read straight from the YAML rather than through `joint_specs`, which casts to
    float32 -- the contract's normalisation is defined on these exact numbers.
    `joint_specs` 를 거치지 않고 YAML 에서 바로 읽는다. 그쪽은 float32 로
    캐스팅하는데, 계약의 정규화는 이 정확한 수치 위에 정의돼 있다.
    """
    rows = sorted(cfg["joints"], key=lambda j: int(j["index"]))
    if len(rows) != N_JOINTS:
        raise ConversionError(f"관절이 {len(rows)}개다. {N_JOINTS}개여야 한다")
    return np.array([[float(j["range_rad"][0]), float(j["range_rad"][1])] for j in rows])


def gap_from_angle(angle_rad: float, curve: Sequence[Sequence[float]]) -> float:
    """Gripper joint angle to pad-to-pad gap in metres. Forward direction of the
    measured curve; `umi.convert.invert_gap_curve` is the inverse.
    그리퍼 관절각을 패드 간 간격[m]으로. 실측 곡선의 정방향."""
    table = np.asarray(curve, dtype=float)
    return float(np.interp(float(angle_rad), table[:, 0], table[:, 1] / 100.0))


def eef_pose_from_joints(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    q_rad: np.ndarray,
    pinch_offset_local: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Forward kinematics to the raw schema's EEF pose: pinch position and orientation.
    raw 스키마의 EEF pose 로 가는 순기구학. 파지점 위치와 자세.

    The orientation follows `umi/raw.py`'s convention, and the sign flips are the
    whole point of writing it out:

        R[:, 2] = approach axis = **minus** the gripper body's local z 🟢
        R[:, 0] = jaw axis      = the gripper body's local x
        R[:, 1] = z x x         = minus the body's local y (keeps it right-handed)

    자세는 `umi/raw.py` 규약을 따르고, 부호 반전이 이 함수를 따로 쓰는 이유다.
    실측상 손가락은 gripper body 로컬 **-z** 를 향한다. 부호를 틀리면 IK 가
    반대편에서 접근하고, 증상은 "IK 가 좀 안 맞네" 로만 보인다.

    Position is the pinch point, not the TCP site -- the same point
    `solve_pose_ik` aims at, so the round trip compares like with like.
    위치는 TCP site 가 아니라 파지점이다. `solve_pose_ik` 가 조준하는 바로 그
    점이라서 왕복이 같은 것끼리 비교된다.
    """
    data.qpos[:N_JOINTS] = np.asarray(q_rad, dtype=float)
    mujoco.mj_forward(model, data)
    pos = grasp_point(model, data, np.asarray(pinch_offset_local, dtype=float))
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, GRIPPER_BODY)
    rot = data.xmat[bid].reshape(3, 3)
    r_eef = np.column_stack([rot[:, 0], -rot[:, 1], -rot[:, 2]])
    # 규약 자기검사: 이 행렬의 z열은 sim 의 approach_axis 와 같아야 한다.
    if not np.allclose(r_eef[:, 2], approach_axis(model, data), atol=1e-12):
        raise ConversionError("EEF 규약과 sim.approach_axis 가 불일치 — 부호 규약을 다시 봐라")
    return np.asarray(pos, dtype=float).copy(), matrix_to_quat(r_eef)


class MujocoIK:
    """`umi.ik.IKSolver` over `sim.mujoco.kinematics.solve_pose_ik`.
    `solve_pose_ik` 를 감싼 `umi.ik.IKSolver` 구현.

    The 6-DOF pose is projected onto the five constraints the arm can hold:
    position (3) and approach direction (2). `wrist_roll` is left free -- with it
    fixed the solve becomes over-determined (5 constraints, 4 free joints) and the
    position error, which is the one with a measured budget, is what gives way.
    6자유도 pose 를 팔이 유지할 수 있는 5개 구속으로 투영한다. 위치 3 + 접근방향 2.
    `wrist_roll` 은 열어둔다 — 고정하면 구속 5개에 자유관절 4개로 과결정이 되고,
    실측 예산이 걸려 있는 위치 오차가 밀린다.

    The discarded roll is measured per step, never reported as zero.
    버린 롤은 스텝마다 계측한다. 0 으로 보고하지 않는다.
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        cfg: dict[str, Any],
        *,
        crosscheck: bool = True,
        pos_tol_m: float = 1e-5,
        axis_tol_deg: float = 0.05,
    ) -> None:
        """`pos_tol_m` and `axis_tol_deg` are tightened from `solve_pose_ik`'s
        defaults (1e-3 m, 1.0 deg) on purpose, and recorded as conditions.

        MEASURED 🟢 2026-09-07: with the defaults, the round-trip position error
        piled up against 1.0mm — median 0.830, p95 0.983, max 0.9989, with 0/450
        non-converged. That distribution is the stopping tolerance, not the
        geometry. 1mm is 20% of the conversion's 5mm budget spent on an early
        exit, and there were iterations left over.
        기본값(1e-3 m, 1.0도)보다 조인 값이고, 실행 조건으로 기록한다.

        실측 🟢 2026-09-07: 기본값으로는 왕복 위치 오차가 1.0mm 에 쌓였다 —
        중앙 0.830, p95 0.983, 최대 0.9989, 미수렴 0/450. 그 분포는 기하가 아니라
        정지 허용오차다. 변환 예산 5mm 의 20% 를 조기종료로 쓰는 것이고, 반복
        여유는 남아 있었다.

        ⚠️ `solve_pose_ik` 의 기본값을 바꾸지 않는다 — 그 함수는 시뮬·정책 대화
           소유이고 스크립트 수집이 그 기본값으로 검증돼 있다. 여기서 인자로만
           덮는다.
        """
        self.model = model
        self.cfg = cfg
        self.pos_tol_m = float(pos_tol_m)
        self.axis_tol_deg = float(axis_tol_deg)
        self.data = mujoco.MjData(model)
        self.pinch = np.asarray(cfg["grasp"]["pinch_offset_local"], dtype=float)
        self.ranges = joint_ranges(cfg)
        if crosscheck:
            self._crosscheck()

    def _crosscheck(self) -> None:
        """Fail now if the config, the compiled model and the two normalisations disagree.
        설정·컴파일된 모델·두 정규화 구현이 어긋나면 지금 실패한다.

        Duplication across the track boundary cannot be removed, so make divergence
        detectable. A converter that normalises differently from the simulator
        produces a dataset whose numbers mean something else, and nothing downstream
        can tell.
        트랙 경계 때문에 중복을 없앨 수 없으니 갈라지는 것을 탐지 가능하게 만든다.
        시뮬과 다르게 정규화하는 변환기는 수치의 의미가 다른 데이터셋을 만들고,
        하류에서는 아무도 그걸 알 수 없다.
        """
        problems = verify_against_config(self.model, self.cfg)
        if problems:
            raise ConversionError(
                "컴파일된 모델과 설정의 관절 한계가 다르다:\n  " + "\n  ".join(problems)
            )
        rng = np.random.default_rng(20260907)
        lo, hi = self.ranges[:, 0], self.ranges[:, 1]
        q = lo + rng.random((64, N_JOINTS)) * (hi - lo)
        mine = np.stack([normalize_joints(row, self.ranges) for row in q])
        theirs = np.stack([sim_normalize(row, self.cfg) for row in q])
        worst = float(np.abs(mine - theirs).max())
        if worst > NORM_CROSSCHECK_ATOL:
            raise ConversionError(
                f"umi.normalize_joints 와 sim.build_scene.normalize 가 최대 {worst:.3e} "
                f"다르다 (허용 {NORM_CROSSCHECK_ATOL:.0e}). 두 구현이 갈라졌다"
            )
        self.norm_crosscheck_max = worst

    def solve(
        self,
        pos_m: np.ndarray,
        quat_wxyz: np.ndarray,
        gripper_gap_m: float,
        q_init: np.ndarray | None = None,
    ) -> IKSolution:
        desired_axis = approach_axis_from_quat(quat_wxyz)
        res: IKResult = solve_pose_ik(
            self.model,
            target_xyz=np.asarray(pos_m, dtype=float),
            offset_local=self.pinch,
            desired_axis=desired_axis,
            q_init=None if q_init is None else np.asarray(q_init, dtype=float),
            pos_tol=self.pos_tol_m,
            axis_tol_deg=self.axis_tol_deg,
        )
        q = np.asarray(res.qpos, dtype=float).copy()

        self.data.qpos[:N_JOINTS] = q
        mujoco.mj_forward(self.model, self.data)
        bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, GRIPPER_BODY)
        rot = self.data.xmat[bid].reshape(3, 3)
        roll = roll_residual_deg(quat_wxyz, -rot[:, 2], rot[:, 0])

        # 수렴 판정은 sim 쪽 상수(5mm, 5도)를 그대로 쓴다. 임계값을 새로 지어내지
        # 않는다. 솔버에 준 정지 허용오차(pos_tol_m)와는 다른 것이다 — 이쪽은
        # "쓸 수 있는 해인가", 저쪽은 "언제 멈추는가" 다.
        converged = (
            res.pos_error_m < IKResult.POS_TOL_M
            and res.axis_error_deg < IKResult.AXIS_TOL_DEG
        )
        return IKSolution(
            q_rad=q,
            pos_error_m=float(res.pos_error_m),
            axis_error_deg=float(res.axis_error_deg),
            roll_residual_deg=float(roll),
            within_limits=bool(res.within_limits),
            converged=bool(converged),
        )


def smooth_joint_trajectory(
    ranges: np.ndarray, n: int, rng: np.random.Generator, *, span_frac: float = 0.45
) -> np.ndarray:
    """A slow trajectory strictly inside the joint limits, one sinusoid per axis.
    관절 한계 안에서만 움직이는 느린 궤적. 축마다 사인 하나.

    Inside the limits by construction, so a limit rejection during the round trip
    means the IK wandered, not that the input was impossible.
    구조적으로 한계 안이므로, 왕복 중 한계 폐기가 나오면 입력이 불가능했던 것이
    아니라 IK 가 헤맨 것이다.
    """
    lo, hi = ranges[:, 0], ranges[:, 1]
    mid = 0.5 * (lo + hi)
    span = span_frac * 0.5 * (hi - lo)
    phase = rng.random(N_JOINTS) * 2.0 * np.pi
    freq = 0.5 + rng.random(N_JOINTS)
    s = np.linspace(0.0, 1.0, n)[:, None]
    return mid + span * np.sin(2.0 * np.pi * freq * s + phase)


def main() -> int:
    ap = argparse.ArgumentParser(description="FK→IK 왕복으로 변환 계측기를 검증한다")
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--steps", type=int, default=90)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    ap.add_argument("--scene", type=Path, default=DEFAULT_SCENE)
    ap.add_argument("--pos-tol-m", type=float, default=1e-5,
                    help="솔버 정지 허용오차. sim 기본값은 1e-3 이고 그게 왕복 오차의 천장이었다")
    ap.add_argument("--axis-tol-deg", type=float, default=0.05)
    args = ap.parse_args()

    cfg = load_config(args.config)
    model = build_model(cfg, args.scene)
    ik = MujocoIK(model, cfg, pos_tol_m=args.pos_tol_m, axis_tol_deg=args.axis_tol_deg)
    data = mujoco.MjData(model)
    ranges = ik.ranges
    curve = cfg["grasp"]["gap_curve"]

    print(f"씬 {args.scene.name} · 설정 {args.config.name}")
    print(f"관절 한계 대조 OK · 정규화 두 구현 최대차 {ik.norm_crosscheck_max:.3e}")
    print(f"솔버 정지 허용오차 pos {ik.pos_tol_m:.1e}m · axis {ik.axis_tol_deg}도 "
          f"(sim 기본값 1e-3m · 1.0도)\n")

    rng = np.random.default_rng(args.seed)
    all_sols = []
    q_err_max = 0.0
    gap_err_max = 0.0

    for e in range(args.episodes):
        q_true = smooth_joint_trajectory(ranges, args.steps, rng)
        poses = [eef_pose_from_joints(model, data, q, ik.pinch) for q in q_true]
        gaps = [gap_from_angle(q[5], curve) for q in q_true]

        sols = []
        q_prev = None
        for i, ((pos, quat), gap) in enumerate(zip(poses, gaps)):
            sol = ik.solve(pos, quat, gap, q_init=q_prev)
            sols.append(sol)
            q_prev = sol.q_rad
        summary = summarize(sols)
        all_sols.extend(sols)

        q_rec = np.stack([s.q_rad for s in sols])
        q_err = float(np.abs(q_rec[:, :5] - q_true[:, :5]).max())
        q_err_max = max(q_err_max, q_err)
        gap_back = [
            gap_from_angle(
                float(np.interp(g, np.asarray(curve)[:, 1] / 100.0, np.asarray(curve)[:, 0])),
                curve,
            )
            for g in gaps
        ]
        gap_err_max = max(gap_err_max, float(np.abs(np.array(gap_back) - np.array(gaps)).max()))

        print(
            f"ep{e}  위치 중앙 {summary.pos_error_median_mm:6.3f}mm  "
            f"p95 {summary.pos_error_p95_mm:6.3f}  최대 {summary.pos_error_max_mm:6.3f}  "
            f"축 {summary.axis_error_median_deg:5.2f}도  "
            f"roll잔차 중앙 {summary.roll_residual_median_deg:6.2f}도  "
            f"미수렴 {summary.n_not_converged:3d}  한계 {summary.n_limit_clamped:3d}  "
            f"5mm초과 {summary.n_over_5mm:3d}"
        )

    total = summarize(all_sols)
    print(f"\n=== 합계 n={total.n} ({args.episodes}편 x {args.steps}스텝) ===")
    print(f"위치 오차   중앙 {total.pos_error_median_mm:.4f}mm · p95 {total.pos_error_p95_mm:.4f} · 최대 {total.pos_error_max_mm:.4f}")
    print(f"접근축 오차 중앙 {total.axis_error_median_deg:.4f}도")
    print(f"roll 잔차   중앙 {total.roll_residual_median_deg:.3f}도 · p95 {total.roll_residual_p95_deg:.3f}")
    print(f"미수렴 {total.n_not_converged}/{total.n} · 한계클램프 {total.n_limit_clamped}/{total.n} · 5mm초과 {total.n_over_5mm}/{total.n}")
    print(f"관절각 복원 최대오차 (팔 5축) {q_err_max:.4f} rad = {np.degrees(q_err_max):.2f}도")
    # 이 검사는 표의 자기일관성만 본다. 실물 리그의 간격과는 무관하다.
    print(f"gap 곡선 자기일관성 최대오차 {gap_err_max * 1000:.4f} mm (실물 리그와 무관)")
    print(f"\n127 게이트 (위치 중앙값 5mm 이내): {'통과' if total.passes_127_gate else '미통과'}")
    print(
        "⚠️ 이 입력은 유효한 관절 배치에서 FK 로 만든 것이라 5자유도 다양체 위에 정확히\n"
        "   놓여 있다. 도달 가능성이 구조적으로 보장된다. 이 수치를 UMI 실행가능성으로\n"
        "   보고하지 않는다 — 재는 것은 좌표계·오프셋·부호가 맞는가다."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
