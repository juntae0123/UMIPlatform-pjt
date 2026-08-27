"""Build the SO-101 pick-place MuJoCo model without touching the official MJCF.
공식 MJCF를 수정하지 않고 SO-101 픽앤플레이스 모델을 조립한다.

The official SO-101 description ships with zero cameras, and MJCF `include`
cannot reopen a body that lives in an included file — so a wrist camera cannot
be declared in the scene XML at all. Both cameras are therefore injected at
load time through `mujoco.MjSpec`, driven entirely by `configs/so101.yaml`.
공식 SO-101 기술서에는 카메라가 하나도 없고, include된 파일 안의 body는 다시
열 수 없다. 그래서 손목 카메라는 씬 XML에 선언 자체가 불가능하다.
두 카메라 모두 configs/so101.yaml 값으로 로드 시점에 주입한다.

Hardware-dependent numbers live only in the config. Nothing here is hardcoded.
하드웨어 의존 값은 설정 파일에만 있다. 여기에 하드코딩된 값은 없다.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
import yaml

REPO_AI_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_AI_ROOT / "configs" / "so101.yaml"
DEFAULT_SCENE = REPO_AI_ROOT / "scenes" / "pick_place.xml"


@dataclass(frozen=True)
class JointSpec:
    """One actuated joint and its hardware limit in radians.
    관절 하나와 그 하드웨어 한계(rad)."""

    name: str
    index: int
    lo: float
    hi: float


def load_config(path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    """Read the single-source-of-truth hardware config.
    하드웨어 값 단일 출처 설정을 읽는다."""
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def joint_specs(cfg: dict[str, Any]) -> list[JointSpec]:
    """Return joints ordered by their state/action vector index.
    state/action 벡터 인덱스 순서대로 관절을 반환한다."""
    specs = [
        JointSpec(j["name"], j["index"], float(j["range_rad"][0]), float(j["range_rad"][1]))
        for j in cfg["joints"]
    ]
    specs.sort(key=lambda s: s.index)
    if [s.index for s in specs] != list(range(len(specs))):
        raise ValueError(f"joint index must be contiguous from 0: {[s.index for s in specs]}")
    return specs


def _set_camera(cam: Any, spec: dict[str, Any], resolution: list[int]) -> None:
    """Apply one camera block from the config onto an MjSpec camera.
    설정의 카메라 블록 하나를 MjSpec 카메라에 적용한다."""
    xyaxes = np.asarray(spec["xyaxes"], dtype=float)
    if xyaxes.shape != (6,):
        raise ValueError(f"xyaxes must have 6 numbers, got {xyaxes.shape}")
    cam.pos = np.asarray(spec["pos"], dtype=float)
    cam.fovy = float(spec["fovy"])
    cam.mode = mujoco.mjtCamLight.mjCAMLIGHT_FIXED
    cam.alt.xyaxes = xyaxes
    cam.alt.type = mujoco.mjtOrientation.mjORIENTATION_XYAXES
    cam.resolution = np.asarray(resolution, dtype=float)


def build_model(
    cfg: dict[str, Any] | None = None,
    scene_path: Path = DEFAULT_SCENE,
) -> mujoco.MjModel:
    """Compile the scene with every camera injected from the config.
    설정에서 읽은 카메라를 전부 주입해 씬을 컴파일한다."""
    cfg = cfg if cfg is not None else load_config()
    cams = cfg["cameras"]
    resolution = cams["resolution"]
    spec = mujoco.MjSpec.from_file(str(scene_path))

    names = [k for k in cams if isinstance(cams[k], dict)]
    if len(names) != int(cams["count"]):
        raise ValueError(f"cameras.count={cams['count']} but {len(names)} camera blocks defined")

    for name in names:
        block = cams[name]
        kind = block["kind"]
        if kind == "fixed":
            parent = spec.worldbody
        elif kind == "attached":
            parent = spec.body(block["parent_body"])
            if parent is None:
                raise ValueError(
                    f"body {block['parent_body']!r} not found in {scene_path.name}; "
                    "공식 MJCF의 body 이름이 바뀌었는지 확인할 것"
                )
        else:
            raise ValueError(f"unknown camera kind {kind!r} for {name!r}")
        cam = parent.add_camera()
        cam.name = name
        _set_camera(cam, block, resolution)

    model = spec.compile()

    found = {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_CAMERA, i) for i in range(model.ncam)}
    if found != set(names):
        raise RuntimeError(f"camera injection failed: expected {set(names)}, got {found}")
    return model


def verify_against_config(model: mujoco.MjModel, cfg: dict[str, Any]) -> list[str]:
    """Compare compiled joint limits with the config; return mismatch messages.
    컴파일된 관절 한계를 설정과 대조하고 불일치 목록을 반환한다."""
    problems: list[str] = []
    for spec in joint_specs(cfg):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, spec.name)
        if jid < 0:
            problems.append(f"{spec.name}: joint not found in compiled model")
            continue
        lo, hi = model.jnt_range[jid]
        if not (np.isclose(lo, spec.lo, atol=1e-9) and np.isclose(hi, spec.hi, atol=1e-9)):
            problems.append(
                f"{spec.name}: mjcf=({lo!r}, {hi!r}) != config=({spec.lo!r}, {spec.hi!r})"
            )
    return problems


def normalize(q_rad: np.ndarray, cfg: dict[str, Any]) -> np.ndarray:
    """Map joint angles (rad) to the [-1, 1] data-contract range.
    관절각(rad)을 데이터 계약의 [-1,1] 범위로 매핑한다."""
    specs = joint_specs(cfg)
    lo = np.array([s.lo for s in specs], dtype=np.float32)
    hi = np.array([s.hi for s in specs], dtype=np.float32)
    return (2.0 * (np.asarray(q_rad, dtype=np.float32) - lo) / (hi - lo) - 1.0).astype(np.float32)


def denormalize(q_norm: np.ndarray, cfg: dict[str, Any]) -> np.ndarray:
    """Inverse of :func:`normalize`.
    normalize의 역변환."""
    specs = joint_specs(cfg)
    lo = np.array([s.lo for s in specs], dtype=np.float32)
    hi = np.array([s.hi for s in specs], dtype=np.float32)
    return ((np.asarray(q_norm, dtype=np.float32) + 1.0) / 2.0 * (hi - lo) + lo).astype(np.float32)


def dls_ik(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    target_xyz: np.ndarray,
    site_name: str = "gripperframe",
    max_iters: int = 300,
    damping: float = 1e-2,
    tol: float = 1e-3,
) -> tuple[np.ndarray, float, bool]:
    """Damped least-squares IK for the gripper site, clamped to joint limits.
    그리퍼 site에 대한 감쇠최소자승 IK. 관절 한계로 클램프한다.

    Returns (qpos_arm, final_position_error_m, within_limits).
    반환: (팔 관절각, 최종 위치오차 m, 관절한계 내 여부).
    이건 '실행가능성 검사'의 씨앗이다 — 목표 자세가 애초에 도달 불가면 여기서 걸린다.
    """
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
    if sid < 0:
        raise ValueError(f"site {site_name!r} not found")
    n_arm = 6
    lo = model.jnt_range[:n_arm, 0].copy()
    hi = model.jnt_range[:n_arm, 1].copy()
    jacp = np.zeros((3, model.nv))
    target = np.asarray(target_xyz, dtype=float)

    for _ in range(max_iters):
        mujoco.mj_forward(model, data)
        err = target - data.site_xpos[sid]
        if np.linalg.norm(err) < tol:
            break
        mujoco.mj_jacSite(model, data, jacp, None, sid)
        j = jacp[:, :n_arm]
        dq = j.T @ np.linalg.solve(j @ j.T + damping * np.eye(3), err)
        data.qpos[:n_arm] = np.clip(data.qpos[:n_arm] + dq, lo, hi)

    mujoco.mj_forward(model, data)
    final_err = float(np.linalg.norm(target - data.site_xpos[sid]))
    q = data.qpos[:n_arm].copy()
    within = bool(np.all(q > lo + 1e-6) and np.all(q < hi - 1e-6))
    return q, final_err, within


if __name__ == "__main__":
    config = load_config()
    mj_model = build_model(config)
    mismatches = verify_against_config(mj_model, config)
    print(f"nq={mj_model.nq} nu={mj_model.nu} ncam={mj_model.ncam}")
    print("cameras:", [
        mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_CAMERA, i)
        for i in range(mj_model.ncam)
    ])
    print("config mismatches:", mismatches or "none")
