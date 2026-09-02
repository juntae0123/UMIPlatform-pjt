"""The AI-part data contract, as code rather than prose.
AI 파트 데이터 계약. 산문이 아니라 코드로 고정한다.

This is the ONE interface between track A (data) and track B (policy). If it
changes, every episode already collected becomes invalid — so changing it needs
both tracks to agree and a D-AI record.
트랙 A(데이터)와 트랙 B(정책)의 **유일한** 접점이다. 이게 바뀌면 이미 수집한
에피소드가 전부 무효가 된다. 변경에는 양 트랙 합의와 D-AI 기록이 필요하다.

Status: PROVISIONAL. Issue S15P21A103-27 is not closed. Track A has not
confirmed the camera layout or the gripper normalisation rule.
상태: **잠정**. S15P21A103-27 미확정. 카메라 구성과 그리퍼 정규화 규칙에 대해
트랙 A 확인이 아직 없다.

Why timestamps are separate fields, even though the simulator makes them equal:
in sim, image and action come from the same step, so the offset is structurally
zero and the fields look pointless. On real hardware they do not, and
S15P21A103-30 has to measure that offset. Without the fields the measurement is
impossible, and a policy trained on misaligned data looks fine in sim and fails
only on the robot — the hardest failure to trace.
시뮬에서는 값이 같은데도 타임스탬프를 따로 두는 이유: 시뮬은 이미지와 액션이
같은 스텝에서 나와 오차가 구조적으로 0이라 이 필드가 무의미해 보인다. 실물은
아니고, S15P21A103-30이 그 오차를 계측해야 한다. 필드가 없으면 계측 자체가
불가능하다. 어긋난 데이터로 학습한 정책은 시뮬에서 멀쩡하고 실물에서만
실패해서, 추적이 가장 어려운 형태의 실패가 된다.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from contract.ids import SKILL_IDS

# 0.2.0 (2026-09-02): EpisodeMeta.skill_id 추가, SKILL_IDS enum 강제.
# 0.1.0 데이터는 이 버전에서 거부된다 — 당시 수집분은 시뮬 스크립트 98편뿐이고
# 재수집에 20분이면 된다. 실시연 데이터가 0편인 지금이 바꿀 수 있는 마지막 구간에
# 가깝다. 750편을 모은 뒤 바꾸면 그때는 전량 폐기다.
CONTRACT_VERSION = "0.2.0-provisional"

IMAGE_SHAPE = (3, 224, 224)  # CHW uint8
STATE_DIM = 6
ACTION_DIM = 6
STATE_RANGE = (-1.0, 1.0)

# How far outside STATE_RANGE a value may sit before the episode is rejected.
# Exposed as a constant because other code has to reason about it: a degree-based
# control API rounds its joint limits, and whether that rounding can invalidate an
# episode depends on exactly this number (sim/mujoco/angle_contract.py).
# STATE_RANGE 를 얼마나 벗어나면 에피소드를 거부하는가. 상수로 노출하는 이유는 다른
# 코드가 이 값을 근거로 판단해야 하기 때문이다 — degree 기반 제어 API 는 관절 한계를
# 반올림하는데, 그 반올림이 에피소드를 무효로 만드는지가 정확히 이 값에 달려 있다
# (sim/mujoco/angle_contract.py).
RANGE_TOLERANCE = 1e-4


@dataclass
class EpisodeMeta:
    """Everything needed to judge whether an episode is usable.
    에피소드를 쓸 수 있는지 판단하는 데 필요한 전부."""

    episode_id: str
    # Which of the five skills this demonstration belongs to. Constrained to
    # `SKILL_IDS` because four parties -- the platform, the user's click, the
    # VLM's selection and the robot's execution -- have to name the same thing,
    # and one of them is a language model that will invent a sixth if nothing
    # stops it.
    # 이 시연이 다섯 스킬 중 어느 것인가. `SKILL_IDS` 로 제한한다. 플랫폼·사용자
    # 클릭·VLM 선택·로봇 실행 네 주체가 같은 것을 같은 이름으로 불러야 하는데,
    # 그중 하나는 막지 않으면 여섯 번째를 지어내는 언어모델이다.
    skill_id: str
    # Free text for humans. `skill_id` is what machines compare.
    # 사람이 읽는 자유 문자열. 기계가 대조하는 것은 `skill_id` 다.
    task: str
    source: str  # "sim" | "real"
    success: bool
    n_steps: int
    control_rate_hz: float
    cameras: list[str]
    contract_version: str = CONTRACT_VERSION
    collected_by: str = ""
    config_sha: str = ""
    git_rev: str = ""
    notes: dict[str, Any] = field(default_factory=dict)


@dataclass
class Episode:
    """One demonstration. Arrays are time-major.
    시연 하나. 배열은 시간축이 첫 축이다."""

    meta: EpisodeMeta
    images: dict[str, np.ndarray]  # camera name -> (T, 3, 224, 224) uint8
    state: np.ndarray  # (T, 6) float32 in [-1, 1]
    state_timestamp: np.ndarray  # (T,) float64 seconds
    action: np.ndarray  # (T, 6) float32 in [-1, 1]
    action_timestamp: np.ndarray  # (T,) float64 seconds


class ContractError(ValueError):
    """An episode violates the data contract.
    에피소드가 데이터 계약을 위반했다."""


def validate(ep: Episode, *, strict_range: bool = True) -> list[str]:
    """Return every contract violation found. Empty list means the episode is valid.
    발견된 계약 위반을 전부 반환한다. 빈 리스트면 유효하다.

    Returning a list rather than raising on the first problem is deliberate: when
    a collection run goes wrong you want to see all of what is wrong at once.
    첫 위반에서 예외를 던지지 않고 목록을 반환하는 것은 의도적이다. 수집이
    잘못됐을 때 무엇이 얼마나 잘못됐는지 한 번에 봐야 한다.
    """
    problems: list[str] = []
    n = ep.meta.n_steps

    if n <= 0:
        problems.append(f"n_steps must be positive, got {n}")

    if set(ep.images) != set(ep.meta.cameras):
        problems.append(f"camera mismatch: arrays={sorted(ep.images)} meta={sorted(ep.meta.cameras)}")

    for name, arr in ep.images.items():
        if arr.dtype != np.uint8:
            problems.append(f"images[{name}].dtype must be uint8, got {arr.dtype}")
        if arr.shape != (n, *IMAGE_SHAPE):
            problems.append(f"images[{name}].shape must be {(n, *IMAGE_SHAPE)}, got {arr.shape}")

    for label, arr, dim in (("state", ep.state, STATE_DIM), ("action", ep.action, ACTION_DIM)):
        if arr.dtype != np.float32:
            problems.append(f"{label}.dtype must be float32, got {arr.dtype}")
        if arr.shape != (n, dim):
            problems.append(f"{label}.shape must be {(n, dim)}, got {arr.shape}")
        if not np.isfinite(arr).all():
            problems.append(f"{label} contains NaN or inf")
        elif strict_range:
            lo, hi = float(arr.min()), float(arr.max())
            if lo < STATE_RANGE[0] - RANGE_TOLERANCE or hi > STATE_RANGE[1] + RANGE_TOLERANCE:
                problems.append(f"{label} out of {STATE_RANGE}: min={lo:.4f} max={hi:.4f}")

    for label, ts in (("state_timestamp", ep.state_timestamp), ("action_timestamp", ep.action_timestamp)):
        if ts.dtype != np.float64:
            problems.append(f"{label}.dtype must be float64, got {ts.dtype}")
        if ts.shape != (n,):
            problems.append(f"{label}.shape must be {(n,)}, got {ts.shape}")
        elif n > 1 and not np.all(np.diff(ts) > 0):
            problems.append(f"{label} is not strictly increasing")

    if ep.state_timestamp.shape == ep.action_timestamp.shape and n > 1:
        offset_ms = np.abs(ep.action_timestamp - ep.state_timestamp) * 1000.0
        if offset_ms.max() > 10.0:
            problems.append(
                f"state/action timestamp offset {offset_ms.max():.2f}ms exceeds 10ms "
                "— S15P21A103-30 게이트 위반"
            )

    if n > 1:
        dt = np.diff(ep.state_timestamp)
        expected = 1.0 / ep.meta.control_rate_hz
        drift = np.abs(dt - expected).max()
        if drift > 0.2 * expected:
            problems.append(
                f"control period drifts by {drift * 1000:.2f}ms from "
                f"{expected * 1000:.2f}ms (>20%)"
            )

    if ep.meta.skill_id not in SKILL_IDS:
        problems.append(
            f"skill_id {ep.meta.skill_id!r} 이 {SKILL_IDS} 에 없다 "
            "— 어느 스킬의 시연인지 알 수 없는 에피소드는 학습에 쓸 수 없다"
        )

    if ep.meta.contract_version != CONTRACT_VERSION:
        problems.append(
            f"contract_version {ep.meta.contract_version!r} != {CONTRACT_VERSION!r}"
        )

    return problems


def write_episode(ep: Episode, out_dir: Path) -> Path:
    """Write one episode as an .npz plus a sidecar .json of its metadata.
    에피소드 하나를 .npz 와 메타데이터 .json 으로 쓴다."""
    problems = validate(ep)
    if problems:
        raise ContractError("계약 위반 상태로는 저장하지 않는다:\n  " + "\n  ".join(problems))

    out_dir.mkdir(parents=True, exist_ok=True)
    npz_path = out_dir / f"{ep.meta.episode_id}.npz"
    arrays: dict[str, np.ndarray] = {
        "state": ep.state,
        "state_timestamp": ep.state_timestamp,
        "action": ep.action,
        "action_timestamp": ep.action_timestamp,
    }
    for name, arr in ep.images.items():
        arrays[f"image__{name}"] = arr
    np.savez_compressed(npz_path, **arrays)
    (out_dir / f"{ep.meta.episode_id}.json").write_text(
        json.dumps(asdict(ep.meta), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return npz_path


def read_episode(npz_path: Path) -> Episode:
    """Read back an episode written by :func:`write_episode`.
    write_episode 로 쓴 에피소드를 다시 읽는다."""
    meta_path = npz_path.with_suffix(".json")
    meta = EpisodeMeta(**json.loads(meta_path.read_text(encoding="utf-8")))
    with np.load(npz_path) as z:
        images = {k[len("image__"):]: z[k] for k in z.files if k.startswith("image__")}
        return Episode(
            meta=meta,
            images=images,
            state=z["state"],
            state_timestamp=z["state_timestamp"],
            action=z["action"],
            action_timestamp=z["action_timestamp"],
        )


def write_dataset_index(out_dir: Path, extra: dict[str, Any] | None = None) -> Path:
    """Write the dataset-level index that describes the contract in force.
    적용 중인 계약을 기술하는 데이터셋 인덱스를 쓴다."""
    episodes = sorted(p.stem for p in out_dir.glob("*.npz"))
    index = {
        "contract_version": CONTRACT_VERSION,
        "status": "PROVISIONAL — S15P21A103-27 미확정, 트랙 A 확인 필요",
        "observation": {
            "image": {"shape": list(IMAGE_SHAPE), "dtype": "uint8", "layout": "CHW"},
            "state": {"shape": [STATE_DIM], "dtype": "float32", "range": list(STATE_RANGE)},
            "state_timestamp": {"shape": [], "dtype": "float64", "unit": "s"},
        },
        "action": {
            "shape": [ACTION_DIM],
            "dtype": "float32",
            "range": list(STATE_RANGE),
            "meaning": "목표 관절각 5개 + 그리퍼, 정규화된 값",
        },
        "action_timestamp": {"shape": [], "dtype": "float64", "unit": "s"},
        "skill_ids_allowed": list(SKILL_IDS),
        "joint_order": [
            "shoulder_pan", "shoulder_lift", "elbow_flex",
            "wrist_flex", "wrist_roll", "gripper",
        ],
        "normalization": "x_norm = 2*(x_rad - lo)/(hi - lo) - 1, lo/hi = configs/so101.yaml joint range",
        "episodes": episodes,
        "n_episodes": len(episodes),
    }
    if extra:
        index.update(extra)
    path = out_dir / "dataset.json"
    path.write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")
    return path
