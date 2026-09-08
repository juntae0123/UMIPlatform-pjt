"""ARCore 세션 기록 → `umi.raw.RawEpisode`. **구현 없음.** S15P21A103-31.
ARCore session recording to `umi.raw.RawEpisode`. NOT IMPLEMENTED.

**이 파일이 트랙 A 가 채워야 하는 유일한 칸이다.** 여기서 `RawEpisode` 를 만들어
주면 아래 전부가 이미 돌아간다 — 변환기·IK·계약 검증기·수용률 리포트.

    track_a/convert/arcore.py   ←  여기만 비어 있다
        ↓  umi.raw.RawEpisode
    umi/convert.py              ←  구현·검증 완료 (검사 26건 통과)
        ↓  contract.episode.Episode
    data/verify.py              ←  위반 0 확인 완료 (합성 20편)
        ↓
    학습 · 롤아웃

## 왜 아직 못 쓰는가 — 지어내면 안 되는 것 셋

구현하기 전에 **재야** 한다. 값이 없는 상태로 코드를 채우면 그 근사값이 정본이
되고, 실물 데이터가 처음 들어오는 날 전량 재작업이 된다.

1. **ARCore 로그 형식.** 샘플 0건이다. 필드 이름을 추측해 파서를 쓰면 그게
   정본이 된다. 로그 1건을 먼저 받아라
2. **hand-eye `T_phone→pinch`** (이슈 28 · LIMITS L30). ARCore 가 주는 것은
   **폰의 pose** 이고 계약이 필요한 것은 **파지점의 pose** 다. 그 사이 변환이
   미실측이다. **상수 편향이라 학습이 지우지 못한다** — 실측상 ±10mm 에서
   재생 성공률이 3/4 로 떨어진다 🟢. 손측정 근사값으로 본수집하지 마라
3. **`T_world→base`** (이슈 29). ARCore 세션 원점과 로봇 베이스의 관계.
   시연 중에는 로봇이 없으므로 이걸 세우는 절차가 수집 프로토콜의 일부다

그리고 그리퍼 간격 측정 방식이 미정이다 — UMI 리그(이슈 39, HW)가 손가락 간격을
무엇으로 알려주는지 정해지지 않았다. 계약이 요구하는 것은 **미터 단위 실측값**이다.

## 시그니처를 미리 못 박아 둔 이유

미실측 변환을 **필수 인자**로 두면, 없는 상태로는 호출 자체가 안 된다.
기본값을 주면 조용히 통과하고 그 값이 정본이 된다.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from umi.raw import RawEpisode

__all__ = ["to_raw"]


def to_raw(
    *,
    pose_log: Path,
    frames: Path,
    recording_id: str,
    skill_id: str,
    calibration_id: str,
    t_phone_to_pinch: np.ndarray,
    t_world_to_base: np.ndarray,
) -> RawEpisode:
    """Build one `RawEpisode` from an ARCore session. NOT IMPLEMENTED.
    ARCore 세션 하나로 `RawEpisode` 를 만든다. **구현 없음.**

    Args:
        pose_log: ARCore pose 스트림 로그. **형식 미정** — 샘플 1건 필요
        frames: RGB 프레임. 계약은 (3,224,224) 를 요구하지만 raw 는 원해상도
            HWC uint8 로 남긴다. 줄이는 것은 변환기의 일이다
        recording_id: 파일명이 된다. 기존 이름 재사용 금지
        skill_id: `contract.ids.SKILL_IDS` 중 하나. 아니면 검증기가 거부한다
        calibration_id: 어느 캘리브레이션이 `t_*` 를 만들었는가.
            **빈 값이면 `umi.raw.validate_raw` 가 실기록을 거부한다**
        t_phone_to_pinch: (4,4) 폰 → 파지점. 이슈 28 · L30 **미실측**
        t_world_to_base: (4,4) ARCore 세션 원점 → 로봇 베이스. 이슈 29 **미실측**

    Returns:
        `frame="robot_base"` 인 `RawEpisode`. 다른 프레임이면 변환기가 거부한다.

    반드시 지킬 것:
      - `pose_timestamp` 와 `image_timestamp[cam]` 을 **따로** 채운다.
        길이가 달라도 된다. 같게 맞추지 마라 — 그건 동기화를 이미 했다고
        가정하는 것이고, 어긋난 데이터가 검증기를 통과해 실물에서만 실패한다
      - `eef_quat` 의 회전 R 규약: `R[:,2]` = 손가락 방향(접근축),
        `R[:,0]` = 턱이 열리는 방향. `umi/raw.py` 가 정본이다
      - `gripper_gap_m` 은 **미터**다. `configs/so101.yaml` 의 `gap_curve` 는 cm 다
      - 저장은 `umi.raw.write_raw` 로 한다. 스키마 위반이면 저장되지 않는다
    """
    raise NotImplementedError(
        "S15P21A103-31 미구현. 착수 전에 다음이 필요하다:\n"
        "  1) ARCore 로그 샘플 1건 (형식 확인 — 추측 금지)\n"
        "  2) hand-eye T_phone→pinch 실측 (이슈 28, LIMITS L30)\n"
        "  3) T_world→base 실측 절차 (이슈 29)\n"
        "  4) UMI 리그의 손가락 간격 측정 방식 (이슈 39, HW)\n"
        "자세한 것은 track_a/HANDOVER_track_a.md"
    )
