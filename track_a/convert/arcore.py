"""`umi_raw/0.1.0` 번들 → `umi.raw.RawEpisode`. **구현 없음.** S15P21A103-31.
The app's raw bundle to the decoded robot-frame intermediate. NOT IMPLEMENTED.

**이 파일이 트랙 A 가 채워야 하는 유일한 칸이다.** 여기서 `RawEpisode` 를 만들어
주면 아래 전부가 이미 돌아간다 — 변환기·IK·계약 검증기·수용률 리포트.

    track_a/convert/arcore.py   ←  여기만 비어 있다
        ↓  umi.raw.RawEpisode  (디코딩 + 로봇 좌표)
    umi/convert.py              ←  완료. 검사 32건 통과
        ↓  contract.episode.Episode
    data/verify.py              ←  완료. 합성 20편 위반 0

입력 형식은 `docs/SPEC_umi_raw_bundle.md` 가 정본이다 (개정 2판, MW 회신 반영 🔵).
ZIP 무압축 1편: `episode.json` · `poses.csv` · `imu.csv` · `frames.csv` ·
`gripper.csv` · `frames/*.jpg` · `summary.txt`. **jpg 재인코딩 금지.**

## 반드시 변환해야 하는 것 — 전부 조용히 틀리는 종류다

| 번들 (SPEC) | `RawEpisode` | 안 바꾸면 |
|---|---|---|
| `quaternion_order: "xyzw"` | **`wxyz`** | 180도급 회전 오류. 손실은 잘 떨어지고 실물에서만 실패 |
| `pose_is: "T_world_camera"` | 파지점 pose | `T_cam→pinch` 상수 편향. 학습이 지우지 못한다 |
| `arcore_world`, **Y = 중력반대** | `robot_base`, **Z-up** | 축 교환을 빠뜨리면 중력 방향이 틀린다 |
| 원점 = 세션 시작 (에피소드마다 다름) | 로봇 베이스 고정 | 상대궤적만으로는 어디로 갈지 알 수 없다 |
| `gripper.csv` `status` D/M/T/X | `gripper_status` 그대로 + T·X 는 `gap_m` **NaN** | 0 으로 채우면 "완전히 닫혔다"가 되어 학습이 그것을 배운다 |

그리고 그냥 옮기면 되는 것:
`summary.txt` 의 `frames_dropped` · `usable_segments` → `RawMeta` 의 같은 이름 필드.
**초기 3~5초는 ARCore 안정화 전이라 폐기 대상이다** (SPEC). 이미지 없는 pose 행도 뺀다.

## 착수 조건 — SPEC 이 2건을 이미 해결했다

| # | 항목 | 상태 |
|---|---|---|
| 1 | 번들 형식 | **해결.** SPEC 개정 2판이 정본이다 🔵 |
| 2 | 그리퍼 간격 측정 방식 | **해결.** 16mm 마커 스케일, `gap_m` 미터, 0~90mm 실측 🔵 |
| 3 | **hand-eye `T_cam→pinch`** | **미실측.** 이슈 28 · LIMITS. 상수 편향이라 학습이 못 지운다 — ±10mm 에서 재생 성공률 3/4 🟢 |
| 4 | **`T_ARCore월드→로봇베이스`** | **미결정.** SPEC 「막는 것」 절이 방안 A(작업대 ArUco 보드)를 권한다. 트랙 A 소관 |

미실측 두 건을 **필수 인자**로 뒀다. 값이 없으면 호출 자체가 안 된다.
기본값을 주면 조용히 통과하고 그 근사값이 정본이 된다.

## 아직 정해지지 않은 것 (SPEC 미해결 절)

- 연속 X 가 몇 프레임 이상이면 에피소드를 버릴지 — 파일럿 데이터 후 결정
- `gap_m` 의 프레임 위치 의존성 (판정 기준 ±2mm, MW 계측 요청 중)
- 핑거 50% 축소의 시각 도메인 영향 (카메라↔핑거 거리비가 0.5 인가)
- `poses.csv` 의 `tracking` 값 종류 (미회신)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from umi.raw import RawEpisode

__all__ = ["to_raw"]


def to_raw(
    *,
    bundle: Path,
    t_cam_to_pinch: np.ndarray,
    t_arcore_world_to_base: np.ndarray,
    recording_id: str | None = None,
    skill_id: str | None = None,
    calibration_id: str,
) -> RawEpisode:
    """Decode one `umi_raw/0.1.0` bundle into a robot-frame `RawEpisode`. NOT IMPLEMENTED.
    번들 하나를 로봇 좌표 `RawEpisode` 로 디코딩한다. **구현 없음.**

    Args:
        bundle: ZIP 또는 풀어놓은 디렉터리. `episode.json` 의 `schema` 가
            `umi.raw.BUNDLE_SCHEMA_SUPPORTED` 에 없으면 거부한다
        t_cam_to_pinch: (4,4) 카메라 → 파지점. 이슈 28 **미실측**
        t_arcore_world_to_base: (4,4) ARCore 세션 원점 → 로봇 베이스.
            이슈 29 **미결정** — Y-up→Z-up 축 교환이 여기 포함된다
        recording_id: None 이면 `episode.json` 의 `episode_id` 를 쓴다
        skill_id: None 이면 번들 값을 쓴다. 번들이 비어 있으면(이슈 109 미확정)
            호출자가 줘야 한다 — `contract.ids.SKILL_IDS` 밖이면 검증기가 거부한다
        calibration_id: `t_*` 를 만든 캘리브레이션 식별자.
            **빈 값이면 `umi.raw.validate_raw` 가 실기록을 거부한다**

    Returns:
        `frame="robot_base"`, `bundle_schema="umi_raw/0.1.0"` 인 `RawEpisode`.

    저장은 `umi.raw.write_raw` 로 한다. 스키마 위반이면 저장되지 않는다.
    """
    raise NotImplementedError(
        "S15P21A103-31 미구현. 착수 전에 남은 것:\n"
        "  1) hand-eye T_cam→pinch 실측 (이슈 28) — 상수 편향이라 학습이 못 지운다\n"
        "  2) T_ARCore월드→로봇베이스 방안 확정 (이슈 29, SPEC 「막는 것」 절 A/B)\n"
        "번들 형식과 gap 측정 방식은 SPEC 개정 2판에서 해결됐다.\n"
        "자세한 것은 track_a/HANDOVER_track_a.md 와 docs/SPEC_umi_raw_bundle.md"
    )
