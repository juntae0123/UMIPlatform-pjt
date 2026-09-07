# SPEC · UMI raw 번들 (앱 → 서버)

작성: 김준태(AI 트랙B) · 2026-09-07 (같은 날 MW 회신 반영해 개정) ·
수신: 신현우(MW), 황도경(AI 트랙A)
상태: **개정 1판 — 트랙 A 확인 대기.** 확정되면 D-AI-{n} 기록
확신도: 🟢 실측 / 🔵 MW 회신·문서 / 🟡 설계 판단

## 이게 무엇이고 무엇이 아닌가

**raw 다. 학습 형식이 아니다.** 학습 계약은 `AI/contract/episode.py`
(`0.2.0-provisional`, 관절각 · 30Hz · [-1,1]) 이고, raw→계약 변환은 AI 파트가 한다.
이 번들의 목적은 **무손실 보존**이다. 게이트 임계값은 파일럿 이후 반드시 바뀌고,
raw 를 버리면 재처리가 불가능해진다.

로더 상태: `AI/track_a/convert/` 는 아직 비어 있다. **로더보다 이 스펙이 먼저다.**

⚠️ **OIS 기본값이 ON 이었으므로 이전 녹화분은 전부 무효다** 🔵. 새 빌드로 재촬영한다.
카메라가 그리퍼에 강체 고정이라는 전제가 UMI 가 캘리브레이션 없이 성립하는 이유이고,
손떨림 보정이 그 전제를 깬다. 이 항목을 요청 목록에 넣은 것이 값을 했다.

## 파일 구성

```
episode.json      메타 (아래 §필드)
poses.csv         index,timestamp_ns,tracking,x,y,z,qx,qy,qz,qw     ← image 컬럼 제거됨
imu.csv           timestamp_ns,sensor,x,y,z
frames.csv        frame_index,timestamp_ns,image
gripper.csv       frame_index,gap_m,status                          ← gap_px → gap_m
frames/*.jpg
```

ZIP 무압축 1편 단위. jpg 재인코딩 금지.

## episode.json 필드

```json
{
  "schema": "umi_raw/0.1.0",
  "episode_id": "...",
  "skill_id": "...",
  "collected_by": "...",
  "started_at_utc": "2026-09-07T12:38:00.000Z",
  "device": { "model": "SM-S911N", "os": "...", "app_version": "..." },
  "clock": { "source": "elapsedRealtimeNanos", "all_streams_same_clock": true,
             "first_frame_offset_ms": -57, "imu_skew_ms": 7 },
  "camera": {
    "which": "main", "hfov_deg": 74,
    "width": 640, "height": 480, "nominal_hz": 30,
    "focus_mode": "fixed", "ae": "locked", "awb": "locked",
    "ois": "off", "vdis": "off",
    "intrinsics": { "fx": 0, "fy": 0, "cx": 0, "cy": 0 },
    "distortion": null
  },
  "pose": {
    "provider": "ARCore", "pose_is": "T_world_camera",
    "frame": "arcore_world", "handedness": "right",
    "units": "m", "quaternion_order": "xyzw"
  },
  "gripper": { "method": "marker_scale", "marker_mm": 16,
               "calib_closed_mm": 0, "calib_open_mm": 90 }
}
```

## 요청 4건 — 전부 해결 (2026-09-07 MW 회신) 🔵

| # | 항목 | 회신 |
|---|---|---|
| 1 | `clock.source` + 세 스트림 동일 시계 | **해결.** pose·frame·IMU·녹화시작 전부 `elapsedRealtimeNanos` 기준 실측 확인. `all_streams_same_clock=true` 기록. 첫 프레임 −47~−68ms, IMU skew 약 7ms |
| 2 | `skill_id` | **필드만 넣고 값은 비움.** 이슈 109 확정 후 채운다 |
| 3 | `ois: off`, `vdis: off`, AE/AF lock | **해결.** ARCore 기본으로는 제어 불가라 `SharedCamera` 로 카메라를 직접 잡았다. 실측: OIS off · EIS off · AF fixed · AE/AWB lock. **OIS 기본값이 ON 이었다 → 이전 녹화분 전부 무효, 새 빌드로 재촬영** |
| 4 | 해상도 + 왜곡계수 | 해상도 **640×480** + intrinsics 제공. **왜곡계수는 ARCore 가 주지 않는다.** 그리고 초광각이 아니라 **메인 카메라(수평 화각 약 74°)** 다 |

## 왜곡계수 — 답: **무시한다.** 단 `gap_m` 만 예외

쓰이는 곳이 세 군데고 답이 다르다.

| 쓰임 | 왜곡계수 |
|---|---|
| **pose** (`T_world_camera`) | **불필요.** ARCore 가 내부 VIO 로 처리한 pose 를 준다. 우리는 픽셀이 아니라 pose 를 소비한다 |
| **정책 입력 이미지** | **불필요하고, 보정하면 안 된다.** 정책은 카메라가 내놓는 그대로를 학습한다. 중요한 것은 정확성이 아니라 **시연과 배포의 일치**다. 로봇 손목 카메라가 같은 왜곡을 갖지 않는데 시연만 rectify 하면 도메인이 벌어진다 🟡 |
| **`gap_m`** | **여기만 문제가 될 수 있다.** 16mm 마커를 프레임별 스케일 기준으로 쓰는 것은 이미지 공간 기하이고, 반경 방향 왜곡은 화면 위치에 따라 겉보기 크기를 바꾼다 |

체커보드 캘리브는 지금 필요 없다. **`gap_m` 의 위치 의존성만 계측**한다 (아래 미해결 1).

## 고쳐달라 한 것 2건 — 전부 반영 🔵

**`poses.csv` 의 `image` 컬럼 제거됨.** 그리고 MW 설명: **pose 와 이미지가 같은
ARCore 프레임에서 나와 `timestamp_ns` 가 동일하다** → join 이 정확하고
pose↔frame 시차는 **구조적으로 0** 이다. 남는 동기화 리스크는 IMU↔pose (약 7ms).

⚠️ 이 설명으로 L29 를 내리려면 근거가 필요하다 → 미해결 3.

**`gap_px` → `gap_m` 으로 교체됨.** 핑거 팁 폭을 미터로 준다. 16mm 마커를
프레임별 스케일 기준으로 써서 거리·각도 변화를 자동 보정한다 (MW 실측 검증).
완전폐쇄 0mm / 완전개방 90mm 캘리브레이션. **프레임마다 스케일이 다르므로
`px_to_m` 단일값은 없다** — 이 설계가 더 낫다.

`status` 값: **D**(양쪽 직접검출) / **M**(중심선 대칭 추정) / **T**(템플릿 보강) /
**X**(실패). `gap_m` 은 D·M 프레임(약 97%)에만 있다.

**AI 파트 처리 규칙:** X 프레임의 `gap_m` 은 **채우지 않고 결측으로 두고 학습에서
제외**한다. 연속 X 가 몇 프레임 이상이면 에피소드를 버릴지는 파일럿 데이터를 보고
정해 이 문서에 추가한다 (미정).

## 미해결 — 계측이 필요한 것

### 1. `gap_m` 의 프레임 위치 의존성 (MW 계측 요청)

그리퍼를 **고정 개구**로 잡고 화면 중앙 → 네 모서리로 옮기며 녹화. 개구 두 지점
(20mm, 60mm), **작업거리 20/30/40cm** 각각.

**판정 기준 (결과 보기 전 확정): 위치에 따른 `gap_m` 편차 ±2mm 이내.**
근거 — 대상물이 1.5~2.5cm 이고, 별도 계측에서 그리퍼 표현의 6.53mm 오차가 게이트와
같은 자릿수여서 못 쓴다고 판정했다 (D-AI-11 초안) 🟢. 넘으면 체커보드 캘리브로 간다.

### 2. 해상도 대비 마커 크기 🟡

640px / 수평 74° → 약 0.116°/px. 작업거리 300mm 에서 **1px ≈ 0.6mm**.
16mm 마커는 약 **26px**. 변 검출이 ±1px 이면 스케일 오차 약 4% → 90mm 게이지에서
**±3.5mm**.

이 어림이 맞다면 위 ±2mm 기준을 **왜곡과 무관하게** 못 넘는다. 미해결 1 의 스윕을
실제 작업거리에서 재면 함께 판정된다. 처방 후보: 마커 확대(25mm) 또는 해상도 상향.

### 3. `poses.csv` ↔ `frames.csv` 타임스탬프 집합 일치 (MW 확인 요청)

한 에피소드 전체에서 두 `timestamp_ns` 집합이 **정확히 같은지** 한 번 확인.
같으면 join 이 1:1 이고 **L29 를 해소**할 수 있다.

### 4. 첫 프레임 −47~−68ms 의 성질 (MW 확인 요청)

**상수 편향인가 에피소드별 지터인가.** 상수면 빼면 되고, 지터면 계측 대상이다.

### 5. 개구 범위 불일치 (MW 확인 요청)

MW 캘리브 **0~90mm** vs 저희 SO-101 공식 MJCF 실측 **팁 간격 4.1~133.4mm** 🟢.
핸드헬드가 SO-101 핑거 STL 을 그대로 쓰는 설계였다면 같아야 한다.

셋 중 무엇인가 — (1) 핑거 형상이 다르다 (2) 90mm 는 물리 최대가 아니라 캘리브
상한이다 (3) 측정 지점이 팁이 아니다.
**범위가 다르면 `gap_m` 을 로봇 명령으로 바로 못 쓰고 매핑이 필요하다.**
계약의 그리퍼 표현은 핑거 팁 간격 [m] 이다 (D-AI-11 초안, 트랙 A 확인 대기).

### 6. `poses.csv` 의 `tracking` 값 종류

정상/유실 등 enum. 품질 게이트에 그대로 쓴다. 미회신.

## 이 스펙이 바뀌면

수집한 데이터가 전부 무효가 될 수 있다. 변경 제안은 **양 트랙 합의 + D- 기록**이
필요하다 (`claude/규칙_동시대화_소유권.md`).
