# SPEC · UMI raw 번들 (앱 → 서버)

작성: 김준태(AI 트랙B) · 2026-09-07 · 수신: 신현우(MW), 황도경(AI 트랙A)
상태: **초안 — 트랙 A 확인 대기.** 확정되면 D-AI-{n} 기록
확신도: 🟢 실측 / 🔵 문서 / 🟡 설계 판단

## 이게 무엇이고 무엇이 아닌가

**raw 다. 학습 형식이 아니다.** 학습 계약은 `AI/contract/episode.py`
(`0.2.0-provisional`, 관절각 · 30Hz · [-1,1]) 이고, raw→계약 변환은 AI 파트가 한다.
이 번들의 목적은 **무손실 보존**이다. 게이트 임계값은 파일럿 이후 반드시 바뀌고,
raw 를 버리면 재처리가 불가능해진다.

로더 상태: `AI/track_a/convert/` 는 아직 비어 있다. **로더보다 이 스펙이 먼저다.**

## 파일 구성

```
episode.json      메타 (아래 §필드)
poses.csv         index,timestamp_ns,tracking,x,y,z,qx,qy,qz,qw
imu.csv           timestamp_ns,sensor,x,y,z
frames.csv        frame_index,timestamp_ns,image
gripper.csv       frame_index,gap_px,status
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
  "clock": { "source": "elapsedRealtimeNanos", "all_streams_same_clock": true },
  "camera": {
    "width": 1280, "height": 720, "nominal_hz": 30,
    "focus_mode": "locked", "ae": "locked", "awb": "locked",
    "ois": "off", "vdis": "off",
    "intrinsics": { "fx": 0, "fy": 0, "cx": 0, "cy": 0 },
    "distortion": { "model": "...", "coeffs": [] }
  },
  "pose": {
    "provider": "ARCore", "pose_is": "T_world_camera",
    "frame": "arcore_world", "handedness": "right",
    "units": "m", "quaternion_order": "xyzw"
  },
  "gripper": { "method": "...", "px_to_m": null }
}
```

## 빠지면 재수집이 되는 것 — 넣는 비용은 거의 0

| # | 항목 | 없으면 |
|---|---|---|
| 1 | `clock.source` + 세 스트림 동일 시계 여부 | pose 와 RGB 의 타임스탬프 기준이 다르면 이미지↔포즈 오차를 **계측 자체가 불가능**하다. 이슈 30. 이 번들의 최대 리스크 |
| 2 | `skill_id` | 어느 스킬 시연인지 없으면 학습 데이터가 아니다. 값 목록은 `contract/ids.py` `SKILL_IDS`, 확정은 이슈 109 대기 — **필드만 먼저** 넣는다 |
| 3 | `ois: off`, `vdis: off`, AE/AF lock | 손떨림 보정이 켜져 있으면 카메라가 그리퍼에 강체 고정이라는 전제가 깨진다. UMI 가 캘리브레이션 없이 성립하는 이유가 그 전제다 🔵 |
| 4 | 해상도 + 왜곡계수 | fx/fy/cx/cy 만으로는 역투영이 안 된다. S23 초광각은 왜곡이 크다 |

## 고쳐야 할 것 두 개

**`poses.csv` 의 `image` 컬럼을 뺀다.**
앱이 pose 를 프레임에 미리 붙이면 두 스트림의 시차가 그 순간 사라진다. 우리가 재야 할
오차를 앱이 지우는 것이다. pose 와 frame 은 각자 `timestamp_ns` 만 갖고 따로 온다.
결합은 AI 파트가 한다.

**`gripper.csv` 의 `gap_px` 만으로는 못 쓴다.**
픽셀은 거리·각도에 따라 변한다. 둘 중 하나가 필요하다.
- 개폐 캘리브레이션 녹화 1회분(완전 개방 / 완전 폐쇄) — Stanford UMI 방식 🔵
- 또는 `gripper.px_to_m` 변환을 episode.json 에 기입

계약의 그리퍼 표현은 **핑거 팁 간격 [m]** 이다 (D-AI-11 초안, 트랙 A 확인 대기).
SO-101 실측 닫힘 간격 1.7cm 🟢 — 원조 UMI 평행 80mm 스트로크와 스케일이 4배 다르므로
`gap_px` 를 그대로 원조 코드에 넣을 수 없다.

## 값 종류를 알려달라

- `poses.csv` 의 `tracking` — 정상/유실 등 enum. 품질 게이트에 그대로 쓴다
- `gripper.csv` 의 `status`

## 이 스펙이 바뀌면

수집한 데이터가 전부 무효가 될 수 있다. 변경 제안은 **양 트랙 합의 + D- 기록**이
필요하다 (`claude/규칙_동시대화_소유권.md`).
