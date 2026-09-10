# MEASURE — UMI raw 번들 1편 수령 검사 (2026-09-10)

- 확신도: 🟢 파일 실측 · 프레임 육안 확인
- 검사자: 김준태(트랙B) · 이슈 S15P21A103-27, S15P21A103-30
- 대상: `rec_20260910_172751.zip` (146파일 · 10MB)
- 대조 기준: `docs/SPEC_umi_raw_bundle.md` (2판)

⚠️ 번들 생성은 MW·트랙 A 영역이다. 이 문서는 **계약 대조 결과만** 적는다.
   상대 트랙 작업을 추정한 서술은 넣지 않았다.

## 한 줄 판정

**실물 데이터이고 앱 파이프라인은 SPEC 요구를 상당 부분 만족한다. 그러나 시연이
아니므로 학습에 쓸 수 없다.**

## 통과 🟢

| 항목 | 실측 | SPEC 대조 |
|---|---|---|
| 카메라 모드 | `ois=0 eis=0 af_mode=0 ae_lock=true awb_lock=true` | 요청 3번 반영. 카메라-그리퍼 강체 고정 전제 유지 |
| pose↔frame | 141행 / 141장, 누락 0, 타임스탬프 유니크·단조증가 | 집합 일치. L29 해소 근거 재확인 |
| 제어 주기 | 30.22Hz · 간격 중앙 33.33ms (min 33.04 / max 33.57) | 30Hz 계약 만족, 지터 ±0.3ms |
| 시계 | 세 스트림 `elapsedRealtimeNanos` · IMU skew 7.34ms | 회신 "약 7ms" 와 일치 |
| tracking | 전 구간 `TRACKING` · `tracking_jumps=0` · `max_jump_mm=0.0` | |
| intrinsics | fx 431.227 fy 430.286 cx 318.966 cy 241.032 → hfov 73.2° / vfov 58.3° | 회신 "메인 약 74°" 와 일치 |
| IMU | accel 1004샘플 212.8Hz · gyro 1003샘플 212.7Hz | 요청 200Hz 초과 |
| pose 표현 | `T_world_camera` · xyzw · m · up=Y | SPEC 일치 |

## 막는 것

### 1. 시연이 아니다 🟢 (프레임 육안 확인)

`frames/000000.jpg`, `frames/000070.jpg` 를 열어 확인했다. 손에 폰을 들고 사무실을
훑은 영상이고 손가락이 렌즈를 일부 가린다. **UMI 그리퍼·16mm 마커·조작 대상 물체가
모두 없다.** 앱 동작 확인용 녹화로 보인다.

pose 궤적도 조작 궤적의 모양이 아니다: 이동 범위 x 96mm · y 69mm · z 93mm,
총 경로 368mm 인데 첫↔끝 직선거리 27mm — 제자리에서 흔든 궤적이다.

### 2. `gripper.csv` 가 없다 — 트랙 B 에는 이게 치명적이다

`episode.json` 이 스스로 `"gripper_width_source": "postprocess: tools/gripper_gap.py"`
라고 적는다. SPEC 이 요구한 `gripper.csv (frame_index, gap_m, status)` 가 번들에 없다.

`gap_m` 이 없으면 **그리퍼 개폐 라벨을 만들 수 없다.** 2026-09-10 실측 🟢 으로
정책의 병목이 정확히 "언제 닫는가" 임이 확인됐다 (v5 seed0 이 100편 중 42편에서
한 번도 닫지 않았고, fixed schedule 로 폐쇄 시각만 강제하자 11% → 37%).
이 신호가 빠진 에피소드는 정책 학습에 투입할 수 없다.

### 3. 길이 4.67초 — 두 판정이 충돌한다

`summary.txt` 는 이렇게 보고한다:

```
duration_sec=4.67
verdict=OK - 전체 구간 사용 가능
usable_segments=0.00~4.67
```

그런데 SPEC 2판의 MW 2차 회신에는 **"초기 3~5초는 ARCore 안정화 전이라 폐기"** 가
있다. 그 규칙을 적용하면 남는 구간이 0~1.7초이고, 사실상 쓸 구간이 없다.

**`usable_segments` 의 "usable" 은 tracking jump·PAUSED 기준이고 ARCore 안정화
기준이 아니다.** 좁은 것을 재고 넓게 보고한다. 이대로면 안정화 전 구간이 조용히
학습에 들어간다 — 계측기가 스스로 고장을 알리지 않는 전형이다.

조치안: `usable_segments` 에서 안정화 폐기 구간을 빼고 계산하거나, 필드 이름을
`tracking_ok_segments` 로 좁힌다. 어느 쪽이든 **MW·트랙 A 확인 필요.**

## 부수 차이 (무해하나 로더 착수 전에 정해야 한다)

| 항목 | 번들 | SPEC 2판 |
|---|---|---|
| schema | `arpose.episode/1` | `umi_raw/0.1.0` |
| `poses.csv` | `image` 컬럼 있음 (+ `jump_mm`, `jump_deg`) | `image` 컬럼 제거 |
| `frames.csv` | 없음 | 있어야 함 |
| ZIP | 일부 DEFLATE | 무압축 |

`jump_mm`·`jump_deg` 는 SPEC 에 없던 추가 진단 컬럼이고 유용하다 — SPEC 에 넣는 것을
제안한다.

## 계약 관점 미해결 (기존 LIMITS 그대로)

- **카메라 1대.** 계약은 2대(`cam_front`, `cam_wrist`) → **L62**
- **vfov 58.3° vs 시뮬 `cam_wrist` 70°, 차이 11.7°.** 기하가 다른 두 데이터셋이
  같은 이름으로 계약을 통과한다 → **L61**
- **pose 원점이 에피소드마다 다르다** → SPEC 「막는 것」(관절각 변환 사슬) 미해결

## 다음에 요청할 것 (구체적으로)

1. **실제 시연 1편** — UMI 그리퍼 장착 · 16mm 마커 부착 · 1.5~2.5cm 물체 집기.
   ARCore 안정화 이후 **유효 구간 20초 이상**
2. **`gripper.csv` 포함** (`frame_index, gap_m, status`). `status=X` 프레임은
   `gap_m` 을 비워 둔다 — 채우지 않는다 (SPEC 처리 규칙)
3. **`usable_segments` 의 정의 확정** — 안정화 폐기 구간 포함 여부

이 3건이 오면 로더(`AI/track_a/convert/`) 착수 조건이 된다. 그 전에는 단계식 수집
규칙(20편 → 검증 → 프로토콜 확정)의 첫 단계에도 들어가지 않는다.
