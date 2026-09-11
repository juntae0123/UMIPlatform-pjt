# MEASURE — ROS2 실행 계층 인계본 판독 (2026-09-11)

- 확신도: 🟢 원문 코드·설정 확인 (`so101_motor_control` ROS2 패키지)
- 판독자: 김준태(트랙B) · 이슈 S15P21A103-34, S15P21A103-42
- 대상: `motor_control/` (파일 33개 · 18MB · 대부분 STL)

핵심 파일: `config/controllers.yaml` · `BACKEND_INTERFACE.md` ·
`src/so101_system.cpp` · `src/st3215_bus.cpp` · `urdf/so101.ros2_control.xacro` ·
`urdf/so101_ver1.urdf`

---

## 1. ROS2 경로의 실제 거부 조건 — 원문 🟢

`src/so101_system.cpp`:

```cpp
// :254  팔 관절
if (!std::isfinite(radians) || radians < joint.min_rad || radians > joint.max_rad)
    throw std::out_of_range(joint.name + ": angle outside configured limits");
// :267, :290  tick 변환 후
if (tick > 4095) throw ...
// :277  그리퍼
if (!std::isfinite(width_m) || width_m < 0.0 || width_m > gripper_.max_width_m) throw ...
// :243  통과 시
bus_->write_positions(targets, speed_, acceleration_);
```

**거부 조건은 셋뿐이다: 비유한값 · URDF 위치 한계 밖 · tick 0..4095 밖.**
`never wraps or clamps` — 하나라도 걸리면 **쓰기 주기 전체가 거부**된다.

### ⚠️ 속도·가속 검사가 없다

`5.774 * dq / dt**2 <= max_accel` 은 **ROS2 경로에 존재하지 않는다.**
속도·가속은 ST3215 서보 펌웨어 파라미터로 넘어간다 — xacro 의
`speed=300`, `acceleration=10` (허용 범위 `speed 0..32767`, `acceleration 0..150`).
**검사기가 거부하는 게 아니라 서보가 그 속도로 갈 뿐이다.**

→ **L67 의 적용 범위를 좁혀야 한다.** `5.774` 가속 검사는 direct-motor 파이썬
파이프라인(`core/models.py`) 전용이고 ROS2 경로에는 안 걸린다. 두 경로는 **다른 검사식**을
쓰므로 S1 은 반드시 분리해서 판정해야 한다.

---

## 2. ⚠️ 관절 범위 불일치 — elbow_flex 하한

| 출처 | elbow_flex 하한 | 상한 |
|---|---:|---:|
| ROS2 xacro `min_rad`/`max_rad` · URDF `<limit>` | **−1.57079632679** | 1.69 |
| `AI/configs/so101.yaml` · `AI/configs/real/so101_ver1.json` | **−1.69** | 1.69 |

**0.119 rad = 6.8° 차이.** 우리 설정이 더 넓다. 즉 **시뮬이 만든 궤적이 elbow_flex
−1.5708 아래로 내려가면 ROS2 가 쓰기 주기 전체를 거부한다.**

실제 궤적이 그 구간을 얼마나 쓰는지 **미측정** — S1 이 제일 먼저 재야 할 숫자다.
`AI/configs/so101.yaml` 을 고치기 전에 HW 확인 필요 (URDF 가 정본인지, 우리 값이
어디서 왔는지).

나머지 관절은 소수점까지 일치한다 (pan ±1.91986 · lift ±1.74533 · wrist_flex ±1.65806 ·
wrist_roll −2.743847297~2.841206309).

---

## 3. 그리퍼 — 대응표가 필요 없다 🟢

```
Topic  /gripper_controller/commands   std_msgs/Float64MultiArray
data[0] = full gripper opening width in metres, 0.0 (closed) ~ 0.09 (open)
tick    1720 closed / 70 open
```

**백엔드가 SI 미터를 직접 받는다.** `gripper` 는 virtual full-width joint 이고
`gripper_right`·`gripper_mirror` 가 각각 0.5 배로 mimic 한다 — 외부에서 2로 나눌 필요 없다.

→ **"실물 slide 의 `gap_m` 대응표" 요청은 불필요하다.** 남는 변환은 하나뿐이다:
**시뮬 hinge 각도 → `gap_m`** (`AI/sim/mujoco/gripper.py` 의 `gap_curve`).
실물 쪽 변환은 백엔드가 한다.

URDF 그리퍼 한계: `<limit effort="30" velocity=".08" lower="0" upper=".09"/>`
→ `so101_ver1.json` 의 `max_gap_m 0.09` · `max_gap_speed_m_s 0.08` 과 일치 🟢

---

## 4. 닫힌 항목 (ASK_hw_addendum_0908 대조)

| ASK 항목 | 상태 | 근거 |
|---|---|---|
| **§1-1 direction 부호** (1순위) | ✅ **닫힘** | J1~J5 전부 `+1`, 물리 인코더 방향 검증표가 `BACKEND_INTERFACE.md` 에 있다 |
| **§1-2 shoulder_lift 영점** | ✅ **닫힘** | `zero_tick = 2048` (2200 아님) |
| **§3 그리퍼 gap** | ✅ **닫힘** | 백엔드가 미터를 받는다. 대응표 불필요 |
| §2 속도·가속 출처 | 🔶 **절반** | URDF `velocity`: 팔 4축 **10**, wrist_roll **1**, gripper **0.08**. 우리가 쓰던 `max_speed_rad_s 1.0` 은 **가장 느린 관절(wrist_roll) 기준 보수값**으로 보인다 🟡 — 측정값인지는 여전히 미확인 |
| §4 카메라 | ❌ **열림** | 이 패키지에도 카메라가 없다 |

## 5. 실행 인터페이스 요약 🟢

```
controller_manager update_rate: 30            ← 계약 30Hz 와 일치
arm_controller   joint_trajectory_controller/JointTrajectoryController
                 command/state interfaces: position 만
                 allow_partial_joints_goal: false
                 관절 순서 shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll
  Action  /arm_controller/follow_joint_trajectory  control_msgs/FollowJointTrajectory
  Topic   /arm_controller/joint_trajectory         trajectory_msgs/JointTrajectory (스트리밍)
gripper_controller  forward_command_controller/ForwardCommandController
  Topic   /gripper_controller/commands             std_msgs/Float64MultiArray
포트 /dev/ttyTHS1 (Jetson UART) · baudrate 1000000 · ST3215 Sync Write 6축 동시
```

## 6. 아직 없는 것

- `core/models.py` 의 `validate_trajectory` 원문 (direct-motor 파이썬 경로) — 이 패키지에 없다
- 실물 로그 (command/measured 관절각, controller state, reject code, 송신 주기)
- `so101_ver1.json` 의 상류 커밋 해시

## 7. S1 에 주는 영향 — 판정식을 두 갈래로 나눈다

| 경로 | 검사 | 출처 |
|---|---|---|
| **ROS2 연속 trajectory** | ① 비유한값 ② URDF 위치 한계 ③ tick 0..4095. **속도·가속 검사 없음** | 원문 🟢 |
| **direct-motor 정지→정지 waypoint** | `0.05<=dt<=60` + `5.774*dq/dt²<=3.0` | 전사 🔵 (원문 미보유) |

**S1 의 첫 숫자는 "elbow_flex 하한 위반 프레임 비율" 이다.** 그것 하나로 ROS2 경로
거부 여부가 갈린다. 가속 초과배수(25.8x)는 ROS2 경로에는 적용되지 않는다.
