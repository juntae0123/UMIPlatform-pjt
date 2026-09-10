# PREREG — E1 원본 v5 + fixed gripper schedule (2026-09-10)

작성자: 김준태(트랙B) · 이슈 S15P21A103-34

## 질문

fixed schedule 의 성공이 DAgger arm correction 덕분인가, 아니면 원래 v5 정책도
팔은 충분히 학습했고 gripper 만 못 배운 것인가.

## 조건

- checkpoints: `sim_pick_v5_seed0/1/2.pt`
- 공식 재현값(EXP_LOG/FINDINGS 2026-09-07 02:13, repeat_runs per_run): 14 / 12 / 2 (각 100편)
- n = 100 / 조건 / checkpoint
- evaluation seeds 3000~3099
- render=True · policy-device=cpu · jitter ±50mm
- 조건: learned, fixed60, fixed67, fixed75, fixed85, oracle
- 단일 잡. 다른 GPU 실험과 병렬 실행하지 않는다 (LIMITS L51)
- 원본 `tools/probe_gripper_schedule.py` 를 직접 사용한다. 소스 복제 금지
- 실행 tool SHA 와 git revision 을 CONDITIONS.txt 에 기록한다

## 사전등록 판정

- 같은 fixed tick 이 3개 checkpoint 중 2개 이상에서 성공률 > 20% 이고
  learned 대비 >= 20%p 개선하면 fixed schedule 효과 재현으로 본다.
- v5 fixed67 이 DAgger fixed67(83/42/43) 과 같은 수준이면
  → DAgger arm correction 기여는 작다.
- v5 fixed67 이 현저히 낮으면 → DAgger arm correction 이 별도로 기여했다.
- 두 조건은 정식 5-run A/B 가 아니므로 유의성 결론은 내리지 않는다.
- oracle 은 계측기 수정(E2) 전 참고값이다. fixed 대비 우열 판정에 쓰지 않는다.

## 해석 제한

- fixed67 은 scripted 전문가의 close 위상 시작 tick 66 과 사실상 같다
  (approach 36 + descend 30). DAgger seed0 fixed75 86% 는 scripted 전체 84% 와 같은 수준이다.
- 따라서 이 실험은 "시계가 좋은 컨트롤러다" 가 아니라
  "정책이 팔은 배웠고 gripper 는 못 배웠다" 를 가르는 **진단**이다.
- 고정 tick 은 시연마다 길이가 다른 실물 UMI 시연에 적용할 수 없다.
  배포 가능한 처방이 아니다.
