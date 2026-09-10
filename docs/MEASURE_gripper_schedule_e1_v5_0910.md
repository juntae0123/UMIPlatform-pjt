# MEASURE — E1 원본 v5 + fixed gripper schedule (2026-09-10)

- 확신도: 🟢 실행·로그 확인
- 사전등록: `docs/PREREG_gripper_schedule_e1_v5_0910.md` (축소판 — seed0 만)
- 작성자: 김준태(트랙B) · 이슈 S15P21A103-34

## 조건

```
checkpoint       sim_pick_v5_seed0.pt
learned fixture  --expected 11  → PASS (오늘 트리 eval_rollout 2회 확인값)
n                100/조건 · evaluation seeds 3000~3099
render           true · policy-device cpu · jitter ±50mm · max_ticks 200
실행             단일 잡 · GPU 2 · AI_THREADS 2 · MUJOCO_GL egl
```

## 결과

| 조건 | 성공률 | learned 대비 | 평균 최대 상승 | close tick |
|---|---:|---:|---:|---:|
| learned | 11/100 = 11.0% | — | 0.88cm | — |
| fixed60 | 18/100 = 18.0% | +7%p | 1.93cm | 60 |
| fixed67 | 35/100 = 35.0% | **+24%p** | 3.67cm | 67 |
| fixed75 | 36/100 = 36.0% | **+25%p** | 3.76cm | 75 |
| fixed85 | 37/100 = 37.0% | **+26%p** | 3.82cm | 85 |
| oracle | 3/100 = 3.0% | −8%p | 0.33cm | 67 |

## 판정

사전등록 기준(성공률 > 20% **그리고** learned 대비 >= 20%p): **fixed67·fixed75·fixed85 통과.**

⚠️ **도구가 출력한 `verdict=gripper_only_not_sufficient` 는 무효다.**
`probe_gripper_schedule.py:315-319` 의 `count >= 2` 가 **체크포인트 개수**를 세는데
임계값이 상수 2로 하드코딩돼 있다. 체크포인트 1개로 실행하면 최대값이 1이므로
어떤 결과가 나와도 `gripper_only_not_sufficient` 가 출력된다.
n에 따라 스케일하지 않는 계측기 결함이다 → LIMITS 등재 대상.

## 이 실험이 답한 것

**v5 팔 + 완벽한 그리퍼 = 37% 가 상한이다.**

DAgger 계열과 나란히 놓으면 (트리 `5fbbcdf`, 같은 시드 블록):

| 체크포인트 | learned | fixed 최고 |
|---|---:|---:|
| sim_pick_v5_seed0 | 11% | **37%** |
| dagger seed0 | 1% | **86%** |
| dagger seed1 | 0% | 49% |
| dagger seed2 | 7% | 44% |

→ **DAgger 팔 보정이 실제로 기여했다.** v5 팔로는 그리퍼를 완벽하게 만들어도
37% 가 천장이고, DAgger 팔은 같은 개입으로 86% 까지 간다. 차이 49%p 가 팔 정밀도의 몫이다.

0909 시점의 미해결 질문("fixed schedule 의 성공이 DAgger 덕분인가 그리퍼 덕분인가")에
대한 답: **둘 다이고, 팔 쪽 몫이 더 크다.**

## oracle

v5 에서도 oracle(3%)이 learned(11%)보다 낮다. 특권정보를 쓰는 조건이 아무것도 안 쓰는
조건보다 나쁘다 — 0909 DAgger 에서 관측된 것과 같은 고장이 재현됐다.
원인 추정 🟡: `grasp_tol_m = 0.005` 가 만족되지 않아 닫는 시점이 밀린다.
**미해결.** E2(tolerance 스윕)를 3일 마감 때문에 폐기했으므로 LIMITS 로 넘긴다.

## 해석 제한

- 체크포인트 1개다. 학습 실행 간 성공률이 25%p 갈린 실측이 있으므로 🟢
  이 37% 는 v5 계열 전체의 상한이 아니라 seed0 의 상한이다
- fixed tick 은 실물 UMI 시연에 적용할 수 없다 (시연마다 길이가 다르다).
  배포 처방이 아니라 진단이다
- n=100 의 95% 구간 반폭 약 ±9%p. fixed67·75·85 (35/36/37%) 는 서로 구분되지 않는다
