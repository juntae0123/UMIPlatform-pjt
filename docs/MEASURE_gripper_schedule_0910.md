# MEASURE — DAgger gripper schedule probe (2026-09-10)

- 확신도: 🟢 실행·로그 확인
- 원본 로그: `out/gripper_schedule_20260910_112251.log`
- 조건: n=100/조건/checkpoint, seed 3000~3099, render=True, policy CPU

| checkpoint | 조건 | 성공률 | 평균 최대 상승 | 중앙 close tick |
|---|---|---:|---:|---:|
| dagger_merged_20260910_092947_dagger_segments_0909_seed0 | learned | 1/100 = 1.0% | 1.05cm | - |
| dagger_merged_20260910_092947_dagger_segments_0909_seed0 | fixed60 | 81/100 = 81.0% | 6.60cm | 60 |
| dagger_merged_20260910_092947_dagger_segments_0909_seed0 | fixed67 | 83/100 = 83.0% | 6.79cm | 67 |
| dagger_merged_20260910_092947_dagger_segments_0909_seed0 | fixed75 | 86/100 = 86.0% | 6.75cm | 75 |
| dagger_merged_20260910_092947_dagger_segments_0909_seed0 | fixed85 | 78/100 = 78.0% | 6.06cm | 85 |
| dagger_merged_20260910_092947_dagger_segments_0909_seed0 | oracle | 48/100 = 48.0% | 3.98cm | 70 |
| dagger_merged_20260910_092947_dagger_segments_0909_seed1 | learned | 0/100 = 0.0% | 0.43cm | - |
| dagger_merged_20260910_092947_dagger_segments_0909_seed1 | fixed60 | 49/100 = 49.0% | 4.47cm | 60 |
| dagger_merged_20260910_092947_dagger_segments_0909_seed1 | fixed67 | 42/100 = 42.0% | 3.95cm | 67 |
| dagger_merged_20260910_092947_dagger_segments_0909_seed1 | fixed75 | 33/100 = 33.0% | 3.29cm | 75 |
| dagger_merged_20260910_092947_dagger_segments_0909_seed1 | fixed85 | 36/100 = 36.0% | 3.43cm | 85 |
| dagger_merged_20260910_092947_dagger_segments_0909_seed1 | oracle | 21/100 = 21.0% | 1.71cm | 92 |
| dagger_merged_20260910_092947_dagger_segments_0909_seed2 | learned | 7/100 = 7.0% | 1.27cm | - |
| dagger_merged_20260910_092947_dagger_segments_0909_seed2 | fixed60 | 38/100 = 38.0% | 3.17cm | 60 |
| dagger_merged_20260910_092947_dagger_segments_0909_seed2 | fixed67 | 43/100 = 43.0% | 3.76cm | 67 |
| dagger_merged_20260910_092947_dagger_segments_0909_seed2 | fixed75 | 44/100 = 44.0% | 3.72cm | 75 |
| dagger_merged_20260910_092947_dagger_segments_0909_seed2 | fixed85 | 30/100 = 30.0% | 2.65cm | 85 |
| dagger_merged_20260910_092947_dagger_segments_0909_seed2 | oracle | 24/100 = 24.0% | 2.04cm | 82 |

## 사전등록 판정

- fixed 통과 checkpoint 수: {'fixed60': 3, 'fixed67': 3, 'fixed75': 3, 'fixed85': 3}
- oracle 통과 checkpoint 수: 2
- viable fixed: ['fixed60', 'fixed67', 'fixed75', 'fixed85']
- 판정: **fixed_schedule_candidate**