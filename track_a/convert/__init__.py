"""Raw recordings to dataset format — NOT IMPLEMENTED. S15P21A103-31.
raw → dataset 포맷 변환 — **구현 없음.** S15P21A103-31.

출력은 `contract/episode.py` 의 `Episode` 를 만족해야 한다. 검증기를 통과하지 못한
에피소드는 저장하지 않는다 — 시뮬 쪽 대응물이 `data/collect.py` 와 `data/verify.py` 이고,
거기서 검증기가 실제로 결함 1건을 저장 전에 걸러냈다.

임포트는 이렇게 한다 (`AI/` 에서 실행):

    from contract.episode import Episode, EpisodeMeta, validate, write_episode

⚠️ 계약의 미확정 항목 3건(정규화 규칙 / 타임스탬프 필드 / 범위 초과값 처리)이
   S15P21A103-27 에 남아 있다. 확정 전에 대량 변환하면 전량 재작업 위험이 있다.
"""
