"""Lightweight VLM for grounding and skill selection — NOT IMPLEMENTED YET.
grounding·스킬선택용 경량 VLM — **아직 구현 없음.**

Issue S15P21A103-36. Fine-tuning code and adapters belong here.
이슈 S15P21A103-36. 파인튜닝 코드와 어댑터가 여기 들어온다.

HARD CONSTRAINT: VLM + policy must fit in 8 GB on Jetson together
(S15P21A103-42, unverified). Growing the VLM leaves no room for the policy.
제약: Jetson 에서 VLM + 정책 합쳐 8GB 안에 들어가야 한다
(S15P21A103-42, 미검증). VLM 을 키우면 정책이 올라갈 자리가 없어진다.
"""
