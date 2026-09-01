"""Why did BC score 0%? Narrow it before changing anything.
BC 는 왜 0% 인가? 무엇을 고치기 전에 원인부터 좁힌다.

A rollout of 0/20 with a validation loss of 0.005 has two very different
explanations, and they lead to opposite fixes:
롤아웃 0/20 에 val_loss 0.005 라는 조합에는 성격이 완전히 다른 두 설명이 있고,
각각 반대 방향의 처방으로 이어진다:

  (A) 정책이 관측을 무시하고 **평균 궤적**을 출력한다.
      시연이 스크립트라 궤적 모양이 거의 같으면, 이미지를 보지 않고 평균만
      뱉어도 손실이 아주 낮게 나온다. 물체가 움직이면 전부 실패한다.
      → 처방: 데이터 다양성, 인코더, 물체 위치에 대한 조건화

  (B) 정책이 관측에 반응은 하는데 **오차가 누적**되어 실패한다.
      한 스텝 오차가 작아도 141 스텝 폐루프에서 분포 밖으로 밀려난다.
      → 처방: 액션 청킹(ACT), DAgger, 더 많은 데이터

These are distinguished by one number: how much the prediction varies ACROSS
episodes at the same timestep, compared with how much the recorded action
varies. A policy outputting the mean trajectory has near-zero variation no
matter where the object is.
이 둘은 수치 하나로 갈린다 — 같은 시각 t 에서 **에피소드들 사이** 예측이 얼마나
달라지는가를, 기록된 행동이 얼마나 달라지는가와 비교한다. 평균 궤적을 뱉는
정책은 물체가 어디 있든 그 변동이 0 에 가깝다.

⚠️ 이 진단은 **개루프**다. 기록된 관측을 그대로 먹인다. 폐루프에서 정책이
   방문하는 상태 분포는 여기 나오지 않는다. 즉 여기서 좋아 보여도 롤아웃이
   실패할 수 있고, 실제로 그것이 (B) 의 정의다.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from contract.episode import read_episode
from policy.bc import BCPolicy
from sim.base import Observation
from sim.mujoco.build_scene import DEFAULT_CONFIG, joint_specs, load_config
from tracking.exp_log import file_digest, log_run

# Fixed before the diagnostic was run.
# 진단 실행 전에 확정했다.
RESPONDS_ABOVE = 0.50
IGNORES_BELOW = 0.20

GATES: dict[str, str] = {
    "responds": (
        f"응답비 >= {RESPONDS_ABOVE:.2f} → 정책이 관측에 반응한다. 실패 원인은 "
        "오차 누적 쪽이다 (가설 B)"
    ),
    "ignores": (
        f"응답비 < {IGNORES_BELOW:.2f} → 정책이 사실상 평균 궤적을 출력한다. "
        "이미지를 쓰지 않고 있다 (가설 A)"
    ),
    "partial": f"{IGNORES_BELOW:.2f} ~ {RESPONDS_ABOVE:.2f} → 부분 반응. 둘 다 작용",
}


@dataclass
class DiagnosisReport:
    """Open-loop behaviour of one checkpoint over a set of episodes.
    체크포인트 하나의 에피소드 묶음에 대한 개루프 동작."""

    n_episodes: int
    n_steps: int
    response_ratio: float
    pred_spread: float
    rec_spread: float
    mae_norm: float
    mae_deg: list[float]
    mae_deg_by_phase: dict[str, float]
    clip_rate: float
    joint_names: list[str]

    def verdict(self) -> str:
        """Which hypothesis the response ratio supports.
        응답비가 지지하는 가설."""
        if self.response_ratio < IGNORES_BELOW:
            return "가설 A — 관측을 무시하고 평균 궤적을 출력한다"
        if self.response_ratio >= RESPONDS_ABOVE:
            return "가설 B — 관측에 반응한다. 실패는 오차 누적 쪽"
        return "부분 반응 — A 와 B 가 함께 작용"


def _predict(policy: BCPolicy, ep: Any, n_steps: int) -> np.ndarray:
    """Feed recorded observations one at a time; collect the actions.
    기록된 관측을 하나씩 먹이고 행동을 모은다."""
    out = np.zeros((n_steps, 6), dtype=np.float32)
    for t in range(n_steps):
        obs = Observation(
            images={cam: ep.images[cam][t] for cam in ep.meta.cameras},
            state=ep.state[t],
            timestamp=float(ep.state_timestamp[t]),
        )
        out[t] = policy.act(obs)
    return out


def diagnose(
    dataset: Path, ckpt: Path, *, limit: int, device: str, cfg: dict[str, Any]
) -> DiagnosisReport:
    """Compare predicted and recorded actions across episodes, timestep by timestep.
    예측 행동과 기록 행동을 에피소드 전체에 걸쳐 시각별로 비교한다."""
    paths = sorted(dataset.glob("*.npz"))[:limit]
    if len(paths) < 2:
        raise SystemExit("에피소드가 2편 미만이면 에피소드 간 변동을 잴 수 없다")

    episodes = [read_episode(p) for p in paths]
    n_steps = min(ep.meta.n_steps for ep in episodes)

    policy = BCPolicy(ckpt, device=device)
    preds = np.stack([_predict(policy, ep, n_steps) for ep in episodes])  # (E,T,6)
    recs = np.stack([ep.action[:n_steps] for ep in episodes]).astype(np.float32)

    # Spread ACROSS episodes at each timestep. This is the discriminating number:
    # the mean trajectory has zero spread by construction.
    # 각 시각에서 **에피소드 사이** 퍼짐. 이것이 판별 수치다 — 평균 궤적은
    # 정의상 퍼짐이 0 이다.
    pred_spread = float(np.mean(preds.std(axis=0)))
    rec_spread = float(np.mean(recs.std(axis=0)))
    ratio = pred_spread / rec_spread if rec_spread > 0 else float("nan")

    err = np.abs(preds - recs)
    specs = sorted(joint_specs(cfg), key=lambda s: s.index)
    half_range_deg = np.array([np.rad2deg((s.hi - s.lo) / 2.0) for s in specs])
    mae_deg = (err.mean(axis=(0, 1)) * half_range_deg).tolist()

    third = max(1, n_steps // 3)
    phases = {
        "접근 (앞 1/3)": err[:, :third],
        "정렬 (중간 1/3)": err[:, third : 2 * third],
        "파지·상승 (뒤 1/3)": err[:, 2 * third :],
    }
    mae_deg_by_phase = {
        k: float(np.mean(v.mean(axis=(0, 1)) * half_range_deg)) for k, v in phases.items()
    }

    return DiagnosisReport(
        n_episodes=len(episodes),
        n_steps=n_steps,
        response_ratio=ratio,
        pred_spread=pred_spread,
        rec_spread=rec_spread,
        mae_norm=float(err.mean()),
        mae_deg=mae_deg,
        mae_deg_by_phase=mae_deg_by_phase,
        clip_rate=policy.n_clipped / max(1, policy.n_actions),
        joint_names=[s.name for s in specs],
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=Path("datasets/sim_pick_v1"))
    parser.add_argument("--ckpt", type=Path, default=Path("checkpoints/bc/bc_sim_pick_v1.pt"))
    parser.add_argument("--limit", type=int, default=20, help="진단에 쓸 에피소드 수")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--author", type=str, default="김준태(트랙B)")
    parser.add_argument("--log", action="store_true")
    args = parser.parse_args()

    cfg = load_config()
    print("게이트 기준 (결과 확인 전 확정):")
    for key, text in GATES.items():
        print(f"  [{key}] {text}")
    print()

    r = diagnose(args.dataset, args.ckpt, limit=args.limit, device=args.device, cfg=cfg)

    print(f"에피소드 {r.n_episodes}편 × {r.n_steps}스텝, 개루프 (기록된 관측 입력)\n")
    print(f"에피소드 간 퍼짐  기록 {r.rec_spread:.5f}  예측 {r.pred_spread:.5f}")
    print(f"응답비 = 예측/기록 = {r.response_ratio:.4f}   →  {r.verdict()}\n")

    print(f"개루프 행동 오차 (정규화 MAE {r.mae_norm:.5f})")
    for name, deg in zip(r.joint_names, r.mae_deg):
        print(f"  {name:15s} {deg:7.3f}°")
    print()
    for phase, deg in r.mae_deg_by_phase.items():
        print(f"  {phase:20s} 평균 {deg:7.3f}°")
    print(f"\n계약 범위 초과 클립 비율 {r.clip_rate * 100:.1f}%")

    print(
        "\n⚠️ 개루프 진단이다. 기록된 관측을 그대로 먹였다. 폐루프에서 정책이 실제로 "
        "방문하는 상태는 여기 없다."
    )

    if args.log:
        rec = log_run(
            experiment="diagnose_bc",
            author=args.author,
            issue="S15P21A103-34",
            conditions={
                "dataset": str(args.dataset),
                "ckpt": str(args.ckpt),
                "n_episodes": r.n_episodes,
                "n_steps": r.n_steps,
                "config_sha": file_digest(DEFAULT_CONFIG),
                "gates": GATES,
                "mode": "open_loop",
            },
            result={
                "response_ratio": r.response_ratio,
                "pred_spread": r.pred_spread,
                "rec_spread": r.rec_spread,
                "mae_norm": r.mae_norm,
                "mae_deg": dict(zip(r.joint_names, r.mae_deg)),
                "mae_deg_by_phase": r.mae_deg_by_phase,
                "clip_rate": r.clip_rate,
                "verdict": r.verdict(),
            },
        )
        print(f"\nEXP_LOG.jsonl 기록 (git {rec['git_rev']}, dirty={rec['git_dirty']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
