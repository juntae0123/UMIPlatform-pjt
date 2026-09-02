"""Train a BC policy. Finishes the loop on random tensors before real data exists.
BC 정책을 학습한다. 실데이터가 있기 전에 랜덤 텐서로 루프를 먼저 완주시킨다.

    python tools/train_bc.py --random 256 --epochs 3      # 루프 검증 (데이터 불필요)
    python tools/train_bc.py --data datasets/sim_teleop_v0

⚠️ 이 스크립트는 성능을 판정하지 않는다. 손실만 낸다.
   판정은 `tools/eval_rollout.py --policy-ckpt <ckpt>` 가 하고,
   넘어야 할 값은 롤아웃 성공률 **20.0%** 다 (eval/rollout.py 의 GATES).
   손실 곡선이 내려간 것은 성과가 아니다.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from contract.episode import CONTRACT_VERSION
from data.dataset import EpisodeDataset, RandomTensorDataset, collate
from paths import AI_ROOT, DEFAULT_CONFIG
from policy.bc import (
    BCNet,
    CheckpointMeta,
    load_train_config,
    save_checkpoint,
    target_scale,
    training_target,
)
from tracking.exp_log import code_digest, file_digest, log_run

DEFAULT_CAMERAS = ["cam_front", "cam_wrist"]


def make_loss(name: str) -> nn.Module:
    """L1 or MSE. Recorded in the checkpoint so a number can be attributed.
    L1 또는 MSE. 수치를 귀속시킬 수 있도록 체크포인트에 기록한다."""
    if name == "l1":
        return nn.L1Loss()
    if name == "mse":
        return nn.MSELoss()
    raise ValueError(f"loss 는 l1|mse 여야 한다: {name}")


def split_indices(n: int, val_fraction: float, seed: int) -> tuple[list[int], list[int]]:
    """Deterministic train/val split.
    결정적인 train/val 분할."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_val = max(1, int(round(n * val_fraction))) if n > 1 else 0
    return idx[n_val:].tolist(), idx[:n_val].tolist()


def run_epoch(
    model: BCNet,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    grad_clip: float = 0.0,
    target_mean: torch.Tensor | None = None,
    target_std: torch.Tensor | None = None,
) -> float:
    """One pass. `optimizer=None` means evaluation.
    한 바퀴. `optimizer=None` 이면 평가.

    ⚠️ 손실은 `model.action_space` 가 정하는 공간에서 계산된다. 절대 목표의 손실과
       델타 목표의 손실은 **서로 비교할 수 없다** — 크기가 두 자릿수 다르다.
       비교는 언제나 롤아웃 성공률로 한다.
    """
    train = optimizer is not None
    model.train(train)
    total, n = 0.0, 0
    with torch.set_grad_enabled(train):
        for images, state, action in loader:
            images = {c: v.to(device) for c, v in images.items()}
            state, action = state.to(device), action.to(device)
            pred = model(images, state)
            target = training_target(action, state, model.action_space)
            if target_std is not None:
                target = (target - target_mean) / target_std
            loss = criterion(pred, target)
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
            total += float(loss.item()) * action.shape[0]
            n += action.shape[0]
    return total / max(n, 1)


def main() -> int:
    parser = argparse.ArgumentParser(description="BC 정책 학습")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--data", type=Path, help="계약 에피소드 디렉터리")
    src.add_argument("--random", type=int, metavar="N",
                     help="랜덤 텐서 N개로 루프만 검증한다 (데이터 불필요)")
    parser.add_argument("--epochs", type=int, default=None, help="설정값을 덮어쓴다")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=None,
                        help="설정값을 덮어쓴다. 학습 3회 반복 시 서로 다른 값을 준다")
    parser.add_argument("--out", type=Path, default=None, help="체크포인트 경로")
    parser.add_argument("--author", type=str, default="김준태(트랙B)")
    parser.add_argument("--log", action="store_true")
    args = parser.parse_args()

    cfg: dict[str, Any] = load_train_config()
    t = cfg["train"]
    epochs = int(args.epochs if args.epochs is not None else t["epochs"])
    batch_size = int(args.batch_size if args.batch_size is not None else t["batch_size"])
    seed = int(args.seed if args.seed is not None else t["seed"])
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device(args.device)

    if args.random is not None:
        dataset: Any = RandomTensorDataset(args.random, cfg, DEFAULT_CAMERAS, seed=seed)
        cameras = DEFAULT_CAMERAS
        trained_on = "random_tensors"
        n_episodes = 0
        print("⚠️ 랜덤 텐서로 학습한다. **손실 값에 의미가 없다.**")
        print("   확인하는 것은 루프가 끝까지 도는가 하나뿐이다.\n")
    else:
        dataset = EpisodeDataset(args.data, cfg)
        cameras = dataset.camera_names
        trained_on = str(args.data)
        n_episodes = len(dataset.episodes)

    print(dataset.summary())

    tr_idx, va_idx = split_indices(len(dataset), float(t["val_fraction"]), seed)
    loaders = {
        "train": DataLoader(Subset(dataset, tr_idx), batch_size=batch_size, shuffle=True,
                            num_workers=int(t["num_workers"]), collate_fn=collate),
        "val": DataLoader(Subset(dataset, va_idx), batch_size=batch_size, shuffle=False,
                          num_workers=int(t["num_workers"]), collate_fn=collate)
        if va_idx else None,
    }

    # Target statistics, computed from the episode arrays rather than by iterating
    # the DataLoader -- the loader would decode 13,818 images to read six numbers.
    # 타겟 통계. DataLoader 를 도는 대신 에피소드 배열에서 직접 계산한다. 로더로 돌면
    # 숫자 여섯 개를 읽으려고 이미지 13,818장을 디코딩한다.
    t_mean = t_std = None
    baseline_norm = float("nan")
    if getattr(dataset, "episodes", None):
        acts = torch.from_numpy(
            np.concatenate([e.action for e in dataset.episodes], axis=0)
        ).float()
        sts = torch.from_numpy(
            np.concatenate([e.state for e in dataset.episodes], axis=0)
        ).float()
        raw_target = training_target(acts, sts, str(cfg["model"].get("action_space", "joint_absolute")))
        if bool(t.get("normalize_target", True)):
            t_mean, t_std = target_scale(raw_target)
            scaled = (raw_target - t_mean) / t_std
        else:
            scaled = raw_target
        # What a network that always outputs zero would score, in the same units
        # the epoch losses are printed in. A loss without this reference cannot be
        # read: 0.0057 looked fine until it turned out zero-output scores 0.0075.
        # 항상 0 을 내는 신경망의 점수. epoch 손실과 같은 단위다. 이 기준 없이는 손실을
        # 읽을 수 없다 — 0.0057 이 괜찮아 보였는데 0 출력이 0.0075 였다.
        baseline_norm = float(scaled.abs().mean())

    model = BCNet(cameras, cfg).to(device)
    if t_mean is not None:
        t_mean, t_std = t_mean.to(device), t_std.to(device)
    n_params = model.n_params()

    # The action space is a one-line config change that silently decides what the
    # loss even means. If it does not take effect, training runs for hours in the
    # wrong space and the loss looks fine. So it gets printed, not assumed.
    # 행동 공간은 설정 한 줄이지만 손실의 의미 자체를 정한다. 이게 안 먹으면 몇 시간을
    # 엉뚱한 공간에서 학습하고도 손실은 멀쩡해 보인다. 그래서 가정하지 않고 출력한다.
    print(f"행동 공간: {model.action_space}", end="")
    if model.action_space == "joint_delta":
        print("  — 목표는 action - state 잔차. 출력에 state 를 더해 행동을 만든다")
        print("  ⚠️ 손실 값을 절대 목표 학습분과 비교하지 마라. 크기가 두 자릿수 다르다")
    else:
        print("  — 목표는 절대 관절각")
        print("  ⚠️ 실측상 이 공간에서는 상태만으로 정답의 86% 가 설명된다"
              " (2026-09-02 image_sensitivity)")
    if t_std is not None:
        print(f"타겟 표준화: 켬 (관절별 std {[round(float(v), 5) for v in t_std.cpu()]})")
    else:
        print("타겟 표준화: 끔")
    if baseline_norm == baseline_norm:  # not NaN
        print(f"⚠️ 자명한 예측기(항상 0) 손실 = {baseline_norm:.5f}")
        print("   epoch 손실이 이 값 근처에서 멈추면 학습이 안 되고 있는 것이다."
              " 손실이 내려간 것만으로 판단하지 마라")
    print(f"파라미터 {n_params:,}개 (~{n_params * 4 / 1024 / 1024:.1f}MB fp32)")
    print("⚠️ Jetson 8GB 에 VLM 과 함께 올라가야 한다 — 이슈 42 미검증\n")

    criterion = make_loss(str(t["loss"]))
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(t["lr"]),
                                  weight_decay=float(t["weight_decay"]))

    out = args.out or (AI_ROOT / cfg["checkpoint"]["dir"] /
                       ("bc_random.pt" if args.random else "bc.pt"))
    best_out = out.with_name(out.stem + "_bestval" + out.suffix)
    best = float("inf")
    history: list[dict[str, float]] = []
    t0 = time.perf_counter()

    for ep in range(1, epochs + 1):
        tr = run_epoch(model, loaders["train"], criterion, device, optimizer,
                       float(t["grad_clip"]))
        va = (run_epoch(model, loaders["val"], criterion, device,
                           target_mean=t_mean, target_std=t_std)
              if loaders["val"] is not None else float("nan"))
        history.append({"epoch": ep, "train_loss": tr, "val_loss": va})
        mark = ""
        if loaders["val"] is not None and va < best:
            best = va
            save_checkpoint(best_out, model, CheckpointMeta(
                camera_names=list(cameras), contract_version=CONTRACT_VERSION,
                action_space=model.action_space,
                target_mean=[float(v) for v in t_mean.cpu()] if t_mean is not None else None,
                target_std=[float(v) for v in t_std.cpu()] if t_std is not None else None,
                train_config=cfg, config_sha=file_digest(DEFAULT_CONFIG),
                code_sha=code_digest(), n_params=n_params, n_episodes=n_episodes,
                n_samples=len(dataset), epochs_run=ep, best_val_loss=best,
                trained_on=trained_on))
            mark = "  ← best-val 저장"
        print(f"  epoch {ep:3d}/{epochs}  train {tr:.5f}  val {va:.5f}{mark}")

    elapsed = time.perf_counter() - t0

    # The main checkpoint is the LAST epoch, not the one with the lowest val loss.
    # This project's own rule says val loss only tells you whether training broke,
    # and yet checkpoint selection was being driven by it -- on a flat val curve
    # that picks an arbitrary early epoch. Measured: a 200-epoch run saved its
    # epoch-1 model, and the rollout that "evaluated the policy" evaluated an
    # untrained network. 🟢 2026-09-02
    # 주 체크포인트는 val loss 가 가장 낮은 epoch 이 아니라 **마지막 epoch** 이다.
    # 이 프로젝트 규칙은 val loss 가 "학습이 망가졌나"만 말한다고 해놓고, 정작
    # 체크포인트 선택을 그것이 하고 있었다. val 곡선이 평평하면 임의의 이른 epoch 이
    # 뽑힌다. 실측: 200 epoch 실행이 epoch 1 모델을 저장했고, "정책을 평가"한 롤아웃이
    # 학습되지 않은 신경망을 평가했다.
    save_checkpoint(out, model, CheckpointMeta(
            camera_names=list(cameras), contract_version=CONTRACT_VERSION,
            action_space=model.action_space,
            target_mean=[float(v) for v in t_mean.cpu()] if t_mean is not None else None,
            target_std=[float(v) for v in t_std.cpu()] if t_std is not None else None,
            train_config=cfg, config_sha=file_digest(DEFAULT_CONFIG),
            code_sha=code_digest(), n_params=n_params, n_episodes=n_episodes,
            n_samples=len(dataset), epochs_run=epochs, best_val_loss=best,
            trained_on=trained_on))

    print(f"\n체크포인트(마지막 epoch): {out}  ({elapsed:.1f}초, {elapsed / epochs:.2f}초/epoch)")
    if best_out.exists():
        print(f"참고용(best val): {best_out}")
        print("  ⚠️ 판정에는 마지막 epoch 을 쓴다. val loss 가 낮은 epoch 이 좋은 정책이라는"
              " 근거가 이 프로젝트에는 없다")
    print("\n" + "=" * 70)
    print("⚠️ 손실은 성과가 아니다. 이 체크포인트가 쓸 만한지는 아직 모른다.")
    print("   판정은 롤아웃 성공률이고, 넘어야 할 값은 20.0% 다:")
    print(f"     python tools/eval_rollout.py --policy-ckpt {out} --render --log")
    if args.random is not None:
        print("   ⚠️ 지금 것은 랜덤 텐서 학습이다. 평가해도 의미 없다.")
    print("=" * 70)

    if args.log:
        rec = log_run(
            experiment="train_bc", author=args.author, issue="S15P21A103-34",
            conditions={
                "trained_on": trained_on, "n_episodes": n_episodes,
                "n_samples": len(dataset), "epochs": epochs, "batch_size": batch_size,
                "lr": t["lr"], "loss": t["loss"], "seed": seed, "device": str(device),
                "encoder_mode": cfg["model"]["encoder_mode"],
                "action_space": model.action_space,
                "normalize_target": t_std is not None,
                "trivial_baseline_loss": baseline_norm, "n_params": n_params,
                "config_sha": file_digest(DEFAULT_CONFIG),
                "metric_note": "손실은 학습이 망가졌는지 확인용. 판정은 롤아웃 성공률(게이트 20.0%)",
            },
            result={
                "best_val_loss": best if best != float("inf") else None,
                "final_train_loss": history[-1]["train_loss"] if history else None,
                "history": history, "seconds": round(elapsed, 2),
                "checkpoint": str(out),
                "success_rate": None,
                "success_rate_note": "여기서 재지 않는다. tools/eval_rollout.py 참조",
            },
        )
        print(f"\nEXP_LOG.jsonl 기록 (code {rec['code_sha']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
