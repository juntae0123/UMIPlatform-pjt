"""Torch dataset over contract episodes, plus a random-tensor stand-in.
계약 에피소드를 읽는 torch 데이터셋과, 랜덤 텐서 대역.

The random-tensor dataset exists so the training loop can be finished and
verified **before** any real data arrives. Debugging a training loop and
debugging a dataset at the same time is how days disappear.
랜덤 텐서 데이터셋이 있는 이유는, 실데이터가 오기 **전에** 학습 루프를 완성하고
검증하기 위해서다. 루프 디버깅과 데이터 디버깅을 동시에 하면 며칠이 사라진다.

⚠️ 랜덤 텐서로 얻은 손실 값은 아무 의미가 없다. 확인하는 것은 "루프가 도는가"
   하나뿐이다. 그 사실을 EXP_LOG 와 체크포인트 메타에 `trained_on` 으로 남긴다.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from contract.episode import IMAGE_SHAPE, read_episode, validate


class EpisodeDataset(Dataset):
    """Flattened (observation, action) pairs from contract episodes.
    계약 에피소드를 (관측, 행동) 쌍으로 펼친 것.

    Every episode is validated on load. A dataset that silently contains a
    contract violation trains a model on something other than what the contract
    says, and the mismatch only shows up at inference on the robot.
    로드할 때 에피소드마다 검증한다. 계약 위반을 조용히 품은 데이터셋은 계약과
    다른 것으로 모델을 학습시키고, 그 불일치는 로봇 앞 추론 시점에야 드러난다.
    """

    def __init__(
        self,
        root: Path,
        cfg: dict[str, Any],
        camera_names: list[str] | None = None,
        strict: bool = True,
    ) -> None:
        self.root = Path(root)
        files = sorted(self.root.glob("*.npz"))
        if not files:
            raise FileNotFoundError(f"에피소드가 없다: {self.root}")

        d = cfg["data"]
        self.mean = float(d["image_mean"])
        self.std = float(d["image_std"])

        self.index: list[tuple[int, int]] = []   # (episode idx, timestep)
        self.episodes: list[Any] = []
        self.rejected: list[tuple[str, list[str]]] = []
        cams: list[str] | None = camera_names

        for ep_path in files:
            ep = read_episode(ep_path)
            problems = validate(ep)
            if problems:
                self.rejected.append((ep_path.name, problems))
                if strict:
                    raise ValueError(
                        f"{ep_path.name} 이 계약을 위반한다. 학습에 쓰지 않는다:\n  "
                        + "\n  ".join(problems)
                    )
                continue
            if cams is None:
                cams = list(ep.meta.cameras)
            elif list(ep.meta.cameras) != cams:
                raise ValueError(
                    f"{ep_path.name} 의 카메라 {ep.meta.cameras} 가 "
                    f"앞선 에피소드의 {cams} 와 다르다"
                )
            i = len(self.episodes)
            self.episodes.append(ep)
            self.index.extend((i, t) for t in range(ep.meta.n_steps))

        if cams is None or not self.index:
            raise ValueError(f"쓸 수 있는 에피소드가 없다: {self.root}")
        self.camera_names = cams

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, i: int) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
        ep_i, t = self.index[i]
        ep = self.episodes[ep_i]
        images = {}
        for cam in self.camera_names:
            arr = torch.from_numpy(ep.images[cam][t].astype(np.float32) / 255.0)
            images[cam] = (arr - self.mean) / self.std
        state = torch.from_numpy(ep.state[t].astype(np.float32))
        action = torch.from_numpy(ep.action[t].astype(np.float32))
        return images, state, action

    def summary(self) -> str:
        return (
            f"{self.root.name}: 에피소드 {len(self.episodes)}개, "
            f"샘플 {len(self.index)}개, 카메라 {self.camera_names}"
            + (f", 계약위반으로 제외 {len(self.rejected)}개" if self.rejected else "")
        )


class RandomTensorDataset(Dataset):
    """Contract-shaped noise. For proving the loop runs, nothing else.
    계약 shape 의 잡음. 루프가 돈다는 것을 증명하는 용도, 그 외 없음.

    Shapes and dtypes come from `contract/episode.py`, so if the contract
    changes this stand-in changes with it and the loop is re-verified against
    the new shape rather than the old one.
    shape 과 dtype 은 `contract/episode.py` 에서 온다. 계약이 바뀌면 이 대역도
    함께 바뀌고, 루프는 옛 shape 이 아니라 새 shape 으로 다시 검증된다.
    """

    def __init__(
        self,
        n_samples: int,
        cfg: dict[str, Any],
        camera_names: list[str],
        seed: int = 0,
        action_dim: int = 6,
        state_dim: int = 6,
    ) -> None:
        self.n = int(n_samples)
        self.camera_names = list(camera_names)
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.seed = seed
        d = cfg["data"]
        self.mean = float(d["image_mean"])
        self.std = float(d["image_std"])

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, i: int):
        # 샘플마다 결정적. 같은 seed 면 같은 데이터가 나온다.
        rng = np.random.default_rng(self.seed * 1_000_003 + i)
        images = {}
        for cam in self.camera_names:
            raw = rng.integers(0, 256, IMAGE_SHAPE, dtype=np.uint8).astype(np.float32) / 255.0
            images[cam] = torch.from_numpy((raw - self.mean) / self.std)
        state = torch.from_numpy(rng.uniform(-1, 1, self.state_dim).astype(np.float32))
        action = torch.from_numpy(rng.uniform(-1, 1, self.action_dim).astype(np.float32))
        return images, state, action

    def summary(self) -> str:
        return (
            f"랜덤 텐서 {self.n}개, 카메라 {self.camera_names} "
            f"(이미지 {IMAGE_SHAPE}) — ⚠️ 손실 값에 의미 없음, 루프 검증용"
        )


def collate(batch):
    """Stack a list of (images dict, state, action) into batched tensors.
    (이미지 dict, state, action) 목록을 배치 텐서로 쌓는다."""
    cams = batch[0][0].keys()
    images = {c: torch.stack([b[0][c] for b in batch]) for c in cams}
    state = torch.stack([b[1] for b in batch])
    action = torch.stack([b[2] for b in batch])
    return images, state, action
