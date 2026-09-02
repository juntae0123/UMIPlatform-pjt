"""Behavior cloning — the baseline learned policy. Deliberately the simplest one.
행동복제 — 기준이 되는 학습 정책. 의도적으로 가장 단순한 것.

BC before ACT before Diffusion. If BC's number is not measured first, a later
model that works cannot be explained and one that fails cannot be diagnosed —
you will not know whether the problem is the data or the model.
BC 다음 ACT 다음 Diffusion. BC 수치를 먼저 재지 않으면, 나중 모델이 잘 돼도
이유를 설명 못 하고 안 돼도 원인(데이터 vs 모델)을 좁힐 수 없다.

⚠️ 통과 기준은 validation loss 가 아니다. 롤아웃 성공률이고, 넘어야 할 값은
   **20.0%** 다 (eval/rollout.py 의 GATES, 결과 보기 전에 확정됨).
   손실은 "학습이 망가지지 않았나" 확인용으로만 쓴다.

⚠️ Jetson 8GB 에 VLM 과 함께 올라가야 한다. 파라미터 수를 체크포인트 메타에
   기록하고, 학습 시작 때 출력한다. 키우기 전에 이슈 42 를 확인한다.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import yaml

from paths import CONFIG_DIR
from policy.base import check_action
from sim.base import Observation

DEFAULT_TRAIN_CONFIG = CONFIG_DIR / "train" / "bc.yaml"


def load_train_config(path: Path = DEFAULT_TRAIN_CONFIG) -> dict[str, Any]:
    """Read the BC training config.
    BC 학습 설정을 읽는다."""
    with path.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)


class ConvEncoder(nn.Module):
    """A small strided CNN. Not pretrained, on purpose — for now.
    작은 스트라이드 CNN. 사전학습 없음, 지금은 의도적으로.

    An ImageNet-pretrained ResNet would very likely help with a small dataset,
    and it is the obvious next thing to measure. It is not here yet because the
    point of this first pass is a loop that runs end to end without a download,
    and because swapping the encoder changes the image normalization statistics
    — which is a change worth recording rather than sliding in.
    작은 데이터셋에서는 ImageNet 사전학습 ResNet 이 도움이 될 가능성이 높고,
    다음에 재볼 후보다. 지금 없는 이유는 이번 단계의 목적이 다운로드 없이
    끝까지 도는 루프이고, 인코더를 바꾸면 이미지 정규화 통계가 함께 바뀌기
    때문이다 — 슬쩍 넣을 것이 아니라 기록할 변경이다.
    """

    def __init__(self, channels: list[int], feature_dim: int) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_ch = 3
        for out_ch in channels:
            layers += [
                nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1, bias=False),
                nn.GroupNorm(num_groups=min(8, out_ch), num_channels=out_ch),
                nn.ReLU(inplace=True),
            ]
            in_ch = out_ch
        self.conv = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Linear(in_ch, feature_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, 3, H, W) float -> (B, feature_dim).
        (B, 3, H, W) float -> (B, feature_dim)."""
        h = self.conv(x)
        h = self.pool(h).flatten(1)
        return self.proj(h)


ACTION_SPACES = ("joint_absolute", "joint_delta")
"""What the network regresses. Not a contract change -- the dataset always stores
absolute joint angles and this is a transform applied at train and inference time.
신경망이 무엇을 회귀하는가. 계약 변경이 아니다. 데이터셋에는 언제나 절대 관절각이
저장되고, 이것은 학습·추론 시점에 적용되는 변환이다."""


def training_target(
    action: torch.Tensor, state: torch.Tensor, action_space: str
) -> torch.Tensor:
    """The tensor the loss is computed against.
    손실을 계산할 대상 텐서.

    With an absolute target most of the loss is spent on "where is the arm now",
    which the state input already answers -- measured at 86% on this dataset. The
    part that actually produces motion, `action - state`, is small and gets left
    wrong. Subtracting the state removes exactly the term the state explains, so
    what is left is the part that requires knowing where the object is.
    절대 목표에서는 손실 대부분이 "팔이 지금 어디 있나"에 쓰이는데, 그건 상태 입력이
    이미 답하고 있다 — 이 데이터셋에서 86% 로 실측됐다. 실제로 움직임을 만드는
    `action - state` 는 작고 틀린 채 남는다. 상태를 빼면 상태가 설명하던 항이 정확히
    사라지고, 남는 것은 물체가 어디 있는지 알아야 설명되는 몫이다.
    """
    if action_space == "joint_delta":
        return action - state
    if action_space == "joint_absolute":
        return action
    raise ValueError(f"action_space 는 {ACTION_SPACES} 중 하나여야 한다: {action_space!r}")


def to_action(
    raw: torch.Tensor, state: torch.Tensor, action_space: str
) -> torch.Tensor:
    """Turn the network's output into a contract-unit action.
    신경망 출력을 계약 단위 행동으로 바꾼다."""
    if action_space == "joint_delta":
        return state + raw
    if action_space == "joint_absolute":
        return raw
    raise ValueError(f"action_space 는 {ACTION_SPACES} 중 하나여야 한다: {action_space!r}")


class BCNet(nn.Module):
    """Images (+ joint state) -> one action. Single step, no chunking.
    이미지 (+ 관절 state) -> 행동 하나. 단일 스텝, 청킹 없음.

    Action chunking is what ACT adds. Doing it here would mean BC and ACT differ
    in two ways at once, and a difference in the number could not be attributed.
    행동 청킹은 ACT 가 더하는 것이다. 여기서 하면 BC 와 ACT 가 두 가지가 동시에
    달라져서, 수치 차이를 무엇 때문인지 귀속시킬 수 없게 된다.
    """

    def __init__(
        self,
        camera_names: list[str],
        cfg: dict[str, Any],
        state_dim: int = 6,
        action_dim: int = 6,
    ) -> None:
        super().__init__()
        m = cfg["model"]
        self.camera_names = list(camera_names)
        self.encoder_mode = m["encoder_mode"]
        self.use_state = bool(m["use_state"])
        self.action_space = str(m.get("action_space", "joint_absolute"))
        if self.action_space not in ACTION_SPACES:
            raise ValueError(
                f"action_space 는 {ACTION_SPACES} 중 하나여야 한다: {self.action_space!r}"
            )
        if self.action_space == "joint_delta" and not self.use_state:
            raise ValueError(
                "joint_delta 는 출력에 state 를 더해 행동을 만든다. use_state=false 로는 "
                "학습 목표와 추론이 어긋난다"
            )
        feat = int(m["feature_dim"])

        if self.encoder_mode == "shared":
            enc = ConvEncoder(m["encoder_channels"], feat)
            self.encoders = nn.ModuleDict({c: enc for c in self.camera_names})
        elif self.encoder_mode == "separate":
            self.encoders = nn.ModuleDict(
                {c: ConvEncoder(m["encoder_channels"], feat) for c in self.camera_names}
            )
        else:
            raise ValueError(f"encoder_mode 는 separate|shared 여야 한다: {self.encoder_mode}")

        in_dim = feat * len(self.camera_names) + (state_dim if self.use_state else 0)
        dims = [in_dim, *[int(d) for d in m["hidden_dims"]]]
        head: list[nn.Module] = []
        for a, b in zip(dims[:-1], dims[1:]):
            head += [nn.Linear(a, b), nn.ReLU(inplace=True), nn.Dropout(float(m["dropout"]))]
        head.append(nn.Linear(dims[-1], action_dim))
        self.head = nn.Sequential(*head)

    def forward(self, images: dict[str, torch.Tensor], state: torch.Tensor) -> torch.Tensor:
        """Raw head output. Under `joint_delta` this is the residual, not the action.
        헤드의 원 출력. `joint_delta` 에서는 이것이 행동이 아니라 잔차다.

        Converting here would hide which space the loss is computed in. The caller
        applies `to_action` when it wants an action.
        여기서 변환하면 손실이 어느 공간에서 계산되는지가 가려진다. 행동이 필요한
        호출자가 `to_action` 을 적용한다."""
        feats = [self.encoders[c](images[c]) for c in self.camera_names]
        if self.use_state:
            feats.append(state)
        return self.head(torch.cat(feats, dim=1))

    def n_params(self) -> int:
        """Trainable parameter count — Jetson budget depends on this.
        학습 파라미터 수. Jetson 예산이 여기 걸린다."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


@dataclass
class CheckpointMeta:
    """What a checkpoint must carry to be interpretable later.
    체크포인트가 나중에 해석되려면 함께 지녀야 하는 것."""

    camera_names: list[str]
    contract_version: str
    # Which space the head regresses. A checkpoint without this predates the
    # delta target and is absolute -- loading it as delta would add the state
    # twice and drive the arm to nonsense.
    # 헤드가 어느 공간을 회귀하는가. 이 값이 없는 체크포인트는 델타 목표 이전 것이라
    # 절대다. 델타로 불러오면 상태가 두 번 더해져 팔이 엉뚱하게 간다.
    action_space: str
    train_config: dict[str, Any]
    config_sha: str
    code_sha: str
    n_params: int
    n_episodes: int
    n_samples: int
    epochs_run: int
    best_val_loss: float
    trained_on: str  # "random_tensors" | dataset path
    note: str = (
        "val_loss 는 학습이 망가지지 않았는지 확인용이다. 성능 판정은 "
        "tools/eval_rollout.py 의 롤아웃 성공률이 한다."
    )


def save_checkpoint(path: Path, model: BCNet, meta: CheckpointMeta) -> Path:
    """Write weights and the metadata needed to load them correctly.
    가중치와, 그것을 올바로 불러오는 데 필요한 메타데이터를 쓴다."""
    from dataclasses import asdict

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "meta": asdict(meta)}, path)
    return path


class BCPolicy:
    """A trained BC checkpoint behind the `Policy` protocol.
    학습된 BC 체크포인트를 `Policy` 프로토콜 뒤에 둔 것.

    It sees only an `Observation` and returns an action in contract units, so
    the rollout harness scores it under exactly the same conditions as the
    baselines. Nothing about it is privileged.
    `Observation` 만 보고 계약 단위의 행동을 돌려준다. 그래서 롤아웃 harness 가
    baseline 과 **정확히 같은 조건**으로 채점한다. 특권 정보는 없다.
    """

    uses_privileged_state = False

    def __init__(self, ckpt_path: Path, device: str = "cpu") -> None:
        blob = torch.load(ckpt_path, map_location=device, weights_only=False)
        self.meta = blob["meta"]
        self.device = torch.device(device)
        self.model = BCNet(self.meta["camera_names"], self.meta["train_config"])
        self.model.load_state_dict(blob["state_dict"])
        self.model.to(self.device).eval()
        d = self.meta["train_config"]["data"]
        self._mean = float(d["image_mean"])
        self._std = float(d["image_std"])
        self.action_space = str(self.meta.get("action_space", "joint_absolute"))
        self._ckpt = Path(ckpt_path)
        # The head is linear, so the network can predict outside the contract's
        # [-1, 1]. Clipping keeps the action valid, and counting how often it
        # happens is a real signal: a model that constantly saturates has not
        # learned the action distribution.
        # 헤드가 선형이라 계약 범위 [-1, 1] 밖을 예측할 수 있다. 클립하면 행동은
        # 유효해지고, 얼마나 자주 그러는지 세는 것은 실제 신호다 — 계속 포화되는
        # 모델은 행동 분포를 배우지 못한 것이다.
        self.n_actions = 0
        self.n_clipped = 0

    @property
    def name(self) -> str:
        return "bc"

    def reset(self, seed: int | None = None) -> None:
        """BC is memoryless — only the clipping counters reset.
        BC 는 상태가 없다. 클립 카운터만 초기화한다."""
        self.n_actions = 0
        self.n_clipped = 0
        return None

    @torch.no_grad()
    def act(self, obs: Observation) -> np.ndarray:
        """One observation in, one contract-unit action out.
        관측 하나 받아 계약 단위 행동 하나를 낸다."""
        images = {}
        for cam in self.model.camera_names:
            if cam not in obs.images:
                raise KeyError(
                    f"체크포인트는 카메라 {self.model.camera_names} 를 기대하는데 "
                    f"관측에는 {sorted(obs.images)} 만 있다"
                )
            arr = torch.from_numpy(obs.images[cam].astype(np.float32) / 255.0)
            images[cam] = ((arr - self._mean) / self._std).unsqueeze(0).to(self.device)
        state = torch.from_numpy(np.asarray(obs.state, dtype=np.float32)).unsqueeze(0)
        state = state.to(self.device)
        raw = self.model(images, state)
        out = to_action(raw, state, self.action_space)
        action = out.squeeze(0).cpu().numpy().astype(np.float32)
        action = check_action(action, self.name)
        clipped = np.clip(action, -1.0, 1.0)
        self.n_actions += 1
        self.n_clipped += int(np.any(clipped != action))
        return clipped

    def describe(self) -> str:
        """One line naming what this checkpoint actually is.
        이 체크포인트가 실제로 무엇인지 한 줄로."""
        m = self.meta
        return (
            f"bc ckpt {self._ckpt.name} · 행동공간 {self.action_space} · "
            f"파라미터 {m['n_params']:,} · "
            f"학습대상 {m['trained_on']} · 에피소드 {m['n_episodes']} · "
            f"샘플 {m['n_samples']} · epochs {m['epochs_run']} · "
            f"best val_loss {m['best_val_loss']:.5f}"
        )

    def clip_report(self) -> str:
        """How often the raw prediction left the contract range.
        원 예측이 계약 범위를 벗어난 빈도."""
        if self.n_actions == 0:
            return "행동 없음"
        pct = 100.0 * self.n_clipped / self.n_actions
        verdict = " — **포화가 잦다. 행동 분포를 배우지 못했을 수 있다**" if pct > 20 else ""
        return f"계약 범위 초과로 클립된 행동 {self.n_clipped}/{self.n_actions} ({pct:.1f}%){verdict}"
