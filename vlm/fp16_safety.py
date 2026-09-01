"""Is a bf16 checkpoint safe to run in fp16 on Volta?
bf16 체크포인트를 Volta 에서 fp16 으로 돌려도 안전한가?

Why this exists.
왜 이것이 있는가.

Our GPUs are Tesla V100 (sm_70), which has no bf16 arithmetic — bf16 Tensor Cores
arrived with Ampere. Every small VLM we are considering ships as a bf16 checkpoint,
so we must cast to fp16. fp16 has the same mantissa width but a far smaller exponent
range, and Qwen-VL family models have documented inf/NaN and gibberish under fp16.
우리 GPU 는 Tesla V100(sm_70)이고 bf16 연산이 없다 — bf16 Tensor Core 는 Ampere 부터다.
검토 중인 소형 VLM 은 전부 bf16 체크포인트라 fp16 으로 캐스팅해야 한다. fp16 은 가수부는
같지만 지수 범위가 훨씬 좁고, Qwen-VL 계열에서 fp16 inf/NaN·gibberish 가 보고돼 있다.

**"되겠지"로 넘길 항목이 아니다.** 안 재고 학습을 시작하면, 손실이 이상해진 뒤에야
원인을 의심하게 되고 그때는 며칠이 지나 있다.

Gate (fixed before looking at any result) / 게이트 (결과 보기 전 확정):
    inf/NaN 0건  AND  argmax 일치율 >= 99%
    미달이면 그 모델은 후보에서 탈락한다.

⚠️ 이 코드는 GPU 가 있는 환경에서만 돌 수 있고, 작성 시점에 실행 검증되지 않았다.
   첫 실행에서 프로세서 관례 차이로 손볼 수 있다.
"""

from __future__ import annotations

import argparse
import gc
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tracking.exp_log import log_run  # noqa: E402

# 게이트 — 결과 보기 전에 확정됐다. 여기 숫자를 결과에 맞춰 고치면 그것은 판정이 아니다.
GATE_ARGMAX_AGREEMENT = 0.99
GATE_MAX_NONFINITE = 0

# 우리 데이터 규격과 같은 크기로 만든다 (image (3,224,224)).
IMAGE_SIZE = 224

PROMPTS: tuple[str, ...] = (
    "빨간 블록을 집어",
    "긁힘 있는 것만 골라내",
    "이거 지그에 넣어",
    "왼쪽 부품을 오른쪽 통에 옮겨",
    "제일 큰 것을 집어",
    "What do you see on the table?",
    "Pick up the red block.",
    "Which object is closest to the gripper?",
    "Sort the defective parts.",
    "Describe the scene in one sentence.",
)


@dataclass
class FP16Report:
    """One model's fp32 vs fp16 comparison.
    모델 하나의 fp32 대 fp16 비교 결과."""

    model_id: str
    n_samples: int
    n_nonfinite_fp16: int
    max_abs_diff: float
    mean_abs_diff: float
    argmax_agreement: float
    passed: bool
    device_name: str
    torch_dtype_fp16: str = "float16"

    def verdict(self) -> str:
        """One line a human can act on.
        사람이 바로 판단할 수 있는 한 줄."""
        if self.passed:
            return f"통과 — 일치율 {self.argmax_agreement:.4f}, inf/NaN {self.n_nonfinite_fp16}건"
        reasons = []
        if self.n_nonfinite_fp16 > GATE_MAX_NONFINITE:
            reasons.append(f"inf/NaN {self.n_nonfinite_fp16}건")
        if self.argmax_agreement < GATE_ARGMAX_AGREEMENT:
            reasons.append(f"일치율 {self.argmax_agreement:.4f} < {GATE_ARGMAX_AGREEMENT}")
        return "탈락 — " + " · ".join(reasons)


def _load_model(model_id: str, dtype: torch.dtype, device: str) -> Any:
    """Load a VLM under whichever auto class this transformers version exposes.
    이 transformers 버전이 제공하는 auto 클래스로 VLM 을 불러온다.

    The class name for vision-language models moved between transformers releases,
    so we try the current name first and fall back rather than pinning a version.
    비전-언어 모델의 auto 클래스 이름이 릴리스마다 바뀌어서, 버전을 고정하는 대신
    최신 이름부터 시도하고 폴백한다.
    """
    import transformers

    candidates = [
        "AutoModelForImageTextToText",
        "AutoModelForVision2Seq",
        "AutoModelForCausalLM",
    ]
    errors: list[str] = []
    for name in candidates:
        cls = getattr(transformers, name, None)
        if cls is None:
            continue
        try:
            model = cls.from_pretrained(
                model_id,
                dtype=dtype,
                attn_implementation="sdpa",  # flash_attention_2 는 sm80+ 전용이라 금지
                low_cpu_mem_usage=True,
            )
            return model.to(device).eval()
        except Exception as exc:  # noqa: BLE001 - 어느 클래스가 맞는지 미리 알 수 없다
            errors.append(f"{name}: {type(exc).__name__} {exc}")
    raise RuntimeError(
        f"{model_id} 를 어떤 auto 클래스로도 불러오지 못했다:\n  " + "\n  ".join(errors)
    )


def _build_inputs(processor: Any, n: int, seed: int = 0) -> list[dict[str, torch.Tensor]]:
    """Fixed, reproducible (image, prompt) pairs — same inputs for both dtypes.
    고정·재현 가능한 (이미지, 지시문) 쌍 — 두 dtype 에 같은 입력을 준다.

    Random images are deliberate: we are measuring numerical behaviour of the cast,
    not task accuracy. Real photos would add a variable we are not asking about.
    랜덤 이미지는 의도적이다. 여기서 재는 것은 캐스팅의 수치 거동이지 태스크 정확도가
    아니다. 실사진을 쓰면 묻지 않은 변수가 하나 늘어난다.
    """
    from PIL import Image

    rng = np.random.default_rng(seed)
    batches: list[dict[str, torch.Tensor]] = []

    for i in range(n):
        arr = rng.integers(0, 256, size=(IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
        img = Image.fromarray(arr)
        prompt = PROMPTS[i % len(PROMPTS)]

        text: str | None = None
        if hasattr(processor, "apply_chat_template"):
            try:
                msgs = [
                    {
                        "role": "user",
                        "content": [{"type": "image"}, {"type": "text", "text": prompt}],
                    }
                ]
                text = processor.apply_chat_template(msgs, add_generation_prompt=True)
            except Exception:  # noqa: BLE001 - 템플릿이 없는 모델도 있다
                text = None

        last_error: Exception | None = None
        for kwargs in (
            {"text": [text or prompt], "images": [img]},
            {"text": [text or prompt], "images": [[img]]},
            {"text": text or prompt, "images": img},
        ):
            try:
                batches.append(processor(return_tensors="pt", **kwargs))
                last_error = None
                break
            except Exception as exc:  # noqa: BLE001
                last_error = exc
        if last_error is not None:
            raise RuntimeError(f"프로세서 입력 구성 실패: {type(last_error).__name__} {last_error}")

    return batches


@torch.no_grad()
def _logits_for(model: Any, batches: list[dict[str, torch.Tensor]], device: str) -> list[torch.Tensor]:
    """Forward every batch and keep logits on CPU in float32.
    모든 배치를 forward 하고 로짓을 CPU float32 로 보관한다.

    Kept in float32 on CPU so the comparison itself never loses precision.
    비교 자체가 정밀도를 잃지 않도록 CPU float32 로 둔다.
    """
    out: list[torch.Tensor] = []
    for batch in batches:
        moved = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        logits = model(**moved).logits
        out.append(logits.detach().float().cpu())
    return out


def compare(model_id: str, device: str = "cuda", n_samples: int = 20, seed: int = 0) -> FP16Report:
    """Run the same inputs through fp32 and fp16 and report the difference.
    같은 입력을 fp32 와 fp16 에 각각 통과시키고 차이를 보고한다."""
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(model_id)
    batches = _build_inputs(processor, n_samples, seed=seed)

    # fp32 를 먼저 돌리고 메모리에서 내린다. 3B 모델이면 fp32 만 12GB 라 둘을 동시에
    # 올릴 이유가 없다.
    model32 = _load_model(model_id, torch.float32, device)
    logits32 = _logits_for(model32, batches, device)
    del model32
    gc.collect()
    torch.cuda.empty_cache()

    model16 = _load_model(model_id, torch.float16, device)
    logits16 = _logits_for(model16, batches, device)
    device_name = torch.cuda.get_device_name(0) if device.startswith("cuda") else "cpu"
    del model16
    gc.collect()
    torch.cuda.empty_cache()

    n_nonfinite = 0
    max_diff = 0.0
    diff_sum = 0.0
    diff_count = 0
    agree = 0
    total = 0

    for a, b in zip(logits32, logits16):
        n_nonfinite += int((~torch.isfinite(b)).sum().item())
        finite = torch.isfinite(a) & torch.isfinite(b)
        if finite.any():
            d = (a[finite] - b[finite]).abs()
            max_diff = max(max_diff, float(d.max()))
            diff_sum += float(d.sum())
            diff_count += int(d.numel())
        agree += int((a.argmax(-1) == b.argmax(-1)).sum().item())
        total += int(a.argmax(-1).numel())

    agreement = agree / total if total else 0.0
    passed = n_nonfinite <= GATE_MAX_NONFINITE and agreement >= GATE_ARGMAX_AGREEMENT

    return FP16Report(
        model_id=model_id,
        n_samples=n_samples,
        n_nonfinite_fp16=n_nonfinite,
        max_abs_diff=max_diff,
        mean_abs_diff=(diff_sum / diff_count) if diff_count else 0.0,
        argmax_agreement=agreement,
        passed=passed,
        device_name=device_name,
    )


def main() -> int:
    """CLI. Exit code 1 means the model failed the gate.
    CLI. 종료코드 1 은 그 모델이 게이트를 통과하지 못했다는 뜻이다."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="HuggingFace 모델 id")
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--author", default="김준태(트랙B)")
    parser.add_argument("--log", action="store_true")
    args = parser.parse_args()

    print(f"M1 fp16 안전성 검사 — {args.model}")
    print(f"게이트: inf/NaN <= {GATE_MAX_NONFINITE} AND argmax 일치율 >= {GATE_ARGMAX_AGREEMENT}")
    print("(결과 보기 전에 확정된 값이다)\n")

    rep = compare(args.model, device=args.device, n_samples=args.n, seed=args.seed)

    print(f"모델          {rep.model_id}")
    print(f"장치          {rep.device_name}")
    print(f"샘플          {rep.n_samples}")
    print(f"inf/NaN       {rep.n_nonfinite_fp16}")
    print(f"최대 절대차   {rep.max_abs_diff:.6f}")
    print(f"평균 절대차   {rep.mean_abs_diff:.6f}")
    print(f"argmax 일치율 {rep.argmax_agreement:.6f}")
    print(f"\n판정: {rep.verdict()}")

    if args.log:
        rec = log_run(
            experiment="vlm_fp16_safety",
            author=args.author,
            issue="S15P21A103-111",
            conditions={
                "model_id": rep.model_id,
                "n_samples": rep.n_samples,
                "seed": args.seed,
                "image_size": IMAGE_SIZE,
                "attn_implementation": "sdpa",
                "device_name": rep.device_name,
                "gate_argmax_agreement": GATE_ARGMAX_AGREEMENT,
                "gate_max_nonfinite": GATE_MAX_NONFINITE,
            },
            result=asdict(rep),
        )
        print(f"\nEXP_LOG.jsonl 기록 (git {rec['git_rev']})")

    return 0 if rep.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
