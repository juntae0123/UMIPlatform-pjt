"""CLI entry point for `vlm.fp16_safety`.
`vlm.fp16_safety` 의 CLI 진입점.

AI/ 디렉터리에서 실행한다. 종료코드 1 은 그 모델이 M1 게이트를 통과하지 못했다는 뜻이다.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vlm.fp16_safety import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
