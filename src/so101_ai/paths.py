"""Filesystem anchors for the AI part — resolved once, imported everywhere.
AI 파트의 파일시스템 기준점. 한 번 정하고 모든 곳에서 임포트한다.

Every module used to compute its own `parents[n]` walk, which silently broke the
moment a file moved. One module owns it now.
예전에는 모듈마다 `parents[n]` 을 각자 계산했는데, 파일이 옮겨지는 순간 조용히
깨지는 구조였다. 이제 한 모듈이 책임진다.
"""

from __future__ import annotations

from pathlib import Path

# .../AI/src/so101_ai/paths.py -> parents[2] == .../AI
AI_ROOT: Path = Path(__file__).resolve().parents[2]

CONFIG_DIR: Path = AI_ROOT / "configs"
SCENE_DIR: Path = AI_ROOT / "scenes"
DOCS_DIR: Path = AI_ROOT / "docs"

# Official SO-101 MJCF/URDF. Vendored, never edited — replaced wholesale on update.
# 공식 SO-101 MJCF/URDF. 벤더링만 하고 수정하지 않는다. 업데이트 시 통째로 교체.
THIRD_PARTY: Path = AI_ROOT / "third_party"
SO101_MJCF_DIR: Path = THIRD_PARTY / "so101_mujoco" / "SO101"

DEFAULT_CONFIG: Path = CONFIG_DIR / "so101.yaml"
DEFAULT_SCENE: Path = SCENE_DIR / "pick_place.xml"
DEFAULT_EXP_LOG: Path = AI_ROOT / "EXP_LOG.jsonl"


def require(path: Path, what: str) -> Path:
    """Return `path`, or fail now with a message that names what is missing.
    `path` 를 반환하거나, 무엇이 없는지 이름을 붙여 지금 실패한다."""
    if not path.exists():
        raise FileNotFoundError(f"{what} 가 없다: {path}")
    return path
