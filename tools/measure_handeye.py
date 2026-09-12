"""Solve T_cam->pinch translation from the demonstration video itself.
시연 영상 자체에서 T_cam->pinch 의 평행이동을 푼다.

사전등록: docs/PREREG_handeye_from_video_0912.md
이슈: S15P21A103-28, S15P21A103-31

왜 있는가.

HW 가 도면 기준 파지점 `(0, +24.6, +176.3)mm` 를 줬다 🔵. 그런데 1프레임 대조에서
**Y 부호가 반대**였다 🟢 — 투영하면 화면 아래인데 실제 마커는 화면 위에 있다.

hand-eye 는 상수 편향이라 학습이 지우지 못한다 (±10mm 에서 재생 성공률 3/4 🟢).
**그리고 파이프라인은 그대로 통과한다** — 손실은 잘 떨어지고 실물에서만 실패한다.
그래서 회신을 기다리는 대신 영상에서 직접 푼다. 필요한 것이 다 있다:

  매 프레임 마커 2개  +  episode.json 의 intrinsics  +  gap->물리간격 교정식

푸는 방법:
    dist_mm = 37.6 + gap/70 * (134 - 37.6)          황도경 교정식 🔵
    Z = fx * dist_mm / dist_px                      닮은꼴로 깊이
    X = (u_mid - cx) * Z / fx   ·  Y = (v_mid - cy) * Z / fy

마커 둘이 턱에 좌우 대칭이므로 중점이 그리퍼 중심선 위다. 접근축 방향 오프셋만 남는다 🟡.

**6 DOF 중 5 개만 나온다.** 마커축 기준 롤은 이 방법으로 못 구한다 — 도면이 필요하다.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.make_gripper_csv import detect  # noqa: E402  같은 검출기를 그대로 쓴다

# HW 가 도면 기준으로 준 값 🔵 (OpenCV 축, mm). 대조 대상이지 입력이 아니다.
HW_CLAIM_MM = (0.0, 24.6, 176.3)

# 사전등록 게이트 — 결과 보기 전에 확정했다.
GATE_ABS_X_MM = 5.0
GATE_EPISODE_STD_MM = 5.0
GATE_DZ_DGAP = 0.3
REFERENCE_GAP_MM = 60.0


def _intrinsics(ep: Path) -> tuple[float, float, float, float]:
    meta = json.loads((ep / "episode.json").read_text(encoding="utf-8"))
    k = meta["camera"]["intrinsics"]
    return float(k["fx"]), float(k["fy"]), float(k["cx"]), float(k["cy"])


def _quat_to_R(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    """xyzw -> (3,3). ARCore 는 scalar-last 다 (episode.json quaternion_order=xyzw)."""
    n = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    if n == 0.0:
        return np.eye(3)
    qx, qy, qz, qw = qx / n, qy / n, qz / n, qw / n
    return np.array([
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
        [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
        [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
    ])


# ARCore(OpenGL: x 오른쪽 · y 위 · -z 앞) -> OpenCV(x 오른쪽 · y 아래 · z 앞)
GL_TO_CV = np.diag([1.0, -1.0, -1.0])


def episode_points(ep: Path, draft: int) -> list[tuple[int, float, float, float, float]]:
    """(frame_index, gap_mm, X, Y, Z) in camera OpenCV mm, for detected frames."""
    fx, fy, cx, cy = _intrinsics(ep)
    out: list[tuple[int, float, float, float, float]] = []
    for f in sorted((ep / "frames").glob("*.jpg")):
        r = detect(f, draft)
        if r.gap_mm is None or r.mid_u is None or r.dist_px is None:
            continue
        # detect() 가 draft 로 줄인 픽셀을 낸다 -- intrinsics 와 맞추려면 되돌린다.
        u, v, dpx = r.mid_u * draft, r.mid_v * draft, r.dist_px * draft
        dist_mm = 37.6 + r.gap_mm / 70.0 * (134.0 - 37.6)
        z = fx * dist_mm / dpx
        out.append((r.index, r.gap_mm,
                    (u - cx) * z / fx, (v - cy) * z / fy, z))
    return out


def approach_axis(ep: Path, close_index: int | None) -> np.ndarray | None:
    """Camera-frame direction the gripper travels while approaching. 🟡 보조 판정용."""
    rows = list(csv.DictReader((ep / "poses.csv").open(encoding="utf-8")))
    if len(rows) < 8:
        return None
    end = close_index if close_index else int(len(rows) * 0.5)
    end = max(4, min(end, len(rows) - 1))
    acc = np.zeros(3)
    n = 0
    for i in range(1, end):
        a, b = rows[i - 1], rows[i]
        d = np.array([float(b["x"]) - float(a["x"]),
                      float(b["y"]) - float(a["y"]),
                      float(b["z"]) - float(a["z"])])
        if np.linalg.norm(d) < 1e-5:
            continue
        r_gl = _quat_to_R(float(a["qx"]), float(a["qy"]), float(a["qz"]), float(a["qw"]))
        acc += GL_TO_CV @ (r_gl.T @ d)
        n += 1
    if n == 0 or np.linalg.norm(acc) < 1e-9:
        return None
    return acc / np.linalg.norm(acc)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bundle-root", type=Path, required=True)
    ap.add_argument("--draft", type=int, default=1,
                    help="JPEG 1/N 디코드. 픽셀은 되돌려 쓰므로 결과 단위는 같다")
    ap.add_argument("--limit", type=int, default=0, help="앞에서 N 편만 (0=전부)")
    ap.add_argument("--out", type=Path, default=Path("out") / "handeye_points.csv")
    args = ap.parse_args()

    eps = sorted(p for p in args.bundle_root.glob("rec_*") if (p / "frames").is_dir())
    if args.limit:
        eps = eps[:args.limit]
    if not eps:
        print(f"✗ 에피소드가 없다: {args.bundle_root}")
        return 2

    intr = {_intrinsics(e) for e in eps}
    print(f"에피소드 {len(eps)} · intrinsics 종류 {len(intr)}")
    if len(intr) != 1:
        print("⚠️ 편마다 intrinsics 가 다르다 — 섞어서 적합하면 안 된다 (사전등록 보조판정 5)")
        for k in sorted(intr):
            print(f"   fx {k[0]:.3f} fy {k[1]:.3f} cx {k[2]:.3f} cy {k[3]:.3f}")

    rows: list[tuple[str, int, float, float, float, float]] = []
    per_ep: list[tuple[str, float, float, float, int]] = []
    axes: list[np.ndarray] = []
    for i, ep in enumerate(eps, 1):
        pts = episode_points(ep, args.draft)
        if not pts:
            print(f"  {ep.name}: 검출 0 — 건너뛴다")
            continue
        arr = np.array([[p[1], p[2], p[3], p[4]] for p in pts])
        close_idx = None
        gaps = arr[:, 0]
        th = (gaps.max() + gaps.min()) / 2.0
        below = np.nonzero(gaps < th)[0]
        if below.size:
            close_idx = int(pts[int(below[0])][0])
        ax = approach_axis(ep, close_idx)
        if ax is not None:
            axes.append(ax)
        per_ep.append((ep.name, float(np.median(arr[:, 1])), float(np.median(arr[:, 2])),
                       float(np.median(arr[:, 3])), len(pts)))
        rows.extend((ep.name, p[0], p[1], p[2], p[3], p[4]) for p in pts)
        if i % 10 == 0 or i == len(eps):
            print(f"  {i}/{len(eps)}", flush=True)

    if not rows:
        print("✗ 검출된 프레임이 없다")
        return 2

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["episode", "frame_index", "gap_mm", "x_mm", "y_mm", "z_mm"])
        w.writerows(rows)

    a = np.array([[r[2], r[3], r[4], r[5]] for r in rows])   # gap, X, Y, Z
    gap = a[:, 0]
    print(f"\n프레임 {len(a)} · gap {gap.min():.1f}~{gap.max():.1f}mm\n")

    print(f"{'':6s}{'중앙값':>10s}{'표준편차':>10s}{'gap 기울기':>12s}{'gap=60 적합':>12s}")
    fit_at_ref: dict[str, float] = {}
    slopes: dict[str, float] = {}
    for j, name in enumerate("XYZ", start=1):
        v = a[:, j]
        slope, intercept = np.polyfit(gap, v, 1)
        at_ref = slope * REFERENCE_GAP_MM + intercept
        fit_at_ref[name] = float(at_ref)
        slopes[name] = float(slope)
        print(f"{name:6s}{np.median(v):10.1f}{v.std():10.1f}{slope:12.3f}{at_ref:12.1f}")

    ep_arr = np.array([[p[1], p[2], p[3]] for p in per_ep])
    print(f"\n편별 중앙값의 표준편차 (n={len(per_ep)}편)")
    for j, name in enumerate("XYZ"):
        print(f"  {name}  {ep_arr[:, j].std():6.2f} mm")

    print(f"\nHW 도면값 🔵        X {HW_CLAIM_MM[0]:+7.1f}  Y {HW_CLAIM_MM[1]:+7.1f}  Z {HW_CLAIM_MM[2]:+7.1f} mm")
    print(f"영상 실측 (gap 60)  X {fit_at_ref['X']:+7.1f}  Y {fit_at_ref['Y']:+7.1f}  Z {fit_at_ref['Z']:+7.1f} mm")
    print(f"차이                X {abs(fit_at_ref['X'] - HW_CLAIM_MM[0]):7.1f}  "
          f"Y {abs(fit_at_ref['Y'] - HW_CLAIM_MM[1]):7.1f}  "
          f"Z {abs(fit_at_ref['Z'] - HW_CLAIM_MM[2]):7.1f} mm")

    if axes:
        ax = np.array(axes)
        m = ax.mean(axis=0)
        m = m / np.linalg.norm(m)
        spread = float(np.degrees(np.arccos(np.clip(ax @ m, -1, 1))).std())
        marker_axis = np.array([1.0, 0.0, 0.0])   # 마커는 이미지 가로로 벌어진다
        ang = float(np.degrees(np.arccos(abs(float(m @ marker_axis)))))
        print(f"\n접근축 추정 🟡 (편 {len(axes)}개 평균)  "
              f"[{m[0]:+.3f} {m[1]:+.3f} {m[2]:+.3f}]  편간 산포 {spread:.1f}°")
        print(f"  마커축과의 사잇각 {ang:.1f}°  "
              + ("OK (90±10)" if abs(ang - 90) <= 10 else "**벗어남 — 이 절은 버린다**"))

    print("\n판정")
    ok = True
    if abs(fit_at_ref["X"]) <= GATE_ABS_X_MM:
        print(f"  [대칭]     |X| {abs(fit_at_ref['X']):.1f} <= {GATE_ABS_X_MM} → 통과")
    else:
        ok = False
        print(f"  [대칭]     |X| {abs(fit_at_ref['X']):.1f} > {GATE_ABS_X_MM} → **실패**. 도면 없이 확정하지 않는다")
    std_max = float(ep_arr.std(axis=0).max())
    if std_max <= GATE_EPISODE_STD_MM:
        print(f"  [안정성]   편별 표준편차 최대 {std_max:.2f} <= {GATE_EPISODE_STD_MM} → 통과")
    else:
        ok = False
        print(f"  [안정성]   편별 표준편차 최대 {std_max:.2f} > {GATE_EPISODE_STD_MM} → **실패**. "
              f"영상만으로 ±5mm 를 못 맞춘다")
    if abs(slopes["Z"]) <= GATE_DZ_DGAP:
        print(f"  [단일 TCP] |dZ/dgap| {abs(slopes['Z']):.3f} <= {GATE_DZ_DGAP} → 통과. TCP 하나로 쓴다")
    else:
        print(f"  [단일 TCP] |dZ/dgap| {abs(slopes['Z']):.3f} > {GATE_DZ_DGAP} → gap 의존 TCP 가 필요하다")
    if fit_at_ref["Y"] < 0:
        print("  [부호]     Y < 0 → **HW 도면값의 부호가 뒤집혀 있다** (보조판정 1)")
    else:
        print("  [부호]     Y > 0 → HW 값과 같은 부호. 1프레임 대조를 다시 본다 (보조판정 2)")

    print(f"\n원자료: {args.out}")
    print("⚠️ 이 측정은 6 DOF 중 5 개만 준다. 마커축 기준 롤은 도면이 필요하다.")
    print("⚠️ 마커 중점 = 파지점으로 가정했다. 접근축 방향 오프셋은 안 나온다 🟡")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
