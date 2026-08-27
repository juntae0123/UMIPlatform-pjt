"""Map where on the table a top-down grasp is kinematically possible.
작업대 위에서 수직 파지가 기구학적으로 가능한 영역을 지도로 만든다.

This is the feasibility check that has to exist BEFORE any demonstration is
collected. If a human demonstrates a motion the arm cannot reproduce, the whole
episode is unusable — and finding that out after collecting 100 of them means
throwing away 100. Run this first, then constrain where objects are placed.
이건 시연을 수집하기 **전에** 있어야 하는 실행가능성 검사다. 로봇이 재현할 수
없는 동작을 사람이 시연하면 그 에피소드는 통째로 못 쓴다. 100개 찍고 나서
알게 되면 100개를 버린다. 이걸 먼저 돌리고, 물체를 놓을 범위를 그에 맞춰 제한한다.

Kinematic reachability only. It says nothing about whether the grasp succeeds —
that is grasp_check.py. A point can be reachable and still fail to grasp.
기구학적 도달성만 본다. 파지 성공 여부는 말해주지 않는다 — 그건 grasp_check.py다.
도달 가능해도 파지에 실패할 수 있다.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from sim.mujoco.build_scene import DEFAULT_CONFIG, build_model, load_config
from tracking.exp_log import file_digest, log_run
from sim.mujoco.kinematics import solve_pose_ik


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--x-range", type=float, nargs=2, default=(0.05, 0.40))
    parser.add_argument("--y-range", type=float, nargs=2, default=(-0.20, 0.20))
    parser.add_argument("--step", type=float, default=0.02, help="grid step, m / 격자 간격")
    parser.add_argument("--wrist-roll", type=float, default=0.0,
                        help="held fixed, matching grasp_check; pass nan to let IK use it "
                             "/ grasp_check 과 동일하게 고정. nan 을 주면 IK 가 자유변수로 쓴다")
    parser.add_argument("--out", type=Path, default=Path("out"))
    parser.add_argument("--author", type=str, default="김준태(트랙B)")
    parser.add_argument("--log", action="store_true")
    args = parser.parse_args()

    cfg = load_config()
    model = build_model(cfg)
    g = cfg["grasp"]
    offset = np.asarray(g["pinch_offset_local"], dtype=float)
    axis = np.asarray(g["approach_axis"], dtype=float)
    half_z = float(cfg["task"]["object"]["half_size_m"][2])
    z = half_z + float(g["grasp_z_offset_m"])

    # The map must be drawn under the SAME wrist_roll policy the grasp script uses,
    # or it advertises reachability the actual pick cannot deliver.
    # 지도는 파지 스크립트와 **같은** wrist_roll 정책으로 그려야 한다.
    # 아니면 실제 파지가 못 하는 도달성을 광고하게 된다.
    wrist_roll = None if np.isnan(args.wrist_roll) else float(args.wrist_roll)

    xs = np.arange(args.x_range[0], args.x_range[1] + 1e-9, args.step)
    ys = np.arange(args.y_range[0], args.y_range[1] + 1e-9, args.step)
    grid = np.zeros((len(ys), len(xs)), dtype=bool)
    pos_err = np.full((len(ys), len(xs)), np.nan)

    for iy, y in enumerate(ys):
        for ix, x in enumerate(xs):
            res = solve_pose_ik(model, np.array([x, y, z]), offset, axis, wrist_roll=wrist_roll)
            grid[iy, ix] = res.ok
            pos_err[iy, ix] = res.pos_error_m * 1000.0

    n_ok = int(grid.sum())
    total = grid.size
    print(f"수직 파지 도달 가능: {n_ok}/{total} = {n_ok / total * 100:.1f}%")
    print(f"격자 x[{xs[0]:.2f},{xs[-1]:.2f}] y[{ys[0]:.2f},{ys[-1]:.2f}] step={args.step}m, 파지높이 z={z:.4f}m")
    print(f"wrist_roll={'자유(IK 변수)' if wrist_roll is None else f'{wrist_roll:.4f} 고정'}"
          "   ⚠️ 기구학만 본다. 테이블·자기충돌·물체 충돌은 검사하지 않는다.")
    print()
    header = "      " + "".join(f"{x * 100:5.0f}" for x in xs)
    print(f"{'y[cm]':>6}{header[6:]}")
    for iy, y in enumerate(ys):
        row = "".join("    O" if v else "    ." for v in grid[iy])
        print(f"{y * 100:6.0f}{row}")
    print("\nO = 도달 가능 (위치오차<5mm, 접근축오차<5deg, 관절한계 내),  . = 불가")

    # Largest axis-aligned rectangle of reachable cells: where to place objects.
    # 도달 가능 셀로 이루어진 최대 사각형 — 물체를 놓아도 되는 범위.
    best = (0, None)
    for y0 in range(len(ys)):
        for y1 in range(y0, len(ys)):
            for x0 in range(len(xs)):
                for x1 in range(x0, len(xs)):
                    block = grid[y0 : y1 + 1, x0 : x1 + 1]
                    if block.all():
                        area = block.size
                        if area > best[0]:
                            best = (area, (xs[x0], xs[x1], ys[y0], ys[y1]))
    if best[1]:
        bx0, bx1, by0, by1 = best[1]
        print(
            f"\n최대 연속 가능 영역: x[{bx0:.3f},{bx1:.3f}] y[{by0:.3f},{by1:.3f}] "
            f"= {(bx1 - bx0) * 100:.0f}cm x {(by1 - by0) * 100:.0f}cm"
        )

    args.out.mkdir(parents=True, exist_ok=True)
    np.savez(args.out / "reach_grid.npz", xs=xs, ys=ys, reachable=grid, pos_err_mm=pos_err)
    print(f"격자 저장: {args.out / 'reach_grid.npz'}")

    if args.log:
        rec = log_run(
            experiment="reach_scan",
            author=args.author,
            issue="S15P21A103-63",
            conditions={
                "x_range": list(args.x_range),
                "y_range": list(args.y_range),
                "step_m": args.step,
                "grasp_height_m": z,
                "config_sha": file_digest(DEFAULT_CONFIG),
                "criterion": "pos_err<5mm AND axis_err<5deg AND 반복 중 관절한계에 걸리지 않음",
                "wrist_roll": args.wrist_roll,
            },
            result={
                "reachable_fraction": n_ok / total,
                "n_reachable": n_ok,
                "n_total": total,
                "largest_rect": None if not best[1] else [float(v) for v in best[1]],
            },
        )
        print(f"EXP_LOG.jsonl 기록 (git {rec['git_rev']})")


if __name__ == "__main__":
    main()
