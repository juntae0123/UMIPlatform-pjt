"""Simulation: scene build, kinematics, environment, grasp/workspace probes.
시뮬레이션: 씬 구성, 기구학, 환경, 파지·작업공간 계측.

Everything MuJoCo-specific lives here. A policy must not import from this
package — it only sees `so101_ai.sim.env.RobotEnv`'s surface.
MuJoCo 고유한 것은 전부 여기 있다. 정책은 이 패키지를 임포트해서는 안 된다.
정책이 보는 것은 `so101_ai.sim.env.RobotEnv` 의 표면뿐이다.
"""
