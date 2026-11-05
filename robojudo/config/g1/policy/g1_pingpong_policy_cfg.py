"""G1 config for the table-tennis policy (PingPongPolicy): HITTER-style whole-body controller, 107 inputs -> 29 joint targets at 50 Hz.

Joint order, default pose and PD gains are the training's (pingpong/robot/actuators.py: holosoma's G1 motor model), NOT the stock G1
values of RoboJuDo: the env takes them from `action_dof` (Environment.update_dof_cfg), so the motors run kp, kd and torque limits the
policy was trained with. The checkpoint's run (params/training.yaml) sets the planner and the swing library.
"""

import numpy as np

from robojudo.config import ASSETS_DIR
from robojudo.policy.policy_cfgs import PolicyCfg
from robojudo.tools.tool_cfgs import DoFConfig
from robojudo.utils.pingpong_path import ensure_pingpong

ensure_pingpong()
from pingpong.robot import actuators as ha  # noqa: E402

# the policy's joint order (the swing library's / Isaac Lab's: breadth first, left and right interleaved)
POLICY_JOINTS = [
    "left_hip_pitch_joint", "right_hip_pitch_joint", "waist_yaw_joint", "left_hip_roll_joint", "right_hip_roll_joint", "waist_roll_joint",
    "left_hip_yaw_joint", "right_hip_yaw_joint", "waist_pitch_joint", "left_knee_joint", "right_knee_joint", "left_shoulder_pitch_joint",
    "right_shoulder_pitch_joint", "left_ankle_pitch_joint", "right_ankle_pitch_joint", "left_shoulder_roll_joint", "right_shoulder_roll_joint",
    "left_ankle_roll_joint", "right_ankle_roll_joint", "left_shoulder_yaw_joint", "right_shoulder_yaw_joint", "left_elbow_joint",
    "right_elbow_joint", "left_wrist_roll_joint", "right_wrist_roll_joint", "left_wrist_pitch_joint", "right_wrist_pitch_joint",
    "left_wrist_yaw_joint", "right_wrist_yaw_joint",
]
_PERM = np.array([ha.JOINTS.index(j) for j in POLICY_JOINTS])


class PingPongDoF(DoFConfig):
    joint_names: list[str] = POLICY_JOINTS
    default_pos: list[float] | None = ha.DEFAULT[_PERM].tolist()
    stiffness: list[float] | None = ha.KP[_PERM].tolist()
    damping: list[float] | None = ha.KD[_PERM].tolist()
    torque_limits: list[float] | None = ha.TORQUE_LIMIT[_PERM].tolist()


# fix6-7 @16500: the real-robot pick (robust to 0-20 ms motor delay; settled gait). stage4-fix6-7, 2026-10-26. Kept in RoboJuDo's
# assets with its params/ (the player reads params/training.yaml next to it); where it came from: SOURCE.md there.
DEFAULT_CHECKPOINT = ASSETS_DIR / "motions" / "g1" / "pingpong_play" / "model_16500.pt"


class PingPongPolicyCfg(PolicyCfg):
    policy_type: str = "PingPongPolicy"
    robot: str = "g1"
    disable_autoload: bool = True  # the actor is loaded by PingPongPolicy (numpy), not as a torch.jit file

    freq: int = 50
    obs_dof: DoFConfig = PingPongDoF()
    action_dof: DoFConfig = PingPongDoF()

    checkpoint: str = str(DEFAULT_CHECKPOINT)
    """training checkpoint (<run>/model_N.pt): its run's params/training.yaml sets the planner and the swing library"""
    actor_path: str | None = None
    """load the actor from this .npz (pingpong.deploy.actor.Actor.export_npz) instead of the checkpoint: a computer without torch"""

    # --- what the mocap link delivers (the sim's OptiTrack preset) ---
    cam_hz: float = 240.0
    noise: float = 0.001  # m per axis: the estimator's gate and fit weights
    latency: float = 0.010  # s from capture to the estimator (schedules the first plan job)
    avoid_late: bool = False  # planner: a footwork step / stroke type that starts on time before a late swing (run_pingpong.py --avoid-late)
    planner_paddle_e: float | None = 0.65  # the paddle restitution the planner assumes (--planner-e); None: ball.yaml's. Between the drop test's
    # 0.74 and the robot's returns' 0.56 (2026-11-04): in simulation it holds up whichever of the two is real
    late_types: list[str] = []  # with avoid_late: for the balls late anyway, these stroke types first
    compute_scale: float = 1.0  # x the measured planner time (1 = this computer is the one that will run it); sim only
    threaded: bool = False  # estimator + planner jobs on a worker thread: True on the robot, so a 20 ms job never holds up the 50 Hz loop
    pelvis_extrapolate: bool = True  # carry the (10-30 ms old) mocap pelvis pose forward to the present
    pelvis_stale_s: float = 0.05  # a pelvis pose older than this counts as lost (the launcher stops the robot after a while)
    ball: str = "stickers"  # perception inside the player (g1_pingpong_inproc only): stickers | covered, as the sim world's ball_stickers
    seed: int = 0
    racket_z_offset: float = 0.0  # m added to the racket target's height in the OBSERVATION only (the arm sags below its target on the
    # robot: 4-6 cm in the 2026-11-02 shot log); the planner and the recorded targets keep the true height
    script_plans: str | None = None  # shadow swings: a shot log whose plans are replayed to the policy without a ball (run_pingpong.py
    # --script-plans; pingpong/deploy/script_plans.py)
    manual_port: int | None = None  # manual planner: swings typed in pingpong scripts/deploy/planner_console.py, UDP to this port
    # (run_pingpong.py --manual; pingpong/deploy/manual_plans.py)
    manual_host: str = "127.0.0.1"  # the interface it listens on (0.0.0.0: a console on another computer)
    record_shots: bool = False  # the runtime's shot recorder: ball samples, plans, paddle face per step (run_pingpong.py --log x.json)
