"""PingPongPolicy: the table-tennis controller (HITTER-style whole-body policy + ball estimator + planner + swing clock) as a RoboJuDo
policy. All of the logic is pingpong.deploy.PingPongRuntime (the same classes scripts/main/live_mujoco.py and
scripts/deploy/sim_runtime.py run in MuJoCo); this class only maps RoboJuDo's data onto it:

    ctrl_data["PingPongPerceptionCtrl"]  {"msgs": [PerceptionMsg], "t": now}: the perception program's table-frame pelvis pose, ball
                                         samples and shot events (single-process simulations: PingPongSimWorldCtrl's raw frames instead)
    env_data                        joint angles / speeds (policy order, the pipeline's adapter), pelvis IMU orientation + gyro
    get_observation -> the 107 inputs (the plans usable now are applied first)      get_action -> joint targets - default pose
    post_step_callback -> the strike clock advances one step (only in the main loop, not in the ramp / blend phases of prepare())

The ramp target of prepare() is the swing library's ready pose (get_init_dof_pos), not the default standing pose: the policy was always
started from the ready pose. While no pelvis pose has arrived the previous joint targets are held.
"""

import logging

import numpy as np

from robojudo.config.g1.policy.g1_pingpong_policy_cfg import PingPongPolicyCfg
from robojudo.policy import Policy, policy_registry
from robojudo.utils.pingpong_path import ensure_pingpong

logger = logging.getLogger(__name__)


@policy_registry.register
class PingPongPolicy(Policy):
    cfg_policy: PingPongPolicyCfg

    def __init__(self, cfg_policy: PingPongPolicyCfg, device: str = "cpu"):
        super().__init__(cfg_policy=cfg_policy, device=device)
        ensure_pingpong()
        from pingpong.deploy.runtime import PingPongRuntime, RuntimeConfig

        c = cfg_policy
        rcfg = RuntimeConfig(dt=1.0 / c.freq, cam_hz=c.cam_hz, noise=c.noise, latency=c.latency, compute_scale=c.compute_scale, threaded=c.threaded,
                             pelvis_extrapolate=c.pelvis_extrapolate, pelvis_timeout=c.pelvis_stale_s, ball=c.ball, record_shots=c.record_shots,
                             racket_z_offset=c.racket_z_offset, script_plans=c.script_plans,
                             manual_port=c.manual_port, manual_host=c.manual_host, avoid_late=c.avoid_late, late_types=tuple(c.late_types))
        self.rt = PingPongRuntime(c.checkpoint, None, rcfg, actor_path=c.actor_path, seed=c.seed)
        if list(self.rt.lib.joint_names) != list(c.obs_dof.joint_names):
            raise ValueError("the swing library's joint order differs from PingPongPolicyCfg.obs_dof (the checkpoint's run changed?)")
        if not np.allclose(self.rt.jmap.default, self.default_pos):
            raise ValueError("pingpong.robot.actuators.DEFAULT differs from PingPongPolicyCfg.action_dof.default_pos")
        self.ready_q = self.rt.lib.ready_q[0].copy()
        self.n_obs = self.rt.actor.obs_dim
        logger.info(f"[PingPongPolicy] {c.checkpoint}: {self.n_obs} inputs; planner and estimator compiled")
        self.reset()

    # ---- RoboJuDo Policy interface ----------------------------------------------------------------------------------------
    def reset(self):
        self.rt.reset()
        self.last_action = np.zeros(self.num_actions)
        self._ok = False
        self._hold = self.ready_q.copy()  # the last joint targets (policy order): held while the pelvis pose is missing
        self.last_info = {}

    def close(self):
        self.rt.close()

    def get_init_dof_pos(self) -> np.ndarray:
        return self.ready_q.copy()

    def get_observation(self, env_data, ctrl_data):
        if "PingPongPerceptionCtrl" in ctrl_data:  # the perception program's messages (the robot, and simulations run like it)
            m = ctrl_data["PingPongPerceptionCtrl"]
            for msg in m["msgs"]:
                self.rt.on_perception(msg)
        elif "PingPongSimWorldCtrl" in ctrl_data:  # a single-process simulation: perception runs here, on the world's raw frames
            m = ctrl_data["PingPongSimWorldCtrl"]
            if self.rt.frame is None and m.get("table_frame") is not None:
                self.rt.set_frame(m["table_frame"])
            for f in m["frames"]:
                self.rt.on_frame(f)
        else:
            raise RuntimeError("PingPongPolicy needs a PingPongPerceptionCtrl (or, in one-process simulations, a PingPongSimWorldCtrl)")
        from pingpong.deploy.table_frame import quat_xyzw_to_mat

        R = quat_xyzw_to_mat(env_data.base_quat)
        obs, info = self.rt.observe(float(m["t"]), env_data.dof_pos, env_data.dof_vel, env_data.base_ang_vel, R.T @ np.array([0.0, 0.0, -1.0]))
        self._ok = obs is not None
        self.last_info = info
        return (obs if self._ok else np.zeros(self.n_obs)), {"CALLBACK": [], "pingpong": info}

    def get_action(self, obs: np.ndarray) -> np.ndarray:
        """Joint targets minus the default pose (the pipeline adds default_pos back)."""
        if self._ok:
            self.last_action, self._hold = self.rt.act(obs)
        return self._hold - self.default_pos

    def post_step_callback(self, commands: list[str] | None = None):
        if self._ok:
            self.rt.advance()
