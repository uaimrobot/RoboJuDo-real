"""The table-tennis PLAYER (estimator, planner, swing clock, policy) on MuJoCo or the real G1. Perception is its own program.

Dual deployment: two programs, the same in simulation and on the robot (see docs/g1_pingpong_deployment.md):

    1. perception (pingpong folder):  python scripts/deploy/run_perception.py run --calibrate-first
    2. this player (RoboJuDo):        python scripts/run_pingpong.py -c g1_pingpong [--window --realtime]   # simulation: serves without end
                                      python scripts/run_pingpong.py -c g1_pingpong_real [--dry-run]        # the robot
    the policy alone, you as the planner:  add --manual, and type swings in the pingpong folder's scripts/deploy/planner_console.py

Either may start first. The player waits (--wait-perception s) for perception to report the pelvis before anything moves: in simulation the
world stands still meanwhile (the robot in its ready pose, no ball) and its cameras run, so that perception can calibrate the table; on
the robot nothing is commanded until then. -c g1_pingpong_inproc: one process, no perception program (fast batch runs in simulation).

Real robot: start it hanging from the gantry (feet just off the floor), then lower it once the policy holds the ready pose. prepare() ramps
the joints from wherever they are to the swing library's READY pose (not the default standing pose: the policy was always started there)
in --prepare-seconds, then blends the policy in over the same time; the policy holds the ready pose while no ball comes.

Stops. On the robot every stop is RoboJuDo's guard stop (rl_pipeline.stop_with_guard_pose: legs and waist to zero stiffness with
damping, at once; arms to the guard pose in 0.4 s). The guard is carried out by the control loop each tick and never clears by
itself, so after a stop this program keeps the loop running, holding it, until Ctrl-C; then it ends and the motors are released
(env.shutdown). What stops it:
    the remote: A, Select or Start (triggers in g1_pingpong_real) - during play and during the start-up ramp and blend (read from
                the robot's raw remote state there: RoboJuDo's prepare() does not read the remote in its ramp)
    Ctrl-C      in the terminal (a second Ctrl-C then ends the program; --log's file is written at the stop and again at the end)
    watchdogs   no fresh pelvis pose from perception for --stale-steps control steps; pelvis below 0.45 m or tilted more than
                --max-tilt-deg; a NaN in the targets
A stop needs this program running: if it is killed or hangs, the last command stays. Unitree's own stop (power) is the last resort.

Pelvis height check (pingpong/deploy/pelvis_height.py). While the robot stands on flat feet, the legs (joint angles + IMU) tell the pelvis
origin's true height above the floor; perception's pelvis z should match it. Every 5 s and at the end the player logs both and what is
left over, which is Motive's pivot of the pelvis rigid body sitting off the pelvis origin: give it to perception as
`run_perception.py ... --pelvis-pivot 0 0 <Z>` (added to the value set now, if any). Valid only with the robot's weight on its feet:
hanging in the gantry with level feet looks the same to the legs.

Simulation viewer (--window): the ball orange, the racket target blue (during a swing), the station green, home grey.
"""

import os

# one math thread per process (set before numpy loads): the actor's matrix products, the planner and the simulator otherwise fight over
# the cores through a multi-threaded BLAS, and a 0.1 ms product takes 8 ms now and then
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import sys as _sys

if "--window" in _sys.argv and os.environ.get("MUJOCO_GL") in ("egl", "osmesa"):
    os.environ.pop("MUJOCO_GL")  # an off-screen GL backend and a window do not mix: the viewer opens its own (GLFW) context

import argparse
import json
import logging
import sys
import threading
import time

import struct

import numpy as np

import robojudo.pipeline  # noqa: F401  (torch before mujoco, see run_pipeline_prepared.py)
from robojudo.config.config_manager import ConfigManager
from robojudo.pipeline.rl_pipeline import RlPipeline

logger = logging.getLogger("robojudo")


class PelvisHeightCheck:
    """Perception's pelvis z against the legs' (forward kinematics on joint angles + IMU), on steps with both soles flat on one level."""

    EVERY = 5  # control steps between samples (the FK is pure python, ~0.2 ms)

    def __init__(self, env):
        from pingpong.deploy.pelvis_height import LegFK

        self.env, self.fk, self.n = env, LegFK(), 0
        self.rows = []  # (legs, perception, cos tilt, truth or nan)
        self.n_not_flat = 0

    def step(self, info, sim):
        """Never raises (an error in the control loop is a guard stop on the robot): a failing check logs once and turns itself off."""
        if self.fk is None:
            return
        try:
            self._step(info, sim)
        except Exception as e:  # noqa: BLE001
            logger.warning(f"[player] pelvis height check off: {e!r}")
            self.fk = None

    def _step(self, info, sim):
        self.n += 1
        if self.n % self.EVERY or "pelvis" not in info:
            return
        q = self.env.base_quat
        h, flat = self.fk.pelvis_height(self.env.joint_names, self.env.dof_pos, q)
        if not flat:
            self.n_not_flat += 1
            return
        x, y = q[0], q[1]
        cz = 1.0 - 2.0 * (x * x + y * y)  # the pelvis z axis's vertical component
        self.rows.append((h, float(info["pelvis"][2]), cz, float(self.env.data.qpos[2]) if sim else np.nan))

    def summary(self, last=None):
        r = np.array(self.rows[-last:] if last else self.rows)
        if not len(r):
            return f"pelvis height: no step with both feet flat yet ({self.n_not_flat} samples with the feet not level)"
        legs, perc, cz = np.median(r[:, 0]), np.median(r[:, 1]), np.median(r[:, 2])
        d = (legs - perc) / max(cz, 0.5)
        msg = (f"pelvis height (feet flat, {len(r)} samples): legs {legs:.3f} m, perception {perc:.3f} m (spread "
               f"{np.std(r[:, 1] - r[:, 0]) * 1000:.1f} mm)")
        if np.isfinite(r[:, 3]).all():
            msg += f", truth {np.median(r[:, 3]):.3f} m"
        if abs(d) < 0.005:
            return msg + " -> agree (within 5 mm)"
        return (msg + f" -> perception reads the pelvis {abs(d) * 100:.1f} cm {'LOW' if d > 0 else 'HIGH'}: Motive's pivot is off the pelvis "
                f"origin; perception: --pelvis-pivot 0 0 {d:+.3f} (added to its pivot z now, if one is set). Valid only standing on the feet")


class ActionMonitor:
    """The dry run's eyes. Warns (once per kind) on a NaN, a torque the PD law would ask beyond the motor's limit, or a big jump between
    two steps; counts, per joint, how often and by how much the target lies outside the joint range (the policy's PD targets do, in
    training too: the joint sits on its stop and the motor pushes into it with kp x the excess)."""

    def __init__(self, env, policy):
        self.env, self.policy, self.prev, self.n = env, policy, None, 0
        self.torque_over = np.zeros(env.num_dofs)
        self.out_n = np.zeros(env.num_dofs)
        self.out_max = np.zeros(env.num_dofs)
        self.warned = set()

    def _warn(self, kind, msg):
        if kind not in self.warned:
            self.warned.add(kind)
            logger.warning(f"[ActionMonitor] step {self.n}: {msg} (further ones counted, not printed)")

    def check(self, target_env, q, dq):
        ok = True
        if not np.all(np.isfinite(target_env)):
            self._warn("nan", "NaN / inf in the joint targets")
            ok = False
        pl = getattr(self.env, "position_limits", None)
        if pl is not None and np.ndim(pl) == 2:
            exc = np.maximum(pl[:, 0] - target_env, 0.0) + np.maximum(target_env - pl[:, 1], 0.0)
            self.out_n += exc > 1e-3
            self.out_max = np.maximum(self.out_max, exc)
        tau = (target_env - q) * self.env.stiffness - dq * self.env.damping
        over = np.abs(tau) > self.env.torque_limits * 1.001
        self.torque_over += over
        if over.sum() > 4:
            self._warn("torque", f"{int(over.sum())} joints would exceed their torque limit")
        if self.prev is not None and np.max(np.abs(target_env - self.prev)) > 1.0:
            self._warn("jump", f"target jump {np.max(np.abs(target_env - self.prev)):.2f} rad in one step")
        self.prev = np.array(target_env)
        self.n += 1
        return ok

    def report(self):
        rows = [(self.env.joint_names[i], self.out_n[i] / max(self.n, 1), self.out_max[i], self.out_max[i] * self.env.stiffness[i], self.env.torque_limits[i])
                for i in np.argsort(-self.out_max) if self.out_n[i] > 0]
        if not rows:
            return
        print("\njoint targets outside the joint range (share of steps, largest excess, motor pushing into the stop at that excess = kp x excess):")
        for name, share, mx, tq, lim in rows[:8]:
            print(f"  {name:26s} {share:6.1%}   {mx:.3f} rad   {tq:5.1f} N m (limit {lim:.0f})")


class StageTimer:
    """Wall time of the stages of one control step (ms): where the 20 ms go. Wraps methods of the pipeline's parts."""

    def __init__(self):
        self.t = {}

    def wrap(self, obj, name, label):
        f = getattr(obj, name)

        def g(*a, **k):
            t0 = time.perf_counter()
            r = f(*a, **k)
            self.t.setdefault(label, []).append((time.perf_counter() - t0) * 1000.0)
            return r

        setattr(obj, name, g)

    def report(self, budget_ms):
        print(f"\nwall time per control step [ms] (budget {budget_ms:.0f}):   p50    p99    max")
        for k, v in self.t.items():
            v = np.array(v)
            print(f"  {k:34s} {np.percentile(v, 50):6.2f} {np.percentile(v, 99):6.2f} {v.max():6.2f}")


def tilt_deg(base_quat_xyzw):
    from robojudo.utils.util_func import get_gravity_orientation

    g = get_gravity_orientation(base_quat_xyzw)
    return float(np.degrees(np.arccos(np.clip(-g[2], -1.0, 1.0))))


STOP_COMMANDS = ("[SHUTDOWN]", "[SOFT_STOP]", "[GUARD_STOP]", "[ESTOP]")
REMOTE_BITS = {b: i for i, b in enumerate(["R1", "L1", "Start", "Select", "R2", "L2", "F1", "F2", "A", "B", "X", "Y", "Up", "Right",
                                           "Down", "Left"])}  # the remote's button word (robojudo/controller/utils/joystick.py)


class StopRequested(Exception):
    pass


def remote_stop_buttons(env, triggers):
    """The stop buttons held right now, read from the robot's raw remote state (not through the controller's event queue, which the
    pipeline drains itself): [] if none or if the state cannot be read."""
    try:
        keys = struct.unpack("H", bytes(env.unitree.get_robot_state().wireless_remote[2:4]))[0]
    except Exception:
        return []
    return [b for b, c in triggers.items() if c in STOP_COMMANDS and b in REMOTE_BITS and (keys >> REMOTE_BITS[b]) & 1]


def prepare_watching_stops(pipeline, env, triggers, prepare_seconds):
    """pipeline.prepare() (ramp to the ready pose, blend the policy in) with the remote's stop buttons watched every tick: a stop
    button held for 2 ticks (one corrupted radio packet does not stop) raises StopRequested before the next command is sent."""
    orig, held, own = env.step, [0], "step" in vars(env)

    def step(*args, **kw):
        b = remote_stop_buttons(env, triggers)
        held[0] = held[0] + 1 if b else 0
        if held[0] >= 2:
            raise StopRequested(f"operator stop {b} during start-up")
        return orig(*args, **kw)

    env.step = step
    try:
        pipeline.prepare(prepare_seconds=prepare_seconds)
    finally:  # the env's own step back
        if own:
            env.step = orig
        else:
            del env.step


def hold_guard(pipeline, env, dt, reason, on_hold=None):
    """After a stop on the robot: arm RoboJuDo's guard stop (unless a stop command already did) and keep stepping the env, which carries
    it out (legs limp and damped, arms to the guard pose and held), until Ctrl-C. on_hold: called once the guard is armed (RunLog.start)."""
    if getattr(env, "_guard_ramp", None) is None:
        pipeline.stop_with_guard_pose(reason)
    logger.error(f"STOPPED ({reason}): legs limp and damped, arms in the guard pose, held. Ctrl-C to end the program "
                 "(the motors are then released).")
    if on_hold is not None:
        try:
            on_hold()
        except Exception:  # whatever it does, the hold goes on
            logger.exception("on_hold failed")
    try:
        while True:
            t0 = time.perf_counter()
            env.update()
            env.step(np.asarray(env.dof_pos, dtype=np.float32))
            time.sleep(max(0.0, dt - (time.perf_counter() - t0)))
    except KeyboardInterrupt:
        logger.warning("ended after the stop")


LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs", "pingpong")  # (logs/ is git-ignored)


def log_path(name, kind):
    """--log's file: a name or relative path inside LOG_DIR (created); "" = <kind>_<date>_<time>.json there. An absolute path is kept
    (with a warning when it is outside this workspace)."""
    if not name:
        name = f"{kind}_{time.strftime('%Y%m%d_%H%M%S')}.json"
    if os.path.isabs(name):
        ws = os.path.dirname(os.path.dirname(LOG_DIR))
        if not os.path.abspath(name).startswith(ws + os.sep):
            logger.warning(f"--log {name} is outside the RoboJuDo workspace (logs go to {LOG_DIR} with a name or relative path)")
        path = name
    else:
        path = os.path.join(LOG_DIR, name)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    return path


def _json_default(o):
    """json.dump's fallback: numpy scalars as floats (as before), arrays as lists (one stray array must not cost the whole record)."""
    return o.tolist() if getattr(o, "ndim", 0) else float(o)


class RunLog:
    """--log's record. build() copies the run's state so far into a plain dict (.json: meta/results/shots; .npz: the per-step lists).
    start() writes it on a thread when a stop begins: the hold state keeps stepping (which carries out the guard), and a program that is
    killed instead of ended with the second Ctrl-C still leaves its record. finish() writes it again, whole, when the program ends."""

    def __init__(self, path, build):
        self.path, self.build, self.threads = path, build, []

    def _write(self, payload):
        tmp = f"{self.path}.{threading.get_ident()}.tmp"  # written whole, then renamed: a kill never leaves half a file
        if self.path.endswith(".json"):
            with open(tmp, "w") as f:
                json.dump(payload, f, default=_json_default)
        else:
            with open(tmp, "wb") as f:  # (a file object: np.savez adds ".npz" to a name)
                np.savez(f, **{k: np.array(v) for k, v in payload.items()})
        os.replace(tmp, self.path)

    def start(self):
        try:
            payload = self.build()
        except Exception:
            logger.exception("--log: could not copy the run's record")
            return

        def run():
            try:
                self._write(payload)
            except Exception:
                logger.exception(f"--log: could not write {self.path}")

        t = threading.Thread(target=run, daemon=True)
        t.start()
        self.threads.append(t)
        logger.info(f"[player] --log: the record so far is being written to {self.path} (written again when the program ends)")

    def finish(self):
        for t in self.threads:
            t.join(timeout=10.0)
        try:
            self._write(self.build())
            return True
        except Exception:
            logger.exception(f"--log: could not write {self.path}")
            return False


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-c", "--config", default="g1_pingpong")
    ap.add_argument("--dry-run", action="store_true", help="compute observations and actions every step, apply no torque")
    ap.add_argument("--prepare-seconds", type=float, default=3.0, help="real: the ramp to the ready pose, and again for the blend-in")
    ap.add_argument("--stale-steps", type=int, default=15, help="stop after this many control steps (20 ms) without a fresh pelvis pose")
    ap.add_argument("--max-tilt-deg", type=float, default=40.0)
    ap.add_argument("--pelvis-stale-ms", type=float, default=None, help="a pelvis pose older than this (camera capture -> this control "
                    "step) counts as stale for the watchdog (default: the config's pelvis_stale_s, 50 ms)")
    ap.add_argument("--max-seconds", type=float, default=0.0, help="stop after this long (sim time in sim); 0: no limit")
    ap.add_argument("--window", action="store_true", help="sim: open the MuJoCo viewer")
    ap.add_argument("--realtime", action="store_true", help="sim: pace to real time (default: as fast as possible)")
    ap.add_argument("--log", type=str, nargs="?", const="", default=None, help="write the run's record: .json = every shot's ball "
                    "samples, plans and paddle face per step (the shot recorder; pingpong scripts/deploy/analyze_shots.py reads it); "
                    ".npz = per-step timing. A name or relative path goes into this workspace's logs/pingpong/; --log alone: "
                    "logs/pingpong/<shots|shadow|manual>_<date>_<time>.json")
    ap.add_argument("--wait-perception", type=float, default=300.0, help="s to wait for the perception program before giving up")
    ap.add_argument("--perception-port", type=int, default=None, help="UDP port perception's messages come to (7780)")
    ap.add_argument("--raw-to", type=str, default=None, help="sim: host:port the simulated cameras send raw frames to (127.0.0.1:7770)")
    # the simulated world's options (PingPongSimWorldCtrlCfg)
    ap.add_argument("--balls", type=str, default=None)
    ap.add_argument("--env-idx", type=int, default=None)
    ap.add_argument("--n-shots", type=int, default=None, help="sim: stop after this many balls (default: no end; Ctrl-C or close the window)")
    ap.add_argument("--random", type=str, default=None, help="sim: fresh balls of this kind (standard, topspin, short, ...) instead of a scenario file")
    ap.add_argument("--sensing", choices=["optitrack", "optitrack120", "lab", "camera", "stereo"], default=None)
    ap.add_argument("--opti-frame", choices=["random", "fixed"], default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--paddle-extra-mass", type=float, default=None, help="sim: kg added to the paddle at the blade centre (a test of "
                    "the arm's sag with the real, heavier paddle)")
    ap.add_argument("--racket-z-offset", type=float, default=None, help="m added to the racket target's height in the policy's "
                    "observation (the arm sags below its target; the planner and the ball keep the true height)")
    ap.add_argument("--far-drop", type=float, default=None, help="sim: each sticker also lost with this probability beyond x 2.2 m "
                    "(the lab's thin far-half coverage: 0.12)")
    ap.add_argument("--ball-noise-mm", type=float, default=None, help="the estimator's ball position noise, mm per axis (its fit "
                    "weights, spin prior and gate; the config's value otherwise)")
    ap.add_argument("--fake-pivot", type=float, nargs=3, default=None, metavar=("X", "Y", "Z"),
                    help="sim: the simulated pelvis rigid body's pivot off the pelvis origin (the pelvis origin seen from it, pelvis axes, m)")
    ap.add_argument("--script-plans", type=str, default=None, help="shadow swings: replay this shot log's plans to the policy without a "
                    "ball (the policy alone; with --log the recorder gives the racket against the targets). Sim: no serves, ends when done")
    ap.add_argument("--manual", type=int, nargs="?", const=7790, default=None, metavar="PORT",
                    help="manual planner: the policy alone, live - you type the swings in pingpong scripts/deploy/planner_console.py "
                    "(another terminal), UDP to this port (7790); every swing's racket result comes back. Sim: no serves, real time (add --window to watch)")
    ap.add_argument("--manual-host", type=str, default=None, help="the interface --manual listens on (default 127.0.0.1; 0.0.0.0 for "
                    "a console on another computer)")
    ap.add_argument("--compute-scale", type=float, default=None, help="x the measured planner time (0 = instant)")
    ap.add_argument("--checkpoint", type=str, default=None)
    ap.add_argument("--actor-path", type=str, default=None)
    ap.add_argument("--start-offset", type=float, nargs=3, default=[0.0, 0.0, 0.0], metavar=("FWD", "LEFT", "YAW_DEG"), help="sim: start off home")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if a.log is not None:
        a.log = log_path(a.log, "manual" if a.manual is not None else "shadow" if a.script_plans else "shots")
    if a.script_plans and not os.path.isabs(a.script_plans) and not os.path.exists(a.script_plans):
        a.script_plans = os.path.join(LOG_DIR, a.script_plans)  # a shot log by its name in logs/pingpong/

    cfg = ConfigManager(config_name=a.config).get_cfg()
    if a.log and a.log.endswith(".json"):
        cfg.policy.record_shots = True
    if a.pelvis_stale_ms is not None:
        cfg.policy.pelvis_stale_s = a.pelvis_stale_ms / 1000.0
    sim = bool(cfg.env.is_sim)
    for c in cfg.ctrl:
        if c.ctrl_type == "PingPongSimWorldCtrl":
            for k in ("balls", "env_idx", "n_shots", "random", "sensing", "opti_frame", "seed"):
                if getattr(a, k, None) is not None:
                    setattr(c, k, getattr(a, k))
            if a.far_drop is not None:
                c.far_drop = a.far_drop
            if a.fake_pivot is not None:
                c.pelvis_pivot = tuple(a.fake_pivot)
            if a.raw_to and c.send_to:
                c.send_to = a.raw_to
        if c.ctrl_type == "PingPongPerceptionCtrl" and a.perception_port:
            c.port = a.perception_port
    if a.paddle_extra_mass is not None and sim:
        cfg.env.paddle_extra_mass = a.paddle_extra_mass
    if a.racket_z_offset is not None:
        cfg.policy.racket_z_offset = a.racket_z_offset
    if a.ball_noise_mm is not None:
        cfg.policy.noise = a.ball_noise_mm / 1000.0
    if a.compute_scale is not None:
        cfg.policy.compute_scale = a.compute_scale
    if a.script_plans:
        cfg.policy.script_plans = a.script_plans
        for c in cfg.ctrl:
            if c.ctrl_type == "PingPongSimWorldCtrl":
                c.serve = False
    if a.manual is not None:
        if a.script_plans:
            ap.error("--manual or --script-plans, not both")
        cfg.policy.manual_port = a.manual
        if a.manual_host:
            cfg.policy.manual_host = a.manual_host
        for c in cfg.ctrl:
            if c.ctrl_type == "PingPongSimWorldCtrl":
                c.serve = False
    if a.checkpoint:
        cfg.policy.checkpoint = a.checkpoint
    if a.actor_path:
        cfg.policy.actor_path = a.actor_path
    if sim:
        cfg.env.headless = not a.window
        if a.manual is not None and not a.realtime:  # typed swings need the simulation at the speed of the person typing
            a.realtime = True
            logger.info("--manual in simulation: paced to real time (--realtime)" + ("" if a.window else "; add --window to watch it"))
    if not sim and not a.dry_run:
        logger.warning("REAL ROBOT, torque will be applied. Gantry rope on? Remote in hand (A / Select / Start = stop: legs limp, arms "
                       "guard pose, held until Ctrl-C)?")

    pipeline = RlPipeline(cfg)
    env, policy = pipeline.env, pipeline._inner_policy()
    ctrls = {c.inst.__class__.__name__: c.inst for c in pipeline.ctrl_manager.controllers.values()}
    sim_ctrl, perc = ctrls.get("PingPongSimWorldCtrl"), ctrls.get("PingPongPerceptionCtrl")
    lib, jmap = policy.rt.lib, policy.rt.jmap
    if sim:
        env.reset_to_ready(float(lib.ready_root_z[0]) + 0.016, lib.ready_root_quat[0], jmap.to_motor(lib.ready_q[0]),
                           xy=a.start_offset[:2], yaw_deg=a.start_offset[2])
    if perc is not None:  # dual deployment: nothing moves before perception reports the pelvis
        print("\n=== waiting for the perception program (it may start after this one). In the pingpong folder:\n"
              "    python scripts/deploy/run_perception.py run --calibrate-first" + ("" if sim else "      (or: run, with a calibration file)")
              + f"\n    (it sends to UDP {perc.rx.port}; {'the simulated world sends it raw frames on 127.0.0.1:7770' if sim else 'Motive relay -> UDP 7770'})\n",
              flush=True)
        tick = None
        if sim and sim_ctrl is not None:  # the world stands still, its cameras run (perception calibrates on them)
            t_wait0 = time.perf_counter()

            last_render = [0.0]

            def tick():
                sim_ctrl.publish_waiting(time.perf_counter() - t_wait0)
                if env.viewer.is_alive and time.perf_counter() - last_render[0] > 1 / 30:  # the window at 30 Hz while waiting
                    last_render[0] = time.perf_counter()
                    env._draw_markers()
                    env.viewer.render()

        m = perc.wait_for_pelvis(a.wait_perception, tick=tick)
        if m is None:
            logger.error(f"no pelvis pose from perception within {a.wait_perception:.0f} s: is run_perception.py running and sending to "
                         f"UDP {perc.rx.port}? Nothing was moved.")
            env.shutdown()
            return 1
        src = perc.rx.last_addr
        msg = (f"perception is up ({'messages from ' + src[0] if src else ''}): pelvis at ({m.pelvis_pos[0]:+.3f}, {m.pelvis_pos[1]:+.3f}, "
               f"{m.pelvis_pos[2]:.3f}) m in the table frame, "
               f"heading {np.degrees(np.arctan2(m.pelvis_R[1, 0], m.pelvis_R[0, 0])):+.1f} deg")
        if sim:  # in simulation the truth is known: how good is perception's calibration?
            from pingpong.deploy.library import quat_mat

            Rt = quat_mat(env.data.qpos[3:7])
            msg += (f" (truth ({env.data.qpos[0]:+.3f}, {env.data.qpos[1]:+.3f}, {env.data.qpos[2]:.3f}), "
                    f"{np.degrees(np.arctan2(Rt[1, 0], Rt[0, 0])):+.1f} deg)")
        logger.info(msg)
    if sim:
        pipeline.policy.reset()
        pipeline.ctrl_manager.reset()
        if sim_ctrl is not None:
            sim_ctrl.start_play()
    elif not a.dry_run:
        ustop = ctrls.get("UnitreeCtrl")
        triggers = dict(getattr(ustop.cfg_ctrl, "triggers", None) or {}) if ustop is not None else {}
        if not any(c in STOP_COMMANDS for c in triggers.values()):
            logger.error("no stop button on the remote (UnitreeCtrl triggers): refusing to apply torque")
            env.shutdown()
            return 1
        try:
            prepare_watching_stops(pipeline, env, triggers, a.prepare_seconds)
        except (StopRequested, KeyboardInterrupt) as e:
            hold_guard(pipeline, env, 1.0 / cfg.policy.freq, str(e) or "Ctrl-C during start-up")
            env.shutdown()
            return 1
    if sim and a.window:
        import mujoco

        def viz():  # the plan the robot is acting on: the racket target (blue, during a swing) and the station (green)
            sw = policy.rt.swing
            out = [dict(type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[0.09, 0.09, 0.003], pos=[sw.station[0], sw.station[1], 0.004],
                        rgba=[0.1, 0.8, 0.2, 0.6])]
            tp = sw.tp if sw.mode == 1 else np.array([0.0, 0.0, -5.0])
            out.append(dict(type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[0.025] * 3, pos=tp, rgba=[0.2, 0.4, 1.0, 0.6]))
            return out

        env.viz_fn = viz
    if a.manual is not None:
        print(f"\n=== manual planner: in another terminal (pingpong folder)\n    python scripts/deploy/planner_console.py --port {a.manual}"
              + ("" if cfg.policy.manual_host == "127.0.0.1" else " --host <this computer>") + "\n", flush=True)
    monitor = ActionMonitor(env, policy)
    height = PelvisHeightCheck(env)
    timer = StageTimer()
    timer.wrap(pipeline.policy, "get_observation", "policy.get_observation (planner jobs)")
    timer.wrap(pipeline.policy, "get_pd_target", "policy.get_pd_target (actor)")
    timer.wrap(pipeline.ctrl_manager, "get_ctrl_data", "controllers (mocap, opponent)")
    timer.wrap(env, "step", "env.step (" + ("MuJoCo + ball" if sim else "LowCmd") + ")")
    timer.wrap(pipeline, "step", "whole control step")

    dt, n, n_stale, drops, t_wall0 = 1.0 / cfg.policy.freq, 0, 0, 0, time.perf_counter()
    rec = dict(t=[], pelvis_age=[], mode=[], tts=[], ok=[])
    from collections import deque

    seen = deque(maxlen=a.stale_steps + 3)  # per step: perception messages so far, with the pelvis so far (the watchdog's diagnosis)
    stop_reason = None
    n_reported = 0
    script_said = False
    t_next = time.perf_counter()

    def build_log():
        if not a.log.endswith(".json"):
            return {k: list(v) for k, v in rec.items()}
        meta = dict(config=a.config, sim=sim, dry_run=a.dry_run, checkpoint=str(cfg.policy.checkpoint), cam_hz=cfg.policy.cam_hz,
                    date=time.strftime("%Y-%m-%d %H:%M:%S"), stop=stop_reason, n_busy=policy.rt.n_busy,
                    ball_noise_m=cfg.policy.noise, racket_z_offset_m=cfg.policy.racket_z_offset,
                    paddle_extra_mass_kg=getattr(cfg.env, "paddle_extra_mass", None) if sim else None,
                    script_plans=cfg.policy.script_plans, manual_port=cfg.policy.manual_port)
        return dict(meta=meta, results=list(env.results) if sim else [], shots=list(policy.rt.shots))

    run_log = RunLog(a.log, build_log) if a.log else None
    on_hold = run_log.start if run_log else None
    try:
        while True:
            pipeline.step(dry_run=a.dry_run)
            n += 1
            info = policy.last_info
            t_now = float(env.data.time) if (sim and not a.dry_run) else time.perf_counter() - t_wall0  # a dry run never advances the simulation
            rec["t"].append(t_now)
            rec["pelvis_age"].append(info.get("pelvis_age", np.inf))
            rec["mode"].append(info.get("mode", -1))
            rec["tts"].append(info.get("time_to_strike", np.nan))
            rec["ok"].append(bool(info.get("ok", False)))
            height.step(info, sim and not a.dry_run)
            # ---- watchdogs
            n_stale = n_stale + 1 if (not info.get("ok", False) or info.get("pelvis_age", np.inf) > cfg.policy.pelvis_stale_s) else 0
            if not sim and n % 250 == 0:  # the robot: how old perception's pelvis pose is here, every 5 s
                ag = np.array([x for x in rec["pelvis_age"][-250:] if np.isfinite(x)]) * 1000
                if len(ag):
                    logger.info(f"[player] pelvis pose age (camera -> this control step) p50 {np.median(ag):.0f} / p95 "
                                f"{np.percentile(ag, 95):.0f} / max {ag.max():.0f} ms; stale above {cfg.policy.pelvis_stale_s * 1000:.0f} ms"
                                + (f"; arrival jitter removed p95 {np.percentile(perc.rx.jitter, 95) * 1000:.1f} ms"
                                   if perc is not None and getattr(perc.rx, "jitter", None) else ""))
                    if perc is not None and getattr(perc.rx, "jitter", None):
                        perc.rx.jitter.clear()
                logger.info("[player] " + height.summary(last=50))
            q, dq = env.dof_pos, env.dof_vel
            target = policy._hold if hasattr(policy, "_hold") else None
            if target is not None:
                tgt_env = pipeline.policy.actions_adapter.fit(target, template=env.dof_cfg.default_pos)
                if not monitor.check(tgt_env, q, dq):
                    stop_reason = "NaN in the joint targets"
            seen.append((getattr(policy.rt, "n_msgs", getattr(policy.rt, "n_frames", 0)), getattr(policy.rt, "n_pelvis_msgs", -1)))
            if n_stale > a.stale_steps and n > 50:
                d_msgs, d_pel = seen[-1][0] - seen[0][0], seen[-1][1] - seen[0][1]
                age = info.get("pelvis_age", np.inf)
                why = ("NO messages from perception in that time: perception or the relay stopped or stalled (CPU load? a crash?)"
                       if d_msgs == 0 else f"{d_msgs} perception messages, none with the pelvis: Motive lost the pelvis rigid body "
                       "(markers hidden?)" if d_pel == 0 else "(perception's runtime is older: no count of pelvis messages)" if seen[-1][1] < 0
                       else f"{d_pel} pelvis poses arrived, but older than "
                       f"{cfg.policy.pelvis_stale_s * 1000:.0f} ms (the last {age * 1000:.0f} ms): a delay between Motive and here "
                       "(the relay or perception falling behind: CPU load?)")
                stop_reason = f"no fresh pelvis pose for {n_stale} steps ({n_stale * dt * 1000:.0f} ms) - {why}"
            pz = float(info["pelvis"][2]) if "pelvis" in info else None
            if (pz is not None and pz < 0.45) or (sim and env.fallen):
                stop_reason = f"pelvis too low ({pz if pz is not None else float(env.data.qpos[2]):.2f} m): fallen"
            tl = tilt_deg(env.base_quat)
            if tl > a.max_tilt_deg:
                stop_reason = f"pelvis tilted {tl:.0f} deg"
            if getattr(pipeline, "last_commands", None) and any(c in STOP_COMMANDS for c in pipeline.last_commands):
                stop_reason = f"operator stop {pipeline.last_commands}"
            if stop_reason:
                logger.error(f"STOP: {stop_reason}")
                if not a.dry_run and not sim:  # the guard is carried out by stepping: hold it until Ctrl-C
                    hold_guard(pipeline, env, dt, stop_reason, on_hold)
                break
            if a.script_plans and policy.rt.script_done and not script_said:
                script_said = True
                logger.info(f"[player] shadow swings done: {len(policy.rt.shots)} played, {policy.rt.pipe.n_skipped} plans skipped"
                            + ("" if sim else " - the robot keeps standing; Ctrl-C to stop (the log is written then)"))
                if sim:
                    break
            if sim and sim_ctrl is not None and sim_ctrl.done:
                for _ in range(25):  # a moment more, so the last ball's outcome is seen through
                    pipeline.step(dry_run=a.dry_run)
                break
            if a.max_seconds and t_now > a.max_seconds:
                logger.info("time limit")
                break
            if sim and not cfg.env.headless and not env.viewer.is_alive:
                logger.info("the window was closed")
                break
            if sim and len(env.results) >= n_reported + 10:  # progress, every 10 balls
                n_reported = len(env.results)
                R = env.results
                last = R[-10:]
                print(f"[play] balls {len(R)}: returned {sum(r['returned'] for r in R) / len(R):.1%} "
                      f"(last 10: {sum(r['returned'] for r in last)}/10), net {sum(r['net'] for r in R)}, no contact {sum(not r['contact'] for r in R)}",
                      flush=True)
            if not sim or a.realtime:
                t_next += dt
                slack = t_next - time.perf_counter()
                if slack > 0:
                    time.sleep(slack)
                else:
                    drops += 1
                    t_next = time.perf_counter()
    except KeyboardInterrupt:
        logger.warning("interrupted")
        if not sim and not a.dry_run:
            hold_guard(pipeline, env, dt, "Ctrl-C", on_hold)
    except Exception as e:  # a bug in the loop with torque on: the guard stop, as for any stop (then the error is shown)
        if not sim and not a.dry_run:
            logger.exception("error in the control loop")
            hold_guard(pipeline, env, dt, f"error: {e!r}", on_hold)
        raise
    finally:
        log_ok = run_log.finish() if run_log else False  # first: nothing below may keep the record from being written
        wall = time.perf_counter() - t_wall0
        logger.info(f"{n} control steps in {wall:.1f} s wall ({n * dt:.1f} s of control); late steps {drops}")
        timer.report(dt * 1000.0)
        monitor.report()
        logger.info("[player] " + height.summary())
        if sim:
            R = env.results
            ret = sum(r["returned"] for r in R)
            print(f"\nshots {len(R)}: returned {ret} ({ret / max(len(R), 1):.1%}), net {sum(r['net'] for r in R)}, no contact {sum(not r['contact'] for r in R)}; "
                  f"fell {env.fallen}")
            for r in (R if len(R) <= 16 else []):
                print(f"  ball {r['idx']}: {'RETURNED' if r['returned'] else 'net' if r['net'] else 'missed' if not r['contact'] else 'not returned'}")
            rt = policy.rt
            if perc is not None:
                print(f"perception messages {perc.n_msgs}; lockstep timeouts {perc.n_lockstep_timeouts}")
            print(f"perception messages used {rt.n_frames}; detected shots {len(rt.shots)}; swings: {[s.get('swing') for s in rt.shots]}; "
                  f"shots while busy {rt.n_busy}; dry-run style torque excess steps (joints): {int((monitor.torque_over > 0).sum())}")
        if log_ok:
            print(f"-> {a.log}")
        env.shutdown()


if __name__ == "__main__":
    sys.exit(main())
