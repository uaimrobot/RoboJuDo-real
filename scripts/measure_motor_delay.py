"""B7: the G1's motor delay, joint by joint. Small steps and a short sine sweep on one joint at a time (all others hold the swing
library's ready pose), the joint angle recorded at 200 Hz. Per joint: dead time (command to 10 % of the step), t63, overshoot, the share
of the step reached, and the sine's gain and delay at 1-5 Hz. Run the same in MuJoCo for the reference: the difference is the robot's
extra delay (the policy, fix6-7, was trained with 0-20 ms).

    python scripts/measure_motor_delay.py -c g1_pingpong_real --dry-run              # the plan; no torque
    python scripts/measure_motor_delay.py -c g1_pingpong_real --joints arms --log    # the robot, ~1.5 min
    python scripts/measure_motor_delay.py -c g1_pingpong_real --joints legs waist --log
    python scripts/measure_motor_delay.py -c g1_pingpong --joints all --log          # the simulation reference (logs/pingpong-sim/)
    python scripts/measure_motor_delay.py -c g1_pingpong_real --joints arms --log --compare <sim log>.json

The robot HANGS from the gantry, feet at least 5 cm off the floor (the legs then carry no load: the motors' and the link's own delay is
what is measured; in simulation the pelvis is held in the air the same way). The gains are the policy's (PingPongDoF: training's kp, kd,
torque limits). Targets are sent at 200 Hz but change at 50 Hz, as the policy's. Steps are 0.10 rad (--amp, at most 0.15), and a side
that would come within 0.05 rad of the joint's limit is skipped.

Stops, as the player's: the remote's stop buttons (held 40 ms), Ctrl-C, or an error -> RoboJuDo's guard stop (legs limp and damped,
arms to the guard pose), held until Ctrl-C; then the motors are released. At the end the ready pose is held until Ctrl-C.

The log (--log): per joint the command and angle series and the results, plus the robot state's field names (motor temperatures and
battery voltage are recorded if the state has them: their names are not known here, so they are looked up, not assumed).
"""

import argparse
import importlib.util
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
_spec = importlib.util.spec_from_file_location("run_pingpong", os.path.join(HERE, "run_pingpong.py"))
rp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rp)  # the player's stop handling, guard stop and log folders (it imports robojudo.pipeline first: torch before mujoco)

from robojudo.config.config_manager import ConfigManager  # noqa: E402
from robojudo.pipeline.rl_pipeline import RlPipeline  # noqa: E402

GROUPS = {
    "arms": ["left_shoulder_pitch_joint", "right_shoulder_pitch_joint", "left_shoulder_roll_joint", "right_shoulder_roll_joint",
             "left_elbow_joint", "right_elbow_joint"],
    "legs": ["left_hip_pitch_joint", "right_hip_pitch_joint", "left_knee_joint", "right_knee_joint", "left_ankle_pitch_joint",
             "right_ankle_pitch_joint"],
    "waist": ["waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"],
}
GROUPS["all"] = GROUPS["arms"] + GROUPS["legs"] + GROUPS["waist"]

LOOP_DT = 0.005  # s: 200 Hz sampling and sending
CMD_EVERY = 4  # loops per target change: 50 Hz, the policy's rate
SETTLE, STEP_HOLD = 0.5, 0.6  # s
FREQS, CYCLES = (1.0, 2.0, 3.0, 4.0, 5.0), 3  # the sine sweep (the first cycle of each is not analysed: its start is a kink)
AMP_MAX, LIMIT_MARGIN = 0.15, 0.05  # rad
STOP_HELD = 8  # loops (40 ms): one corrupted radio packet does not stop


def plan_joint(q0, lim, amp):
    """The offsets from q0 (rad) for one joint, as a function of time, and the analysed segments. A step side that would come within
    LIMIT_MARGIN of the limit is left out (both sides out: no steps); the sine is amp/2 (dropped likewise)."""
    sides = [s for s in (+1, -1) if lim[0] + LIMIT_MARGIN <= q0 + s * amp <= lim[1] - LIMIT_MARGIN]
    pieces, segs, t = [(SETTLE, lambda tt: 0.0)], [], SETTLE
    for _ in range(2):
        for s in sides:
            for off in (s * amp, 0.0):
                prev = pieces[-1][1](pieces[-1][0]) if pieces else 0.0
                pieces.append((STEP_HOLD, lambda tt, o=off: o))
                segs.append(dict(kind="step", t0=t, t1=t + STEP_HOLD, delta=off - prev))
                t += STEP_HOLD
    a = amp / 2.0
    if all(lim[0] + LIMIT_MARGIN <= q0 + s * a <= lim[1] - LIMIT_MARGIN for s in (+1, -1)):
        for f in FREQS:
            d = CYCLES / f
            pieces.append((d, lambda tt, f=f: a * np.sin(2 * np.pi * f * tt)))
            segs.append(dict(kind="sine", t0=t + 1.0 / f, t1=t + d, f=f, amp=a))
            t += d
    pieces.append((SETTLE, lambda tt: 0.0))
    return pieces, segs, t + SETTLE, sides


def offset_at(pieces, t):
    for d, fn in pieces:
        if t < d:
            return float(fn(t))
        t -= d
    return 0.0


def analyse(t, cmd, q, segs):
    """Per step: dead time and t63 (ms after the command changed), overshoot and the share of the step reached at the end of the hold;
    per sine frequency: gain and delay (ms) of the angle against the command, from a sinusoid fit on the analysed cycles."""
    t, cmd, q = map(np.asarray, (t, cmd, q))
    steps, sines = [], []
    for s in segs:
        if s["kind"] == "step":
            win = (t >= s["t0"] - 0.01) & (t <= s["t0"] + 0.03)  # the command changes at the first 50 Hz tick of the segment
            ch = np.nonzero(win & (np.abs(np.diff(cmd, prepend=cmd[0])) > 1e-9))[0]
            if not len(ch) or abs(s["delta"]) < 1e-6:
                continue
            tc = t[ch[0]]
            before = (t >= tc - 0.1) & (t < tc)
            m = (t >= tc) & (t < s["t1"])
            if not before.any() or m.sum() < 10:
                continue
            r = (q[m] - np.median(q[before])) / s["delta"]
            tt = t[m] - tc
            first = lambda x: float(tt[np.argmax(r >= x)] * 1000) if (r >= x).any() else np.nan  # noqa: E731
            steps.append(dict(dead_ms=first(0.1), t63_ms=first(0.63), overshoot=float(r.max() - 1.0),
                              reached=float(np.median(r[tt >= tt[-1] - 0.2]))))
        else:
            m = (t >= s["t0"]) & (t < s["t1"])
            if m.sum() < 10:
                continue
            w = 2 * np.pi * s["f"]
            X = np.stack([np.sin(w * t[m]), np.cos(w * t[m]), np.ones(m.sum())], 1)
            c = np.linalg.lstsq(X, cmd[m], rcond=None)[0]
            y = np.linalg.lstsq(X, q[m], rcond=None)[0]
            pc, py = np.arctan2(c[1], c[0]), np.arctan2(y[1], y[0])
            lag = (pc - py) % (2 * np.pi)
            sines.append(dict(f=s["f"], gain=float(np.hypot(*y[:2]) / max(np.hypot(*c[:2]), 1e-9)), delay_ms=float(lag / w * 1000)))
    return dict(steps=steps, sines=sines, **summary(steps, sines))


def summary(steps, sines):
    med = lambda k, L: float(np.nanmedian([x[k] for x in L])) if L else np.nan  # noqa: E731
    return dict(dead_ms=med("dead_ms", steps), t63_ms=med("t63_ms", steps), overshoot=med("overshoot", steps),
                reached=med("reached", steps), sine_delay_ms=med("delay_ms", sines), sine_gain=med("gain", sines))


def state_fields(env):
    """The robot state's field names and anything that looks like motor temperatures / battery voltage (looked up, not assumed)."""
    try:
        st = env.unitree.get_robot_state()
    except Exception:
        return {}
    names = lambda o: sorted(n for n in dir(o) if not n.startswith("_"))  # noqa: E731
    out = dict(robot_state=names(st))
    ms = getattr(st, "motor_state", None)
    if ms is not None:
        out["motor_state"] = names(ms)
        for k in ("temperature", "temp"):
            if hasattr(ms, k):
                out["motor_" + k] = np.asarray(getattr(ms, k), float).tolist()
    for k in out["robot_state"]:
        if any(w in k.lower() for w in ("bms", "battery", "voltage", "power_v")):
            v = getattr(st, k)
            try:
                out[k] = np.asarray(v, float).tolist()
            except Exception:
                out[k] = str(v)
    return out


class Runner:
    """Drives the env at LOOP_DT with a target vector, watching the stops (the robot) and holding the pelvis in the air (simulation)."""

    def __init__(self, env, sim, triggers, pin=None, clock=time.perf_counter, sleep=time.sleep):
        self.env, self.sim, self.triggers, self.pin = env, sim, triggers, pin
        self.clock, self.sleep, self.held, self.n = clock, sleep, 0, 0

    def now(self):
        return float(self.env.data.time) if self.sim else self.clock()

    def tick(self, target):
        """One loop: read the state, check the stops, send the target. Returns (time of the reading, joint angles)."""
        self.env.update()
        t, q = self.now(), np.array(self.env.dof_pos, float)
        if not self.sim:
            b = rp.remote_stop_buttons(self.env, self.triggers)
            self.held = self.held + 1 if b else 0
            if self.held >= STOP_HELD:
                raise rp.StopRequested(f"operator stop {b}")
        self.env.step(np.asarray(target, np.float32))
        if self.pin is not None:
            self.env.data.qpos[:7] = self.pin
            self.env.data.qvel[:6] = 0.0
        self.n += 1
        return t, q

    def run(self, duration, target_fn):
        """target_fn(k) -> the target vector at loop k (it should change only every CMD_EVERY loops). Returns t, q, targets."""
        T, Q, C = [], [], []
        t_next = self.clock()
        for k in range(int(round(duration / LOOP_DT))):
            c = target_fn(k)
            t, q = self.tick(c)
            T.append(t), Q.append(q), C.append(np.array(c, float))
            if not self.sim:
                t_next += LOOP_DT
                slack = t_next - self.clock()
                if slack > 0:
                    self.sleep(slack)
                else:
                    t_next = self.clock()
        return np.array(T), np.array(Q), np.array(C)


def test_joint(runner, ready, j, amp, lim):
    """One joint's steps and sweep from the ready pose. Returns the record (series of that joint, results)."""
    pieces, segs, dur, sides = plan_joint(ready[j], lim, amp)
    held = [ready[j]]

    def target(k):
        if k % CMD_EVERY == 0:
            held[0] = ready[j] + offset_at(pieces, k * LOOP_DT)
        c = ready.copy()
        c[j] = held[0]
        return c

    T, Q, C = runner.run(dur, target)
    t = T - T[0]
    res = analyse(t, C[:, j], Q[:, j], segs)
    others = np.delete(np.abs(Q - ready), j, axis=1).max()
    return dict(t=np.round(t, 5).tolist(), cmd=np.round(C[:, j], 5).tolist(), q=np.round(Q[:, j], 5).tolist(), sides=sides,
                others_max_dev=float(others), **res)


def print_row(name, r, ref=None):
    x = lambda v, f: f.format(v) if np.isfinite(v) else "   -"  # noqa: E731
    line = (f"  {name:28s} {x(r['dead_ms'], '{:5.0f}')} {x(r['t63_ms'], '{:5.0f}')} {x(r['overshoot'] * 100, '{:5.0f}')}% "
            f"{x(r['reached'], '{:6.2f}')} {x(r['sine_delay_ms'], '{:6.0f}')} {x(r['sine_gain'], '{:6.2f}')}")
    if ref is not None:
        line += (f"   | vs sim: dead {x(r['dead_ms'] - ref['dead_ms'], '{:+5.0f}')} t63 {x(r['t63_ms'] - ref['t63_ms'], '{:+5.0f}')} "
                 f"sine delay {x(r['sine_delay_ms'] - ref['sine_delay_ms'], '{:+5.0f}')} ms")
    print(line, flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-c", "--config", default="g1_pingpong_real")
    ap.add_argument("--joints", nargs="+", default=["arms"], help="groups (arms legs waist all) and/or joint names")
    ap.add_argument("--amp", type=float, default=0.10, help=f"step size, rad (at most {AMP_MAX})")
    ap.add_argument("--ramp-seconds", type=float, default=3.0, help="the ramp from the hanging pose to the ready pose")
    ap.add_argument("--hang-height", type=float, default=0.15, help="sim: the pelvis held this far above its standing height")
    ap.add_argument("--dry-run", action="store_true", help="print the plan and the robot state's fields; no torque")
    ap.add_argument("--log", type=str, nargs="?", const="", default=None, help="the record: logs/pingpong/ (robot) or logs/pingpong-sim/")
    ap.add_argument("--compare", type=str, default=None, help="a simulation log of this script: print the robot's extra delay per joint")
    a = ap.parse_args()
    if not 0.0 < a.amp <= AMP_MAX:
        sys.exit(f"--amp must be in (0, {AMP_MAX}] rad")

    cfg = ConfigManager(config_name=a.config).get_cfg()
    sim = bool(cfg.env.is_sim)
    if sim:  # one env step = one loop (the physics at its own rate inside)
        cfg.env.sim_decimation = int(round(LOOP_DT / cfg.env.sim_dt))
    for c in cfg.ctrl:  # the policy needs its perception receiver, but nothing is read from it: any free port (none taken from a
        if c.ctrl_type == "PingPongPerceptionCtrl":  # player or perception left running)
            c.port = 0
    pipeline = RlPipeline(cfg)
    env, policy = pipeline.env, pipeline._inner_policy()
    lib = policy.rt.lib
    ready_by_name = dict(zip(lib.joint_names, np.asarray(lib.ready_q)[0]))
    ready = np.array([ready_by_name[n] for n in env.joint_names], float)
    limits = np.asarray(env.position_limits, float)

    joints = []
    for g in a.joints:
        for n in GROUPS.get(g, [g]):
            if n not in env.joint_names:
                sys.exit(f"unknown joint or group: {n} (groups: {', '.join(GROUPS)})")
            if n not in joints:
                joints.append(n)
    log = rp.log_path(a.log, "motor_delay", sim) if a.log is not None else None

    print(f"\n=== motor delay test ({'simulation' if sim else 'THE ROBOT'}; {len(joints)} joints, steps {a.amp:.2f} rad, sine {a.amp / 2:.2f} rad "
          f"at {', '.join(f'{f:g}' for f in FREQS)} Hz; ~{sum(plan_joint(ready[env.joint_names.index(n)], limits[env.joint_names.index(n)], a.amp)[2] for n in joints) + a.ramp_seconds + 1:.0f} s)")
    for n in joints:
        j = env.joint_names.index(n)
        _, _, dur, sides = plan_joint(ready[j], limits[j], a.amp)
        print(f"  {n:28s} ready {np.degrees(ready[j]):+6.1f} deg, limits [{np.degrees(limits[j, 0]):+6.0f}, {np.degrees(limits[j, 1]):+6.0f}], "
              f"steps {''.join('+' if x > 0 else '-' for x in sides) or 'NONE (too near a limit)'}, "
              f"kp x step {env.stiffness[j] * a.amp:4.1f} N m, {dur:.1f} s")

    triggers = {}
    if not sim:
        ustop = {c.inst.__class__.__name__: c.inst for c in pipeline.ctrl_manager.controllers.values()}.get("UnitreeCtrl")
        triggers = dict(getattr(ustop.cfg_ctrl, "triggers", None) or {}) if ustop is not None else {}
    fields = state_fields(env) if not sim else {}
    if fields:
        print("robot state fields:", ", ".join(fields.get("robot_state", [])))
        print("motor state fields:", ", ".join(fields.get("motor_state", [])))
    if a.dry_run:
        print("dry run: nothing sent")
        env.shutdown()
        return 0
    if not sim and not any(c in rp.STOP_COMMANDS for c in triggers.values()):
        env.shutdown()
        sys.exit("no stop button on the remote (UnitreeCtrl triggers): refusing to apply torque")

    pin = None
    if sim:
        env.reset_to_ready(float(lib.ready_root_z[0]) + 0.016 + a.hang_height, lib.ready_root_quat[0], ready)
        pin = env.data.qpos[:7].copy()
    runner = Runner(env, sim, triggers, pin=pin)
    results, stop = {}, None
    t_start = time.strftime("%Y-%m-%d %H:%M:%S")
    try:
        env.update()
        q_start = np.array(env.dof_pos, float)
        n_ramp = max(int(round(a.ramp_seconds / LOOP_DT)), 1)
        print(f"ramp to the ready pose ({a.ramp_seconds:.0f} s) ...", flush=True)
        runner.run(a.ramp_seconds, lambda k: q_start + (ready - q_start) * min((k // CMD_EVERY * CMD_EVERY) / n_ramp, 1.0))
        runner.run(1.0, lambda k: ready)
        print(f"\n  {'joint':28s} {'dead':>5} {'t63':>5} {'over':>6} {'reach':>6} {'sine':>6} {'gain':>6}   (ms, ms, %, share, ms, ratio)")
        ref = json.load(open(a.compare))["joints"] if a.compare else {}
        for n in joints:
            j = env.joint_names.index(n)
            t0 = state_fields(env) if not sim else {}
            r = test_joint(runner, ready, j, a.amp, limits[j])
            if not sim:
                r["state_before"], r["state_after"] = {k: v for k, v in t0.items() if k.startswith(("motor_t", "bms", "battery"))}, \
                    {k: v for k, v in state_fields(env).items() if k.startswith(("motor_t", "bms", "battery"))}
            results[n] = r
            print_row(n, r, ref.get(n))
        if ref:
            common = [n for n in results if n in ref]
            d = [results[n]["dead_ms"] - ref[n]["dead_ms"] for n in common]
            s = [results[n]["sine_delay_ms"] - ref[n]["sine_delay_ms"] for n in common]
            print(f"\n  extra delay vs the simulation, median over {len(common)} joints: dead time {np.nanmedian(d):+.0f} ms, "
                  f"sine delay {np.nanmedian(s):+.0f} ms (training covered 0-20 ms)")
        if not sim:
            print("\ndone: holding the ready pose. Ctrl-C to release the motors (the gantry holds the robot).", flush=True)
            while True:
                runner.tick(ready)
                time.sleep(LOOP_DT)
    except (rp.StopRequested, KeyboardInterrupt, Exception) as e:  # noqa: BLE001
        stop = str(e) or ("Ctrl-C" if isinstance(e, KeyboardInterrupt) else repr(e))
        done = isinstance(e, KeyboardInterrupt) and len(results) == len(joints)
        if not sim and not done:
            rp.hold_guard(pipeline, env, 1.0 / cfg.policy.freq, stop)
        if not isinstance(e, (rp.StopRequested, KeyboardInterrupt)):
            import traceback

            traceback.print_exc()
    finally:
        if log:
            meta = dict(config=a.config, sim=sim, date=t_start, joints_asked=joints, amp=a.amp, loop_dt=LOOP_DT, cmd_every=CMD_EVERY,
                        freqs=list(FREQS), hang_height=a.hang_height if sim else None, stop=stop,
                        stiffness={n: float(env.stiffness[env.joint_names.index(n)]) for n in joints}, state_fields=fields)
            with open(log, "w") as f:
                json.dump(dict(meta=meta, joints=results), f)
            print(f"-> {log}")
        env.shutdown()
    return 0 if stop is None or (stop == "Ctrl-C" and len(results) == len(joints)) else 1


if __name__ == "__main__":
    sys.exit(main())
