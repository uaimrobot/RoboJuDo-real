# G1 table tennis: deployment guide

Two programs (dual deployment), the same in simulation and on the robot. Only what feeds PERCEPTION and what the PLAYER drives change.

```
 simulation:  MuJoCo world's simulated cameras --.                                 .-- MuJoCo robot
                                                 >-- raw --> PERCEPTION --table--> PLAYER --<
 real robot:  Motive --> natnet_relay.py --------'  UDP 7770   (one program)  frame  (one     '-- G1 (unitree_cpp)
                                                                              UDP 7780 program)
```

| | PERCEPTION | PLAYER |
|---|---|---|
| program | `scripts/deploy/run_perception.py` (pingpong folder) | `scripts/run_pingpong.py` (RoboJuDo) |
| in | raw OptiTrack frames: the pelvis rigid body + unlabeled markers, OptiTrack's coordinates (UDP 7770) | perception's messages (UDP 7780) + the robot's IMU and joints |
| does | table frame (calibration), pelvis pose in the table frame, which marker is the incoming ball, shot start / end | ball estimator, planner, swing clock, policy (`pingpong.deploy.PingPongRuntime`) |
| out | one message per frame, table frame (`pingpong/deploy/perception.py`) | joint targets |
| sim vs real | identical command | `-c g1_pingpong` (MuJoCo) or `-c g1_pingpong_real` (G1) |

The player never sees OptiTrack's coordinates; perception never sees the robot's joints. Timestamps need no clock synchronization on the
robot (each message carries its age) and follow the simulated world's clock in simulation (`pingpong/deploy/mocap.py`), where the player
and perception run in lockstep, so a simulation may run at any speed.

**Status: runs in simulation, never on the robot.** What was run and what was not: "What is tested".

Python: RoboJuDo's environment, `conda_env/robojudo/bin/python` (written `python` below); it has numba. Folders: RoboJuDo =
`humanoid_deployment/RoboJuDo`, pingpong = `playground/humanoid_pingpong/pingpong-version-0`.

## A. Simulation

**A1. Unit tests** (pingpong folder), each prints `all passed`:
```bash
for t in shot_detector table_frame emulator mocap_calibration perception table_rigid_body table_markers_layouts ball_markers natnet_relay; do
  python tests/deploy/test_$t.py | tail -1; done
```

**A2. The two programs.** Either may start first; the player waits for perception before anything moves.
```bash
# terminal 1 (RoboJuDo): the player; its MuJoCo world sends the simulated cameras' raw frames to UDP 7770
python scripts/run_pingpong.py -c g1_pingpong --window --realtime
# terminal 2 (pingpong): perception, calibrating on the simulated table markers first
python scripts/deploy/run_perception.py run --calibrate-first --out calibration/table_frame_sim.json
```
What you see: perception prints the calibrated table (height, width, length against `table.yaml`), then a status line every 2 s (pelvis
seen, the pelvis in the table frame, shots and how they ended). The player prints `perception is up: pelvis at (...)` next to the simulated
truth, then plays. Window: the ball orange, the racket target blue, the station green, home grey. At the end: each ball's outcome, the
stage timings, joint targets outside the joint range, perception messages and lockstep timeouts (should be 0). Drop `--window --realtime`
for a headless run as fast as the computer goes.

The opponent serves without end: the scenario's 64 sequences of 8 balls in turn (512 balls, then again), or fresh balls with
`--random standard` (or `topspin`, `short`, `deep_fast`, ...). Stop with Ctrl-C or by closing the window; `--n-shots N` or `--max-seconds S`
set a limit. A progress line every 10 balls; the summary at the end.

**A3. Variations** (player flags): `--sensing camera` (100 Hz, 5 mm, 30 ms; give perception `--noise 0.005`), `--start-offset 0.2 0.1 10`
(start off home), `--balls topspin`, `--env-idx 5` (another ball sequence), `--opti-frame fixed`.
`--avoid-late [TYPE ...]` (planner: a footwork step or another stroke type that starts on time before a late swing; default types
forehand backhand). Opt-in, OFF by default: on fix6-7 @16500 (2026-11-03, 128 paired balls, lab sensing) it halved late plans (32 -> 17 %)
but returned 90.6 vs 94.5 % nominal and 67.2 vs 68.8 % with 20 ms motor delay + 300 g paddle - no gain.

**A4. Fast batch runs** (one process, perception inside, planner inline): `python scripts/run_pingpong.py -c g1_pingpong_inproc --n-shots 64
--compute-scale 0`. Same play; use it for statistics.

## B. The real robot

Nothing changes in the player but `-c g1_pingpong_real`; nothing changes in perception at all. What is new is what feeds them.

**B1. Install** on the control PC (an external PC on the robot's Ethernet; the G1's own computer is too slow for the planner): RoboJuDo's
environment + `numba`, `unitree_sdk2` + `unitree_cpp` (`docs/unitree_setup.md`), the pingpong folder (or `pingpong/`, `configs/`,
`data/motions/npz/swings_mix.npz`, `data/planner_tables/`, `calibration/`; `PINGPONG_ROOT` if it lives elsewhere). Since 2026-11-03 the robot's pingpong files come from GitHub (public): `git clone https://github.com/uaimrobot/humanoid-pingpong-uaim ~/humanoid-pingpong-uaim`, `export PINGPONG_ROOT=~/humanoid-pingpong-uaim`, updates with `git pull` (the robot's own `calibration/table_frame.json` is tracked there: `git stash` before a pull if perception rewrote it). The policy
itself is in RoboJuDo: `assets/motions/g1/pingpong_play/model_16500.pt` + its `params/` (stage4-fix6-7 @16500, the default for
simulation and the robot; `--checkpoint` for another). Check:
`python -c "from robojudo.environment import UnitreeCppEnv"`. Without torch there: `Actor('model_16500.pt').export_npz('actor.npz')` and
`--actor-path actor.npz`.

**B2. Network.** The control PC on the G1's subnet (192.168.123.x): `NET_IF` in `robojudo/config/g1/g1_pingpong_cfg.py`. Perception may run
on the control PC (default ports, localhost) or elsewhere (`--send <player host>:7780`).

**B3. OptiTrack.** Ground plane set so z = 0 is the floor; stream **Z-up**; a pelvis rigid body with its pivot at the pelvis origin and the
pelvis axes (x forward, z up). The table: markers on (at least) the table top's near-left, near-right and a far corner, at one height, plus any
others you like (e.g. the lab table: 4 corners on the top + the middle of each side on the side faces, 8 markers; calibration finds the
corners by the table's size, and the simulation uses this layout), made a rigid body
named `table` (recommended: its markers then travel apart from the unlabeled ones, can never pass for the ball, stay on during
play, and perception's status line says if the table moves). Without a table rigid body, plain corner markers work too: calibrate, then
take them off. **The ball** stays unlabeled (no rigid body: it spins too fast for Motive to keep one). A ball with dot stickers gives
one unlabeled marker per sticker the cameras see (often 2-5); the relay sends them all and perception makes them one ball (B4).
The relay reads Motive with the lab's unicast NatNet client (`scripts/deploy/natnet_client.py`) and runs on any computer that reaches
Motive, simplest the control PC next to perception (pingpong folder):
```bash
python scripts/deploy/natnet_relay.py    # defaults: Motive 192.168.123.200, this computer 192.168.123.164, to 127.0.0.1:7770
# elsewhere: --client <its address> (or '' for any), --to <perception's computer>
```
Motive > Streaming: Unicast; Rigid Bodies, Labeled Markers, Unlabeled Markers on. Motive's default y-up axes are turned to z-up by the
relay (`--up y`, the default; `--up z` if Motive streams z-up). Its status line every 2 s: Motive's frame rate, latency (from Motive's
camera timestamps), pelvis tracked, `table markers 8.0/frame`, unlabeled markers per frame. Tested only against a fake Motive
(`tests/deploy/test_natnet_relay.py`), not a real one.

**B4. Perception.** The robot standing at home:
```bash
python scripts/deploy/run_perception.py calibrate --marker-height <marker radius>   # -> calibration/table_frame.json; within 1 cm of table.yaml?
python scripts/deploy/run_perception.py run              # status every 2 s: carry the robot, the pelvis follows; roll a ball, a shot starts
```
**The robot must face the table squarely during `calibrate`**: Motive gives a new rigid body its world's axes, whichever way the
robot faced, so calibration measures the pelvis rigid body's heading and corrects it to 0 (`pelvis: the rigid body's x axis was at
+37.0 deg ...`; saved in table_frame.json; a placement error stays as a heading error: use a line on the floor). A rigid body whose
up axis is more than 20 deg from vertical is refused (fix it in Motive). `--keep-pelvis-axes` takes Motive's axes as they are.
**The heading is measured once and kept** (since 2026-11-02): later calibrations redo the table only and print `heading correction
... kept from the previous calibration`, so a robot hanging a few degrees turned that day does not become a heading error.
`--recalibrate-pelvis` measures it again (robot standing still, facing the table squarely). **Live, while the player runs:** type
`h` + Enter in perception's terminal once the robot STANDS at home facing the table squarely, BEFORE THE FIRST BALL: it measures for
2 s while perception keeps sending, turns the correction in over 1 s and saves it (refused above 15 deg or if the robot moves). After
balls `h` is refused (`h!` forces it): the robot turns during play (in simulation ~13 deg after a few balls, as on the robot), and
measuring then makes that turn its "facing the table" - in simulation that cost 98 % -> 62 % returned. A ball on: refused.
**How exact the heading must be** (2026-11-03, the deployed policy fix6-7 @16500 in Isaac, the same 1905 balls with the heading the
policy sees off by exactly +-N deg; `eval_closed_loop.py --heading_bias N`): 0 / 2 / 4 / 6 / 8 / 10 deg = 90.1 / 89.4 / 85.9 / 77.1 /
58.2 / 37.5 % returned. **Keep it within 2 deg** (free); 4 deg costs 4 points, 6 deg 13. The racket lands where a turned robot would
put it (10 cm and 9 deg off at 10 deg), and no policy can notice: a robustness-trained policy (fix7c, +-6 deg in training) did the
same at 10 deg and lost 5-8 points everywhere else. A pelvis position 4 cm off costs 2 points (`--pelvis_offset 0.04`).
With a table rigid body the status line ends in `table steady (0.1 mm)`; `table MOVED 12 mm: STOP and recalibrate` means it was bumped.
The ball (`--ball`, `pingpong/deploy/ball_markers.py`): `stickers` (the default) groups the markers within one ball diameter into one
ball and fits the sphere of the ball's radius through them, which finds the centre; the markers' mean would sit ~6 mm toward the
cameras, since they see the stickers that face them. A sample needs 3 or more markers (1-2 are 8-20 mm off; simulation: with them the
robot returned 2 points fewer balls). The status line shows `markers per ball sample 3:27% 4:52% 5:21%`: if shots are not detected or
few samples come, too few stickers are seen - add stickers, spread evenly. `covered`: a ball covered in reflective material, one
marker at its centre (with `stickers` such a ball is never seen: shots 0). Stickers add mass and change the bounce: weigh the ball
(`ball.yaml`: 2.7 g) and fit its model to real flights (B8).
With plain corner markers, take them off after `calibrate`.
(or `run --calibrate-first`, as in simulation). Redo the calibration whenever the table may have moved.

**B4b. Seeing it in rviz2 (optional).** `run --publish-data` also publishes perception's data to ROS 2 (`pingpong/deploy/ros_viz.py`,
60 Hz, `--publish-hz`), all in the table frame `pp_table`:

| | |
|---|---|
| TF | `optitrack -> pp_table` (the calibration, static), `pp_table -> pp_pelvis` |
| `/pingpong/pelvis` | `geometry_msgs/PoseStamped` |
| `/pingpong/ball` | `geometry_msgs/PointStamped`, while a shot is on |
| `/pingpong/ball_centre` | `geometry_msgs/PointStamped`, the ball's fitted centre whenever it is seen (held, rolled, in flight) |
| `/pingpong/table` | `geometry_msgs/PolygonStamped`, the top's 4 corners |
| `/pingpong/markers` | `visualization_msgs/MarkerArray`: table top + outline + centre line, net, home, the ball's centre (one orange dot per ball), during a shot a red shell + its trail, the table rigid body's markers (green); with `--publish-raw-markers` also every unlabeled marker (white: each sticker) |

rclpy on Foxy is built for Python 3.8, so perception runs on a Python 3.8: the conda env `scripts/deploy/g1_perception_env.yml`
(python 3.8, numpy 1.24, PyYAML: all that the relay and perception import) with ROS sourced after activating, or the system
python3 with `python3-numpy python3-yaml`. Perception's code runs on 3.8 (tested):
```bash
conda activate pingpong_perception && source /opt/ros/foxy/setup.bash
cd <pingpong folder> && python scripts/deploy/run_perception.py run --publish-data
# on a computer with a screen, same network and ROS_DOMAIN_ID:
source /opt/ros/foxy/setup.bash && rviz2 -d <pingpong folder>/scripts/deploy/pingpong.rviz
```
Check `ros2 topic list` / `ros2 topic hz /pingpong/markers` where rviz2 runs. A display without data: set its topic in the Displays
panel (Add > By topic lists them). Tested against stand-in ROS modules only (`tests/deploy/test_ros_viz.py`), not a real ROS 2.

**B5. Player, dry run** (no torque; robot hanging or sitting): `python scripts/run_pingpong.py -c g1_pingpong_real --dry-run`. It waits
for perception, prints the pelvis pose, then computes everything every step. Look at the stage timings (the robot's part under ~5 ms),
`ActionMonitor` warnings, a rolled ball's shot and swing.

**B6. Gantry, feet off the floor:** `python scripts/run_pingpong.py -c g1_pingpong_real`. After perception is up: the joints ramp to the
READY pose in 3 s, the policy blends in over 3 s, then holds the ready pose. Remote: **A** = shutdown, **Select** = soft stop, **Start** =
guard pose.

**B7. On the floor** (rope slack, attached): stand 1-2 min; measure the motor delay (command to motion; fix6-7 trained with 0-20 ms).

**B7-delay. The motor delay, joint by joint** (`scripts/measure_motor_delay.py`, 2026-11-03). The robot HANGS from the gantry, feet at
least 5 cm off the floor; the gains are the policy's. One joint at a time (all others hold the ready pose): steps of 0.10 rad and a sine
of 0.05 rad at 1-5 Hz, sent at 200 Hz but changing at 50 Hz like the policy's targets; the angle recorded at 200 Hz. Stops as the
player's (remote A / Select / Start, Ctrl-C -> the guard stop); at the end it holds the ready pose until Ctrl-C.
```bash
python scripts/measure_motor_delay.py -c g1_pingpong_real --dry-run                     # the plan and the robot state's fields, no torque
python scripts/measure_motor_delay.py -c g1_pingpong_real --joints arms --log \
    --compare logs/pingpong-sim/motor_delay_sim_reference.json                         # ~1.5 min; then: --joints legs waist
```
Per joint: dead time (command to 10 % of the step), t63, overshoot, share of the step reached, the sine's delay and gain; with
`--compare` the difference to the simulation (the robot's EXTRA delay: training covered 0-20 ms). The simulation reference is
`-c g1_pingpong --joints all --log` (logs/pingpong-sim/). Read: one joint much slower than its mirror = that motor; all joints alike =
the system's latency; growing over a session = heat (repeat at the end of a session, and once on a full battery). Tested: a stand-in
robot with a known delay (20 ms recovered within 0-3 ms; the remote's stop held 40 ms stops, a one-packet blip does not; the guard stop
runs), and MuJoCo with 20 ms added to every motor (measured +20 ms on all 15 joints). Not yet run on the robot.

**B7a. Pelvis pivot (do it at the first floor stand).** Motive's pivot of the pelvis rigid body is where Motive put it (the markers'
centre unless moved), not the pelvis origin the policy was trained with; every target the policy sees is measured from the pelvis, so
an offset shifts them all. In simulation a pivot 10 cm low cut the returns from 100 % to 42 % (64 balls; corrected: 98 %). The player
checks it: while the robot stands on flat feet, the legs (joint angles + IMU, `pingpong/deploy/pelvis_height.py`) give the true pelvis
height, and every 5 s and at the end it logs
```
[player] pelvis height (feet flat, ...): legs 0.761 m, perception 0.668 m ... -> perception reads the pelvis 9.3 cm LOW: ...;
perception: --pelvis-pivot 0 0 +0.093 (added to its pivot z now, if one is set)
```
Restart perception with that value (`run --calibrate-first --pelvis-pivot 0 0 0.093`); it is saved in table_frame.json and kept by
later calibrations (`0 0 0` clears it). The next run should log `agree (within 5 mm)`. Only valid with the robot's weight on its feet
(hanging in the gantry with level feet looks the same to the legs); the sim's ready stance is 0.76 m. Redo it whenever the pelvis rigid
body is edited in Motive. Only the vertical part is measured; x / y stay Motive's.

**B8. Balls:** slow, well-placed serves from a launcher first; fit the ball and rubber models (`ball.yaml`) to real flights before judging.

**B9. Recording a session (where do the misses come from?).** Run the player with `--log` (logs go inside this workspace, git-ignored: `logs/pingpong/` for the robot, **`logs/pingpong-sim/` for every simulation player**: `--log` alone names the file `<shots|shadow|manual>_<date>_<time>.json`, `--log name.json` or a relative path puts it there too; `--script-plans name.json` also finds a shot log there by name): the runtime's shot recorder
keeps, per ball, the samples perception sent, every plan applied (the predicted ball at the strike, the racket target) and, every
control step, the paddle face from the joint angles. Then, on any computer with the pingpong folder:
```bash
python scripts/run_pingpong.py -c g1_pingpong_real --log      # robot -> logs/pingpong/shots_<date>_<time>.json, written when it ends
python scripts/deploy/analyze_shots.py <RoboJuDo>/logs/pingpong/shots_20261102_1430.json --shots    # pingpong folder
```
It splits each shot into the ball model (predicted vs seen ball at the strike), timing (swing vs ball at the hitting plane), the racket
(closest approach of the face to its target) and the ball's closest approach to the face. Compare with simulation, same command with
`-c g1_pingpong --log sim.json` (8 balls, all returned: ball model 5 mm, racket 27 mm ~40 ms after the commanded strike, ball 42 mm from
the face centre, 8/8 within the blade). The racket comes from the joints, not a tracked paddle. Motive at 120 Hz (the lab, 2026-11-02)
gives the estimator half the samples of the simulations (240 Hz): `--sensing optitrack120` simulates it.
**Perception guards (2026-11-02, from the first real shot logs).** Perception now (1) turns a pelvis that Motive solved upside down back
(a 180 deg turn about one of the rigid body's axes; the status line says `PELVIS UPSIDE DOWN in Motive N frames` - fix the rigid body:
more markers, asymmetric, clear of the arms), (2) drops ball candidates below the table top over the table (the glossy table's mirror
image; `table reflections dropped N`), (3) declares "turned" from the last 50 ms (a line fit, 3 cm back from the ball's nearest point)
instead of the last 3 samples: with 6 mm ball noise the old rule fired on every slow ball rising after the bounce (80/80 in a
synthetic test, 0/80 now) and stopped its plans; real hits are still detected (80/80, ~7 cm after the hit). "stalled" now also needs the
samples to sit within 2 cm. The recorder adds `joint_names`, `q`, `q_target` per step and `tilt` / `grav_err` (the pelvis pose against the
IMU's gravity: ~0 good, 100-180 upside down); analyze_shots prints both (joint target minus angle at the strike; shots with a bad pose).
`--racket-z-offset M` (player) adds M to the racket target's height in the policy's observation only: in simulation with a +90 g paddle
it did NOT help (0 cm 98.4 %, +4 cm 92.2 %, +7 cm 78.1 %: the policy follows only ~30-55 % of the shift and the sim arm barely sags) -
default 0; use it on the robot only as a logged experiment (the metadata records it) after the joint log shows where the arm sags.

**Weak tracking (opt-in, `run_perception.py ... --track-weak`).** A stickered ball seen with only 1-2 stickers (centre 1-2 cm off) still
helps the shot detector start and follow the ball - bridging frames the camera coverage loses (the lab: 17-29 % of frames without a
3-sticker sample over the far half) - but is never sent as a ball sample: the estimator sees 3+ stickers only; a shot still needs 2
good samples to start and "turned" looks at good samples only. Status line: `ball followed through 1-2 stickers N frames`. In
simulation with the lab's noise and far-half losses (`--sensing lab --far-drop 0.12`, 64 balls each): lost shots 4 -> 1, late plans
25 -> 21, returns 96.9 vs 95.3 % (1 ball, not significant). Try it on the robot and compare the logs' start x and lost shots.

**Ball noise (2026-11-02).** The lab's stickered ball scatters ~6 mm per axis after the sphere fit (first shot log), not ~1.4 mm as
simulated; with the estimator told 1 mm it reads the scatter as spin and its first plans miss by 23-31 cm. `g1_pingpong_real` now tells
it 7 mm (`PingPongPolicyCfg(noise=0.007)`; `--ball-noise-mm` overrides): in simulation with the lab's noise (`--sensing lab`: 120 Hz,
5.5 mm per sticker marker) first plans 235 -> 34 mm off, returned 89 % -> 94-97 % (64 balls). Perception's `--noise` can stay 1 mm.
**Capture times (2026-11-02).** Times used to be the arrival minus the sender's age, which carried the arrival's jitter (120 Hz frames
logged 6.6-8.3 ms apart, some 0.6 ms). Both receivers (perception from the relay, the player from perception) now take the sender's own
times - Motive's frame times, perception's capture times - plus the median offset to their clock (`pingpong/deploy/mocap.py` ClockMap):
the frames are back on the exact grid, the mean latency is unchanged. Perception's status and the player's 5 s line show `arrival jitter
removed p95 x ms` (how much was taken out). A relay without Motive's timestamps falls back to arrival minus age.

**Planner timing (2026-11-02).** On the robot the first plan's job took 36 ms median, 80 ms p95 (session 1751; re-plans 7 / 37 ms), each
20 ms one step of preparation lost. The jobs already run on a worker thread (the 50 Hz loop never waits for them, `n_busy` 0), so the
question is whether the CPU is slow or the thread waits for Python's interpreter lock. On the robot, player stopped, pingpong folder:
```bash
python scripts/deploy/planner_timing.py <RoboJuDo>/logs/pingpong/shots_1751.json     # robojudo env, PINGPONG_ROOT set; ~3 min
python scripts/deploy/planner_timing.py <RoboJuDo>/logs/pingpong/shots_1751.json --modes thread,process --extra-thread 500 0.2   # if inconclusive
```
It replays the log's balls through the player's runtime and times the jobs `alone` (nothing else running), as `thread` (as deployed, with
the 50 Hz loop and a receiver thread) and as a separate `process`, beside the live times. alone ~ live: the CPU is slow (the planner
must do less); thread ~ live >> alone: the lock (a planner process helps if `process` ~ `alone`); both << live: something the replay
does not run (the SDK / ROS threads: `--extra-thread`). On an x86 desktop (6 shots): first plan 13 ms alone, 18 thread, 22 process; with
a 500 Hz x 0.5 ms Python thread added, thread mode stalled once (re-plan 69 ms, the loop 50 ms late), process mode not at all.

**B10. Testing the planner and the policy apart (2026-11-02).** They meet only at the 12 command numbers of the swing clock.
*The planner alone* (offline, any computer with torch): would the ball have gone back with a perfect racket?
```bash
python scripts/deploy/planner_check.py shots_1751.json --checkpoint <run>/model_N.pt --shots     # pingpong folder
```
It fits the real ball in hindsight (all its samples, bounce and spin), puts an exact racket where and when each plan said, and flies
the ball with the sim's contact physics. Session 1751: of the 20 balls seen after their bounce with a sound fit, the LAST plan returns
18 (90 %; 16/17 when the ball was seen up to the strike, ball error 15 mm), the FIRST plan 3 (ball error 139 mm: the early plans
are far off, the re-plans fix them). 10 balls were lost at or before the bounce, 7 tracks were not one ball's flight (perception).
*The policy alone* - shadow swings, no ball: the player replays a log's plans to the swing clock at the times they were applied live
(`pingpong/deploy/script_plans.py`), ignoring balls; perception runs as usual for the pelvis. The recorder then gives the racket
against the same targets:
```bash
python scripts/run_pingpong.py -c g1_pingpong_real --script-plans shots_1751.json --log      # robot (the shot log by name from logs/pingpong/)
python scripts/run_pingpong.py -c g1_pingpong_inproc --script-plans shots_1751.json --log    # sim reference -> logs/pingpong-sim/shadow_*.json (finds the robot's shots_1751.json in logs/pingpong/)
python scripts/deploy/analyze_shots.py <RoboJuDo>/logs/pingpong/shadow_<date>_<time>.json        # pingpong folder: its racket rows
```
The first scripted swing starts 3 s after the policy runs, each next one 1 s after the last swing is over (37 shots ~1.5 min); the
robot steps where the plans say. On the robot it keeps standing when done (Ctrl-C writes the log); in simulation it serves no ball and
ends. Same plans (1751): simulation racket 40 mm from its target (median; p90 172), normal 5.6 deg, +47 ms after the commanded strike;
live on the robot 101 mm (flips corrected), normal 18 deg, z -58 mm. The robot's shadow run separates the policy + robot from
perception's pelvis pose and the ball. A paddle rigid body in Motive (3-4 markers on the blade back) would check the joints' FK.

*The policy alone, live: you are the planner* (2026-11-03). The player takes swings typed in a console instead of the planner's
(`pingpong/deploy/manual_plans.py`; balls ignored, perception runs for the pelvis):
```bash
python scripts/run_pingpong.py -c g1_pingpong_real --manual --log    # terminal 1 -> logs/pingpong/manual_<date>_<time>.json
# simulation: -c g1_pingpong_inproc --manual --window --log   (--window opens the MuJoCo viewer; --manual paces it to real time; log in logs/pingpong-sim/)
python scripts/deploy/planner_console.py                                                          # terminal 2, pingpong folder
```
A bare swing name (`fh_t100`, `bh_t125`, `bh_flick_t100`, ...; `list` shows them) is that swing's DEFAULT target: the typical row
of the planner table the run trained on (not the clip's own strike: the planner's face normal comes from the ball's physics and
differs from the clip's by 35-105 deg). Nudge it: `fh_t100 dz=-3cm dy=2cm`, `in=1.2` (s to the strike), `step=0.2,0` (forward,
left; the target moves with it), `speed=1.2`, `up=10` / `yaw=-5` (face normal, deg), `random [swing]` (a random training row),
`replan dz=2cm` (during the swing), `again`, `status`, `help`. Refused, with the reason: y/z beyond the planner's reach box (y +-25,
z -30..+10 cm from the swing's strike), a step beyond the run's footwork, a strike sooner than the swing's lead, a second swing while
one is on, a swing the run never trained with targets (`bh_banana_t100`). Allowed with a WARNING: x, normal or speed outside the
training rows. After each swing the console gets its result: the paddle face's closest approach to the target and when (as
`analyze_shots.py`), the normal's error, the pelvis against the station. The player listens on 127.0.0.1:7790; a console on another
computer: `--manual-host 0.0.0.0` on the player, `--host <robot>` on the console.
Simulation (inproc, 2026-11-03), defaults: fh_t100 3-13 mm, bh_t100 8, fh_t125 19, flick_t100 17, push 52, fh_high 13 mm at
+33..+50 ms, normal 3-11 deg; nudged fh_t100 dz -5 / +5 / dy -8 cm: 1 / 19 / 22 mm; step 0.3 m: 10 mm; up 10 deg: 6 mm, 3.4 deg;
a re-plan dz -4 cm mid-swing: 4 mm. FIRST FINDING: `random bh_flick_t125` row 600 (racket 2.3 m/s mostly sideways) makes the
policy present the OTHER face (normal 160 deg off, reproducible); the default and 5 other rows 9-15 deg.

## Watchdogs and stops

Every stop on the robot is RoboJuDo's guard stop: **legs and waist go limp at once (damped), the arms brace in the guard pose. The
robot falls unless the gantry or rope holds it** (in MuJoCo the pelvis drops from 0.79 to 0.39 m in 1.5 s). The control loop carries
the guard out, so after a stop the player keeps running and holds it **until Ctrl-C**; then it ends and the motors are released.
With `--log`, the record is written the moment the stop begins (on a thread, the stepping goes on) and again, whole, when the program
ends: a player that is killed instead of ended with that second Ctrl-C still leaves its log in `logs/pingpong/` (`logs/pingpong-sim/` in simulation).

| what stops | when |
|---|---|
| remote **A**, **Select** or **Start** | during play, and during the start-up ramp and blend (read from the robot's raw remote state there; held for 2 control steps, 40 ms, so one corrupted radio packet does not stop it) |
| **Ctrl-C** in the player's terminal | any time; a second Ctrl-C ends the program |
| watchdogs | perception's pelvis pose not fresh for `--stale-steps` steps (default 15 = 0.3 s), pelvis below 0.45 m or tilted more than `--max-tilt-deg` (40), a NaN in the targets |

Before perception is up nothing is commanded (the player gives up after `--wait-perception` s without moving). Without a stop button
on the remote the player refuses to apply torque. **A stop needs the player running**: if it is killed or hangs, the last command
stays - keep the rope on and Unitree's own stop (power) at hand. Not run on the robot: tested against a stand-in robot (the remote's
button word, start-up, the hold) and RoboJuDo's MuJoCo environment. A ball that comes while the robot is still in a swing is not
planned (`shots while busy`).

## Things to know before the first run

* **Targets beyond the joint range:** in about 5 % of steps the policy's targets lie past the stop (left ankle pitch up to ~0.6 rad,
  waist pitch ~0.5, waist roll ~0.4; printed at the end of a run). Training does the same, so it is what the policy expects; the motor
  pushes into the stop with up to ~14-17 N m. Watch ankle and waist temperatures; decide on clipping after a dry run.
* **Pelvis pose latency:** OptiTrack's pose is 10-30 ms old; the player carries it forward to the present at its fitted velocity.
* **Planner jobs on a worker thread** (`threaded=True` on the robot and in `g1_pingpong`), with numba's `nogil`, so a 20 ms job never holds
  up the 50 Hz loop; one math thread per process.
* **The ball model** (`ball.yaml`, from published values) predicts every plan: fit it to your ball, table and rubber.
* **Paddle mount:** about 2 degrees is tolerable (1 degree about the blade's long axis); weigh the parts `robot_parts.yaml` marks `estimate`.
* The robot must stand roughly at home, facing the table (within ~10 degrees), when the first ball comes.

## What is tested

| | |
|---|---|
| unit tests (A1) | shot detector (real flights, clutter, dropouts, a stray marker next to a static one), table frame, emulator, raw packet + UDP + calibration, perception messages + their wire format in both clock modes, table rigid body + marker layouts, stickered ball (centre within 1.4 mm p50 from 4+ markers, 2 mm from 3) |
| stickered ball, `sim_runtime.py --ball_stickers 6` (pingpong), 512 balls | returned 90.0 % vs 90.6 % with a one-marker ball (standard balls; 5 vs 2 balls differ, not significant), 86.9 % vs 86.5 % (topspin); every ball detected. Simulated: 8 cameras round the table 2.7 m up, a sticker seen when 2 face it within 65 deg, 10 % missed |
| `live_mujoco.py` on the shared code | identical to the original script shot for shot (224 shots, 4 settings) |
| `scripts/deploy/sim_runtime.py` (pingpong) | 512 balls: 90.8 % with emulated OptiTrack vs 92.8 % with ideal inputs (not significant); splitting perception out changed none of the 512 shots |
| the two programs in simulation (A2) | 8/8 and 7/8 on two ball sequences, the same as the one-process run; perception calibrated exactly; either start order; 0 lockstep timeouts |
| perception against `fake_relay.py` (real-time timestamps) | calibrates the relay's frame, 240 frames/s, shots detected |
| `natnet_relay.py` against a fake Motive (pingpong `tests/deploy/test_natnet_relay.py`) | NatNet 4.1 packets as the lab client parses them, y-up: pelvis pose, the table's 8 markers (occluded left out), unlabeled markers, latency 4.5 ms from the camera timestamps; the chain fake Motive -> relay -> perception calibrates the table exactly and detects the stickered ball's serves |
| pelvis pivot (`tests/deploy/test_pelvis_pivot.py`, dual sim) | legs' FK = MuJoCo's to 0.0000 mm (100 random poses); `--pelvis-pivot` reads the truth within 0.06 mm at any heading / tilt / rigid body axes; dual sim with a 10 cm fake pivot (`--fake-pivot 0 0 0.10`): the player measured +0.100, returns 100 % / 42.2 % / 98.4 % (no pivot / uncorrected / corrected, 64 balls each) |
| **not tested** | anything with a real Motive or on the G1: the relay on a real stream, `UnitreeCppEnv` with this policy, `prepare()` to the ready pose, the remote's stops, motor delay, the real ball model, the watchdogs against a real fault |
