# Control Pipeline Guide

> 中文版:[deploy/docs/control_cn.md](control_cn.md)

How [deploy/run_pi0_robot.py](../run_pi0_robot.py) gets from "policy output" to "arm moves + gripper closes" on the real robot. Read this before tuning anything.

## One-glance overview

```
┌──────────────────────┐    state(7) + 2 RGB images    ┌──────────────────┐
│  RealSense × 2       │ ────────────────────────────▶ │                  │
│  + UR getActualQ()   │                               │   pi0 + LoRA     │ action chunk
│  + Robotiq GET POS   │                               │   policy         │ (50 × 7)
└──────────────────────┘                               └──────────────────┘
                                                                │
                            ┌───────────────────────────────────┤ pop one (7,) per step
                            │                                   │
                            ▼                                   ▼
              ┌───────────────────────────┐
              │ Safety clamp              │
              │ |Δq|.max() < 0.10 rad?    │
              │ over → hold prev target   │
              └─────────┬─────────────────┘
                        │ q[:6] joints           │ q[6] gripper
                        ▼                         ▼
              ┌─────────────────────┐   ┌──────────────────────┐
              │ rtde_control        │   │ socket :63352        │
              │   .servoJ(...)      │   │   "SET POS 0..255\n" │
              │   30Hz, gain=300    │   │                      │
              └────────┬────────────┘   └─────────┬────────────┘
                       │ RTDE @ 500Hz             │ TCP / ASCII
                       ▼                          ▼
            ┌──────────────────────┐   ┌──────────────────────────┐
            │ UR controller        │   │ Robotiq URCap → driver   │
            │ internal interp      │   │                          │
            └────────┬─────────────┘   └─────────┬────────────────┘
                     │                            │
                     ▼                            ▼
              6 joint servos                 gripper open/close
```

---

## Stage 1 — Policy inference (pi0 + chunk queue)

[deploy/run_pi0_robot.py:404-409](../run_pi0_robot.py#L404-L409)

```python
with torch.no_grad():
    action = policy.select_action(obs)   # 7-d numpy
target_joints = action[:6]               # 6 joints (radians, absolute position)
target_gripper = float(action[6])        # gripper [0, 1]
```

pi0 does NOT run a forward pass per control tick. It maintains an internal
**action queue of length 50** (`chunk_size=50`, ≈ 1.66 s at 30 Hz):

- **Queue non-empty**: `select_action` pops one frame (a few ms — pure
  GPU↔CPU copy).
- **Queue empty**: re-runs the diffusion forward to generate the next 50-frame
  chunk (~430 ms measured).

This is what lets pi0 hold a 30 Hz closed loop. Cost: every 50 steps you take
a ~430 ms latency spike (the `step 51 / 101 / 151... control loop slow`
entries in the log). Can't be removed without architectural changes.

Steady-state ceiling: `50 steps × 33 ms + 1 chunk × 430 ms = 2080 ms / 50
steps = 24 Hz`. To hit a sustained 30 Hz you'd need async chunk generation
(background thread pre-generates the next chunk; main loop only pops).

---

## Stage 2 — Safety clamp

[deploy/run_pi0_robot.py:412-417](../run_pi0_robot.py#L412-L417)

```python
if prev_target is not None:
    delta = np.abs(target_joints - prev_target).max()
    if delta > args.max_joint_delta_rad:    # default 0.10 rad
        print("⚠ refusing to send.")
        target_joints = prev_target          # freeze on last frame
```

- Threshold `0.10 rad ≈ 5.7°`, equivalent to **3 rad/s ≈ 172°/s** at 30 Hz.
  About the upper edge of "normal demo speed".
- Trigger: predicted next target diverges from the **last actually-sent**
  joint angles by more than the threshold.
- On trigger: don't send, restore `prev_target`. Arm holds still.
- Subtle: **pi0's chunk queue still advances** — `select_action` pops a frame
  every iteration regardless. The frame just gets ignored. So during a
  "frozen" stretch the chunk is still being consumed.

When to relax?
- Model is well-trained AND the action is genuinely fast (e.g. quick avoidance)
  → `0.15 ~ 0.20`.
- **Undertrained model → DON'T** relax. Relaxing means letting the arm flail.

When to tighten?
- First-time test, low trust → `0.05`.
- Heavy or fragile payload on the EE → `0.05`.

---

## Stage 3 — Joint control (servoJ + RTDE)

[deploy/run_pi0_robot.py:328-339](../run_pi0_robot.py#L328-L339)

```python
def send_joint_target(self, q, dt=None):
    sj = self.servoj
    t = dt if (dt is not None and dt > 0) else (1.0 / self.control_hz)
    self.ctrl.servoJ(q.tolist(),
                     sj["velocity"], sj["acceleration"],
                     t,                       # critical!
                     sj["lookahead_time"], sj["gain"])
```

UR's official **`servoJ`** — the realtime mode designed for streaming
trajectory waypoints online. Underneath: **RTDE protocol (Real-Time Data
Exchange, 500 Hz)**, wrapped by the `ur_rtde` C++/Python library through
`rtde_control.RTDEControlInterface`.

| Param | Meaning | Reasonable start | Effect of changing |
|---|---|---|---|
| `q` | 6 target joint angles (rad) | — | Model output, don't tune |
| `velocity` | Controller-internal speed cap (rad/s) | 0.5 | Up=more aggressive |
| `acceleration` | Controller-internal accel cap (rad/s²) | 0.5 | Same |
| **`time`** | **"Reach `q` in this many seconds"** | **measured loop dt** | **Mismatch = jerk, see below** |
| `lookahead_time` | Controller's predict horizon (s) | 0.10 | Up=smoother but laggier; down=snappier but jittery (0.03-0.20) |
| `gain` | Closed-loop stiffness | 300 | Up=responsive but oscillates; down=damped (100-600) |

### `time` MUST equal the actual loop period

`time` is "blend to target over this many seconds". If your loop **actually
runs at 80 ms** but you tell `servoJ time=33ms`, the controller will:

- t=0: get cmd A, **sprint** to A in 33 ms → idle for 47 ms
- t=80: get cmd B, **sprint** to B in 33 ms → idle for 47 ms
- ...

→ **Sprint-then-idle jitter**, not smooth servoing.

Our implementation [deploy/run_pi0_robot.py:467](../run_pi0_robot.py#L467)
feeds the previous iteration's actual wall-clock period `prev_loop_dt` to the
next `servoJ`'s `time`:

```python
prev_loop_dt = max(time.perf_counter() - tick, period)
```

So no matter how slow the loop runs, motion is **continuous and smooth** —
just played back at 0.8x / 0.5x / etc.

### RTDE's two interfaces

- `RTDEControlInterface(ip)` → send commands (servoJ / movel / forceMode /
  ...), `servoStop()` / `stopScript()` to exit.
- `RTDEReceiveInterface(ip)` → read state (`getActualQ()` /
  `getActualTCPPose()` / ...).

Two separate connections, share the UR's RTDE port 30004.

---

## Stage 4 — Gripper control (Robotiq URCap socket)

[deploy/run_pi0_robot.py:218-260](../run_pi0_robot.py#L218-L260)

The gripper does **NOT** go through RTDE. The Robotiq URCap (the PolyScope
plugin) opens its own TCP port `63352` inside the UR control box. The
protocol is dead-simple ASCII:

| Command | Meaning |
|---|---|
| `SET ACT 1\n` | Activate (first time triggers a full open-close self-test) |
| `SET GTO 1\n` | Enable go-to-position mode |
| `SET SPE 200\n` | Speed (0-255) |
| `SET FOR 100\n` | Force (0-255) |
| `SET POS 0\n` | Fully open |
| `SET POS 255\n` | Fully closed |
| `GET POS\n` | Returns current `POS <0..255>\n` |

Our wrapper [deploy/run_pi0_robot.py:238-251](../run_pi0_robot.py#L238-L251):

```python
def write_position(self, value: float) -> None:
    """value ∈ [0, 1]: 0=open, 1=closed"""
    raw = int(round(max(0.0, min(1.0, value)) * 255))
    self._cmd(f"SET POS {raw}")
```

Policy outputs gripper in `[0, 1]` float; we map to `[0, 255]` and send.

### Binarization in the main loop

[deploy/run_pi0_robot.py:421](../run_pi0_robot.py#L421)

```python
robot.send_gripper(1.0 if target_gripper > args.gripper_threshold else 0.0)
```

The 3D-printer dataset's gripper is essentially binary (open the door = grab
handle), so thresholding is more robust. For tasks with continuous gripper
control (e.g. squishy objects), change to:

```python
robot.send_gripper(target_gripper)   # continuous
```

### URCap socket has a non-obvious prerequisite

**Port 63352 is open ONLY when a program is RUNNING on the pendant.** This
is by URCap design — the toolbar code is embedded into the UR script that
the program executes; no program → nobody listening on 63352.

Common first-deploy gotcha: gripper powered, URCap installed, but no
program running on PolyScope → `connection refused`.
[deploy/preflight_check.py](../preflight_check.py)'s step 3 catches
exactly this.

---

## Full control loop

[deploy/run_pi0_robot.py:382-468](../run_pi0_robot.py#L382-L468)

```python
period = 1.0 / args.control_hz   # 33.3 ms
prev_loop_dt = period            # last iteration's measured period

while time.perf_counter() - t0 < max_seconds:
    tick = time.perf_counter()

    # 1. observe
    joints = robot.get_joints()       # RTDE read 6 joint angles
    gripper = robot.get_gripper()     # Robotiq socket read position
    img_g, img_w = cams.read()        # 2 RealSenses, parallel

    # 2. inference
    obs = build_observation(joints, gripper, img_g, img_w, task)
    action = policy.select_action(pre(obs))   # pi0 chunk pop
    action = post(action).numpy()              # de-normalize → (7,)

    # 3. safety
    if abs(action[:6] - prev_target).max() > MAX_DELTA:
        action[:6] = prev_target              # refuse, freeze

    # 4. send
    robot.send_joint_target(action[:6], dt=prev_loop_dt)   # servoJ
    robot.send_gripper(action[6] > THRESHOLD)              # Robotiq SET POS

    prev_target = action[:6]

    # 5. pace + record this iteration's actual period
    elapsed = time.perf_counter() - tick
    if elapsed < period:
        time.sleep(period - elapsed)
    prev_loop_dt = max(time.perf_counter() - tick, period)
```

---

## Tuning cheat-sheet (edit [deploy/configs/run_pi0_robot.yaml](../configs/run_pi0_robot.yaml))

| Symptom | Knob | How |
|---|---|---|
| Clamp fires every step, arm doesn't move | `control.max_joint_delta_rad` | **Don't** relax it — it's almost always undertraining. Train more. |
| Arm moves stiffly, sprint-then-idle | `robot.servoj.lookahead_time` | 0.10 → 0.15 |
| Arm jitters at high frequency | `robot.servoj.gain` | 300 → 200 |
| Arm lags target | `robot.servoj.gain` | 300 → 400 |
| Gripper too slow | `RobotiqGripper._cmd("SET SPE ...")` | 200 → 255 (in code) |
| Gripper grip too weak | `RobotiqGripper._cmd("SET FOR ...")` | 100 → 200 |
| Loop runs slow (<20 Hz) | Profile chunk boundaries / cameras / GPU | dry-run, watch `dt=` |
| Task completely fails | **Undertrained** | `train/train_pi0.py --steps=30000 --resume=true` |

---

## Troubleshooting

| Symptom | Most likely cause |
|---|---|
| `RTDEControlInterface` hangs / errors | Pendant not in Remote Control mode, or another RTDE client still connected |
| `connection refused` to port 63352 | No program with the Robotiq toolbar running on the pendant |
| `servoJ(): incompatible function arguments` | `ur_rtde` version changed servoJ signature; we now use positional args |
| Gripper opens-closes once then crashes | `SET ACT 1` self-test; the crash itself is elsewhere — check stack |
| 100% of steps trigger `max joint delta` | Undertrained model; see above |
| `control loop slow` every 50 steps | pi0 chunk boundary; unavoidable without async |
| `control loop slow` every step | Camera I/O or GPU bottleneck; check `dt=` in dry-run |
