# Evaluation

Every method — the baseline and any learned replacement — runs the same
scenarios and reports the same metrics.

## Ground truth

The leader wears a tag the system never reads: an AprilTag on the back, seen by
a separate camera, or a UWB tag. It gives the true leader position per frame.
Bystander positions come from the lidar tracks, checked by hand on a sample.

## Scenarios

| Scenario | What it stresses |
|---|---|
| Leader alone, straight and turning | Distance keeping, controller tuning |
| One passer-by crosses between robot and leader | Short full occlusion, recovery |
| Look-alike distractor walks beside the leader | Appearance-only identity |
| Leader and distractor cross paths | Track swap |
| Leader stops among standing people | Stopping distance, clearance |
| Crowd at increasing density (e.g. 3, 6, 10 people) | All of the above together |

Each scenario: at least 10 repetitions per method, same route and timing script.

## Metrics

| Metric | Definition |
|---|---|
| Follow success rate | Runs that reach the route end still locked on the true leader |
| Correct-lock time | % of the run where the lock is on the true leader |
| ID switches | Times the lock moves to a different person |
| Wrong-follow time | Seconds the robot moves while locked on someone else (target: 0) |
| Recovery time | Seconds from occlusion end to a correct re-lock |
| Distance-keeping error | RMS of leader distance minus the set distance, while locked |
| Minimum clearance | Smallest distance from the robot to any bystander |
| Safety stops | Stops triggered by the safety stage |

Report per scenario: median and spread over repetitions, plus every failed run
with its bag.
