# identity

Who the leader is, over time.

- `lock_on.py` — `LeaderLock`, the SEARCHING → LOCKED → UNCERTAIN → LOST state
  machine with an appearance gallery. It never hands the lock to a different
  person. Candidates come from injected `acquire` / `confirm` callables.
- Planned: "nearest person in front at start" acquisition, and track
  continuity from `tracking` as a second identity cue next to appearance.
