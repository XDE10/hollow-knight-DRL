# ADR-0011: Resumable bounded CPU baselines

- Status: Accepted
- Date: 2026-11-18

The game PC has CPU inference and no training GPU. Before remote/world-model
experiments, measure a bounded recurrent PPO baseline on real observations.

`scripts/train.py --run-dir ... --updates N` uses N as the cumulative target.
The session saves model, Adam, Python/NumPy/Torch CPU RNG, recurrent minibatch
RNG, counters and fully resolved train/task configs in the existing hash-checked
checkpoint registry. It rejects config drift. It resumes with a fresh game reset:
Unity, the unfinished episode and recurrent hidden state are not restored.

Periodic checkpoints default to every five updates, plus initial/final/safe-stop
checkpoints. A process crash can lose updates since the last checkpoint. A STOP
file requests exit after the current complete rollout and optimizer update.
An OS file lock prevents two local sessions from controlling the same game.

Synchronous updates pause the game through its existing control protocol. Pause,
resume, ping and timescale replies do not update the reward tracker or drain its
event buffer. The next STEP receives those events and measures ticks from the
previous transition. These are behavioral fixes, not schema changes; rebuild
the mod before using the session launcher. Input is suspended between actions
by the existing mod; physics still advances during local policy inference.

Multi-enemy evaluation counts a win only on a terminal, non-truncated transition
with a kill signal and no player death. Earlier entity kills do not establish a
win. This relies on the mod's encounter termination; phase-transition correctness
must also be checked before claiming a high Sisters of Battle win rate.

Current RESET does not send a seed into Unity. Python training RNG is seeded,
but game trajectories are not deterministic or identically seeded across methods.
Use repeated independent evaluation episodes and confidence intervals. The
existing evaluator's `time_to_kill` field is decision steps, not seconds.

A new local PPO session accepts `--init-weights` with a raw model state dict
produced by `scripts/clone_sisters_policy.py`. The manifest pins its SHA-256.
Step zero stores the initialized policy with a fresh PPO optimizer. Later
launches restore the full session checkpoint and can omit the initializer.

Local sampling now retries only TimeoutError/ConnectionError, at most two retries
per failed rollout by default. Recovery discards the received partial prefix,
reconnects, resets the episode and recurrent state, and keeps the current model
and optimizer. Logs record discarded received steps; timed-out actions may have
advanced the game without an observation, so this is a lower bound on lost
interactions. Optimizer/model errors are never silently retried. Local sessions
use a 15-second response timeout and preserve a traceback on permanent failure.

The mod explicitly enables background execution and restores the original Unity
setting when destroyed. Buffered terminal events bypass the input-commit wait:
death or scene transitions can remove the input handler before its acknowledgement.
This fixes a possible indefinite wait without fabricating a successful action.
