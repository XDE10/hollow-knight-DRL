# ADR-0016 — Mixed-difficulty live Sisters curriculum

The 50,000-decision Double DQN run collected zero observations of the simultaneous Sisters phase. This campaign starts with the frozen recurrent PPO that averaged 137.3 damage in ten original-speed fights. It collects one rollout from each of three local instances before every PPO update.

- Port 5555 and save slot 3 use original game damage throughout.
- Port 5556 runs with threefold player nail damage against active Sisters.
- Port 5557 runs with sixfold player nail damage against active Sisters.

The environment variable `HKRL_SISTERS_NAIL_MULTIPLIER` selects the training-copy multiplier at process startup. The mod rejects values above one on port 5555 or save slot 3. The game continues to execute its own hit, boss death, phase transition and victory logic. Events capture the resulting actual HP changes. Assisted fights contribute data for later attack patterns, so their damage and win metrics use their own difficulty.

Compile the mod and install it on the two disposable copies. Back up the saves before staging independent copies of slot 3 into training slots. Leave the multiplayer launcher settings on localhost. Launch the main game at port 5555 and the two copies with their documented multipliers. Run `scripts/windows/train_sisters_curriculum.ps1` with the source checkpoint and optionally a separate run directory. The launcher first verifies that the sixfold copy reaches the simultaneous phase with the frozen policy.

The first long-run budget is 204,000 real decisions, or 400 PPO updates of 510 decisions. Checkpoint every ten updates. Automatic evaluations at 51,000, 122,400 and 204,000 decisions each use ten original-speed, unassisted fights on port 5555 with the same seeds as the frozen baseline. Compare first-phase clears, full wins, damage and spike HP loss. Assisted wins measure curriculum coverage. Only unassisted wins count toward the original boss objective.
