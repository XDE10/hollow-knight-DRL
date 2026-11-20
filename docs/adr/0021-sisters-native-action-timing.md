# ADR 0021. Learn from the timing of Sisters attacks and executed controls

## Decision

The optional timing encoder measures the current Knight, up to three Sisters and up to two enemy projectiles on the same game update that selects the next input. For a Sister it records her current game-reported animation state, its observed duration and the Knight's relative position at the last observed state change. A Sister can begin an attack before her collision box appears. In that case the encoder uses her transform position and records the missing box. Entity IDs link successive observations. Within each family the current slots are ordered by distance, and absent slots are masked. Previously executed key transitions are timed per fight. The resulting 192 values are computed locally from the existing observation and action interfaces.

The actor shares one learned encoder across Sisters and another across discs. Masked maximum pooling and the ordered observations feed a two-layer policy. Joint action logits include learned movement, aim and button terms. The native binary stores the combined action head and the action catalogue. Earlier 56-feature policies retain their original weights and native layout.

## Learning

The optional `--return-seconds` horizon credits an executed input with the discounted rewards that followed it on the same development tape. With a positive horizon the critic averages its existing one-step world-model target and the short real-trajectory target. The latter bootstraps at the last recorded successor and stops before an unobserved interval. Death and arena completion end the return. Subsequent actions came from the development policy, so this target estimates outcomes under those continuations. The training report includes held-out errors for the action and state values against this target. The replay packer's `--original-speed-only` option selects by recorded game speed, retaining wins and losses alike.

The development recorder asks the teacher for an action on the state actually visited. It records that label, the student's choice and the applied input separately. The recorder can hold a sampled teacher/student selection for a short segment to visit states that diverge from a demonstration. A whole episode goes into exactly one data split. The collector records individual damage and arena completion events.

The actor first learns to reproduce the teacher's input. Offline IQL then fits two action-value functions and an expectile state value using the executed controls and measured rewards. The dynamics model estimates the next physical features. Its predicted successor contributes one quarter of the value bootstrap as described in [ADR 0020](0020-sisters-native-offline-rl.md). The actor's RL update weights applied actions by estimated advantage while retaining corrective teacher supervision. Only the actor is needed to play. The learner saves the actor, both critics, value and dynamics parameters.

The trainer supports two data choices for experiments. Horizontal mirroring assumes a symmetric teacher. Use `--no-mirror` to retain the original left and right labels when its preferred resting position is asymmetric. Setting `--switch-weight` above one increases the corrective loss on states where the teacher asks to change the last executed input. The report includes accuracy on these states and on states where keys should remain held. Only a live fight tests the consequences of missed or repeated keys.

An optional supervised sampling filter, `--teacher-agreement-seconds`, addresses a property of this stateful teacher. When its recommended attack is overridden, it can still start an internal attack timer. The filter keeps the label on the first disagreement and temporarily excludes later labels after a disagreement. It starts afresh for each fight. All eligible executed transitions remain available for the world model and IQL. The interval is an experiment setting and does not establish exactly when every teacher timer recovers.

## Reproduction and evaluation

An optional auxiliary loss trains the policy encoder to anticipate where enemies will be. `--forecast-weight` enables a training head that predicts proximity to observed body and disc collision boxes 0.12, 0.32 and 0.60 game seconds later, at nine locations around the Knight's current horizontal position. Targets are computed separately within each recorded fight. Missing future frames have zero loss weight. These are forecasts under the recorded behavior. The one-step action-conditioned dynamics model still supplies the IQL bootstrap. Only the current observation and earlier executed keys enter the exported actor; the future-box head is saved with the learner for inspection.

Collect development fights using `scripts/collect_sisters_native_labels.py` with the native timing recorder enabled. Pack the resulting manifest paths into a new archive and extract it before training.

For phase-specific training, `--opener-policy` accepts a frozen 192-feature actor. The trainer selects transitions from the first observed second-phase Sister onward for both fitting and held-out metrics. `--initial-learner` optionally initializes the student encoder and heads from an earlier checkpoint, matching joint-action rows by their actual movement, aim and button tuple. Both actors are embedded in the exported file. The mod uses the opening actor until it observes a second-phase Sister, then uses the late actor for the rest of that fight. Each network selects every button for its phase. The opening actor's exact binary and its checksum are preserved, so a change to the late policy cannot change the opening policy weights.

```sh
python scripts/pack_sisters_timing_replay.py --manifests "$MANIFEST" --output development.zip
python -m zipfile -e development.zip timing-data
python scripts/train_sisters_timing_iql.py --data timing-data --output candidate
```

For an independent battle, launch the mod with `HKRL_NATIVE_ACTOR` pointing to the exported `world_iql.bin`, an episode receipt path in `HKRL_NATIVE_ACTOR_LEDGER`, and a nail multiplier of one. Clear the teacher and development recorder settings. Run `scripts/eval_sisters_native.py --checkpoint candidate/world_iql.bin --port "$PORT" --episodes 100 --seed "$FIRST_SEED" --output evaluation` on fresh evaluation seeds. The evaluator checks the opening health, episode identity, original speed including normal hit-stop, model digest and a live arena completion event. It records health at the first observed trio frame, healing, sampled dash starts and the source of each damage event. Short dashes between observations can be missed. Boss HP in the final trio snapshot covers the entities seen at that time. Evaluation, training and teacher-controlled collection use separate output directories.
