# ADR 0021. Learn from the timing of Sisters attacks and executed controls

## Decision

The optional timing encoder measures the current Knight, up to three Sisters and up to two enemy projectiles on the same game update that selects the next input. For a Sister it records her current game-reported animation state, its observed duration and the Knight's relative position at the last observed state change. A Sister can begin an attack before her collision box appears. In that case the encoder uses her transform position and records the missing box. Entity IDs link successive observations. Within each family the current slots are ordered by distance, and absent slots are masked. Previously executed key transitions are timed per fight. The resulting 192 values are computed locally from the existing observation and action interfaces.

The actor shares one learned encoder across Sisters and another across discs. Masked maximum pooling and the ordered observations feed a two-layer policy. Joint action logits include learned movement, aim and button terms. The native binary stores the combined action head and the action catalogue. Earlier 56-feature policies retain their original weights and native layout.

## Learning

`--teacher-win-min-hz` restricts imitation and movement-ranking labels to completed development victories whose measured input cadence meets the requested threshold in the fitted phase. Rate uses the elapsed game time between the first and last native decision. Failed games and slower fights still supply their executed actions, real damage, terminal outcomes and next observations to the world model and offline IQL updates. Zero uses all eligible teacher labels.

The optional `--no-hit-cost` training objective keeps the attack, death and victory rewards. It charges an additional cost for each actual mask lost and once for the first damaging event. A live arena completion with continuously observed full health and zero recorded damage earns twice that cost. Healing cannot restore eligibility for the clean-finish bonus. These terms are computed on the full development fight before selecting a phase. The independent evaluator counts the actual damage events and complete wins.

The opening specialist uses `--late-policy` to keep the deployed second-phase network byte for byte. Its subtask ends on the first observation of a 750-HP Sister. In addition to normal damage, death and attack rewards, it receives 1.25 per net mask gained and 1.5 per full soul meter gained, with signed changes for losses. Reaching the trio awards five plus half a point per remaining mask and up to 1.5 for remaining soul. Deaths before the trio remain terminal losses. The full-game evaluator still requires the Knight to survive to arena completion.

For development collectors, `HKRL_NATIVE_PLAN_TRACE=1` also records the lowest teacher-planner cost for each movement, jump and dash combination that the planner considered. Each row is aligned to the native input by episode and game time. A positive `--plan-weight` trains the actor to rank the teacher's selected motion ahead of scored alternatives using their relative costs. Only decisions where the teacher issued the same motion after its later attack and recovery logic supply this loss. Unscored alternatives and tied costs carry zero weight. The critic continues to receive real rewards for executed inputs. Deployment uses the usual neural action head, with no planner inputs or traces.

With `--motion-history`, two shared neural projections read the arena as last observed at least 0.16 and 0.48 game seconds earlier. Their input includes the measured age. The projection weights are shared between the two times. Missing history contributes zero. A gap above 0.24 game seconds or a new episode clears history. The late-phase actor receives the recent observations from the opener when phases change. These observations use the same entity masks as the current frame.

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

### Training on upcoming movement changes

The four-clear actor kept its earlier movement on 1,397 of 5,107 observed
movement changes in the held-out demonstrations. Of those missed changes, 696
persisted for another closely spaced decision with a trusted label. Given the
next real observation, the actor selected the desired movement in 489 cases.
These counts describe predictions on recorded states.

With `--future-key-weight`, an auxiliary head uses the shared actor representation
to predict direction, jump and dash 0.04, 0.12 and 0.28 game seconds ahead.
Targets come from observed controls that agreed with the teacher throughout
that interval. The trainer masks gaps, phase changes and missing labels, and
gives additional weight to an upcoming movement change. The action-value
targets still use the rewards of executed inputs. The exported actor uses only
observations and past controls available when it chooses the current input.

### Credit for the final arena outcome

The ordinary IQL discount is 0.995 per 0.04 game seconds. Its half-life is
about 5.5 seconds. In the 105-fight development archive, the 36 training wins
that reached the trio took a median of 126.7 game seconds from trio entry to
the terminal event. Short-term damage and the final win have very different
time scales.

The optional `--terminal-advantage-weight` gives executed inputs additional
credit for the observed victory or death. A Monte Carlo return uses a 90-second
half-life. A ridge regression estimates the expected return from current HP,
SOUL, observed enemy HP and elapsed phase time. Every complete training fight
has equal weight in this regression. Its own outcome is held out when estimating
its baseline. The advantage relative to that baseline multiplies the IQL actor
weight, bounded between exp(−1) and exp(1). Censored fights retain the ordinary
IQL weight. The learned world model and twin critics still use the actual
transition rewards. The exported actor chooses primitive controls as before.
