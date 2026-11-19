# ADR-0018 — Distilling object-world forecasts for Sisters of Battle

The Sisters arena runs on the local game PC. The world model trains on a GPU and the distilled policy runs inside the local actor. This keeps controller timing independent of the network. The frozen first-stage movement head remains available under `models/sisters-tactics-v1`.

## Replay

Use whole real fights with verified outcomes, timestamped player and entity snapshots, and the native controls actually applied by the tactics actor. The outer Gym action is neutral in this configuration. A training example integrates the held controls in four bins per observation interval. Local entity IDs match bodies and discs across frames. The replay retains all observed entities and masks padded slots. Each complete fight belongs to one split, and the split registry persists across additions to the corpus.

The last interval can contain death or arena completion after the final usable combat observation. Train those event labels and mask the missing next geometry. Reserve formal evaluation fights for the final score.

## Cloud training

Pack verified development trajectories on the game PC. Run these commands from the repository root, supplying your collection directories and a writable output directory.

```bash
python scripts/pack_sisters_world_replay.py --root "$RUNS" --populations "$DEVELOPMENT" --output "$DATA"
```

Copy the packed directory, `scripts/sisters_world_data.py`, both cloud training scripts and the frozen base head to Colab. From the directory containing the scripts on an attached GPU:

```bash
python train_sisters_object_world.py --data "$DATA" --output "$TEACHERS" --updates 2200 --batch 160
python distill_sisters_object_world.py --data "$DATA" --models "$TEACHERS" --base "$BASE_HEAD" --output "$STUDENTS" --samples 55000 --steps 1600
```

Three independently initialized teachers encode the player and masked entity set with attention. An action-conditioned GRU forecasts five future intervals. The readouts predict player motion and resources, matched entity geometry and presence, entity state, spatial occupancy and eight combat events. Calibration uses complete held-out fights. These short forecasts are used to compare 12 movement, jump and dash alternatives while retaining the observed offensive controls. The ensemble disagreement reduces the correction on uncertain alternatives. The uniform-weight comparison uses the same training windows.

The student consumes the last four native observations and returns 12 centered movement preference adjustments. It runs alongside the saved movement head and the local tactics planner. It has no API for changing damage, health or game state. Separate real-game trials measure the performance of this combined policy. A native counter records how often the student's residual changed the selected movement plan. The real game also applies the planner's final collision and input rules.

With `--object-history`, the teacher additionally receives each entity's visible age, time in its current FSM state and two velocity derivatives. These are computed from current and earlier observations of the same entity. A disappearance resets the derivative and both clocks. This lets the forecast distinguish the beginning of an attack telegraph from its final frames. The student format stays compatible with the same local game actor.

The optional recovery rules spend SOUL on a heal during a paired wall attack only while the Knight stands near the platform centre before a disc is visible. For a downward spell, the optional timing rule checks collision clearance during both the vulnerable windup and the return to ground control. It also checks where an emergency shadow dash will end. All three actions use the ordinary game input path.

## Evaluation

Freeze the policy configuration, both weight files and the compiled actor before the first evaluation reset. The arena is Attuned Sisters of Battle at 1×, with the normal 21-damage nail, nine masks and empty SOUL. Each trial begins with the 500 HP opening Sister and the next phase has three 750 HP Sisters. A win requires the arena-complete event, a living Knight and normal termination. Count deaths and timeouts as losses. Record actual full-game outcomes, first-phase clears, damage, body/projectile/spike masks, and resets with infrastructure errors. The world model's event probabilities and the wins in its replay corpus never enter the win-rate numerator.
