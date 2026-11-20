# ADR 0020. Offline RL at the Sisters native control frequency

## Decision

For the Sisters of Battle course experiment the Windows mod evaluates a frozen neural policy locally on a new input commit, at most once every two physics ticks. The GPU learner uses recorded game transitions. The native actor chooses movement, aim and buttons together. It reads the same 56 privileged game features as the recorder, including up to three bosses and two discs. Two historical snapshots and twelve features computed from previously executed keys describe motion and input timing. State resets at every fight boundary. This fixed-slot MLP is specific to this arena.

The first stage learns to imitate the observed controls. The second fits two action-value networks, an expectile state-value network and an advantage-weighted policy using real rewards. A separately fitted dynamics network predicts the next physical features given the state and action. The Bellman bootstrap combines value at the recorded successor (75%) and at the successor with predicted physical features (25%). Control history in both successors comes from the recorded transition. This is a one-step model-assisted offline RL experiment. It does not implement the recurrent stochastic state-space model or long imagined rollouts used by DreamerV3. [IQL paper](https://arxiv.org/abs/2110.06169)

## Data and evaluation

Split entire development fights before fitting or mirroring. Map each event to the last native decision before the end of its reporting interval. Mask incomplete transitions and end bootstrapping at a terminal event. Keep the catalogue restricted to controls represented in training. The optional corrective collection mode records the teacher's label on states visited by a mixture of teacher and learner. It stores the actually executed input separately for rewards, dynamics fitting and temporal-difference learning. [DAgger paper](https://proceedings.mlr.press/v15/ross11a.html)

A frozen evaluation uses a fresh block of fights, standard speed and damage and a process without a teacher. The evaluator verifies checkpoint digest and episode identity. It records phase progression, output damage, minimum health and separate health losses to bosses, projectiles and spikes. A mixed-control collection win never counts towards the independent policy win rate. Damage comparisons with the first warm-start control have unequal numbers of actor updates. A further control in corrective mode continues supervised updates from the same warm start for the same number of batches as the RL arm. This controls actor update count but does not isolate the contribution of the dynamics model from the rest of IQL.

## Consequences

No game binary or retail game assets are required for offline training. Running live evaluation requires an installed game, the mod and a prepared save. Previous PPO and hybrid controllers retain their individual checkpoints and evaluation records. [Results, weights and reproduction](../../models/sisters-native-iql/README.md)
