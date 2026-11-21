# ADR 0025. Train a recurrent actor and critic on native Sisters controls

The starting policy uses the present observation and two snapshots taken 0.16 and 0.48 game seconds ago. To learn how earlier movement and enemy attack phases affect an action, we keep its opening policy and train a 24-unit GRU alongside its trio controller. The GRU reads the 192 current state variables, including the previous executed keys. Each decision updates its hidden state. Memory starts fresh at trio entry, after an observation gap over 0.081 game seconds, and at the first decision at least 3.2 game seconds into a block.

The actor adds a learned correction between −3 and +3 to each of the original 79 native action scores. The game chooses the highest score and sends its movement, aim and button combination. The C# controller performs inference locally.

The critic predicts a correction to the original state value. Its target combines a four-second observed discounted return, bootstrapped with the frozen value at the endpoint, and the original one-step world-assisted target. These terms receive 75% and 25% weight. In the one-step term, the frozen value reads the observed successor with 75% weight and a successor with predicted physical variables with 25% weight. A confirmed terminal stops bootstrapping. The actor gives higher weight to executed actions with positive estimated advantage. A penalty for diverging from the original action distribution and extra weight on actual teacher-controlled inputs and changes of key stabilize the update.

Two auxiliary outputs predict whether the Knight will lose health within 0.72 game seconds, and whether it will damage the boss without losing health within 2.4 game seconds. Only the current observation and earlier game inputs enter the policy. Future outcomes supply training targets.

The experiment uses 183 completed development fights with whole-fight splits, including 148 for training and 35 held out. Three models receive identical batches. One GRU reads the current raw state at every decision, one reads the frozen encoder representation, and a control reads only one current raw state with a fresh hidden state. The original actor supplies the same base scores to all three. Training samples fights uniformly, then samples blocks within each fight.

Only the raw-state recurrent variant was nominated for the independent arena. The predeclared screen compares both future event prediction errors with the one-observation control on teacher-free held fights. It also measures how often native inputs change and how well the policy retains the horizontal inputs actually issued by the teacher. These comparisons measure prediction and imitation. A frozen controller's ability to win and avoid damage is scored in complete game runs with new arena seeds.

The Colab L4 completed 1,500 updates for each of the three variants. On 11 teacher-free held fights, the raw-state variant reduced the damage-taken prediction's mean Brier error by 0.000313 and increased the safe-output prediction's mean Brier error by 0.002382. It changed 1.69% of eligible held inputs. The latter prediction missed the predeclared condition, so this checkpoint was archived without a formal arena win rate. [Per-game results and frozen parameters](../../models/sisters-native-iql/timing-recurrent-control/)

For reproduction, restore the 157-fight assigned archive as described in the model README. Merge it with the two descent demonstration fights and the 24 randomized descent fights, keeping their original splits. Then run the following commands from the repository root. [157-fight reconstruction](../../models/sisters-native-iql/README.md)

```sh
m=models/sisters-native-iql
python scripts/merge_sisters_development.py --archives runs/assigned.zip \
  "$m/timing-descent-demonstrations/development.zip" \
  "$m/timing-randomized-descent/development.zip" --output runs/sequence-replay.zip
python scripts/train_sisters_memory_residual.py --archive runs/sequence-replay.zip \
  --parent "$m/timing-executed-movement/learner.pt" \
  --deployed "$m/timing-executed-movement/world_iql.bin" --output runs/reproduced-memory
```
