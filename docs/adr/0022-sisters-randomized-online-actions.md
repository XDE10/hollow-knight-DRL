# ADR 0022. Use real randomized inputs to refine the frozen Sisters actor

The offline IQL actors use a learned world model during value training. Their estimates for rarely tried actions are sensitive to the teacher and learner trajectories in the replay. The online experiment measures the consequences of two plausible inputs on the actual game PC. This is an optional training stage after the offline policy has learned to clear fights.

During development the opening network plays its usual greedy inputs. In the three-Sister phase the late network occasionally proposes its highest-scoring joint input and the highest-scoring input with a different movement, jump or dash. If their logit gap is at most three, a fair coin chooses the input for one native decision. The unchanged network controls subsequent decisions. Trials are at least 0.72 game seconds apart. The recorder stores the time, both action indices, the draw, its probability and the original logit gap. It verifies the action actually issued. The game runs at original speed and damage with zero teacher-controlled decisions.

For each draw the learner measures rewards in complete observation intervals inside the next 0.60 game seconds. An interval crossing the draw is excluded because its damage could have occurred before the input. The return includes a discounted continuation from the existing world-assisted IQL value network. Terminal states have zero continuation. Whole fights form separate training and holdout sets.

The policy-gradient update uses the logged assignment probability and retains a KL penalty to the parent actor. It fits a low-rank correction to movement logits using the last hidden layer. The correction folds into the existing native action head. Holdout selection estimates the change in return for the greedy policy over the full catalogue and assigns the worst bounded change when the proposed input falls outside the logged pair. Zero correction remains a candidate. These local return estimates guide development. A stronger policy still needs fresh full-game evidence.

Use `HKRL_NATIVE_ONLINE_CHOICES=1` on a timing-development worker with teacher ratio zero. Collect and pack a separate set of episodes from one frozen behavior checksum. The following command trains on the extracted development archive.

```sh
python scripts/train_sisters_online_choices.py --data online-data --parent parent/world_iql.bin --learner late/learner.pt --output online-candidate
```

Evaluation disables the recorder and randomized choices. The two original-condition acceptance cohorts remain defined in [the acceptance configuration](../../configs/experiments/sisters_acceptance.json).
