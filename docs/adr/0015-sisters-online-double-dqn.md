# ADR-0015 — Online Double DQN for Sisters of Battle

Accepted on 2026-11-19. The frozen recurrent PPO policy is the control.

The local policy selects among the 48 combinations of horizontal movement, jump
hold, nail direction and dash. The live component mask determines which choices
are legal on each decision. A held jump may span several decisions. A release
and a later press can trigger Monarch Wings. The wire protocol and other
policies retain their existing action layouts.

Each transition retains the privileged player and global observations, all 64
entity slots and their presence mask, the next observation, legal actions,
elapsed physics ticks, actual damage events and episode boundaries. A query
from the player and global encoders pools the visible entities. The model also
receives the existing landing features derived from the spike colliders. The
network learns how much to use them. The encoders exclude run identifiers from
the global features. The availability of an input field depends on the mod's
sensor implementation.

Three game instances run local inference and act at 2× speed. Each request
holds the chosen input for at least four physics ticks. Elapsed game ticks
include scheduling and transport gaps, so discounting uses the recorded tick
count. Every 2,500 real decisions, Colab draws uniform minibatches from all
completed shards. The online network selects the next legal action and a
separate target network values it. The value target follows the full fight
across damage events and stops at a true terminal state. A time-limit
truncation can bootstrap from its final observation. Replay consists entirely of
transitions collected by the successive online policies.

The frozen 50,000-step policy runs greedily on the original-speed slot-3
instance for ten complete fights. The historical PPO control used the same
save, difficulty and Python seeds. The game's internal random sequence and
the actual decision cadence can differ. The comparison measures the deployed
policies, including their action spaces and cadence.

## Reproducing the run

Configure the existing localhost game transport as described in
[ADR-0013](0013-local-parallel-sisters.md). Use a dedicated artifact directory
on each host. Run commands from the repository root, with the project's Python
package installed.

1. Run `scripts/probe_sisters_ddqn_controls.py --port 5555 --output controls.json`
   on the game PC and inspect the heights and action-state flags. Run
   `scripts/sisters_ddqn_collect.py bench --steps 900 --output bench.json`
   with the three local instances available.
2. Run `scripts/sisters_ddqn_collect.py collect --start 0 --steps 2500 --output chunk-01.npz`
   on the game PC. Send the shard to the learner and check its SHA-256 at
   both ends of the transfer.
3. On the learner, run `scripts/sisters_ddqn_learn.py --root . --settings configs/train/sisters_ddqn.yaml --round 1 --device cuda`.
   The policy ZIP contains its weights and SHA-256. The full learner checkpoint
   retains the online and target networks, optimizer and update count.
4. Return the verified weights to the game PC. For round 2 use
   `--start 2500 --checkpoint policy-01.pt --steps 2500 --output chunk-02.npz`.
   Continue with sequential rounds through 20, retaining all shards.
5. Close the two game copies, restore their original saves, and run
   `scripts/sisters_ddqn_collect.py eval --ports 5555 --start 50000 --checkpoint policy-20.pt --output evaluation.json`.
   Validate its replay with
   `scripts/score_sisters_eval.py evaluation.json evaluation.jsonl --output score.json`.
