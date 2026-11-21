# ADR 0023. Learn when to switch between two frozen Sisters policies

The opening uses the shared learned policy. During the trio phase a learned linear head chooses one of two frozen neural controllers for the next 0.8 game seconds. Both controllers receive the observed key and entity histories. The selected controller produces the movement, aim and buttons locally. The frozen checkpoint contains the opening, both late-phase networks, the projection and all selector weights. The first, possibly partial trio block continues with the primary controller until the next scheduled choice. The game clock starts with the first observation of the episode.

The development experiment randomly assigned the controller every 0.8 game seconds, with equal probability. Nine entire fights trained the selector. Three other fights supplied the offline check. Each decision uses a projection of the primary encoder at the start of the block. Its training target uses the observed discounted block reward and a continuation from the primary world-assisted IQL value network. Ridge regressions estimate the two conditional returns, using whole-fight cross-fitting on the training split. A second ridge fit learns the return difference. At deployment the sign of that difference selects the controller.

The held-out report shows single-block gains evaluated on the randomized behavior's state distribution, for both observed reward alone and the critic-assisted target. It compares the learned selector with either fixed controller and with equal random assignment. Three held-out fights gave positive estimated measured-reward gains relative to the random assignment. In one of those fights always selecting the alternate controller had a higher estimated measured reward. Only independent game runs can establish a full-fight win rate. [Training and per-fight estimates](../../models/sisters-native-iql/timing-learned-options/training.json)

Run this command from the repository root. The small selector fits on CPU. Its two controllers and the primary value network came from earlier trained checkpoints.

```sh
m=models/sisters-native-iql
python scripts/train_sisters_option_router.py \
    --data "$m/timing-neural-blocks/development.zip" \
    --experiment "$m/timing-neural-blocks/experiment.json" \
    --primary "$m/timing-preferences/world_iql.bin" \
    --alternate "$m/timing-recovery-blocks/world_iql.bin" \
    --learner "$m/timing-preferences/learner.pt" \
    --output runs/reproduced-sisters-options
```

Replaying 12,448 decisions from the three held-out fights through the compiled game library reproduced the selected controllers and primitive keys at every decision. The independent evaluation binds both the composite checkpoint and the game library by SHA-256 and verifies seed receipts from Unity. [Runtime replay](../../models/sisters-native-iql/timing-learned-options/runtime-parity.json) · [Frozen original-condition plan](../../models/sisters-native-iql/timing-learned-options/goal1-plan.json)
