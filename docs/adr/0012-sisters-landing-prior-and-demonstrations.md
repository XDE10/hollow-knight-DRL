# ADR-0012. Landing-aware movement and demonstration replay for Sisters of Battle

Accepted for the Sisters experiment on 2026-11-18.

In the first survival experiment, the GRU lost its ability to release the direction key during PPO. Ten game evaluations recorded 70 HP lost to the spike strips and zero damage dealt. The previous offensive checkpoint retained reliable nail hits. Both checkpoints are retained for comparison.

The opt-in `sisters_landing_gru` adds a movement preference to the existing GRU distribution. It uses the two observed spike colliders, player position and velocity to reduce outward movement when a low or descending trajectory approaches either lip. At a safe height or in the arena center the preference is zero. All legal movements remain available. The game continues to mask unavailable actions, and the GRU selects jump and attack independently. Sampling, PPO log probabilities and deterministic evaluation use the same distribution. The learned parameter layout matches the offensive model, so its weights can initialize the experiment.

With `--demo-replay`, each recurrent PPO minibatch also supervises three sampled demonstration windows. The loss uses movement and the available jump and attack buttons. Positive button labels receive four times the weight of negative labels. Each window includes preceding episode-local context, and padding and warm-up frames have zero loss weight. Every fourth whole demonstration is held out. The manifest pins the bytes of the training episodes, and the checkpoint stores the sampler RNG.

The evaluation table in [the game report](../sisters_survival.md) compares the combined policy with both previous runs. It records actual HP changes and survival ticks. Starting from the offensive checkpoint, changing its reward, replaying demonstrations and adding the explicit spatial preference are a combined intervention. Isolating their individual contributions requires separate game evaluations.
