# ADR-0017 — Local trajectory planning with a learned Sisters movement head

Status: accepted for the opt-in Sisters actor.

## Context

The 51,000-step baseline did not complete the opening phase in ten original-condition encounters. Accelerated collectors and the original game sampled at different intervals under host load. A safe jump through a charge depends on both simulation time and the next InControl input commit.

## Decision

A dedicated game instance may enable the local actor with `HKRL_TACTICS_POLICY` at startup. It uses the ordinary player and entity observation collectors. In the Sisters scene it selects controller inputs after acknowledged InControl updates while an authenticated client is stepping task 2010. Standard clients retain the existing action path. Disconnect, timeout and task changes release the actor's controls.

The model learns preferences over twelve movement, jump and dash combinations from native input decisions and game outcomes. A receding-horizon controller also scores prospective motion against visible object trajectories. It plans charge and drop timing, tracks each disc's observed acceleration and considers a future direction change when escaping multiple sisters. Attack, spell and focus inputs use ordinary game abilities. The actor has no damage, health or arena-transition privileges.

Training copies can enable `HKRL_TACTICS_TRACE=1`. Each binary trace begins with HKS1, a little-endian episode identity and the input width. Every row contains the game time, 56 model features, movement axis, aim axis and selected button bits as float32. The parallel Gym replay holds the full emitted entity rows and game events. Its heartbeat action is a no-op. Actual decision labels come from the matching native trace.

## Acceptance

The first frozen policy, weights and original-condition ledger are in [models/sisters-tactics-v1](../../models/sisters-tactics-v1). Trials use the game-owned arena completion event, an alive player and isolated save slots. Health restoration during the post-victory scene return is excluded from combat healing.
