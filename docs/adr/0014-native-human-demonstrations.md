# ADR-0014 — Native input capture for Sisters demonstrations

Accepted on 2026-11-19.

The existing Bilibili video provides tactical annotations. The scripted teacher provides state and action pairs but has never reached the simultaneous Sisters phase. Training a clone of human play requires the inputs consumed by the game together with the corresponding structured observations.

A local launcher arms one recording on a normal slot-3 arena reset and requests 0.65× speed. The authenticated local client then disconnects, which releases the input injector. During that episode the mod samples InControl bindings and the existing structured observation on each FixedUpdate. A local file next to the mod selects a fresh destination. Normal policy clients send and receive the existing protocol. A reconnect aborts an active capture. A completed or aborted recording restores normal speed.

The raw file starts with the four bytes `HKD1`. Each tick consists of a little-endian 16-bit native button word followed by an existing size-prefixed FlatBuffers StepResponse. From low to high the bits are left, right, down, up, jump, dash, attack, quick cast, focus/cast and dream nail. The observation and the input word are sampled on the same physics boundary. Events at the next boundary label the transition. The file is renamed from `.partial` only after a terminal or time-limit frame has been written.

Conversion preserves native keys and ticks alongside normalized observations, legal primitive labels and rewards. Attack and dash presses use rising edges. Jump can stay held across ticks. Button requests excluded by the action mask are counted. Full wins and other attempts are stored separately, with phase coverage and the arena victory event checked before admitting a demonstration to supervised training. Offline validation holds out whole fights. Original-speed play measures transfer across the recording and deployment cadences.

The encoder, mask layout and network schema are unchanged. Raw recordings retain the evidence needed to revisit the mapping of physical input to policy controls.
